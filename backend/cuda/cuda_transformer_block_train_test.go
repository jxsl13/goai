//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestTransformerBlockTrainsOnGPU trains a COMPLETE pre-norm transformer block (attention + SwiGLU FFN,
// two RMSNorms, two residuals) entirely on the GPU and checks the loss collapses — the full-block proof:
//
//	y   = x + Attn(RMSNorm(x, g1))·Wo
//	out = y + (SwiGLU(RMSNorm(y,g2)·Wg, RMSNorm(y,g2)·Wu))·Wd
//	loss = MSE(out, target)
//
// Backward runs on device through both residual branches — attention (RMSNorm + softmax attention) and
// the SwiGLU FFN — and DeviceAdam updates all nine parameter groups. Convergence to ~0 proves the whole
// block's composed backward is correct, including SwiGLU backward in a training loop.
func TestTransformerBlockTrainsOnGPU(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const L, D, H, steps = 16, 32, 64, 800
	const eps = 1e-5
	scale := float32(1.0 / math.Sqrt(float64(D)))
	rng := rand.New(rand.NewSource(37))
	must := func(err error) {
		if err != nil {
			t.Fatal(err)
		}
	}
	rm := func(r, c int, f func() float32) *cuda.DeviceF32 {
		d, err := cuda.NewDeviceF32(r, c)
		must(err)
		h := make([]float32, r*c)
		for i := range h {
			h[i] = f()
		}
		must(d.UploadF32(h))
		return d
	}
	n := func(r, c int) *cuda.DeviceF32 { d, err := cuda.NewDeviceF32(r, c); must(err); return d }
	small := func() float32 { return float32(rng.NormFloat64() * 0.1) }
	one := func() float32 { return 1 }

	x := rm(L, D, func() float32 { return float32(rng.NormFloat64()) })
	target := rm(L, D, func() float32 { return float32(rng.NormFloat64()) })
	Wq, Wk, Wv, Wo := rm(D, D, small), rm(D, D, small), rm(D, D, small), rm(D, D, small)
	Wg, Wu := rm(D, H, small), rm(D, H, small)
	Wd := rm(H, D, small)
	g1, g2 := rm(1, D, one), rm(1, D, one)
	params := []*cuda.DeviceF32{Wq, Wk, Wv, Wo, Wg, Wu, Wd, g1, g2}
	defer x.Free()
	defer target.Free()
	for _, p := range params {
		defer p.Free()
	}

	h1, Q, K, V := n(L, D), n(L, D), n(L, D), n(L, D)
	S := n(L, L)
	a, y, h2, out := n(L, D), n(L, D), n(L, D), n(L, D)
	gate, up, ff := n(L, H), n(L, H), n(L, H)
	dout, dy, dA := n(L, D), n(L, D), n(L, D)
	dQ, dK, dV := n(L, D), n(L, D), n(L, D)
	dh1, dh2, tmp := n(L, D), n(L, D), n(L, D)
	dff, dgate, dup, tmpH := n(L, H), n(L, H), n(L, H), n(L, H)
	dxN := n(L, D)
	dWq, dWk, dWv, dWo := n(D, D), n(D, D), n(D, D), n(D, D)
	dWg, dWu := n(D, H), n(D, H)
	dWd := n(H, D)
	dg1, dg2 := n(1, D), n(1, D)
	grads := []*cuda.DeviceF32{dWq, dWk, dWv, dWo, dWg, dWu, dWd, dg1, dg2}
	for _, b := range append([]*cuda.DeviceF32{h1, Q, K, V, S, a, y, h2, out, gate, up, ff, dout, dy, dA, dQ, dK, dV, dh1, dh2, tmp, dff, dgate, dup, tmpH, dxN}, grads...) {
		defer b.Free()
	}

	opt, err := cuda.NewDeviceAdam([]int{D * D, D * D, D * D, D * D, D * H, D * H, H * D, D, D}, 3e-3, 0.9, 0.999, 1e-8, 0)
	must(err)
	defer opt.Free()

	hostO := make([]float32, L*D)
	hostT := make([]float32, L*D)
	must(target.DownloadF32(hostT))
	loss := func() float64 {
		must(out.DownloadF32(hostO))
		var s float64
		for i := range hostO {
			d := float64(hostO[i] - hostT[i])
			s += d * d
		}
		return s / float64(L*D)
	}

	forward := func() {
		must(cuda.RMSNormForward(h1, x, g1, eps))
		must(cuda.MatMul(h1, Wq, Q))
		must(cuda.MatMul(h1, Wk, K))
		must(cuda.MatMul(h1, Wv, V))
		must(cuda.MatMulGradX(Q, K, S))
		must(cuda.Scale(S, scale))
		must(S.Softmax())
		must(cuda.MatMul(S, V, a))
		must(cuda.MatMul(a, Wo, y))
		must(y.Add(x)) // residual 1
		must(cuda.RMSNormForward(h2, y, g2, eps))
		must(cuda.MatMul(h2, Wg, gate))
		must(cuda.MatMul(h2, Wu, up))
		must(cuda.SwiGLUForward(ff, gate, up))
		must(cuda.MatMul(ff, Wd, out))
		must(out.Add(y)) // residual 2
	}

	forward()
	l0 := loss()
	for step := 0; step < steps; step++ {
		forward()
		must(cuda.SubScaled(dout, out, target, 2.0/float32(L*D)))
		// FFN branch: out = y + ff·Wd ; dffOut = dout
		must(cuda.MatMulGradX(dout, Wd, dff)) // dff = dout·Wdᵀ  [L,H]
		must(cuda.MatMulGradW(ff, dout, dWd)) // dWd = ffᵀ·dout   [H,D]
		must(cuda.SwiGLUBackward(dgate, dup, gate, up, dff))
		must(cuda.MatMulGradX(dgate, Wg, dh2))
		must(cuda.MatMulGradX(dup, Wu, tmp))
		must(dh2.Add(tmp))
		must(cuda.MatMulGradW(h2, dgate, dWg))
		must(cuda.MatMulGradW(h2, dup, dWu))
		// RMSNorm2: dy = norm2_dx + dout (residual)
		must(cuda.RMSNormBackward(dy, dg2, y, dh2, g2, eps))
		must(dy.Add(dout))
		// attention branch: y = x + a·Wo ; daProj = dy
		must(cuda.MatMulGradX(dy, Wo, dA))
		must(cuda.MatMulGradW(a, dy, dWo))
		must(cuda.AttentionBackward(dQ, dK, dV, Q, K, V, dA, scale))
		must(cuda.MatMulGradX(dQ, Wq, dh1))
		must(cuda.MatMulGradX(dK, Wk, tmp))
		must(dh1.Add(tmp))
		must(cuda.MatMulGradX(dV, Wv, tmp))
		must(dh1.Add(tmp))
		must(cuda.MatMulGradW(h1, dQ, dWq))
		must(cuda.MatMulGradW(h1, dK, dWk))
		must(cuda.MatMulGradW(h1, dV, dWv))
		must(cuda.RMSNormBackward(dxN, dg1, x, dh1, g1, eps)) // dxN discarded (x fixed)
		must(opt.Step(params, grads))
	}
	forward()
	l1 := loss()

	t.Logf("transformer block GPU training: MSE %.4e -> %.4e over %d steps (all on device)", l0, l1, steps)
	if l1 > l0*0.2 {
		t.Fatalf("transformer block did not train on GPU: MSE %.4e -> %.4e (want < %.4e)", l0, l1, l0*0.2)
	}
}
