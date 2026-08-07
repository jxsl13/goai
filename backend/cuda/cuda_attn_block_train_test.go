//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestAttentionBlockTrainsOnGPU trains a single-head pre-norm attention block ENTIRELY on the GPU and
// checks the loss collapses — the capstone proof that the composed GPU backward (RMSNorm + attention +
// residual + matmul projections) and DeviceAdam work together:
//
//	h  = RMSNorm(x, g1)
//	Q,K,V = h·Wq, h·Wk, h·Wv
//	a  = softmax((Q·Kᵀ)/√d)·V
//	y  = x + a·Wo               (residual)
//	loss = MSE(y, target)
//
// Backward flows through every one of those ops on device; DeviceAdam updates Wq,Wk,Wv,Wo,g1. If any VJP
// or the wiring were wrong the loss would not drop.
func TestAttentionBlockTrainsOnGPU(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const L, D, steps = 16, 32, 600
	const eps = 1e-5
	scale := float32(1.0 / math.Sqrt(float64(D)))
	rng := rand.New(rand.NewSource(29))

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
	g1 := rm(1, D, one)
	defer x.Free()
	defer target.Free()
	for _, w := range []*cuda.DeviceF32{Wq, Wk, Wv, Wo, g1} {
		defer w.Free()
	}

	// Activations / scratch (reused each step).
	h := n(L, D)
	Q, K, V := n(L, D), n(L, D), n(L, D)
	S := n(L, L)
	a := n(L, D)
	y := n(L, D)
	dy := n(L, D)
	dA := n(L, D)
	dQ, dK, dV := n(L, D), n(L, D), n(L, D)
	dh := n(L, D)
	tmp := n(L, D)
	// Grads for the 5 params.
	dWq, dWk, dWv, dWo := n(D, D), n(D, D), n(D, D), n(D, D)
	dg1 := n(1, D)
	dxNorm := n(L, D) // RMSNorm dx (unused input grad, but the kernel writes it)
	for _, b := range []*cuda.DeviceF32{h, Q, K, V, S, a, y, dy, dA, dQ, dK, dV, dh, tmp, dWq, dWk, dWv, dWo, dg1, dxNorm} {
		defer b.Free()
	}

	opt, err := cuda.NewDeviceAdam([]int{D * D, D * D, D * D, D * D, D}, 5e-3, 0.9, 0.999, 1e-8, 0)
	must(err)
	defer opt.Free()

	hostY := make([]float32, L*D)
	hostT := make([]float32, L*D)
	must(target.DownloadF32(hostT))
	loss := func() float64 {
		must(y.DownloadF32(hostY))
		var s float64
		for i := range hostY {
			d := float64(hostY[i] - hostT[i])
			s += d * d
		}
		return s / float64(L*D)
	}

	forward := func() {
		must(cuda.RMSNormForward(h, x, g1, eps)) // h = RMSNorm(x,g1)
		must(cuda.MatMul(h, Wq, Q))
		must(cuda.MatMul(h, Wk, K))
		must(cuda.MatMul(h, Wv, V))
		must(cuda.MatMulGradX(Q, K, S)) // S = Q·Kᵀ
		must(cuda.Scale(S, scale))
		must(S.Softmax())           // S = softmax(S) row-wise
		must(cuda.MatMul(S, V, a))  // a = P·V
		must(cuda.MatMul(a, Wo, y)) // y = a·Wo
		must(y.Add(x))              // y = x + a·Wo  (residual)
	}

	forward()
	l0 := loss()
	for step := 0; step < steps; step++ {
		forward()
		// dy = (2/(L*D))(y - target)
		must(cuda.SubScaled(dy, y, target, 2.0/float32(L*D)))
		// o-proj: aProj = a·Wo ; dA = dy·Woᵀ ; dWo = aᵀ·dy
		must(cuda.MatMulGradX(dy, Wo, dA))
		must(cuda.MatMulGradW(a, dy, dWo))
		// attention backward: dQ,dK,dV from dA
		must(cuda.AttentionBackward(dQ, dK, dV, Q, K, V, dA, scale))
		// QKV proj: dh = dQ·Wqᵀ + dK·Wkᵀ + dV·Wvᵀ ; dW* = hᵀ·d*
		must(cuda.MatMulGradX(dQ, Wq, dh))
		must(cuda.MatMulGradX(dK, Wk, tmp))
		must(dh.Add(tmp))
		must(cuda.MatMulGradX(dV, Wv, tmp))
		must(dh.Add(tmp))
		must(cuda.MatMulGradW(h, dQ, dWq))
		must(cuda.MatMulGradW(h, dK, dWk))
		must(cuda.MatMulGradW(h, dV, dWv))
		// RMSNorm backward: dg1 (dh -> dxNorm discarded; residual dx not trained)
		must(cuda.RMSNormBackward(dxNorm, dg1, x, dh, g1, eps))
		// Adam step over Wq,Wk,Wv,Wo,g1
		must(opt.Step(
			[]*cuda.DeviceF32{Wq, Wk, Wv, Wo, g1},
			[]*cuda.DeviceF32{dWq, dWk, dWv, dWo, dg1},
		))
	}
	forward()
	l1 := loss()

	t.Logf("attention block GPU training: MSE %.4e -> %.4e over %d steps (all on device)", l0, l1, steps)
	if l1 > l0*0.5 {
		t.Fatalf("attention block did not train on GPU: loss %.4e -> %.4e (want < %.4e)", l0, l1, l0*0.5)
	}
}
