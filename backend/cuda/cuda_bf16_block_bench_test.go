//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

type gemmFns struct {
	mm    func(a, b, out *cuda.DeviceF32) error // out = a·b
	gradW func(x, dY, dW *cuda.DeviceF32) error // dW = xᵀ·dY
	gradX func(dY, w, dX *cuda.DeviceF32) error // dX = dY·wᵀ
}

// BenchmarkTransformerBlockTrainStep_f32_vs_bf16 measures the END-TO-END training step (forward +
// backward + AdamW) of a single-head transformer block at realistic dims, with the GEMMs run in f32 vs
// bf16 tensor cores (elementwise norm/SwiGLU/softmax stay f32). This is the "how much faster is bf16
// training" number that quantifies the whole bf16 recipe (#1036/#1037) at the model level.
func BenchmarkTransformerBlockTrainStep_f32_vs_bf16(b *testing.B) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const L, D, H = 512, 1024, 2816
	const eps = 1e-5
	scale := float32(1.0 / math.Sqrt(float64(D)))
	rng := rand.New(rand.NewSource(51))
	must := func(e error) {
		if e != nil {
			b.Fatal(e)
		}
	}
	rm := func(r, c int, s float32) *cuda.DeviceF32 {
		d, err := cuda.NewDeviceF32(r, c)
		must(err)
		h := make([]float32, r*c)
		for i := range h {
			h[i] = float32(rng.NormFloat64()) * s
		}
		must(d.UploadF32(h))
		return d
	}
	n := func(r, c int) *cuda.DeviceF32 { d, err := cuda.NewDeviceF32(r, c); must(err); return d }

	x, target := rm(L, D, 1), rm(L, D, 1)
	Wq, Wk, Wv, Wo := rm(D, D, 0.05), rm(D, D, 0.05), rm(D, D, 0.05), rm(D, D, 0.05)
	Wg, Wu, Wd := rm(D, H, 0.05), rm(D, H, 0.05), rm(H, D, 0.05)
	g1, g2 := rm(1, D, 0), rm(1, D, 0)
	g1.UploadF32(ones(D))
	g2.UploadF32(ones(D))
	params := []*cuda.DeviceF32{Wq, Wk, Wv, Wo, Wg, Wu, Wd, g1, g2}

	h1, Q, K, V, S := n(L, D), n(L, D), n(L, D), n(L, D), n(L, L)
	a, y, h2, out := n(L, D), n(L, D), n(L, D), n(L, D)
	gate, up, ff := n(L, H), n(L, H), n(L, H)
	dout, dy, dA := n(L, D), n(L, D), n(L, D)
	dQ, dK, dV, dP, dS := n(L, D), n(L, D), n(L, D), n(L, L), n(L, L)
	dh1, dh2, tmp, dxN := n(L, D), n(L, D), n(L, D), n(L, D)
	dff, dgate, dup := n(L, H), n(L, H), n(L, H)
	dWq, dWk, dWv, dWo := n(D, D), n(D, D), n(D, D), n(D, D)
	dWg, dWu, dWd := n(D, H), n(D, H), n(H, D)
	dg1, dg2 := n(1, D), n(1, D)
	grads := []*cuda.DeviceF32{dWq, dWk, dWv, dWo, dWg, dWu, dWd, dg1, dg2}
	opt, err := cuda.NewDeviceAdam([]int{D * D, D * D, D * D, D * D, D * H, D * H, H * D, D, D}, 1e-3, 0.9, 0.999, 1e-8, 0)
	must(err)
	defer opt.Free()

	step := func(g gemmFns) {
		// forward
		must(cuda.RMSNormForward(h1, x, g1, eps))
		must(g.mm(h1, Wq, Q))
		must(g.mm(h1, Wk, K))
		must(g.mm(h1, Wv, V))
		must(g.gradX(Q, K, S)) // S = Q·Kᵀ
		must(cuda.Scale(S, scale))
		must(S.Softmax())
		must(g.mm(S, V, a)) // a = P·V
		must(g.mm(a, Wo, y))
		must(y.Add(x))
		must(cuda.RMSNormForward(h2, y, g2, eps))
		must(g.mm(h2, Wg, gate))
		must(g.mm(h2, Wu, up))
		must(cuda.SwiGLUForward(ff, gate, up))
		must(g.mm(ff, Wd, out))
		must(out.Add(y))
		// backward
		must(cuda.SubScaled(dout, out, target, 2.0/float32(L*D)))
		must(g.gradX(dout, Wd, dff))
		must(g.gradW(ff, dout, dWd))
		must(cuda.SwiGLUBackward(dgate, dup, gate, up, dff))
		must(g.gradX(dgate, Wg, dh2))
		must(g.gradX(dup, Wu, tmp))
		must(dh2.Add(tmp))
		must(g.gradW(h2, dgate, dWg))
		must(g.gradW(h2, dup, dWu))
		must(cuda.RMSNormBackward(dy, dg2, y, dh2, g2, eps))
		must(dy.Add(dout))
		must(g.gradX(dy, Wo, dA))
		must(g.gradW(a, dy, dWo))
		// attention backward (inline, using P=S)
		must(g.gradW(S, dA, dV)) // dV = Pᵀ·dO
		must(g.gradX(dA, V, dP)) // dP = dO·Vᵀ
		must(cuda.SoftmaxBackward(dS, S, dP))
		must(cuda.Scale(dS, scale))
		must(g.mm(dS, K, dQ))    // dQ = dS·K
		must(g.gradW(dS, Q, dK)) // dK = dSᵀ·Q
		must(g.gradX(dQ, Wq, dh1))
		must(g.gradX(dK, Wk, tmp))
		must(dh1.Add(tmp))
		must(g.gradX(dV, Wv, tmp))
		must(dh1.Add(tmp))
		must(g.gradW(h1, dQ, dWq))
		must(g.gradW(h1, dK, dWk))
		must(g.gradW(h1, dV, dWv))
		must(cuda.RMSNormBackward(dxN, dg1, x, dh1, g1, eps))
		must(opt.Step(params, grads))
	}

	f32 := gemmFns{cuda.MatMul, cuda.MatMulGradW, cuda.MatMulGradX}
	bf16 := gemmFns{cuda.MatMulBf16, cuda.MatMulGradWBf16, cuda.MatMulGradXBf16}
	run := func(name string, g gemmFns) {
		b.Run(name, func(b *testing.B) {
			step(g)
			cuda.GraphSync()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				step(g)
			}
			cuda.GraphSync()
			b.StopTimer()
			b.ReportMetric(b.Elapsed().Seconds()*1e3/float64(b.N), "ms/step")
		})
	}
	run("f32", f32)
	run("bf16", bf16)
}

func ones(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = 1
	}
	return s
}
