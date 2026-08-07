//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// wgemm parameterizes the block's GEMMs by whether the weight operand is cached in bf16. proj/gx take a
// weight index (0..6 = Wq,Wk,Wv,Wo,Wg,Wu,Wd); mm/gw/gxAct take two activations.
type wgemm struct {
	refresh func() // convert all f32 weights -> bf16 caches (once/step)
	proj    func(act *cuda.DeviceF32, wi int, out *cuda.DeviceF32) error
	gx      func(dY *cuda.DeviceF32, wi int, dX *cuda.DeviceF32) error
	mm      func(a, b, out *cuda.DeviceF32) error
	gw      func(x, dY, dW *cuda.DeviceF32) error
	gxAct   func(dY, act, dX *cuda.DeviceF32) error
}

// BenchmarkBf16WeightCache measures the transformer-block training step three ways — f32, bf16
// (re-converts weights every GEMM), and bf16-CACHED (converts each weight to bf16 once per step, then
// reuses it) — to quantify the weight-cache win the e2e bench (#1038) pointed at.
func BenchmarkBf16WeightCache(b *testing.B) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const L, D, H = 512, 1024, 2816
	const eps = 1e-5
	scale := float32(1.0 / math.Sqrt(float64(D)))
	rng := rand.New(rand.NewSource(53))
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
	nf := func(r, c int) *cuda.DeviceF32 { d, err := cuda.NewDeviceF32(r, c); must(err); return d }

	x, target := rm(L, D, 1), rm(L, D, 1)
	// weights[0..6] = Wq,Wk,Wv,Wo (D×D), Wg,Wu (D×H), Wd (H×D)
	W := []*cuda.DeviceF32{rm(D, D, 0.05), rm(D, D, 0.05), rm(D, D, 0.05), rm(D, D, 0.05), rm(D, H, 0.05), rm(D, H, 0.05), rm(H, D, 0.05)}
	g1, g2 := rm(1, D, 0), rm(1, D, 0)
	g1.UploadF32(onesF(D))
	g2.UploadF32(onesF(D))
	params := append(append([]*cuda.DeviceF32{}, W...), g1, g2)

	h1, Q, K, V, S := nf(L, D), nf(L, D), nf(L, D), nf(L, D), nf(L, L)
	a, y, h2, out := nf(L, D), nf(L, D), nf(L, D), nf(L, D)
	gate, up, ff := nf(L, H), nf(L, H), nf(L, H)
	dout, dy, dA := nf(L, D), nf(L, D), nf(L, D)
	dQ, dK, dV, dP, dS := nf(L, D), nf(L, D), nf(L, D), nf(L, L), nf(L, L)
	dh1, dh2, tmp, dxN := nf(L, D), nf(L, D), nf(L, D), nf(L, D)
	dff, dgate, dup := nf(L, H), nf(L, H), nf(L, H)
	dW := []*cuda.DeviceF32{nf(D, D), nf(D, D), nf(D, D), nf(D, D), nf(D, H), nf(D, H), nf(H, D)}
	dg1, dg2 := nf(1, D), nf(1, D)
	grads := append(append([]*cuda.DeviceF32{}, dW...), dg1, dg2)
	opt, err := cuda.NewDeviceAdam([]int{D * D, D * D, D * D, D * D, D * H, D * H, H * D, D, D}, 1e-3, 0.9, 0.999, 1e-8, 0)
	must(err)
	defer opt.Free()

	// bf16 weight caches (for the cached mode).
	Wc := make([]*cuda.DeviceBf16, len(W))
	for i, w := range W {
		Wc[i], err = cuda.NewDeviceBf16(w.Rows(), w.Cols())
		must(err)
		defer Wc[i].Free()
	}

	step := func(g wgemm) {
		if g.refresh != nil {
			g.refresh()
		}
		must(cuda.RMSNormForward(h1, x, g1, eps))
		must(g.proj(h1, 0, Q))
		must(g.proj(h1, 1, K))
		must(g.proj(h1, 2, V))
		must(g.gxAct(Q, K, S)) // S = Q·Kᵀ
		must(cuda.Scale(S, scale))
		must(S.Softmax())
		must(g.mm(S, V, a))
		must(g.proj(a, 3, y)) // Wo
		must(y.Add(x))
		must(cuda.RMSNormForward(h2, y, g2, eps))
		must(g.proj(h2, 4, gate))
		must(g.proj(h2, 5, up))
		must(cuda.SwiGLUForward(ff, gate, up))
		must(g.proj(ff, 6, out)) // Wd
		must(out.Add(y))
		must(cuda.SubScaled(dout, out, target, 2.0/float32(L*D)))
		must(g.gx(dout, 6, dff))
		must(g.gw(ff, dout, dW[6]))
		must(cuda.SwiGLUBackward(dgate, dup, gate, up, dff))
		must(g.gx(dgate, 4, dh2))
		must(g.gx(dup, 5, tmp))
		must(dh2.Add(tmp))
		must(g.gw(h2, dgate, dW[4]))
		must(g.gw(h2, dup, dW[5]))
		must(cuda.RMSNormBackward(dy, dg2, y, dh2, g2, eps))
		must(dy.Add(dout))
		must(g.gx(dy, 3, dA))
		must(g.gw(a, dy, dW[3]))
		must(g.gw(S, dA, dV))    // dV = Pᵀ·dO
		must(g.gxAct(dA, V, dP)) // dP = dO·Vᵀ
		must(cuda.SoftmaxBackward(dS, S, dP))
		must(cuda.Scale(dS, scale))
		must(g.mm(dS, K, dQ))
		must(g.gw(dS, Q, dK))
		must(g.gx(dQ, 0, dh1))
		must(g.gx(dK, 1, tmp))
		must(dh1.Add(tmp))
		must(g.gx(dV, 2, tmp))
		must(dh1.Add(tmp))
		must(g.gw(h1, dQ, dW[0]))
		must(g.gw(h1, dK, dW[1]))
		must(g.gw(h1, dV, dW[2]))
		must(cuda.RMSNormBackward(dxN, dg1, x, dh1, g1, eps))
		must(opt.Step(params, grads))
	}

	f32 := wgemm{
		proj: func(act *cuda.DeviceF32, wi int, o *cuda.DeviceF32) error { return cuda.MatMul(act, W[wi], o) },
		gx:   func(dY *cuda.DeviceF32, wi int, dX *cuda.DeviceF32) error { return cuda.MatMulGradX(dY, W[wi], dX) },
		mm:   cuda.MatMul, gw: cuda.MatMulGradW, gxAct: cuda.MatMulGradX,
	}
	bf16 := wgemm{
		proj: func(act *cuda.DeviceF32, wi int, o *cuda.DeviceF32) error { return cuda.MatMulBf16(act, W[wi], o) },
		gx:   func(dY *cuda.DeviceF32, wi int, dX *cuda.DeviceF32) error { return cuda.MatMulGradXBf16(dY, W[wi], dX) },
		mm:   cuda.MatMulBf16, gw: cuda.MatMulGradWBf16, gxAct: cuda.MatMulGradXBf16,
	}
	cached := wgemm{
		refresh: func() {
			for i, w := range W {
				must(Wc[i].FromF32(w))
			}
		},
		proj: func(act *cuda.DeviceF32, wi int, o *cuda.DeviceF32) error { return cuda.MatMulWBf16(act, Wc[wi], o) },
		gx: func(dY *cuda.DeviceF32, wi int, dX *cuda.DeviceF32) error {
			return cuda.MatMulGradXWBf16(dY, Wc[wi], dX)
		},
		mm: cuda.MatMulBf16, gw: cuda.MatMulGradWBf16, gxAct: cuda.MatMulGradXBf16,
	}
	run := func(name string, g wgemm) {
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
	run("bf16cached", cached)
}

func onesF(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = 1
	}
	return s
}
