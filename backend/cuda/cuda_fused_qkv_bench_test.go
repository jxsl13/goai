//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// BenchmarkFusedQKV isolates the "concat-fuse shared-LHS projections" lever (research L2). A transformer
// block's QKV projections share LHS h1 and its gate/up projections share LHS h2, so 3 (resp. 2) separate
// GEMMs can be one wider GEMM — fewer launches AND bigger N for better SM occupancy on GA106's 28 SMs.
// This benches the WHOLE projection cluster (forward + weight-grad + input-grad) unfused (15 GEMMs + 3
// adds) vs fused (6 wider GEMMs), all bf16-resident so the ONLY variable is GEMM shape/count.
func BenchmarkFusedQKV(b *testing.B) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const L, D, H = 512, 1024, 2816
	rng := rand.New(rand.NewSource(71))
	must := func(e error) {
		if e != nil {
			b.Fatal(e)
		}
	}
	bf := func(r, c int) *cuda.DeviceBf16 { // random bf16 buffer (via an f32 upload + convert)
		f, err := cuda.NewDeviceF32(r, c)
		must(err)
		h := make([]float32, r*c)
		for i := range h {
			h[i] = float32(rng.NormFloat64()) * 0.05
		}
		must(f.UploadF32(h))
		d, err := cuda.NewDeviceBf16(r, c)
		must(err)
		must(d.FromF32(f))
		f.Free()
		return d
	}
	nf := func(r, c int) *cuda.DeviceF32 { d, err := cuda.NewDeviceF32(r, c); must(err); return d }

	// Shared-LHS activations and their upstream grads (bf16-resident).
	h1, h2 := bf(L, D), bf(L, D)
	// Unfused weights + grads.
	Wq, Wk, Wv := bf(D, D), bf(D, D), bf(D, D)
	Wg, Wu := bf(D, H), bf(D, H)
	dQ, dK, dV := bf(L, D), bf(L, D), bf(L, D)
	dg, du := bf(L, H), bf(L, H)
	// Fused weights + grads (concatenated).
	Wqkv, WGU := bf(D, 3*D), bf(D, 2*H)
	dQKV, dGU := bf(L, 3*D), bf(L, 2*H)

	// Outputs (f32).
	Q, K, V := nf(L, D), nf(L, D), nf(L, D)
	g, u := nf(L, H), nf(L, H)
	QKV, GU := nf(L, 3*D), nf(L, 2*H)
	dWq, dWk, dWv := nf(D, D), nf(D, D), nf(D, D)
	dWg, dWu := nf(D, H), nf(D, H)
	dWqkv, dWGU := nf(D, 3*D), nf(D, 2*H)
	dh1, dh2, tmp1, tmp2 := nf(L, D), nf(L, D), nf(L, D), nf(L, D)

	unfused := func() {
		// forward: 3 + 2 projections
		must(cuda.MatMulBB(h1, Wq, Q))
		must(cuda.MatMulBB(h1, Wk, K))
		must(cuda.MatMulBB(h1, Wv, V))
		must(cuda.MatMulBB(h2, Wg, g))
		must(cuda.MatMulBB(h2, Wu, u))
		// weight-grad: xᵀ·dY
		must(cuda.MatMulBBGradW(h1, dQ, dWq))
		must(cuda.MatMulBBGradW(h1, dK, dWk))
		must(cuda.MatMulBBGradW(h1, dV, dWv))
		must(cuda.MatMulBBGradW(h2, dg, dWg))
		must(cuda.MatMulBBGradW(h2, du, dWu))
		// input-grad: dY·Wᵀ, accumulated into dh
		must(cuda.MatMulBBGradX(dQ, Wq, dh1))
		must(cuda.MatMulBBGradX(dK, Wk, tmp1))
		must(dh1.Add(tmp1))
		must(cuda.MatMulBBGradX(dV, Wv, tmp1))
		must(dh1.Add(tmp1))
		must(cuda.MatMulBBGradX(dg, Wg, dh2))
		must(cuda.MatMulBBGradX(du, Wu, tmp2))
		must(dh2.Add(tmp2))
	}
	fused := func() {
		must(cuda.MatMulBB(h1, Wqkv, QKV)) // fwd QKV
		must(cuda.MatMulBB(h2, WGU, GU))   // fwd gate/up
		must(cuda.MatMulBBGradW(h1, dQKV, dWqkv))
		must(cuda.MatMulBBGradW(h2, dGU, dWGU))
		must(cuda.MatMulBBGradX(dQKV, Wqkv, dh1))
		must(cuda.MatMulBBGradX(dGU, WGU, dh2))
	}
	run := func(name string, fn func()) {
		b.Run(name, func(b *testing.B) {
			fn()
			cuda.GraphSync()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fn()
			}
			cuda.GraphSync()
			b.StopTimer()
			b.ReportMetric(b.Elapsed().Seconds()*1e6/float64(b.N), "us/cluster")
		})
	}
	run("unfused", unfused)
	run("fused", fused)
}
