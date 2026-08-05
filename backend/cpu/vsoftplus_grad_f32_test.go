package cpu

import (
	"math"
	"math/rand/v2"
	"testing"
)

// softplusGradRefF64 is the exact f64 softplus derivative gradient g·σ(x) (softplus'(x)=σ(x)).
func softplusGradRefF64(x, g float32) float64 {
	xd, gd := float64(x), float64(g)
	var s float64
	if xd >= 0 {
		s = 1 / (1 + math.Exp(-xd))
	} else {
		z := math.Exp(xd)
		s = z / (1 + z)
	}
	return gd * s
}

// TestSoftplusGradF32Accuracy is the verify-before-bench parity gate for the OpSoftplusBackward F32
// fast path: vsoftplusGradF32 (build-selected SIMD or scalar) vs the exact f64 g·σ(x).
func TestSoftplusGradF32Accuracy(t *testing.T) {
	rng := rand.New(rand.NewPCG(71, 73))
	check := func(xs, gs []float32, label string) {
		t.Helper()
		got := make([]float32, len(xs))
		vsoftplusGradF32(got, xs, gs)
		var maxAbs float64
		for i, x := range xs {
			w := softplusGradRefF64(x, gs[i])
			gv := float64(got[i])
			abs := math.Abs(gv - w)
			if abs > maxAbs {
				maxAbs = abs
			}
			if abs > 1e-6*math.Abs(float64(gs[i]))+1e-7+2e-4*math.Abs(w) {
				t.Fatalf("%s[%d] x=%g g=%g: got %g want %g (abs %g)", label, i, x, gs[i], gv, w, abs)
			}
		}
		t.Logf("%s: maxAbs %.2e (n=%d)", label, maxAbs, len(xs))
	}
	n := 4099 // not a multiple of 8 → exercises the SIMD tail too
	xs := make([]float32, n)
	gs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(rng.NormFloat64() * 4)
		gs[i] = float32(rng.NormFloat64())
	}
	check(xs, gs, "random")
	edge := []float32{0, 1e-20, -1e-20, 30, -30, 88, -88}
	eg := make([]float32, len(edge))
	for i := range eg {
		eg[i] = 1
	}
	check(edge, eg, "edge")
}

// vsoftplusGradSerialF32 is the serial scalar baseline (what OpSoftplusBackward F32 fell back to on
// the reference backend) — the A/B counterpart for the SIMD kernel.
func vsoftplusGradSerialF32(dst, x, g []float32) {
	for i, v := range x {
		dst[i] = float32(softplusGradRefF64(v, g[i]))
	}
}

func BenchmarkSoftplusGradF32_SIMD(b *testing.B) {
	const n = 512 * 5120 // Mamba Δ [B·L, d_inner]
	x, g, d := make([]float32, n), make([]float32, n), make([]float32, n)
	for i := range x {
		x[i] = float32(i%1000)/250 - 2
		g[i] = 1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vsoftplusGradF32(d, x, g)
	}
}

func BenchmarkSoftplusGradF32_Serial(b *testing.B) {
	const n = 512 * 5120
	x, g, d := make([]float32, n), make([]float32, n), make([]float32, n)
	for i := range x {
		x[i] = float32(i%1000)/250 - 2
		g[i] = 1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vsoftplusGradSerialF32(d, x, g)
	}
}
