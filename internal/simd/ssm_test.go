package simd

import (
	"math"
	"math/rand"
	"testing"
)

// ssmInputs builds a well-conditioned S6 problem: A<0 (stability ⟹ exp arg ≤ 0), the
// other tensors standard-normal. Returns (u, delta>0, A, B, C, dsk).
func ssmInputs(rng *rand.Rand, L, D, N int, skip bool) (u, delta, a, b, c, dsk []float64) {
	u = make([]float64, L*D)
	delta = make([]float64, L*D)
	a = make([]float64, D*N)
	b = make([]float64, L*N)
	c = make([]float64, L*N)
	for i := range u {
		u[i] = rng.NormFloat64()
		delta[i] = 0.1 + 0.5*rng.Float64() // Δ>0 (softplus output)
	}
	for i := range a {
		a[i] = -0.1 - rng.Float64() // A<0
	}
	for i := range b {
		b[i] = rng.NormFloat64()
		c[i] = rng.NormFloat64()
	}
	if skip {
		dsk = make([]float64, D)
		for i := range dsk {
			dsk[i] = rng.NormFloat64()
		}
	}
	return
}

// TestSSMScanF64Accuracy checks the state-dim-vectorized scan against the scalar
// reference over the full recurrence (error accumulates across L steps). N=17 exercises
// the scalar state-dim tail; the D-skip path is covered both ways.
func TestSSMScanF64Accuracy(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, dims := range [][3]int{{64, 128, 16}, {32, 64, 128}, {40, 48, 17}, {50, 32, 4}} {
		L, D, N := dims[0], dims[1], dims[2]
		for _, skip := range []bool{false, true} {
			u, delta, a, b, c, dsk := ssmInputs(rng, L, D, N, skip)
			got := make([]float64, L*D)
			hGot := make([]float64, D*N)
			SSMScanF64(u, delta, a, b, c, dsk, got, hGot, L, D, N)
			want := make([]float64, L*D)
			hWant := make([]float64, D*N)
			ssmScanScalar(u, delta, a, b, c, dsk, want, hWant, L, D, N, 0, N)
			var maxRel float64
			for i := range got {
				den := math.Max(1e-6, math.Abs(want[i]))
				if rel := math.Abs(got[i]-want[i]) / den; rel > maxRel {
					maxRel = rel
				}
			}
			t.Logf("L=%d D=%d N=%d skip=%v maxRel=%.3e", L, D, N, skip, maxRel)
			if maxRel > 1e-10 {
				t.Fatalf("L=%d D=%d N=%d skip=%v: SSMScanF64 maxRel=%.3e exceeds 1e-10", L, D, N, skip, maxRel)
			}
		}
	}
}

func benchSSMScan(b *testing.B, L, D, N int, useSimd bool) {
	rng := rand.New(rand.NewSource(1))
	u, delta, a, bb, c, _ := ssmInputs(rng, L, D, N, false)
	out := make([]float64, L*D)
	h := make([]float64, D*N)
	b.ResetTimer()
	for range b.N {
		for i := range h {
			h[i] = 0
		}
		if useSimd {
			SSMScanF64(u, delta, a, bb, c, nil, out, h, L, D, N)
		} else {
			ssmScanScalar(u, delta, a, bb, c, nil, out, h, L, D, N, 0, N)
		}
	}
}

func BenchmarkSSMScan_SIMD_512x2048x16(b *testing.B)    { benchSSMScan(b, 512, 2048, 16, true) }
func BenchmarkSSMScan_Scalar_512x2048x16(b *testing.B)  { benchSSMScan(b, 512, 2048, 16, false) }
func BenchmarkSSMScan_SIMD_512x2048x128(b *testing.B)   { benchSSMScan(b, 512, 2048, 128, true) }
func BenchmarkSSMScan_Scalar_512x2048x128(b *testing.B) { benchSSMScan(b, 512, 2048, 128, false) }
