//go:build darwin && arm64 && goexperiment.simd

package cpu

import (
	"math"
	"math/rand"
	"testing"
)

// gemmF32Ref is the f64-accumulating scalar reference (the §V10 semantics the
// tolerant policy measures against).
func gemmF32Ref(A, B []float32, m, k, n int) []float64 {
	ref := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for p := 0; p < k; p++ {
			a := float64(A[i*k+p])
			for j := 0; j < n; j++ {
				ref[i*n+j] += a * float64(B[p*n+j])
			}
		}
	}
	return ref
}

func checkTolerance(t *testing.T, got []float32, ref []float64, k int) {
	t.Helper()
	const rtol, atol = 2e-3, 1e-4 // ADR-0021 tolerance contract
	worst := 0.0
	for i := range got {
		diff := math.Abs(float64(got[i]) - ref[i])
		lim := atol + rtol*math.Abs(ref[i])
		if diff > lim {
			t.Fatalf("elem %d: got %v want %v (diff %g > %g, k=%d)", i, got[i], ref[i], diff, lim, k)
		}
		if r := diff / math.Max(math.Abs(ref[i]), 1e-30); r > worst {
			worst = r
		}
	}
	t.Logf("max rel err %.3g", worst)
}

// TestGemmAMXMatchesReference exercises the raw-AMX path (ADR-0027 Path B)
// against the f64 reference across the tile-boundary shapes: 32-aligned
// interiors, m%32 / n%32 NEON tails, odd k (the pair-load remainder step),
// and k=1 (the do-while edge).
func TestGemmAMXMatchesReference(t *testing.T) {
	if gemmF32AMX == nil {
		t.Skip("no AMX on this host (needs Apple M1/M2/M3)")
	}
	shapes := []struct{ m, k, n int }{
		{32, 1, 32}, {32, 2, 32}, {32, 3, 32}, {32, 64, 32},
		{64, 33, 96}, {96, 128, 64}, {33, 65, 47}, {61, 100, 130},
		{32, 511, 32}, {128, 257, 224}, {40, 96, 40}, {160, 320, 288},
	}
	for _, s := range shapes {
		rng := rand.New(rand.NewSource(int64(s.m*1000003 + s.k*1009 + s.n)))
		A := make([]float32, s.m*s.k)
		B := make([]float32, s.k*s.n)
		for i := range A {
			A[i] = rng.Float32()*2 - 1
		}
		for i := range B {
			B[i] = rng.Float32()*2 - 1
		}
		C := make([]float32, s.m*s.n)
		for i := range C {
			C[i] = float32(math.NaN()) // store semantics: every element written
		}
		if !gemmF32AMXCompute(A, B, C, s.m, s.k, s.n) {
			t.Fatalf("%dx%dx%d: AMX path unexpectedly punted", s.m, s.k, s.n)
		}
		checkTolerance(t, C, gemmF32Ref(A, B, s.m, s.k, s.n), s.k)
	}
}
