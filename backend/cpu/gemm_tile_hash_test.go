//go:build !(amd64 && goexperiment.simd) && !(arm64 && goexperiment.simd)

package cpu

import (
	"math"
	"math/rand"
	"testing"
)

// TestGemmF32PortableHash freezes the exact bits of the portable f32 GEMM across geometries that
// exercise BOTH the 4x4 register tile and every remainder it can leave: n not divisible by 4 takes
// the column tail, and m not divisible by 4 takes the single-row tail.
func TestGemmF32PortableHash(t *testing.T) {
	const wantHash uint64 = 0xad374cc7cd2a5ae4
	var h uint64 = 14695981039346656037
	for _, g := range []struct{ m, k, n int }{
		{8, 8, 8}, {7, 5, 9}, {4, 3, 6}, {13, 11, 15}, {1, 1, 1}, {5, 4, 3},
	} {
		rng := rand.New(rand.NewSource(int64(g.m*100 + g.n)))
		A := make([]float32, g.m*g.k)
		B := make([]float32, g.k*g.n)
		C := make([]float32, g.m*g.n)
		for i := range A {
			A[i] = rng.Float32()*2 - 1
		}
		for i := range B {
			B[i] = rng.Float32()*2 - 1
		}
		gemmF32(A, B, C, g.m, g.k, g.n)
		for _, v := range C {
			h = (h ^ uint64(math.Float32bits(v))) * 1099511628211
		}
	}
	if wantHash == 0 {
		t.Fatalf("CAPTURE: %#x", h)
	}
	if h != wantHash {
		t.Fatalf("portable gemmF32 hash %#x, want %#x", h, wantHash)
	}
}
