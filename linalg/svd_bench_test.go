package linalg_test

import (
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkSVDPCA times the one-sided Jacobi SVD on a tall [2000,50] matrix — the
// PCA / low-rank regime (m≫n) where the working-buffer column access dominates.
// Deterministic xorshift fill (no math/rand) so the A/B is stable across runs.
func BenchmarkSVDPCA(b *testing.B) {
	const m, n = 2000, 50
	a := tensor.New(tensor.F64, tensor.Shape{m, n})
	x := uint64(0x9e3779b97f4a7c15)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			x ^= x << 13
			x ^= x >> 7
			x ^= x << 17
			a.SetF64(float64(int64(x))/(1<<62), i, j)
		}
	}
	b.ResetTimer()
	for range b.N {
		u, s, v, err := linalg.SVD(a)
		if err != nil {
			b.Fatal(err)
		}
		_, _, _ = u, s, v
	}
}
