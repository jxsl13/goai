package linalg_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// Solve with many right-hand-side columns: the per-column scratch is what is under
// test, so allocs/op and B/op are the primary metrics. Inverse() is the extreme
// case (cols == n), which is how this path is reached in practice.
func benchSolve(b *testing.B, n, cols int) {
	rng := rand.New(rand.NewSource(1))
	ad := make([]float64, n*n)
	for i := range ad {
		ad[i] = rng.NormFloat64()
	}
	for i := range n { // diagonally dominant so the factorization is well conditioned
		ad[i*n+i] += float64(n)
	}
	a := tensor.FromFloat64(tensor.Shape{n, n}, ad)
	f, err := linalg.Factor(a)
	if err != nil {
		b.Fatal(err)
	}
	bd := make([]float64, n*cols)
	for i := range bd {
		bd[i] = rng.NormFloat64()
	}
	rhs := tensor.FromFloat64(tensor.Shape{n, cols}, bd)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := f.Solve(rhs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLUSolve_64x64(b *testing.B)   { benchSolve(b, 64, 64) }
func BenchmarkLUSolve_128x128(b *testing.B) { benchSolve(b, 128, 128) }
func BenchmarkLUSolve_128x1(b *testing.B)   { benchSolve(b, 128, 1) } // control: single RHS

// Sizes where the [n,cols] output stops fitting in cache — 2MB at 512, 4.5MB at 768,
// against this host's 128-byte lines. The 64 and 128 cases above cannot see a traversal
// -order defect in this path at all (131KB of output sits in L2 whichever way it is
// walked), which is why the strided back-substitution read went unnoticed while they
// were green. Any benchmark meant to guard cache behavior needs a size that exceeds it.
func BenchmarkLUSolve_512x512(b *testing.B) { benchSolve(b, 512, 512) }
func BenchmarkLUSolve_768x768(b *testing.B) { benchSolve(b, 768, 768) }
