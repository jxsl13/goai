package linalg_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/linalg"
)

// Cholesky factorization at sizes where the O(n³) inner dot dominates — measures the
// flat row-major factor vs the prior [][]float64 (scattered rows + slice-header deref).
func benchCholesky(b *testing.B, n int) {
	rng := rand.New(rand.NewPCG(1, 2))
	a := spd(rng, n)
	b.ResetTimer()
	for range b.N {
		if _, err := linalg.Cholesky(a); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCholesky128(b *testing.B) { benchCholesky(b, 128) }
func BenchmarkCholesky256(b *testing.B) { benchCholesky(b, 256) }
func BenchmarkCholesky512(b *testing.B) { benchCholesky(b, 512) }

// CholSolve (factor + fwd/back substitution) with a modest RHS block.
func BenchmarkCholSolve256x8(b *testing.B) {
	rng := rand.New(rand.NewPCG(3, 4))
	a := spd(rng, 256)
	rhs := randRect(rng, 256, 8)
	b.ResetTimer()
	for range b.N {
		if _, err := linalg.CholSolve(a, rhs); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkCholSolve256x128 is the WIDE right-hand side. BenchmarkCholSolve256x8 cannot see a
// change to the substitution's memory access at all: with 8 columns the whole solution matrix is
// 16 KB and every strided read lands in L1 regardless of the stride, so the factorization
// dominates and the layout is free. The cost of a column-strided read only appears once the
// stride pushes consecutive reads onto different cache lines and the working set past L1 — which
// is the same "size the cell past L1" lesson the QR and SVD kernels recorded.
func BenchmarkCholSolve256x128(b *testing.B) {
	rng := rand.New(rand.NewPCG(3, 4))
	a := spd(rng, 256)
	rhs := randRect(rng, 256, 128)
	b.ResetTimer()
	for range b.N {
		if _, err := linalg.CholSolve(a, rhs); err != nil {
			b.Fatal(err)
		}
	}
}
