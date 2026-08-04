package linalg

import (
	"math/rand"
	"testing"
)

func benchSymEig(b *testing.B, n int) {
	rng := rand.New(rand.NewSource(2))
	a := make([][]float64, n)
	for i := range a {
		a[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			v := rng.NormFloat64()
			a[i][j], a[j][i] = v, v
		}
		a[i][i] += float64(n)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		SymEig(a)
	}
}
func BenchmarkSymEig_96(b *testing.B)  { benchSymEig(b, 96) }
func BenchmarkSymEig_128(b *testing.B) { benchSymEig(b, 128) }
func BenchmarkSymEig_192(b *testing.B) { benchSymEig(b, 192) }
func BenchmarkSymEig_256(b *testing.B) { benchSymEig(b, 256) }
func BenchmarkSymEig_512(b *testing.B) { benchSymEig(b, 512) }

// benchSymEigClustered builds A = Qᵀ diag(λ) Q with a tightly CLUSTERED spectrum — the
// input class on which cyclic Jacobi's sweep count blows up (rotations between near-equal
// diagonal entries make almost no progress). tridiag+QL is immune (its iteration count is
// bounded by the Wilkinson shift regardless of clustering).
func benchSymEigClustered(b *testing.B, n int) {
	rng := rand.New(rand.NewSource(3))
	q := make([][]float64, n)
	for i := range n {
		q[i] = make([]float64, n)
		for j := range n {
			q[i][j] = rng.NormFloat64()
		}
	}
	for i := range n { // modified Gram-Schmidt → orthonormal rows
		for p := 0; p < i; p++ {
			var d float64
			for r := range n {
				d += q[i][r] * q[p][r]
			}
			for r := range n {
				q[i][r] -= d * q[p][r]
			}
		}
		var nrm float64
		for r := range n {
			nrm += q[i][r] * q[i][r]
		}
		nrm = 1.0 / (nrm + 1e-300)
		for r := range n {
			q[i][r] *= nrm // note: unnormalized scale is irrelevant to the timing pattern
		}
	}
	a := make([][]float64, n)
	for i := range n {
		a[i] = make([]float64, n)
		for j := range n {
			var acc float64
			for k := range n {
				lam := 1.0 + float64(k)*1e-9 // clustered near 1
				acc += q[k][i] * lam * q[k][j]
			}
			a[i][j] = acc
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		SymEig(a)
	}
}
func BenchmarkSymEigClustered_256(b *testing.B) { benchSymEigClustered(b, 256) }
