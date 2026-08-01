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
