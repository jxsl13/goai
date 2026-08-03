package linalg_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/linalg"
)

// BenchmarkLstsq times an overdetermined least-squares solve: Householder QR of the [m,n] system
// once, then Qᵀb and a back substitution per right-hand side.
//
// Two widths on purpose. The narrow one is dominated by the factorization, which is O(m·n²) and
// paid once no matter how many columns follow; only the wide one lets the per-column work — which
// is where the allocation and the substitution live — be a visible share. Sizing a cell so the
// part under test dominates is the same lesson CholSolve's 8-column benchmark taught by NOT
// showing a 74% change.
func benchLstsq(b *testing.B, m, n, cols int) {
	rng := rand.New(rand.NewPCG(7, 8))
	a := randRect(rng, m, n)
	rhs := randRect(rng, m, cols)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := linalg.Lstsq(a, rhs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLstsq_256x64x8(b *testing.B)   { benchLstsq(b, 256, 64, 8) }
func BenchmarkLstsq_256x64x128(b *testing.B) { benchLstsq(b, 256, 64, 128) }
