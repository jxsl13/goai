package linalg_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// pinvAccessors is the ORIGINAL accessor-driven accumulation, kept as the oracle for
// the flat-view rewrite. Written before the change (PROC-009): a one-ulp
// perturbation of the inner product passed linalg's existing suite, so derived_test
// verifies Pinv only to a tolerance — the fifth kernel this session found unguarded
// at the level a rewrite touches, and the first that HAS a test file.
func pinvAccessors(u, s, v *tensor.Tensor, m, n int) []float64 {
	p := s.Numel()
	cutoff := 1e-15 * s.AtF64(0)
	out := make([]float64, n*m)
	for k := range p {
		sk := s.AtF64(k)
		if sk <= cutoff {
			continue
		}
		inv := 1 / sk
		for i := range n {
			vik := v.AtF64(i, k) * inv
			for j := range m {
				out[i*m+j] += vik * u.AtF64(j, k)
			}
		}
	}
	return out
}

func TestPinvBitIdentical(t *testing.T) {
	for _, sz := range [][2]int{{1, 1}, {4, 2}, {5, 5}, {9, 3}} {
		m, n := sz[0], sz[1]
		a := bench.RandF64(tensor.Shape{m, n}, uint64(m*100+n))
		got, err := linalg.Pinv(a)
		if err != nil {
			t.Fatal(err)
		}
		u, s, v, err := linalg.SVD(a)
		if err != nil {
			t.Fatal(err)
		}
		want := pinvAccessors(u, s, v, m, n)
		for i := range n {
			for j := range m {
				g := got.AtF64(i, j)
				w := want[i*m+j]
				if math.Float64bits(g) != math.Float64bits(w) {
					t.Fatalf("%dx%d Pinv[%d,%d]: got %v want %v", m, n, i, j, g, w)
				}
			}
		}
	}
}
