package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// TestMatTmulRectInterchangeIsBitIdentical locks the contraction-outermost xᵀ·y to the column-walk
// form it replaced. Both are exact sums of the same p products in ascending k, so the assertion is
// exact bits.
func TestMatTmulRectInterchangeIsBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	for _, sz := range [][2]int{{1, 1}, {3, 2}, {8, 8}, {17, 5}, {40, 13}} {
		p, q := sz[0], sz[1]
		x := make([][]float64, p)
		y := make([][]float64, p)
		for k := range p {
			x[k], y[k] = make([]float64, q), make([]float64, q)
			for i := range q {
				x[k][i], y[k][i] = rng.NormFloat64(), rng.NormFloat64()
			}
		}
		got := matTmulRect(x, y, p, q)
		for i := range q {
			for j := range q {
				var want float64
				for k := range p {
					want += x[k][i] * y[k][j]
				}
				if math.Float64bits(got[i][j]) != math.Float64bits(want) {
					t.Fatalf("p=%d q=%d [%d,%d]: interchanged %v, column-walk %v — not bit-identical",
						p, q, i, j, got[i][j], want)
				}
			}
		}
	}
}

// TestSVDVJPProjectionHoistIsBitIdentical locks the hoisted projection to the arithmetic it
// replaced. The original rebuilt the projection row INSIDE the j loop — n times over, which is what
// made this term O(m·n³) — and the hoist computes it once per i.
//
// The two are compared term by term on raw bits rather than through the whole rule, because the
// change is local: proj[b] is the same subtraction sequence over ascending a, and each add still
// accumulates over ascending b. A whole-rule comparison would also be dominated by parts that did
// not change.
func TestSVDVJPProjectionHoistIsBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	for _, sz := range [][2]int{{1, 1}, {4, 2}, {9, 6}, {33, 16}} {
		m, n := sz[0], sz[1]
		mk := func(rows, cols int) [][]float64 {
			out := make([][]float64, rows)
			for i := range rows {
				out[i] = make([]float64, cols)
				for j := range cols {
					out[i][j] = rng.NormFloat64()
				}
			}
			return out
		}
		w, u, utw, v := mk(m, n), mk(m, n), mk(n, n), mk(n, n)

		// The SHIPPED function, not a copy of it. The first version of this test defined its own
		// hoisted loop and compared that against the rebuilt form: it proved the identity and
		// detected nothing, because mutating the real code could not affect it. Calling
		// addTallCorrection is what makes this a gate.
		abar := tensor.New(tensor.F64, tensor.Shape{m, n})
		addTallCorrection(abar, w, u, utw, v, m, n)

		// original, rebuilt per (i,j)
		for i := range m {
			for j := range n {
				var add float64
				for b := range n {
					projib := w[i][b]
					for a := range n {
						projib -= u[i][a] * utw[a][b]
					}
					add += projib * v[j][b]
				}
				if g := abar.AtF64(i, j); math.Float64bits(g) != math.Float64bits(add) {
					t.Fatalf("m=%d n=%d [%d,%d]: hoisted %v, rebuilt %v — not bit-identical",
						m, n, i, j, g, add)
				}
			}
		}
	}
}
