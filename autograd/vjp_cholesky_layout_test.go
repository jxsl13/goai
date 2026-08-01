package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestCholeskyVJPLayoutIsBitIdentical locks the column-major relayout of the Cholesky VJP's
// intermediates to the row-major arithmetic it replaced. The reference below is that older code
// written out verbatim; only where it reads an operand differs, never which operand or in what
// order, so the assertion is exact bits rather than a tolerance.
//
// Sizes cross the boundaries that matter: 1 and 2 are degenerate, 7 is odd, 63/64/65 straddle a
// cache line and a page-stride multiple, and 128 is large enough that all four cubic loops
// dominate.
func TestCholeskyVJPLayoutIsBitIdentical(t *testing.T) {
	for _, n := range []int{1, 2, 3, 7, 63, 64, 65, 128} {
		// A deterministic lower-triangular L with a comfortably positive diagonal, and a lower
		// triangular cotangent. These are the only two inputs the rule reads.
		lt := tensor.New(tensor.F64, tensor.Shape{n, n})
		g := tensor.New(tensor.F64, tensor.Shape{n, n})
		l := make([][]float64, n)
		lbar := make([][]float64, n)
		for i := range n {
			l[i], lbar[i] = make([]float64, n), make([]float64, n)
			for j := 0; j <= i; j++ {
				v := math.Sin(float64(i*31+j*7)) * 0.5
				if i == j {
					v = 1.5 + 0.25*math.Cos(float64(i))
				}
				b := math.Cos(float64(i*17+j*5)) * 0.3
				l[i][j], lbar[i][j] = v, b
				lt.SetF64(v, i, j)
				g.SetF64(b, i, j)
			}
		}

		want := choleskyVJPRowMajor(n, l, lbar)

		vjp := vjps[backend.OpCholesky]
		if vjp == nil {
			t.Fatal("no VJP registered for OpCholesky")
		}
		got, err := vjp(nil, nil, []*tensor.Tensor{lt}, nil, g)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		for i := range n {
			for j := range n {
				gv, wv := got[0].AtF64(i, j), want[i][j]
				if math.Float64bits(gv) != math.Float64bits(wv) {
					t.Fatalf("n=%d [%d,%d]: column-major %v, row-major %v — not bit-identical",
						n, i, j, gv, wv)
				}
			}
		}
	}
}

// choleskyVJPRowMajor is the pre-relayout implementation, kept verbatim as the reference. It reads
// every intermediate along a column, which is exactly the property the shipped code changed.
func choleskyVJPRowMajor(n int, l, lbar [][]float64) [][]float64 {
	p := make([][]float64, n)
	for i := range n {
		pi := make([]float64, n)
		p[i] = pi
		for j := 0; j <= i; j++ {
			var m float64
			for k := i; k < n; k++ {
				m += l[k][i] * lbar[k][j]
			}
			if i == j {
				pi[j] = 0.5 * m
			} else {
				pi[j] = m
			}
		}
	}
	linv := make([][]float64, n)
	for i := range n {
		linv[i] = make([]float64, n)
	}
	for j := range n {
		linv[j][j] = 1 / l[j][j]
		for i := j + 1; i < n; i++ {
			li := l[i]
			var s float64
			for k := j; k < i; k++ {
				s += li[k] * linv[k][j]
			}
			linv[i][j] = -s / li[i]
		}
	}
	tmp := make([][]float64, n)
	for i := range n {
		ti := make([]float64, n)
		tmp[i] = ti
		pi := p[i]
		for j := range n {
			var s float64
			for k := j; k <= i; k++ {
				s += pi[k] * linv[k][j]
			}
			ti[j] = s
		}
	}
	out := make([][]float64, n)
	for i := range n {
		out[i] = make([]float64, n)
	}
	for i := range n {
		for j := i; j < n; j++ {
			var sij, sji float64
			for k := i; k < n; k++ {
				sij += linv[k][i] * tmp[k][j]
			}
			for k := j; k < n; k++ {
				sji += linv[k][j] * tmp[k][i]
			}
			if i == j {
				out[i][j] = sij
			} else {
				v := 0.5 * (sij + sji)
				out[i][j], out[j][i] = v, v
			}
		}
	}
	return out
}
