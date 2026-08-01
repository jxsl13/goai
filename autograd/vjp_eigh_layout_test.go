package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestEighVJPTriangleAndLayoutAreBitIdentical locks both changes to the arithmetic they replaced:
// the transposed intermediates (where an operand lives) and the triangle-only final loop (which of
// two identical sums is computed). Neither may move a bit, so the assertion is exact.
//
// The reference below is the original nest written out verbatim — full double loop, column-walked
// operands. Sizes include 1 and 2, where the triangle is degenerate, and odd n so the mirror is
// exercised off a square boundary.
func TestEighVJPTriangleAndLayoutAreBitIdentical(t *testing.T) {
	vjp := vjpsMulti[backend.OpEigh]
	if vjp == nil {
		t.Fatal("no multi-output VJP registered for OpEigh")
	}
	for _, n := range []int{1, 2, 3, 7, 16, 33} {
		w := tensor.New(tensor.F64, tensor.Shape{n})
		wbar := tensor.New(tensor.F64, tensor.Shape{n})
		v := tensor.New(tensor.F64, tensor.Shape{n, n})
		vbar := tensor.New(tensor.F64, tensor.Shape{n, n})
		wv := make([]float64, n)
		wbv := make([]float64, n)
		vv := make([][]float64, n)
		vbv := make([][]float64, n)
		for i := range n {
			wv[i] = 1 + 0.5*float64(i) // ascending and spaced: F_ij never blows up
			wbv[i] = math.Sin(float64(i)*0.7) * 0.3
			w.SetF64(wv[i], i)
			wbar.SetF64(wbv[i], i)
			vv[i], vbv[i] = make([]float64, n), make([]float64, n)
			for j := range n {
				a := math.Sqrt(2/float64(n)) * math.Cos(math.Pi*float64(i)*(float64(j)+0.5)/float64(n))
				b := math.Cos(float64(i*5+j*3)) * 0.2
				vv[i][j], vbv[i][j] = a, b
				v.SetF64(a, i, j)
				vbar.SetF64(b, i, j)
			}
		}
		got, err := vjp(nil, nil, []*tensor.Tensor{w, v}, nil, []*tensor.Tensor{wbar, vbar})
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		want := eighVJPFullColumnWalk(n, wv, wbv, vv, vbv)
		for i := range n {
			for j := range n {
				g, wnt := got[0].AtF64(i, j), want[i][j]
				if math.Float64bits(g) != math.Float64bits(wnt) {
					t.Fatalf("n=%d [%d,%d]: triangle+relayout %v, original %v — not bit-identical",
						n, i, j, g, wnt)
				}
			}
		}
	}
}

// eighVJPFullColumnWalk is the pre-change implementation, kept verbatim as the reference: both
// operands of the inner product read down a column, and every (i,j) pair formed twice.
func eighVJPFullColumnWalk(n int, w, wb []float64, v, vb [][]float64) [][]float64 {
	inner := make([][]float64, n)
	for i := range n {
		inner[i] = make([]float64, n)
		for j := range n {
			var p float64
			for r := range n {
				p += v[r][i] * vb[r][j]
			}
			if i != j {
				inner[i][j] = p / (w[j] - w[i])
			}
		}
		inner[i][i] += wb[i]
	}
	tmp := make([][]float64, n)
	for a := range n {
		tmp[a] = make([]float64, n)
		for j := range n {
			var s float64
			for b := range n {
				s += inner[a][b] * v[j][b]
			}
			tmp[a][j] = s
		}
	}
	out := make([][]float64, n)
	for i := range n {
		out[i] = make([]float64, n)
	}
	for i := range n {
		for j := range n {
			var g, gt float64
			for a := range n {
				g += v[i][a] * tmp[a][j]
				gt += v[j][a] * tmp[a][i]
			}
			out[i][j] = 0.5 * (g + gt)
		}
	}
	return out
}
