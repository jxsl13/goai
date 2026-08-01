package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestSolveSPDInterchangeIsBitIdentical locks the c-innermost substitution to the arithmetic it
// replaced. The reference is the pre-interchange loop nest written out verbatim: c outermost, the
// accumulator in a register, one L element fetched per innermost iteration.
//
// The claim is that only the interleaving of independent (i,c) results changed, so the assertion is
// exact bits. Shapes cross k=1 (where the interchange is a no-op) and k>n (where a row of the
// right-hand side is longer than the factor).
func TestSolveSPDInterchangeIsBitIdentical(t *testing.T) {
	for _, sz := range [][2]int{{1, 1}, {2, 3}, {5, 1}, {8, 16}, {33, 7}, {64, 16}} {
		n, k := sz[0], sz[1]
		// A diagonally dominant SPD matrix, so the forward Cholesky inside the kernel is stable.
		a := tensor.New(tensor.F64, tensor.Shape{n, n})
		for i := range n {
			for j := range n {
				v := math.Cos(float64(i*13+j*7)) * 0.2
				if i == j {
					v = float64(n) + 2
				}
				a.SetF64(v, i, j)
			}
		}
		b := tensor.New(tensor.F64, tensor.Shape{n, k})
		for i := range n {
			for c := range k {
				b.SetF64(math.Sin(float64(i*5+c*3))*1.7, i, c)
			}
		}
		got, err := backend.Execute(backend.NewContext(), backend.OpSolveSPD,
			[]*tensor.Tensor{a, b}, nil)
		if err != nil {
			t.Fatalf("n=%d k=%d: %v", n, k, err)
		}
		want := solveSPDColumnOuter(t, a, b, n, k)
		for i := range n {
			for c := range k {
				g, w := got[0].AtF64(i, c), want[i*k+c]
				if math.Float64bits(g) != math.Float64bits(w) {
					t.Fatalf("n=%d k=%d [%d,%d]: interchanged %v, column-outer %v — not bit-identical",
						n, k, i, c, g, w)
				}
			}
		}
	}
}

// solveSPDColumnOuter reproduces the pre-interchange substitution, using the kernel's own Cholesky
// so the reference differs from the shipped code in exactly one respect: the loop order.
func solveSPDColumnOuter(t *testing.T, a, b *tensor.Tensor, n, k int) []float64 {
	t.Helper()
	out, err := backend.Execute(backend.NewContext(), backend.OpCholesky, []*tensor.Tensor{a}, nil)
	if err != nil {
		t.Fatal(err)
	}
	l := out[0]
	x := make([]float64, n*k)
	y := make([]float64, n*k)
	for c := range k {
		for i := range n {
			s := b.AtF64(i, c)
			for p := range i {
				s -= l.AtF64(i, p) * y[p*k+c]
			}
			y[i*k+c] = s / l.AtF64(i, i)
		}
		for i := n - 1; i >= 0; i-- {
			s := y[i*k+c]
			for p := i + 1; p < n; p++ {
				s -= l.AtF64(p, i) * x[p*k+c]
			}
			x[i*k+c] = s / l.AtF64(i, i)
		}
	}
	return x
}
