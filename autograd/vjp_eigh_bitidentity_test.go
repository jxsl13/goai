package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestEighVJPIsBitIdentical freezes the eigendecomposition adjoint on the output of a REAL
// OpEigh (so the VJP is exercised on genuine eigenvector/eigenvalue layout, not a synthetic
// basis). The two changes it gates — reading the intermediate through a transposed mirror, and
// banding the triangular loop — both claim to move no value, so the optimized VJP must match the
// verbatim column-walk reference (eighVJPFullColumnWalk) BIT for bit.
//
// It compares against that independent reference rather than a frozen absolute digest so the
// assertion is invariant to the SymEig algorithm and to platform FMA rounding (the previous
// absolute digest was pinned to one platform's eigendecomposition bits and could not survive a
// change to the eigensolver): both the optimized path and the reference consume the SAME OpEigh
// output, so their bit-identity is a property of the VJP arithmetic alone.
//
// The sizes straddle the fan-out gate: 6 and 12 run the triangular loop inline, 48 and 64 band
// it, and 48 is not a multiple of the worker count.
func TestEighVJPIsBitIdentical(t *testing.T) {
	fn := vjpsMulti[backend.OpEigh]
	ctx := backend.NewContext()
	for _, n := range []int{6, 12, 48, 64} {
		a := tensor.New(tensor.F64, tensor.Shape{n, n})
		for i := range n {
			for j := range n {
				v := math.Sin(float64(i*13+j*7)) * 0.5
				a.SetF64(v, i, j)
				a.SetF64(v, j, i) // symmetric, as eigh requires
			}
			a.SetF64(float64(n)+float64(i)*0.25, i, i) // distinct eigenvalues: F_ij stays finite
		}
		out, err := backend.Execute(ctx, backend.OpEigh, []*tensor.Tensor{a}, nil)
		if err != nil {
			t.Fatal(err)
		}
		wb := tensor.New(tensor.F64, tensor.Shape{n})
		vb := tensor.New(tensor.F64, tensor.Shape{n, n})
		for i := range n {
			wb.SetF64(math.Cos(float64(i))*0.3, i)
			for j := range n {
				vb.SetF64(math.Cos(float64(i*5+j*3))*0.2, i, j)
			}
		}
		g, err := fn(ctx, []*tensor.Tensor{a}, out, nil, []*tensor.Tensor{wb, vb})
		if err != nil {
			t.Fatal(err)
		}
		// verbatim column-walk reference, fed the SAME OpEigh (w, V) and adjoints
		wv := make([]float64, n)
		wbv := make([]float64, n)
		vv := make([][]float64, n)
		vbv := make([][]float64, n)
		for i := range n {
			wv[i] = out[0].AtF64(i)
			wbv[i] = wb.AtF64(i)
			vv[i], vbv[i] = make([]float64, n), make([]float64, n)
			for j := range n {
				vv[i][j] = out[1].AtF64(i, j)
				vbv[i][j] = vb.AtF64(i, j)
			}
		}
		want := eighVJPFullColumnWalk(n, wv, wbv, vv, vbv)
		for i := range n {
			for j := range n {
				if math.Float64bits(g[0].AtF64(i, j)) != math.Float64bits(want[i][j]) {
					t.Fatalf("n=%d [%d,%d]: optimized %v, column-walk reference %v — not bit-identical",
						n, i, j, g[0].AtF64(i, j), want[i][j])
				}
			}
		}
	}
}
