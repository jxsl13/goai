package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestEighVJPIsBitIdentical freezes the eigendecomposition adjoint. The two changes it gates —
// reading the intermediate through a transposed mirror, and banding the triangular loop — both
// claim to move no value, and a gradient is the last place a small drift would be noticed
// because training absorbs it.
//
// The sizes straddle the fan-out gate: 6 and 12 run the triangular loop inline, 48 and 64 band
// it, and 48 is not a multiple of the worker count.
func TestEighVJPIsBitIdentical(t *testing.T) {
	cases := []struct {
		n    int
		want uint64
	}{
		{6, 275940546675640912}, {12, 11743736518440278093}, {48, 11927342901561703116}, {64, 13416730849446074620},
	}
	fn := vjpsMulti[backend.OpEigh]
	ctx := backend.NewContext()
	for _, c := range cases {
		n := c.n
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
		h := uint64(14695981039346656037)
		for i := range n {
			for j := range n {
				b := math.Float64bits(g[0].AtF64(i, j))
				for s := 0; s < 64; s += 8 {
					h = (h ^ (b>>s)&0xff) * 1099511628211
				}
			}
		}
		if h != c.want {
			t.Fatalf("n=%d digest %d, want %d", n, h, c.want)
		}
	}
}
