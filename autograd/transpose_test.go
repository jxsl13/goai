package autograd_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V2: the transpose gradient is the transpose of the upstream gradient. With a scalar
// loss Σ W⊙xᵀ, ∂/∂x[i,j] = W[j,i]; verified against finite differences.
func TestTransposeGradcheck(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, -2, 0.5, 3, -1, 2})
	w := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{0.7, -1.1, 0.4, 1.3, -0.6, 0.9}) // over xᵀ [3,2]

	tape := autograd.NewTape()
	xt, _ := backend.Execute(tape.Context(), backend.OpTranspose, []*tensor.Tensor{x}, nil)
	p, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{xt[0], w}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{p[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(x)
	const h = 1e-6
	lossOf := func() float64 {
		xt, _ := backend.Execute(backend.NewContext(), backend.OpTranspose, []*tensor.Tensor{x}, nil)
		var s float64
		for i := range 3 {
			for j := range 2 {
				s += w.AtF64(i, j) * xt[0].AtF64(i, j)
			}
		}
		return s
	}
	for i := range 2 {
		for j := range 3 {
			o := x.AtF64(i, j)
			x.SetF64(o+h, i, j)
			lp := lossOf()
			x.SetF64(o-h, i, j)
			lm := lossOf()
			x.SetF64(o, i, j)
			if num, ana := (lp-lm)/(2*h), g.AtF64(i, j); math.Abs(num-ana) > 1e-6 {
				t.Errorf("grad[%d,%d]: numeric %.8g vs analytic %.8g", i, j, num, ana)
			}
		}
	}
}
