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

// §V2: the product-reduction gradient (∂prod/∂xᵢ = ∏_{j≠i} xⱼ) matches central finite
// differences, per-axis, on zero-free data.
func TestProdGradcheck(t *testing.T) {
	xv := []float64{1.5, -2, 0.5, 3, -1, 2}
	x := tensor.FromFloat64(tensor.Shape{2, 3}, xv)
	w := tensor.FromFloat64(tensor.Shape{2}, []float64{0.7, -1.3}) // weight the [2] axis-1 output

	tape := autograd.NewTape()
	p, _ := backend.Execute(tape.Context(), backend.OpProd, []*tensor.Tensor{x}, backend.ReduceAttrs{Axes: []int{1}})
	wc, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{p[0], w}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{wc[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(x)
	const h = 1e-6
	lossOf := func() float64 {
		p, _ := backend.Execute(backend.NewContext(), backend.OpProd, []*tensor.Tensor{x}, backend.ReduceAttrs{Axes: []int{1}})
		return w.AtF64(0)*p[0].AtF64(0) + w.AtF64(1)*p[0].AtF64(1)
	}
	for i := range 6 {
		idx := tensor.Unravel(i, x.Shape())
		o := x.AtF64(idx...)
		x.SetF64(o+h, idx...)
		lp := lossOf()
		x.SetF64(o-h, idx...)
		lm := lossOf()
		x.SetF64(o, idx...)
		if num, ana := (lp-lm)/(2*h), g.AtF64(idx...); math.Abs(num-ana) > 1e-5*math.Max(1, math.Abs(num)) {
			t.Errorf("grad[%v]: numeric %.8g vs analytic %.8g", idx, num, ana)
		}
	}
}

// §V16 tier-1: the zero-handling of the product gradient — with a single zero in the
// group only that element gets a nonzero derivative (the product of the others); with
// two or more zeros every gradient is 0.
func TestProdGradZeros(t *testing.T) {
	// group [2,0,3]: prod=0; ∂/∂x = [0, 2·3=6, 0]
	x := tensor.FromFloat64(tensor.Shape{3}, []float64{2, 0, 3})
	tape := autograd.NewTape()
	p, _ := backend.Execute(tape.Context(), backend.OpProd, []*tensor.Tensor{x}, backend.ReduceAttrs{})
	if err := tape.Backward(p[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(x)
	for i, want := range []float64{0, 6, 0} {
		if g.AtF64(i) != want {
			t.Errorf("single-zero grad[%d] = %g, want %g", i, g.AtF64(i), want)
		}
	}

	// two zeros [0,5,0]: all gradients 0
	x2 := tensor.FromFloat64(tensor.Shape{3}, []float64{0, 5, 0})
	tape2 := autograd.NewTape()
	p2, _ := backend.Execute(tape2.Context(), backend.OpProd, []*tensor.Tensor{x2}, backend.ReduceAttrs{})
	if err := tape2.Backward(p2[0]); err != nil {
		t.Fatal(err)
	}
	g2 := tape2.Grad(x2)
	for i := range 3 {
		if g2.AtF64(i) != 0 {
			t.Errorf("two-zero grad[%d] = %g, want 0", i, g2.AtF64(i))
		}
	}
}
