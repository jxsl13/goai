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

// gradcheckSelect verifies ∂(Σ op(a,b))/∂{a,b} by central finite differences for a
// two-input op whose extra fixed inputs are given (e.g. a condition mask for Where).
// The differentiable inputs are the LAST two of `ins`.
func gradcheckSelect(t *testing.T, op backend.Op, ins ...*tensor.Tensor) {
	t.Helper()
	na := len(ins)
	ai, bi := na-2, na-1
	shape := ins[ai].Shape()
	buildInputs := func(ad, bd []float64) []*tensor.Tensor {
		cp := append([]*tensor.Tensor(nil), ins...)
		cp[ai] = tensor.FromFloat64(shape, ad)
		cp[bi] = tensor.FromFloat64(shape, bd)
		return cp
	}
	flat := func(x *tensor.Tensor) []float64 {
		d := make([]float64, x.Numel())
		for i := range d {
			d[i] = x.AtF64(tensor.Unravel(i, x.Shape())...)
		}
		return d
	}
	ad, bd := flat(ins[ai]), flat(ins[bi])
	lossAt := func(inp []*tensor.Tensor) float64 {
		out, err := backend.Execute(backend.NewContext(), op, inp, nil)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range out[0].Numel() {
			s += out[0].AtF64(tensor.Unravel(i, out[0].Shape())...)
		}
		return s
	}
	tape := autograd.NewTape()
	tin := buildInputs(append([]float64(nil), ad...), append([]float64(nil), bd...))
	// rebuild the differentiable leaves on the tape context so Grad can find them
	aT, bT := tensor.FromFloat64(shape, ad), tensor.FromFloat64(shape, bd)
	tin[ai], tin[bi] = aT, bT
	out, err := backend.Execute(tape.Context(), op, tin, nil)
	if err != nil {
		t.Fatal(err)
	}
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{out[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	for name, side := range map[string]struct {
		data []float64
		grad *tensor.Tensor
		set  func(v []float64) []*tensor.Tensor
	}{
		"a": {append([]float64(nil), ad...), tape.Grad(aT), func(v []float64) []*tensor.Tensor { return buildInputs(v, bd) }},
		"b": {append([]float64(nil), bd...), tape.Grad(bT), func(v []float64) []*tensor.Tensor { return buildInputs(ad, v) }},
	} {
		for i := range side.data {
			o := side.data[i]
			side.data[i] = o + h
			lp := lossAt(side.set(side.data))
			side.data[i] = o - h
			lm := lossAt(side.set(side.data))
			side.data[i] = o
			num := (lp - lm) / (2 * h)
			ana := side.grad.AtF64(i)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%v %s grad[%d]: numeric %.8g vs analytic %.8g", op, name, i, num, ana)
			}
		}
	}
}

// §V2: max/min route the gradient to the selected input (no ties in the data).
func TestMaximumGradcheck(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{4}, []float64{1, 5, 3, -2})
	b := tensor.FromFloat64(tensor.Shape{4}, []float64{4, 2, 6, -5})
	gradcheckSelect(t, backend.OpMaximum, a, b)
}

func TestMinimumGradcheck(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{4}, []float64{1, 5, 3, -2})
	b := tensor.FromFloat64(tensor.Shape{4}, []float64{4, 2, 6, -5})
	gradcheckSelect(t, backend.OpMinimum, a, b)
}

// §V2: Where routes the gradient by the mask (da where cond≠0, db elsewhere); cond
// is not differentiated.
func TestWhereGradcheck(t *testing.T) {
	cond := tensor.FromFloat64(tensor.Shape{4}, []float64{1, 0, 1, 0})
	a := tensor.FromFloat64(tensor.Shape{4}, []float64{2, 3, 4, 5})
	b := tensor.FromFloat64(tensor.Shape{4}, []float64{-1, -2, -3, -4})
	gradcheckSelect(t, backend.OpWhere, cond, a, b)
}

// Explicit routing: for max, dg goes entirely to the larger input.
func TestMaximumRouting(t *testing.T) {
	aT := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 5, 3})
	bT := tensor.FromFloat64(tensor.Shape{3}, []float64{4, 2, 6})
	tape := autograd.NewTape()
	out, _ := backend.Execute(tape.Context(), backend.OpMaximum, []*tensor.Tensor{aT, bT}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{out[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	ga, gb := tape.Grad(aT), tape.Grad(bT)
	wantA := []float64{0, 1, 0} // a<b, a>b, a<b
	wantB := []float64{1, 0, 1}
	for i := range 3 {
		if ga.AtF64(i) != wantA[i] || gb.AtF64(i) != wantB[i] {
			t.Errorf("route[%d]: da=%g db=%g, want %g/%g", i, ga.AtF64(i), gb.AtF64(i), wantA[i], wantB[i])
		}
	}
}
