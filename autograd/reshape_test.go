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

// §V2: reshape is a pure re-layout, so its gradient reshapes straight back. Finite
// differences of ½·Σ(reshape)² w.r.t. x match the analytic VJP, and the gradient
// lands at the corresponding flat position.
func TestReshapeGradcheck(t *testing.T) {
	xd := []float64{1, -2, 3, -4, 5, -6} // [2,3]
	target := backend.ReshapeAttrs{Shape: tensor.Shape{3, 2}}
	lossAt := func(d []float64) float64 {
		out, err := backend.Execute(backend.NewContext(), backend.OpReshape, []*tensor.Tensor{tensor.FromFloat64(tensor.Shape{2, 3}, d)}, target)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range out[0].Numel() {
			v := out[0].AtF64(tensor.Unravel(i, out[0].Shape())...)
			s += v * v
		}
		return s / 2
	}
	xT := tensor.FromFloat64(tensor.Shape{2, 3}, append([]float64(nil), xd...))
	tape := autograd.NewTape()
	out, err := backend.Execute(tape.Context(), backend.OpReshape, []*tensor.Tensor{xT}, target)
	if err != nil {
		t.Fatal(err)
	}
	sq, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{out[0], out[0]}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{sq[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(xT)
	data := append([]float64(nil), xd...)
	const h = 1e-6
	for i := range data {
		o := data[i]
		data[i] = o + h
		lp := lossAt(data)
		data[i] = o - h
		lm := lossAt(data)
		data[i] = o
		num := (lp - lm) / (2 * h)
		ana := 0.5 * g.AtF64(tensor.Unravel(i, xT.Shape())...)
		if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad[%d]: numeric %.8g vs analytic %.8g", i, num, ana)
		}
	}
	// Σ(reshape)² has grad 2·element at its flat position (order preserved, not scrambled)
	for i := range xd {
		if got := g.AtF64(tensor.Unravel(i, xT.Shape())...); math.Abs(got-2*xd[i]) > 1e-9 {
			t.Errorf("grad[%d]=%g, want %g (2·element at preserved flat position)", i, got, 2*xd[i])
		}
	}
}
