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

// §V2: the cumsum gradient is the reverse cumulative sum. With loss=Σ cumsum(x),
// each x[j] feeds every output i≥j, so grad[j] = (L−j) — a reverse cumsum of ones.
func TestCumsumGradIsReverseCount(t *testing.T) {
	xT := tensor.FromFloat64(tensor.Shape{4}, []float64{5, 1, 9, 2})
	tape := autograd.NewTape()
	c, _ := backend.Execute(tape.Context(), backend.OpCumsum, []*tensor.Tensor{xT}, backend.CumsumAttrs{Axis: 0})
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{c[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(xT)
	for i, w := range []float64{4, 3, 2, 1} {
		if g.AtF64(i) != w {
			t.Errorf("grad[%d]=%g, want %g (reverse count)", i, g.AtF64(i), w)
		}
	}
}

// §V2: finite differences of ½·Σ(cumsum(x)⊙W)² w.r.t. x match the analytic (reverse
// cumsum) VJP, along a non-trivial axis of a matrix.
func TestCumsumGradcheck(t *testing.T) {
	xd := []float64{0.5, -1, 2, 0.3, -0.7, 1.1}
	w := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, -2, 0.5, 2, -1, 0.7})
	attrs := backend.CumsumAttrs{Axis: 1}
	lossAt := func(d []float64) float64 {
		c, err := backend.Execute(backend.NewContext(), backend.OpCumsum, []*tensor.Tensor{tensor.FromFloat64(tensor.Shape{2, 3}, d)}, attrs)
		if err != nil {
			t.Fatal(err)
		}
		wc, _ := backend.Execute(backend.NewContext(), backend.OpMul, []*tensor.Tensor{c[0], w}, nil)
		s, _ := backend.Execute(backend.NewContext(), backend.OpSum, []*tensor.Tensor{wc[0]}, nil)
		return s[0].AtF64()
	}
	xT := tensor.FromFloat64(tensor.Shape{2, 3}, append([]float64(nil), xd...))
	tape := autograd.NewTape()
	c, _ := backend.Execute(tape.Context(), backend.OpCumsum, []*tensor.Tensor{xT}, attrs)
	wc, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{c[0], w}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{wc[0]}, nil)
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
		if num, ana := (lp-lm)/(2*h), g.AtF64(tensor.Unravel(i, xT.Shape())...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad[%d]: numeric %.8g vs analytic %.8g", i, num, ana)
		}
	}
}
