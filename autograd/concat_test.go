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

// §V2: the concat gradient routes each output element back to its source input.
// Finite differences of ½·Σ(w ⊙ concat)² w.r.t. each input match the analytic VJP,
// confirming concat is differentiable (so a learned prefix can train through it).
func TestConcatGradcheck(t *testing.T) {
	ad := []float64{0.5, -1.5, 2.0, 1.0, -0.5, 0.25} // [2,3]
	bd := []float64{0.3, -0.8, 1.1, -0.7}            // [2,2]
	// fixed weights so the loss depends on the concat output non-trivially
	wd := []float64{1, -2, 0.5, 3, -1, 0.7, 2, -0.4, 1.3, 0.9} // [2,5]
	w := tensor.FromFloat64(tensor.Shape{2, 5}, wd)

	loss := func(ctx *backend.Context, a, b *tensor.Tensor) *tensor.Tensor {
		c, err := backend.Execute(ctx, backend.OpConcat, []*tensor.Tensor{a, b}, backend.ConcatAttrs{Axis: 1})
		if err != nil {
			t.Fatal(err)
		}
		wc, _ := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{c[0], w}, nil)
		s, _ := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{wc[0]}, nil)
		return s[0]
	}
	lossVal := func(av, bv []float64) float64 {
		return loss(backend.NewContext(), tensor.FromFloat64(tensor.Shape{2, 3}, av), tensor.FromFloat64(tensor.Shape{2, 2}, bv)).AtF64()
	}

	aT := tensor.FromFloat64(tensor.Shape{2, 3}, append([]float64(nil), ad...))
	bT := tensor.FromFloat64(tensor.Shape{2, 2}, append([]float64(nil), bd...))
	tape := autograd.NewTape()
	l := loss(tape.Context(), aT, bT)
	if err := tape.Backward(l); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	sides := []struct {
		name string
		d    []float64
		grad *tensor.Tensor
		set  func(v []float64) float64
	}{
		{"a", append([]float64(nil), ad...), tape.Grad(aT), func(v []float64) float64 { return lossVal(v, bd) }},
		{"b", append([]float64(nil), bd...), tape.Grad(bT), func(v []float64) float64 { return lossVal(ad, v) }},
	}
	for _, side := range sides {
		for i := range side.d {
			o := side.d[i]
			side.d[i] = o + h
			lp := side.set(side.d)
			side.d[i] = o - h
			lm := side.set(side.d)
			side.d[i] = o
			num := (lp - lm) / (2 * h)
			ana := side.grad.AtF64(tensor.Unravel(i, side.grad.Shape())...)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad[%d]: numeric %.8g vs analytic %.8g", side.name, i, num, ana)
			}
		}
	}
}
