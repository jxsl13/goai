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

// gradcheckUnary checks that ∂(Σ op(x))/∂x = op'(x) via central finite differences
// (loss is linear in the op output, so the tape grad equals the raw derivative).
func gradcheckUnary(t *testing.T, op backend.Op, attrs backend.Attrs, xd []float64) {
	t.Helper()
	shape := tensor.Shape{len(xd)}
	lossAt := func(d []float64) float64 {
		out, err := backend.Execute(backend.NewContext(), op, []*tensor.Tensor{tensor.FromFloat64(shape, d)}, attrs)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range out[0].Numel() {
			s += out[0].AtF64(tensor.Unravel(i, out[0].Shape())...)
		}
		return s
	}
	xT := tensor.FromFloat64(shape, append([]float64(nil), xd...))
	tape := autograd.NewTape()
	out, err := backend.Execute(tape.Context(), op, []*tensor.Tensor{xT}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{out[0]}, nil)
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
		ana := g.AtF64(i)
		if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("%v grad[%d]: numeric %.8g vs analytic %.8g", op, i, num, ana)
		}
	}
}

// §V2: Sqrt gradient 1/(2√x) matches finite differences (positive x only).
func TestSqrtGradcheck(t *testing.T) {
	gradcheckUnary(t, backend.OpSqrt, nil, []float64{0.25, 1, 4, 9, 2.5})
}

// §V2: Abs gradient sign(x) matches finite differences (away from the x=0 kink).
func TestAbsGradcheck(t *testing.T) {
	gradcheckUnary(t, backend.OpAbs, nil, []float64{-2, 3, -0.5, 1.5, -4})
}

// §V2: Clip gradient passes through inside [Lo,Hi] and is 0 for clamped elements —
// the finite-difference test uses values strictly inside/outside so the boundary
// kink is not probed.
func TestClipGradcheck(t *testing.T) {
	gradcheckUnary(t, backend.OpClip, backend.ClipAttrs{Lo: -1, Hi: 1}, []float64{-2, -0.5, 0.3, 2, 0.9})
}

// The clamped elements (below Lo or above Hi) get exactly zero gradient.
func TestClipZeroGradWhenClamped(t *testing.T) {
	xd := []float64{-3, 0.5, 4} // −3 < Lo, 0.5 in range, 4 > Hi
	xT := tensor.FromFloat64(tensor.Shape{3}, xd)
	tape := autograd.NewTape()
	out, _ := backend.Execute(tape.Context(), backend.OpClip, []*tensor.Tensor{xT}, backend.ClipAttrs{Lo: -1, Hi: 1})
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{out[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(xT)
	want := []float64{0, 1, 0} // clamped ends → 0, in-range → passthrough (1)
	for i, w := range want {
		if g.AtF64(i) != w {
			t.Errorf("clip grad[%d]=%g, want %g", i, g.AtF64(i), w)
		}
	}
}
