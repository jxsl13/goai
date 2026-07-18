package autograd_test

import (
	"strings"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// opBadDouble is a synthetic custom op (y = 2x) whose registered VJP is
// DELIBERATELY wrong — it claims dy/dx = 1 instead of 2. This is the exact
// scenario the exported GradCheck exists to catch: a custom kernel whose
// hand-derived backward rule disagrees with its forward.
const opBadDouble = backend.Op(10001)

func init() {
	autograd.RegisterVJP(opBadDouble, func(_ *backend.Context, _, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		return []*tensor.Tensor{g}, nil // WRONG: the true rule is 2·g
	})
}

// badDoubleForward is the custom kernel: it computes y = 2x directly (no
// Execute dispatch — an out-of-tree kernel) and records itself on the tape the
// way Execute would, so Backward routes through the registered (bad) VJP.
func badDoubleForward(ctx *backend.Context, xs []*tensor.Tensor) (*tensor.Tensor, error) {
	x := xs[0]
	out := tensor.New(tensor.F64, x.Shape())
	for i := 0; i < x.Numel(); i++ {
		out.SetF64(2*x.AtF64(i), i)
	}
	if ctx.Recorder != nil {
		ctx.Recorder.Record(opBadDouble, []*tensor.Tensor{x}, []*tensor.Tensor{out}, nil)
	}
	return out, nil
}

// A known-good composition (mul → tanh, closed by GradCheck's internal Sum)
// passes with the default tolerances.
func TestGradCheckPassesComposition(t *testing.T) {
	f := func(ctx *backend.Context, xs []*tensor.Tensor) (*tensor.Tensor, error) {
		y, err := exec(ctx, backend.OpMul, nil, xs[0], xs[1])
		if err != nil {
			return nil, err
		}
		return exec(ctx, backend.OpTanh, nil, y)
	}
	inputs := []*tensor.Tensor{
		tensor.FromFloat64(tensor.Shape{2, 3}, []float64{0.8, -1.2, 2.1, 0.3, -0.5, 1.7}),
		tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1.1, 0.4, -0.9, 2.2, 0.6, -1.3}),
	}
	if err := autograd.GradCheck(f, inputs); err != nil {
		t.Errorf("known-good composition must pass: %v", err)
	}
}

// The wrong VJP is caught with a message that names the input, the element,
// and both the finite-difference and analytic values.
func TestGradCheckCatchesBadVJP(t *testing.T) {
	inputs := []*tensor.Tensor{tensor.FromFloat64(tensor.Shape{3}, []float64{0.4, -1.1, 2.0})}
	err := autograd.GradCheck(badDoubleForward, inputs)
	if err == nil {
		t.Fatal("GradCheck must reject a VJP that claims dy/dx=1 for y=2x")
	}
	for _, want := range []string{"input 0", "element 0", "finite-difference", "analytic 1", "disagrees"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("mismatch error missing %q: %v", want, err)
		}
	}
}

// Non-f64 inputs are rejected up front: finite differences at h≈1e-6 are
// rounding noise in f32 and below, so the check would reject correct VJPs.
func TestGradCheckRequiresF64(t *testing.T) {
	f := func(ctx *backend.Context, xs []*tensor.Tensor) (*tensor.Tensor, error) {
		return exec(ctx, backend.OpTanh, nil, xs[0])
	}
	err := autograd.GradCheck(f, []*tensor.Tensor{tensor.Full(tensor.F32, tensor.Shape{2}, 0.5)})
	if err == nil || !strings.Contains(err.Error(), "f64") {
		t.Errorf("f32 input must be rejected with an f64-requirement error, got %v", err)
	}
	if err := autograd.GradCheck(f, nil); err == nil {
		t.Error("empty inputs must be rejected")
	}
	err = autograd.GradCheck(f, []*tensor.Tensor{tensor.Full(tensor.F64, tensor.Shape{2}, 0.5)}, autograd.WithGradCheckEps(0))
	if err == nil || !strings.Contains(err.Error(), "positive") {
		t.Errorf("non-positive eps must be rejected, got %v", err)
	}
}

// Tolerance options are honored: an rtol far tighter than f64 FD noise makes
// even a correct VJP fail, proving the option reaches the comparison.
func TestGradCheckTolerances(t *testing.T) {
	f := func(ctx *backend.Context, xs []*tensor.Tensor) (*tensor.Tensor, error) {
		return exec(ctx, backend.OpExp, nil, xs[0])
	}
	inputs := []*tensor.Tensor{tensor.FromFloat64(tensor.Shape{3}, []float64{0.3, 1.1, -0.7})}
	if err := autograd.GradCheck(f, inputs); err != nil {
		t.Fatalf("exp must pass at default tolerances: %v", err)
	}
	if err := autograd.GradCheck(f, inputs, autograd.WithGradCheckRTol(1e-16)); err == nil {
		t.Error("rtol=1e-16 sits below f64 finite-difference noise and must fail")
	}
}
