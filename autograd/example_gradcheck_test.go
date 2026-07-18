// This file is a complete, runnable custom-op author's workflow: define a fused
// forward kernel, hand-derive its backward rule, register it with RegisterVJP,
// and prove the calculus with GradCheck before trusting it in training.
package autograd_test

import (
	"fmt"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// opFusedPoly is our custom op: y = x + x² fused into one kernel.
// Custom op IDs live outside the built-in backend.Op range.
const opFusedPoly = backend.Op(10002)

// The hand-derived backward rule: dy/dx = 1 + 2x, so dx = g·(1+2x).
// Registration happens once at package init, exactly like the built-in rules.
func init() {
	autograd.RegisterVJP(opFusedPoly, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		dx := tensor.New(tensor.F64, x.Shape())
		for i := 0; i < x.Numel(); i++ {
			dx.SetF64(g.AtF64(i)*(1+2*x.AtF64(i)), i)
		}
		return []*tensor.Tensor{dx}, nil
	})
}

// fusedPoly is the forward kernel. It computes the result directly and then
// records itself on the tape (when one is listening) the same way
// backend.Execute records built-in ops — that single Record call is what
// routes Backward through the VJP registered above.
func fusedPoly(ctx *backend.Context, x *tensor.Tensor) *tensor.Tensor {
	out := tensor.New(tensor.F64, x.Shape())
	for i := 0; i < x.Numel(); i++ {
		v := x.AtF64(i)
		out.SetF64(v+v*v, i)
	}
	if ctx.Recorder != nil {
		ctx.Recorder.Record(opFusedPoly, []*tensor.Tensor{x}, []*tensor.Tensor{out}, nil)
	}
	return out
}

// ExampleGradCheck verifies the custom rule: GradCheck runs fusedPoly under a
// fresh tape for the analytic gradients and compares every element against a
// central finite difference of the forward — the unforgiving ground truth. A
// nil error means the hand-derived calculus matches the kernel.
func ExampleGradCheck() {
	x := tensor.FromFloat64(tensor.Shape{4}, []float64{0.4, -1.2, 2.0, 0.1})
	err := autograd.GradCheck(func(ctx *backend.Context, xs []*tensor.Tensor) (*tensor.Tensor, error) {
		return fusedPoly(ctx, xs[0]), nil
	}, []*tensor.Tensor{x})
	fmt.Println("custom VJP verified:", err == nil)
	// Output: custom VJP verified: true
}
