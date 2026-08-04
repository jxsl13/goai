package autograd

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Exported numeric gradient checking (torch.autograd.gradcheck counterpart).
// The same central finite-difference harness that validates this package's own
// VJP registry (§V2) is exposed here for CUSTOM op authors: anyone who wires a
// new kernel + RegisterVJP pair can prove their backward rule against the
// forward, instead of trusting hand-derived calculus.

// gradCheckCfg holds GradCheck's tolerances (see the options for the defaults).
type gradCheckCfg struct {
	eps  float64 // central-difference step h
	rtol float64 // relative tolerance, with a floor of 1 on the scale (|Δ| ≤ rtol·max(1,|fd|))
}

// GradCheckOption configures [GradCheck].
type GradCheckOption func(*gradCheckCfg)

// WithGradCheckEps sets the central-difference step h (default 1e-6). Smaller h
// reduces the O(h²) truncation error but amplifies f64 roundoff in the
// difference quotient; 1e-6 balances the two for well-scaled inputs.
func WithGradCheckEps(eps float64) GradCheckOption {
	return func(c *gradCheckCfg) { c.eps = eps }
}

// WithGradCheckRTol sets the relative tolerance (default 1e-5, the bound this
// package's own op suite holds; the spec's acceptance bound is 1e-4, §V2). An
// element fails when |fd−analytic| > rtol·max(1, |fd|) — the max(1,·) floor
// makes the comparison absolute for near-zero gradients.
func WithGradCheckRTol(rtol float64) GradCheckOption {
	return func(c *gradCheckCfg) { c.rtol = rtol }
}

// GradCheck verifies the tape's analytic gradients of f against central finite
// differences, element by element. f builds the computation on ctx from inputs
// and returns its output; a non-scalar output is reduced to a scalar loss via
// Sum (so every registered reduction/VJP composition is exercised the same way
// the package's own §V2 suite does it). GradCheck runs f once under a fresh
// recording tape for the analytic gradients, then re-evaluates f at x±h per
// input element for the numeric ones, and returns nil when every element of
// every input gradient matches within the tolerances.
//
// On a mismatch the error pinpoints the failure: which input, which element
// (flat index and coordinates), the finite-difference and analytic values, and
// the tolerances in force — everything needed to debug a hand-derived VJP.
//
// Inputs must be f64: at h≈1e-6 a central difference in f32 (machine epsilon
// ≈1.2e-7) is pure rounding noise, and f16/bf16 are hopeless — the check would
// reject correct VJPs. Non-f64 inputs are rejected with an error (the same
// discipline as torch.autograd.gradcheck, which requires double precision).
//
// f is evaluated 2·N+1 times for N total input elements — this is a TEST-TIME
// tool for small synthetic inputs, not something to run inside a training loop.
//
// For the newcomer: a gradient has an unforgiving ground truth — nudge one
// input a hair, watch the output move, divide. GradCheck automates exactly
// that (with a symmetric nudge, which cancels the leading error term) and
// compares it against what Backward computed through calculus. If the two
// agree everywhere, the backward rule is almost certainly right; if they
// disagree, the error tells you exactly where to look. Run it whenever you
// write a custom gradient rule ([RegisterVJP]) — see the example.
func GradCheck(f func(ctx *backend.Context, inputs []*tensor.Tensor) (*tensor.Tensor, error), inputs []*tensor.Tensor, opts ...GradCheckOption) error {
	cfg := gradCheckCfg{eps: 1e-6, rtol: 1e-5}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.eps <= 0 || cfg.rtol <= 0 {
		return fmt.Errorf("autograd: GradCheck: eps and rtol must be positive (eps=%g rtol=%g)", cfg.eps, cfg.rtol)
	}
	if f == nil {
		return fmt.Errorf("autograd: GradCheck: nil function")
	}
	if len(inputs) == 0 {
		return fmt.Errorf("autograd: GradCheck: no inputs")
	}
	for i, x := range inputs {
		if x == nil {
			return fmt.Errorf("autograd: GradCheck: input %d is nil", i)
		}
		if x.Dtype() != tensor.F64 {
			return fmt.Errorf("autograd: GradCheck: input %d is %v; finite differences need f64 (a central difference at h=%g is rounding noise in smaller dtypes)",
				i, x.Dtype(), cfg.eps)
		}
	}

	// Snapshot the inputs' logical values (views/non-contiguous handled by the
	// logical accessor); FD evals rebuild fresh contiguous tensors from these.
	data := make([][]float64, len(inputs))
	shapes := make([]tensor.Shape, len(inputs))
	for i, x := range inputs {
		shapes[i] = x.Shape()
		//perfscan:ignore PS2008,PS3064 gradcheck snapshot alloc, resource-only test util | alloc, gradcheck verification utility
		data[i] = make([]float64, x.Numel())
		//perfscan:ignore PS1001,PS4006 gradcheck snapshot AtF64, reference test tool | gradcheck snapshot loop, test utility
		for j := range data[i] {
			data[i][j] = x.AtF64(tensor.Unravel(j, shapes[i])...)
		}
	}

	// Analytic pass: f under a fresh recording tape, reduced to a scalar loss.
	tape := NewTape()
	loss, err := gradCheckScalar(tape.Context(), f, inputs)
	if err != nil {
		return fmt.Errorf("autograd: GradCheck: forward: %w", err)
	}
	if err := tape.Backward(loss); err != nil {
		return fmt.Errorf("autograd: GradCheck: backward: %w", err)
	}
	grads := make([]*tensor.Tensor, len(inputs))
	for i, x := range inputs {
		if grads[i] = tape.Grad(x); grads[i] == nil {
			return fmt.Errorf("autograd: GradCheck: input %d received no gradient (the output does not depend on it, or it feeds a non-differentiable slot)", i)
		}
	}

	// Numeric pass: central difference per element on rebuilt inputs.
	eval := func(d [][]float64) (float64, error) {
		xs := make([]*tensor.Tensor, len(d))
		for i := range d {
			xs[i] = tensor.FromFloat64(shapes[i], append([]float64(nil), d[i]...))
		}
		out, err := gradCheckScalar(backend.NewContext(), f, xs)
		if err != nil {
			return 0, err
		}
		return out.AtF64(), nil
	}
	for i := range inputs {
		//perfscan:ignore PS1001 finite-diff checker, O(Numel) by design, test tool
		for j := range data[i] {
			orig := data[i][j]
			data[i][j] = orig + cfg.eps
			fp, err := eval(data)
			if err != nil {
				return fmt.Errorf("autograd: GradCheck: forward at input %d element %d +eps: %w", i, j, err)
			}
			data[i][j] = orig - cfg.eps
			fm, err := eval(data)
			if err != nil {
				return fmt.Errorf("autograd: GradCheck: forward at input %d element %d -eps: %w", i, j, err)
			}
			data[i][j] = orig
			fd := (fp - fm) / (2 * cfg.eps)
			coords := tensor.Unravel(j, shapes[i])
			analytic := grads[i].AtF64(coords...)
			//perfscan:ignore PS3082 gradcheck compare, verification utility not hot path
			if math.Abs(fd-analytic) > cfg.rtol*math.Max(1, math.Abs(fd)) {
				return fmt.Errorf(
					"autograd: GradCheck: input %d element %d %v: finite-difference %.10g vs analytic %.10g (|Δ|=%.3g > rtol %g · max(1,|fd|), eps %g) — the VJP producing this gradient disagrees with the forward",
					i, j, coords, fd, analytic, math.Abs(fd-analytic), cfg.rtol, cfg.eps)
			}
		}
	}
	return nil
}

// gradCheckScalar runs f on ctx and reduces a non-scalar output to Sum(output).
func gradCheckScalar(ctx *backend.Context, f func(ctx *backend.Context, inputs []*tensor.Tensor) (*tensor.Tensor, error), xs []*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := f(ctx, xs)
	if err != nil {
		return nil, err
	}
	if out == nil {
		return nil, fmt.Errorf("f returned a nil output")
	}
	if out.Ndim() == 0 && out.Numel() == 1 {
		return out, nil
	}
	sum, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{out}, nil)
	if err != nil {
		return nil, fmt.Errorf("scalarizing output via sum: %w", err)
	}
	return sum[0], nil
}
