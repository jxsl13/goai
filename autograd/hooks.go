package autograd

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// Gradient hooks (torch's tensor.register_hook counterpart).
//
// # For the AI professional
//
// A gradient hook observes — and may replace — the gradient of one specific
// tensor in the middle of the backward pass, at the exact moment that gradient
// is fully computed. This is the primitive behind per-tensor gradient clipping,
// gradient logging/histograms, straight-through-style surgery, and debugging
// "which layer blew up".
//
// FIRING POINT. The tape walk gives a finality guarantee that picks the firing
// point for free: nodes are recorded in forward execution order, and Backward
// walks them strictly in reverse, so every CONSUMER of a tensor sits at a later
// tape index than its producer and is therefore processed EARLIER in the
// reverse walk. When the walk reaches node n, every consumer of n's outputs has
// already contributed its VJP into the gradient map — the accumulated cotangent
// of each output of n is FINAL. Hooks on those outputs fire right there, BEFORE
// n's VJP consumes the cotangent, so a replacement propagates through the VJP
// to everything upstream (the torch-consistent behavior: torch tensor hooks are
// pre-hooks of the producing grad_fn, called once with the fully-accumulated
// gradient). Two corollaries the tests pin down:
//
//   - Fan-out: a tensor consumed by several ops gets ONE hook call with the
//     SUMMED final cotangent, never one call per consumer — the sum is complete
//     by the time the producer node is reached.
//   - Upstream-only effect: a replacement affects gradients of tensors BEFORE
//     the hooked tensor in forward order; gradients already computed for
//     downstream tensors are untouched.
//
// LEAF tensors (model parameters, inputs — tensors never produced by a taped
// op) have no producer node, so the walk itself never makes their gradient
// "final" at a node; it is final when the walk ends. Leaf hooks fire then, in
// hook-registration order across tensors. [Tape.Grad] always returns the
// POST-hook gradient.
//
// Multiple hooks on one tensor compose in registration order (torch semantics):
// each hook receives the current gradient — the previous hook's replacement if
// there was one — and nil returns leave it unchanged.
//
// INTERACTIONS.
//
//   - Checkpoints ([Checkpoint]): the outer tape's hooks fire for outer-tape
//     tensors — a checkpoint segment's INPUTS and OUTPUTS are hookable, and a
//     hook on a segment output fires before the segment's sub-backward, so the
//     replacement seeds the recomputation. The segment's INTERNAL activations
//     are NOT hookable: they are rematerialized as fresh tensors on a private
//     sub-tape during every backward, so no stable *tensor.Tensor key for them
//     exists at registration time (the first-forward intermediates are
//     discarded — that is the point of checkpointing).
//   - Anomaly mode ([WithAnomalyDetection]) scans the POST-hook gradient: a
//     hook that manufactures NaN/±Inf from a finite gradient is caught and
//     attributed to that hook (index and firing site), not blamed on the
//     neighboring ops.
//   - Zero overhead when unused: a tape with no registered hooks pays one nil
//     map test per backward node (see BenchmarkForwardBackwardHooksOff).
//
// A hook fires only if the tensor's gradient is actually computed (a tensor
// that does not influence the backward target gets no gradient and no call),
// and fires once per Backward* call; hooks persist across calls on the same
// tape.
//
// # For the newcomer
//
// Backward normally runs as a black box: ops in, gradients out. A hook lets you
// put a probe on one tensor and say "when its gradient is ready, show it to me
// — and if I hand back a different tensor, use that instead". Observe-only
// hooks (return nil) are for looking: log the gradient's size, check it for
// surprises. Replacing hooks are for intervening: the classic use is clipping —
// if this layer's gradient explodes, shrink it before it poisons the layers
// below (see the RegisterGradHook example). The machinery guarantees your hook
// sees the finished gradient (all paths summed), exactly once, at the moment it
// is about to be used.

// GradHook is a per-tensor gradient hook (see [Tape.RegisterGradHook]): it is
// called with the tensor's fully-accumulated gradient and returns nil to keep
// it (observe-only) or a replacement tensor of the same dtype and shape. The
// hook must not mutate grad in place — return a new tensor instead (the same
// tensor may already be shared with other gradient-map entries, e.g. the
// BackwardGrad seed).
type GradHook func(grad *tensor.Tensor) *tensor.Tensor

// RegisterGradHook registers fn as a gradient hook on x: during every
// subsequent Backward/BackwardScaled/BackwardGrad on this tape, fn is called
// exactly once with x's fully-accumulated (final) gradient, at the moment that
// gradient is complete — for a tensor produced by a taped op, right before the
// producing op's VJP consumes it (so a replacement propagates upstream); for a
// leaf, at the end of the walk. fn returns nil to observe only, or a
// replacement gradient (dtype and shape must match — a mismatch fails the
// Backward call). Multiple hooks on the same tensor run in registration order,
// each seeing the previous one's result. Hooks never fire for tensors whose
// gradient is not computed. See the package discussion in hooks.go for the
// finality guarantee, and the example for per-tensor clipping.
//
// Like Grad, x is matched by pointer identity: hook the exact tensor that
// participates in the taped forward.
func (t *Tape) RegisterGradHook(x *tensor.Tensor, fn GradHook) {
	if t.hooks == nil {
		t.hooks = make(map[*tensor.Tensor][]GradHook)
	}
	if _, seen := t.hooks[x]; !seen {
		t.hookOrder = append(t.hookOrder, x)
	}
	t.hooks[x] = append(t.hooks[x], fn)
}

// fireOutputHooks runs the hooks of node n's outputs whose gradients are
// non-nil. Called from runBackward right before n's VJP (or checkpoint
// sub-backward) consumes those gradients — the point where the reverse walk
// guarantees they are final (every consumer already contributed; see the
// finality discussion above). idx is n's tape index (error messages).
func (t *Tape) fireOutputHooks(idx int, n node) error {
	for _, o := range n.outputs {
		if len(t.hooks[o]) == 0 {
			continue
		}
		t.hookFired[o] = true
		if t.grads[o] == nil {
			continue // o does not influence the backward target: no gradient, no hook call
		}
		var where string
		if n.ckpt != nil {
			where = fmt.Sprintf("output of checkpoint segment (tape index %d)", idx)
		} else {
			where = fmt.Sprintf("output of op %v (tape index %d)", n.op, idx)
		}
		if err := t.applyHooks(o, where); err != nil {
			return err
		}
	}
	return nil
}

// fireLeafHooks runs the hooks of tensors the walk did not fire — leaves,
// whose gradients only become final when the walk ends. Iteration follows
// hook-registration order (deterministic across runs).
func (t *Tape) fireLeafHooks() error {
	for _, x := range t.hookOrder {
		if t.hookFired[x] || t.grads[x] == nil {
			continue
		}
		if err := t.applyHooks(x, "leaf tensor (end of backward walk)"); err != nil {
			return err
		}
	}
	return nil
}

// applyHooks runs x's hook chain over its current gradient in registration
// order (nil return = keep, non-nil = validated replacement) and stores the
// result. In anomaly mode each replacement is scanned so a hook that
// manufactures a non-finite value from a finite gradient is attributed to that
// hook rather than to the neighboring ops.
func (t *Tape) applyHooks(x *tensor.Tensor, where string) error {
	g := t.grads[x]
	for hi, fn := range t.hooks[x] {
		r := fn(g)
		if r == nil {
			continue // observe-only: gradient unchanged
		}
		if r.Dtype() != g.Dtype() {
			return fmt.Errorf("autograd: grad hook %d on %s: replacement dtype %v != grad dtype %v",
				hi, where, r.Dtype(), g.Dtype())
		}
		if !r.Shape().Equal(g.Shape()) {
			return fmt.Errorf("autograd: grad hook %d on %s: replacement shape %v != grad shape %v",
				hi, where, r.Shape(), g.Shape())
		}
		if t.anomaly {
			if pos, v, bad := firstNonFinite(r); bad {
				if _, _, inBad := firstNonFinite(g); !inBad {
					return fmt.Errorf(
						"autograd: anomaly: grad hook %d on %s CREATED %s at element %d %v (the gradient passed to the hook was finite — the hook manufactured the non-finite value)",
						hi, where, classify(v), pos, tensor.Unravel(pos, r.Shape()))
				}
				// The incoming gradient was already non-finite: the walk's
				// created/propagated scans attribute the true source.
			}
		}
		g = r
	}
	t.grads[x] = g
	return nil
}
