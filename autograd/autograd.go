// Package autograd is layer L2: tape-based reverse-mode automatic
// differentiation (ADR-0006). A Tape implements backend.Recorder, so any op
// executed through a recording context (tape.Context()) is captured without any
// change to L1 kernels (§T5 seam). Backward walks the tape in reverse and
// applies per-op VJP rules from the registry (vjp.go, filled by §T14).
package autograd

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// node is one recorded forward op.
type node struct {
	op      backend.Op
	inputs  []*tensor.Tensor
	outputs []*tensor.Tensor
	attrs   backend.Attrs
}

// Tape records forward ops and computes gradients on Backward. Gradients are
// keyed by tensor pointer (ADR-0006). Not safe for concurrent recording.
type Tape struct {
	nodes   []node
	grads   map[*tensor.Tensor]*tensor.Tensor
	backend backend.Backend  // backend for forward (via Context) + backward (exec)
	exec    *backend.Context // non-recording context for backward computation
}

// NewTape returns an empty tape executing on the default backend.
func NewTape() *Tape { return NewTapeOn(backend.Default()) }

// NewTapeOn returns a tape whose forward AND backward ops dispatch to b (§T30:
// pass the metal/CUDA backend to run the heavy GEMMs of training on the GPU;
// ops b lacks fall back to the reference/cpu backend via Execute, §I4).
func NewTapeOn(b backend.Backend) *Tape {
	return &Tape{backend: b, exec: backend.NewContext().WithBackend(b)}
}

// Context returns a recording context on the tape's backend: ops executed
// through it are taped and run on that backend (GPU when set).
func (t *Tape) Context() *backend.Context {
	return backend.NewContext().WithBackend(t.backend).WithRecorder(t)
}

// Record implements backend.Recorder; Execute calls it after each forward op.
func (t *Tape) Record(op backend.Op, inputs, outputs []*tensor.Tensor, attrs backend.Attrs) {
	t.nodes = append(t.nodes, node{op: op, inputs: inputs, outputs: outputs, attrs: attrs})
}

// Len returns the number of recorded ops (test/introspection).
func (t *Tape) Len() int { return len(t.nodes) }

// Backward computes gradients of out with respect to every taped tensor,
// seeding ∂out/∂out = 1 (ones of out's shape, ADR-0006). Backward ops run in a
// non-recording context — the tape does not grow. It is an error if a needed op
// has no registered VJP.
func (t *Tape) Backward(out *tensor.Tensor) error { return t.backward(out, 1) }

// BackwardScaled is Backward with the output cotangent seeded to `scale` instead
// of 1 — the loss-scaling primitive for mixed-precision training (Micikevicius
// et al. 2018 §3): every gradient is scaled by `scale`, keeping small values out
// of the fp16 underflow range. Callers unscale by 1/scale before the optimizer
// step (see nn.MixedPrecision). Mathematically identical to scaling the loss.
func (t *Tape) BackwardScaled(out *tensor.Tensor, scale float64) error {
	return t.backward(out, scale)
}

func (t *Tape) backward(out *tensor.Tensor, seed float64) error {
	t.grads = map[*tensor.Tensor]*tensor.Tensor{out: scaledLike(out, seed)}

	for i := len(t.nodes) - 1; i >= 0; i-- {
		n := t.nodes[i]
		// Single-output ops for now; multi-output VJPs arrive with such ops.
		gout := t.grads[n.outputs[0]]
		if gout == nil {
			continue // this node does not influence `out`
		}
		rule, ok := vjps[n.op]
		if !ok {
			return fmt.Errorf("autograd: no VJP registered for op %v", n.op)
		}
		gins, err := rule(t.exec, n.inputs, n.outputs, n.attrs, gout)
		if err != nil {
			return fmt.Errorf("autograd: VJP %v: %w", n.op, err)
		}
		if len(gins) != len(n.inputs) {
			return fmt.Errorf("autograd: VJP %v returned %d grads for %d inputs", n.op, len(gins), len(n.inputs))
		}
		for k, gin := range gins {
			if gin == nil {
				continue // non-differentiable input slot
			}
			if err := t.accumulate(n.inputs[k], gin); err != nil {
				return err
			}
		}
	}
	return nil
}

// Grad returns the gradient of the last Backward target w.r.t. x, or nil if x
// was not reached.
func (t *Tape) Grad(x *tensor.Tensor) *tensor.Tensor { return t.grads[x] }

// accumulate adds g into the stored gradient of x (sum at fan-out points).
func (t *Tape) accumulate(x *tensor.Tensor, g *tensor.Tensor) error {
	prev := t.grads[x]
	if prev == nil {
		t.grads[x] = g
		return nil
	}
	sum, err := backend.Execute(t.exec, backend.OpAdd, []*tensor.Tensor{prev, g}, nil)
	if err != nil {
		return fmt.Errorf("autograd: grad accumulate: %w", err)
	}
	t.grads[x] = sum[0]
	return nil
}

// Variable pairs a value tensor with the tape that tracks it — a small
// convenience for user code.
type Variable struct {
	Value *tensor.Tensor
	tape  *Tape
}

// Var wraps x as a Variable on this tape.
func (t *Tape) Var(x *tensor.Tensor) *Variable { return &Variable{Value: x, tape: t} }

// Grad returns the gradient computed for this variable by the last Backward.
func (v *Variable) Grad() *tensor.Tensor { return v.tape.Grad(v.Value) }

// scaledLike returns a tensor of x's dtype/shape filled with v.
func scaledLike(x *tensor.Tensor, v float64) *tensor.Tensor {
	out := tensor.New(x.Dtype(), x.Shape())
	for i := range out.Numel() {
		out.SetF64(v, tensor.Unravel(i, out.Shape())...)
	}
	return out
}
