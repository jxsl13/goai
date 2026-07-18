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

// node is one recorded forward op. A node with a non-nil ckpt is a gradient-checkpoint segment
// (§T162): its intermediate activations were NOT retained on the forward pass and are
// rematerialized by re-running ckpt.fn during backward (see checkpoint.go).
type node struct {
	op      backend.Op
	inputs  []*tensor.Tensor
	outputs []*tensor.Tensor
	attrs   backend.Attrs
	ckpt    *checkpoint
}

// Tape records forward ops and computes gradients on Backward. Gradients are
// keyed by tensor pointer (ADR-0006). Not safe for concurrent recording.
type Tape struct {
	nodes   []node
	grads   map[*tensor.Tensor]*tensor.Tensor
	backend backend.Backend  // backend for forward (via Context) + backward (exec)
	exec    *backend.Context // non-recording context for backward computation

	anomaly    bool  // anomaly-detection mode (anomaly.go); off by default
	anomalyErr error // first forward-pass anomaly; surfaced by Err and Backward*
}

// NewTape returns an empty tape executing on the default backend. Options
// (e.g. [WithAnomalyDetection]) configure the tape.
func NewTape(opts ...TapeOption) *Tape { return NewTapeOn(backend.Default(), opts...) }

// NewTapeOn returns a tape whose forward AND backward ops dispatch to b (§T30:
// pass the metal/CUDA backend to run the heavy GEMMs of training on the GPU;
// ops b lacks fall back to the reference/cpu backend via Execute, §I4).
func NewTapeOn(b backend.Backend, opts ...TapeOption) *Tape {
	t := &Tape{backend: b, exec: backend.NewContext().WithBackend(b)}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// Context returns a recording context on the tape's backend: ops executed
// through it are taped and run on that backend (GPU when set).
func (t *Tape) Context() *backend.Context {
	return backend.NewContext().WithBackend(t.backend).WithRecorder(t)
}

// Record implements backend.Recorder; Execute calls it after each forward op.
// In anomaly mode ([WithAnomalyDetection]) it additionally scans the op's
// outputs for NaN/±Inf; a hit is stored (first wins) because Recorder cannot
// return an error — see [Tape.Err] and anomaly.go.
func (t *Tape) Record(op backend.Op, inputs, outputs []*tensor.Tensor, attrs backend.Attrs) {
	t.nodes = append(t.nodes, node{op: op, inputs: inputs, outputs: outputs, attrs: attrs})
	if t.anomaly && t.anomalyErr == nil {
		t.anomalyErr = t.checkForward(len(t.nodes)-1, op, inputs, outputs)
	}
}

// Len returns the number of recorded ops (test/introspection).
func (t *Tape) Len() int { return len(t.nodes) }

// Backward computes gradients of out with respect to every taped tensor,
// seeding ∂out/∂out = 1 (ones of out's shape, ADR-0006). Backward ops run in a
// non-recording context — the tape does not grow. It is an error if a needed op
// has no registered VJP.
//
// Each call REPLACES the tape's gradient map: gradients from a previous
// Backward/BackwardScaled/BackwardGrad on the same tape are discarded, not
// added to. To accumulate gradients across several backward passes (micro-
// batching), sum the per-call [Tape.Grad] results into your own buffers —
// [github.com/jxsl13/goai/nn.GradAccumulator] does exactly that.
//
// Backward is BackwardGrad with a ones cotangent.
func (t *Tape) Backward(out *tensor.Tensor) error {
	return t.BackwardGrad(out, scaledLike(out, 1))
}

// BackwardScaled is Backward with the output cotangent seeded to `scale` instead
// of 1 — the loss-scaling primitive for mixed-precision training (Micikevicius
// et al. 2018 §3): every gradient is scaled by `scale`, keeping small values out
// of the fp16 underflow range. Callers unscale by 1/scale before the optimizer
// step (see nn.MixedPrecision). Mathematically identical to scaling the loss.
func (t *Tape) BackwardScaled(out *tensor.Tensor, scale float64) error {
	return t.BackwardGrad(out, scaledLike(out, scale))
}

// BackwardGrad computes gradients of out with an explicit output cotangent —
// the general vector-Jacobian-product (VJP) primitive. Where [Tape.Backward]
// answers "how does each input move the (scalar) loss?", BackwardGrad answers
// "how does each input move the specific output direction c?": for a forward
// map y = f(x) it computes ∂(c·y)/∂x = Jᶠ(x)ᵀ·c, pulling a covector c in
// OUTPUT space back to gradients in every INPUT space. It is the tape
// counterpart of PyTorch's y.backward(gradient=c) and the function JAX's
// vjp returns.
//
// For the newcomer: Backward asks the tape "what nudges reduce this one final
// number?". Sometimes the final result is not one number but many (say, a
// whole row of activations), and you care about a particular weighted
// combination of them. The cotangent c holds those weights: BackwardGrad(y, c)
// gives exactly the gradients Backward would give for the single number
// sum(y⊙c), without your having to build that product into the graph.
//
// The cotangent must match out's dtype and shape. As with Backward, the call
// replaces the tape's gradient map, and the seeded cotangent itself becomes
// Grad(out) (the tensor is referenced, not copied — do not mutate it until
// the gradients have been consumed).
func (t *Tape) BackwardGrad(out *tensor.Tensor, cotangent *tensor.Tensor) error {
	if cotangent == nil {
		return fmt.Errorf("autograd: BackwardGrad: nil cotangent")
	}
	if cotangent.Dtype() != out.Dtype() {
		return fmt.Errorf("autograd: BackwardGrad: cotangent dtype %v != output dtype %v", cotangent.Dtype(), out.Dtype())
	}
	if !cotangent.Shape().Equal(out.Shape()) {
		return fmt.Errorf("autograd: BackwardGrad: cotangent shape %v != output shape %v", cotangent.Shape(), out.Shape())
	}
	if t.anomalyErr != nil {
		return t.anomalyErr // a forward-pass anomaly must never be silently swallowed
	}
	t.grads = map[*tensor.Tensor]*tensor.Tensor{out: cotangent}
	return t.runBackward()
}

// runBackward walks the tape in reverse applying each node's VJP (or, for a checkpoint node,
// rematerializing its activations), assuming t.grads is already seeded with the output cotangents.
// Split out from backward so a checkpoint's recomputation sub-tape can be driven with a
// pre-seeded, multi-output gradient map.
func (t *Tape) runBackward() error {
	for i := len(t.nodes) - 1; i >= 0; i-- {
		n := t.nodes[i]
		if n.ckpt != nil {
			if err := t.backwardCheckpoint(n); err != nil {
				return err
			}
			continue
		}
		// Multi-output ops (e.g. QR → Q,R) route through the multi-output VJP,
		// which receives every output's cotangent (zero where unused).
		if len(n.outputs) > 1 {
			if err := t.backwardMulti(i, n); err != nil {
				return err
			}
			continue
		}
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
		if t.anomaly {
			if err := t.checkBackward(i, n, []*tensor.Tensor{gout}, gins); err != nil {
				return err
			}
		}
		if err := t.accumulateGrads(n, gins); err != nil {
			return err
		}
	}
	return nil
}

// backwardMulti applies a multi-output op's VJP, gathering every output cotangent
// (a zero tensor where the output did not reach the loss). Skipped when no output
// influences the loss. i is the node's tape index (anomaly reporting).
func (t *Tape) backwardMulti(i int, n node) error {
	gouts := make([]*tensor.Tensor, len(n.outputs))
	any := false
	for j, o := range n.outputs {
		if g := t.grads[o]; g != nil {
			gouts[j] = g
			any = true
		} else {
			gouts[j] = tensor.New(o.Dtype(), o.Shape()) // zero cotangent for an unused output
		}
	}
	if !any {
		return nil // this node does not influence `out`
	}
	rule, ok := vjpsMulti[n.op]
	if !ok {
		return fmt.Errorf("autograd: no multi-output VJP registered for op %v", n.op)
	}
	gins, err := rule(t.exec, n.inputs, n.outputs, n.attrs, gouts)
	if err != nil {
		return fmt.Errorf("autograd: multi-output VJP %v: %w", n.op, err)
	}
	if t.anomaly {
		if err := t.checkBackward(i, n, gouts, gins); err != nil {
			return err
		}
	}
	return t.accumulateGrads(n, gins)
}

// accumulateGrads adds each returned input gradient into the tape (nil = a
// non-differentiable input slot), validating the count.
func (t *Tape) accumulateGrads(n node, gins []*tensor.Tensor) error {
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
	return nil
}

// Grad returns the gradient of the last Backward target w.r.t. x, or nil if x
// was not reached.
//
// Gradients are keyed by *tensor.Tensor POINTER identity (ADR-0006): x must be
// the very Tensor value that participated in the taped forward, not a copy.
// In particular, a view (reshape/slice/transpose result) and its base share
// storage but are distinct pointers — the tape does NOT unify them, so a
// gradient stored for a view is not visible through the base tensor (and vice
// versa); query the exact tensor you fed to the recorded op.
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
	Value *tensor.Tensor // the wrapped forward-pass value tensor
	tape  *Tape
}

// Var wraps x as a Variable on this tape.
func (t *Tape) Var(x *tensor.Tensor) *Variable { return &Variable{Value: x, tape: t} }

// Grad returns the gradient computed for this variable by the last Backward.
func (v *Variable) Grad() *tensor.Tensor { return v.tape.Grad(v.Value) }

// scaledLike returns a tensor of x's dtype/shape filled with v.
func scaledLike(x *tensor.Tensor, v float64) *tensor.Tensor {
	return tensor.Full(x.Dtype(), x.Shape(), v)
}
