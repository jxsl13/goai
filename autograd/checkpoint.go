package autograd

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// CheckpointFunc is the forward of a checkpointed segment: it builds ops on ctx from inputs and
// returns the segment outputs. It MUST be deterministic (same inputs → same outputs) and free of
// side effects, because Checkpoint runs it a second time during backward to regenerate the
// intermediate activations; a stochastic segment (e.g. containing dropout with fresh RNG) would
// recompute different activations and yield wrong gradients (§R109).
type CheckpointFunc func(ctx *backend.Context, inputs []*tensor.Tensor) ([]*tensor.Tensor, error)

// checkpoint holds a checkpoint node's recomputation closure and its inputs.
type checkpoint struct {
	fn     CheckpointFunc
	inputs []*tensor.Tensor
}

// Checkpoint runs a segment's forward WITHOUT recording its intermediate activations, trading
// memory for one extra forward pass — gradient checkpointing / activation rematerialization (Chen
// et al. 2016, "Training Deep Nets with Sublinear Memory Cost", arXiv:1604.06174; §R109). The
// forward runs on the tape's non-recording context, so only the segment's inputs and final
// outputs are retained (the intermediates are garbage-collected); a single checkpoint node is
// recorded. On Backward the segment's forward is RE-RUN — this time recording into a private
// sub-tape — to rematerialize the activations, the incoming output cotangents seed the sub-tape,
// and its reverse pass produces the input gradients. Because the ops are deterministic, the
// gradients are IDENTICAL (bit-for-bit) to running the segment without checkpointing; only memory
// and compute change. Placing a checkpoint every √n blocks of an n-block network cuts activation
// memory from O(n) to O(√n) for ~one extra forward (§R109).
//
// fn must be deterministic (see CheckpointFunc). Wrap the memory-heavy repeated segments of a
// deep model — e.g. each transformer block — in Checkpoint to train longer sequences or bigger
// models within a fixed activation-memory budget.
func Checkpoint(t *Tape, fn CheckpointFunc, inputs ...*tensor.Tensor) ([]*tensor.Tensor, error) {
	// First forward: NON-recording, so the segment's intermediate activations are not retained.
	outputs, err := fn(t.exec, inputs)
	if err != nil {
		return nil, fmt.Errorf("autograd: checkpoint forward: %w", err)
	}
	if len(outputs) == 0 {
		return nil, fmt.Errorf("autograd: checkpoint segment produced no outputs")
	}
	in := append([]*tensor.Tensor(nil), inputs...)
	t.nodes = append(t.nodes, node{
		inputs:  in,
		outputs: outputs,
		ckpt:    &checkpoint{fn: fn, inputs: in},
	})
	return outputs, nil
}

// backwardCheckpoint rematerializes a checkpoint node's activations by re-running its forward on a
// fresh recording sub-tape, seeds that sub-tape with the incoming output cotangents, runs its
// reverse pass, and propagates the resulting input gradients into the parent tape.
func (t *Tape) backwardCheckpoint(n node) error {
	// Skip if no downstream gradient reached any of this segment's outputs.
	any := false
	for _, o := range n.outputs {
		if t.grads[o] != nil {
			any = true
			break
		}
	}
	if !any {
		return nil
	}
	sub := NewTapeOn(t.backend)
	subOuts, err := n.ckpt.fn(sub.Context(), n.ckpt.inputs)
	if err != nil {
		return fmt.Errorf("autograd: checkpoint recompute: %w", err)
	}
	if len(subOuts) != len(n.outputs) {
		return fmt.Errorf("autograd: checkpoint recompute produced %d outputs, first forward gave %d",
			len(subOuts), len(n.outputs))
	}
	// Seed the sub-tape with the parent's cotangents on the matching (positional) outputs.
	sub.grads = make(map[*tensor.Tensor]*tensor.Tensor, len(subOuts))
	for i, o := range n.outputs {
		if g := t.grads[o]; g != nil {
			sub.grads[subOuts[i]] = g
		}
	}
	if err := sub.runBackward(); err != nil {
		return fmt.Errorf("autograd: checkpoint sub-backward: %w", err)
	}
	// Propagate gradients w.r.t. the segment inputs back into the parent tape.
	for k, in := range n.ckpt.inputs {
		if g := sub.Grad(in); g != nil {
			if err := t.accumulate(n.inputs[k], g); err != nil {
				return err
			}
		}
	}
	return nil
}
