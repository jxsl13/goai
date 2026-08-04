package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// MultiTokenLoss is the multi-token-prediction training objective (Gloeckle et al. 2024,
// arXiv:2404.19737, §R134) — instead of predicting only the next token, the model predicts the next n
// tokens at each position via n parallel heads on a shared trunk. Given the n heads' logits (heads[i]
// ∈ [seq, vocab] is the head that predicts i+1 steps ahead) and the target token sequence
// targets ∈ [seq], the loss is the plain UNWEIGHTED sum of the n cross-entropies, where head i's
// logits at position t are matched to the target i steps ahead (targets shifted left by i):
//
//	L = Σ_{i=1}^{n} CrossEntropy( heads[i−1][0:seq−i] , targets[i:seq] )   (Eq. 2, equal weight)
//
// Positions near the end that have no valid target i steps ahead are dropped (hence the slice to
// seq−i). With n = 1 this is exactly the standard next-token cross-entropy. The extra heads improve
// sample efficiency in training and enable self-speculative decoding at inference (or are discarded,
// using only head 1). All heads must share [seq, vocab]; the sequence must be longer than n.
func MultiTokenLoss(ctx *backend.Context, heads []*tensor.Tensor, targets *tensor.Tensor) (*tensor.Tensor, error) {
	n := len(heads)
	if n == 0 {
		return nil, fmt.Errorf("nn: MultiTokenLoss needs ≥1 head")
	}
	if targets.Ndim() != 1 {
		return nil, fmt.Errorf("nn: MultiTokenLoss targets must be [seq], got %v", targets.Shape())
	}
	seq := targets.Shape()[0]
	if seq <= n {
		return nil, fmt.Errorf("nn: MultiTokenLoss sequence length %d must exceed n=%d heads", seq, n)
	}
	for i, h := range heads {
		if h.Ndim() != 2 || h.Shape()[0] != seq {
			return nil, fmt.Errorf("nn: MultiTokenLoss head %d must be [%d, vocab], got %v", i, seq, h.Shape())
		}
		if !h.Shape().Equal(heads[0].Shape()) {
			return nil, fmt.Errorf("nn: MultiTokenLoss head %d shape %v != head 0 %v", i, h.Shape(), heads[0].Shape())
		}
	}

	exec := func(op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
		out, err := backend.Execute(ctx, op, ins, attrs)
		if err != nil {
			return nil, err
		}
		return out[0], nil
	}

	var total *tensor.Tensor
	//perfscan:ignore PS4011 per-head architectural fan-out; CrossEntropy matmul dominates dispatch
	for i, h := range heads {
		shift := i + 1
		//perfscan:ignore PS3024 resource-only variadic-alloc, no wall-clock win
		logits, err := exec(backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: 0, End: seq - shift}, h)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 resource-only variadic-alloc, no wall-clock win
		tgt, err := exec(backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: shift, End: seq}, targets)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 resource-only variadic-alloc, no wall-clock win
		ce, err := exec(backend.OpCrossEntropy, nil, logits, tgt)
		if err != nil {
			return nil, err
		}
		if total == nil {
			total = ce
			//perfscan:ignore PS3024 resource-only variadic-alloc, no wall-clock win
		} else if total, err = exec(backend.OpAdd, nil, total, ce); err != nil {
			return nil, err
		}
	}
	return total, nil
}
