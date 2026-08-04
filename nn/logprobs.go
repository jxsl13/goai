package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TokenLogProbs returns the per-token log-probabilities [seq] of the target ids
// under logits [seq, vocab] (§T464) — the differentiable bridge from a language
// model's logits to every preference/RL objective in this package (DPO/IPO/KTO/
// SimPO/CPO take sequence log-probs, GRPOLoss/PPOClipLoss take per-token ones).
// Computed as a stable log-softmax gather composed from dispatched ops
// (max-shift → exp → sum → log, then a one-hot row gather), so gradients flow
// into logits through the standard VJPs and the computation runs on the active
// backend.
func TokenLogProbs(ctx *backend.Context, logits *tensor.Tensor, targets []int) (*tensor.Tensor, error) {
	if logits.Ndim() != 2 {
		return nil, fmt.Errorf("nn: TokenLogProbs needs logits [seq,vocab], got %v", logits.Shape())
	}
	seq, vocab := logits.Shape()[0], logits.Shape()[1]
	if len(targets) != seq {
		return nil, fmt.Errorf("nn: TokenLogProbs targets len %d != seq %d", len(targets), seq)
	}
	onehot := tensor.New(logits.Dtype(), tensor.Shape{seq, vocab})
	//perfscan:ignore PS1005 O(seq) sparse one-hot set « [seq,vocab] ops
	for i, t := range targets {
		if t < 0 || t >= vocab {
			return nil, fmt.Errorf("nn: TokenLogProbs target[%d]=%d out of vocab %d", i, t, vocab)
		}
		onehot.SetF64(1, i, t)
	}
	ex := func(op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
		out, err := backend.Execute(ctx, op, ins, attrs)
		if err != nil {
			return nil, err
		}
		return out[0], nil
	}
	lastAxis := backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}
	m, err := ex(backend.OpMax, lastAxis, logits)
	if err != nil {
		return nil, err
	}
	z, err := ex(backend.OpSub, nil, logits, m)
	if err != nil {
		return nil, err
	}
	e, err := ex(backend.OpExp, nil, z)
	if err != nil {
		return nil, err
	}
	s, err := ex(backend.OpSum, lastAxis, e)
	if err != nil {
		return nil, err
	}
	lse, err := ex(backend.OpLog, nil, s)
	if err != nil {
		return nil, err
	}
	logp, err := ex(backend.OpSub, nil, z, lse)
	if err != nil {
		return nil, err
	}
	picked, err := ex(backend.OpMul, nil, logp, onehot)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{1}}, picked)
}

// SequenceLogProbs returns the summed log-probability (a scalar tensor) of the
// whole target sequence under logits [seq, vocab] — the per-response quantity the
// preference losses (DPO/IPO/KTO/SimPO/CPO) consume; run it once per response and
// assemble their [batch] inputs from the results.
func SequenceLogProbs(ctx *backend.Context, logits *tensor.Tensor, targets []int) (*tensor.Tensor, error) {
	lp, err := TokenLogProbs(ctx, logits, targets)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3038 SequenceLogProbs trivial wrapper, single OpSum, no loop
	out, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{lp}, backend.ReduceAttrs{})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
