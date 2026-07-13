package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// PrefixTuning is a prefix-tuning adapter (Li & Liang 2021, arXiv:2101.00190, §R133) — the per-layer
// key/value counterpart of prompt tuning. It prepends p learned continuous vectors (the "prefix") to
// an attention layer's keys and values: for K, V ∈ [seq, d] the attention then operates over
// [P_K ; K] and [P_V ; V] ∈ [p + seq, d]. The prefix is input-independent (the same for every input)
// and the query is unchanged; only the prefix parameters train, with all pretrained weights frozen
// (≈ 0.1 % of the model).
//
// Directly optimizing the prefix is unstable, so (paper §4.3) it is REPARAMETERIZED: a small learned
// embedding P' ∈ [p, k] is mapped up by an MLP — Linear(k→h) → tanh → Linear(h→2d) — whose output is
// split into P_K and P_V ∈ [p, d]. The MLP is only for training stability; at inference one can cache
// the resulting P_K/P_V and drop it. This is the single-attention-layer form; a full model uses one
// PrefixTuning per layer (each layer's K/V gets its own prefix).
//
// Contrast: prompt tuning (§R127) adds soft tokens only at the input embedding layer; adapters (§R132)
// insert bottleneck modules in the residual stream; prefix tuning injects them into attention K/V.
type PrefixTuning struct {
	Pemb           *tensor.Tensor // [p, k], the learned prefix embedding P'
	Trans1, Trans2 *Linear        // reparameterization MLP: k→hidden, hidden→2·d
	dModel         int
}

// NewPrefixTuning builds a prefix of prefixLen (p) virtual tokens for attention of width dModel (d),
// with a projDim-wide (k) prefix embedding and an encHidden-wide reparameterization MLP.
func NewPrefixTuning(dtype tensor.Dtype, prefixLen, dModel, projDim, encHidden int, seed uint64) (*PrefixTuning, error) {
	if prefixLen <= 0 || dModel <= 0 || projDim <= 0 || encHidden <= 0 {
		return nil, fmt.Errorf("nn: PrefixTuning needs positive prefixLen/dModel/projDim/encHidden, got %d/%d/%d/%d", prefixLen, dModel, projDim, encHidden)
	}
	pemb := tensor.New(dtype, tensor.Shape{prefixLen, projDim})
	fillUniform(pemb, -0.5, 0.5, seed)
	return &PrefixTuning{
		Pemb:   pemb,
		Trans1: NewLinear(dtype, projDim, encHidden, seed+1),
		Trans2: NewLinear(dtype, encHidden, 2*dModel, seed+2),
		dModel: dModel,
	}, nil
}

func (p *PrefixTuning) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// KV computes the prefix keys and values P_K, P_V ∈ [p, d] from the reparameterization MLP.
func (p *PrefixTuning) KV(ctx *backend.Context) (pk, pv *tensor.Tensor, err error) {
	h, err := p.Trans1.Forward(ctx, p.Pemb) // [p, hidden]
	if err != nil {
		return nil, nil, err
	}
	a, err := p.exec(ctx, backend.OpTanh, nil, h)
	if err != nil {
		return nil, nil, err
	}
	kv, err := p.Trans2.Forward(ctx, a) // [p, 2d]
	if err != nil {
		return nil, nil, err
	}
	if pk, err = p.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: p.dModel}, kv); err != nil {
		return nil, nil, err
	}
	if pv, err = p.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: p.dModel, End: 2 * p.dModel}, kv); err != nil {
		return nil, nil, err
	}
	return pk, pv, nil
}

// Apply prepends the learned prefix to an attention layer's keys and values: k,v ∈ [seq, d] →
// [p+seq, d]. Pass the returned k′,v′ (and the unchanged query) to the attention op.
func (p *PrefixTuning) Apply(ctx *backend.Context, k, v *tensor.Tensor) (kp, vp *tensor.Tensor, err error) {
	if k.Ndim() != 2 || k.Shape()[1] != p.dModel || !v.Shape().Equal(k.Shape()) {
		return nil, nil, fmt.Errorf("nn: PrefixTuning.Apply expects k,v [seq,%d] equal shapes, got k%v v%v", p.dModel, k.Shape(), v.Shape())
	}
	pk, pv, err := p.KV(ctx)
	if err != nil {
		return nil, nil, err
	}
	if kp, err = p.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 0}, pk, k); err != nil {
		return nil, nil, err
	}
	if vp, err = p.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 0}, pv, v); err != nil {
		return nil, nil, err
	}
	return kp, vp, nil
}

// PrefixLen returns the number of virtual prefix tokens p.
func (p *PrefixTuning) PrefixLen() int { return p.Pemb.Shape()[0] }

// Params returns the trainable prefix parameters (the embedding and the reparameterization MLP); the
// base model is frozen and external.
func (p *PrefixTuning) Params() []*tensor.Tensor {
	return []*tensor.Tensor{p.Pemb, p.Trans1.W, p.Trans1.B, p.Trans2.W, p.Trans2.B}
}
