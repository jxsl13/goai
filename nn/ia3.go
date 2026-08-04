package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// IA3 is an (IA)³ rescaling adapter (Liu et al. 2022, arXiv:2205.05638, §R122) —
// "Infused Adapter by Inhibiting and Amplifying Inner Activations". It is the
// lightest parameter-efficient fine-tuning primitive: a single learned per-feature
// vector l[d] that element-wise rescales an inner activation,
//
//	y[..., j] = l[j] · x[..., j]
//
// with l broadcast over every leading (sequence/batch) position. l is INITIALIZED
// TO ONES, so a fresh adapter is an exact identity and the model starts equal to
// the frozen pretrained base; only l trains. A full (IA)³ adaptation of a
// Transformer block places THREE of these — on the attention keys (l_k), the
// attention values (l_v) and the feed-forward hidden activations (l_ff, applied to
// γ(W1·x) before the down-projection) — while all pretrained weights stay frozen.
// At d parameters per adapter it is far cheaper than LoRA (§R27) or DoRA (§R70).
type IA3 struct {
	Scale *tensor.Tensor // [d], trainable per-feature rescaling, init 1
}

// NewIA3 builds an (IA)³ adapter for activations whose last dimension is d, with
// the rescaling vector initialized to ones (identity at start).
func NewIA3(d int, dtype tensor.Dtype) (*IA3, error) {
	if d <= 0 {
		return nil, fmt.Errorf("nn: IA3 dim must be positive, got %d", d)
	}
	s := tensor.New(dtype, tensor.Shape{d})
	for j := range d {
		s.SetF64(1, j)
	}
	return &IA3{Scale: s}, nil
}

// Forward rescales x[..., d] by the learned vector: y[..., j] = Scale[j]·x[..., j].
func (a *IA3) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	//perfscan:ignore PS3038 resource-only, time flat (alloc-only)
	out, err := backend.Execute(ctx, backend.OpIA3, []*tensor.Tensor{x, a.Scale}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Params returns the single trainable rescaling vector (the base model is frozen).
func (a *IA3) Params() []*tensor.Tensor { return []*tensor.Tensor{a.Scale} }
