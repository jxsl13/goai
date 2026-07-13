package ops

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SoftCap applies the logit soft-cap y = cap·tanh(x/cap) elementwise (Gemma-2,
// arXiv:2408.00118): a smooth, differentiable bound that leaves small values almost
// unchanged (≈ x for |x| ≪ cap) and saturates gently to ±cap as x grows, keeping
// logits in the open interval (−cap, cap) without the hard kink of clipping. cap
// must be > 0.
//
// Gemma-2 uses it in the forward pass at two points: on the attention scores
// QKᵀ/√d before softmax (cap = 50) and on the final LM-head logits before the loss
// (cap = 30). It is fully differentiable — dy/dx = 1 − (y/cap)² — so it participates
// in training, not just inference.
func SoftCap(x *tensor.Tensor, cap float64) (*tensor.Tensor, error) {
	return eagerA(backend.OpSoftCap, backend.SoftCapAttrs{Cap: cap}, x)
}
