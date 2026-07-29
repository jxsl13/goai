package nlp

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// partialRoPE applies rotary position embedding to only the first rotaryDim channels of
// each head's headDim slice, leaving the remaining channels unrotated — the "partial
// rotary" of GPT-NeoX / StableLM / Phi (config partial_rotary_factor / rotary_pct < 1).
// x is [seq, heads·headDim]; rope carries Base and PosOffset (Heads is set internally).
// When rotaryDim >= headDim it degrades to a full [OpRoPE] over every channel.
//
// GPT-NeoX-family partial rotary uses the split-half convention on the rotated portion —
// the same convention as GoAI's OpRoPE — so the rotated channels need no interleave
// permutation (unlike Cohere's fully-interleaved rotary, see permuteInterleaveToSplit).
// Every step is a differentiable op (reshape/slice/rope/concat all carry VJPs), so a
// partial-rotary model built on this helper remains trainable.
func partialRoPE(ctx *backend.Context, x *tensor.Tensor, heads, rotaryDim int, rope backend.RoPEAttrs) (*tensor.Tensor, error) {
	seq := x.Shape()[0]
	hd := x.Shape()[1] / heads
	if rotaryDim >= hd {
		r := rope
		r.Heads = heads
		return exec1a(ctx, backend.OpRoPE, r, x)
	}
	// Reshape to one row per (position, head) so each head's channels are contiguous.
	flat, err := exec1(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{seq * heads, hd}}, x)
	if err != nil {
		return nil, err
	}
	// Split each head into the rotated prefix [0,rotaryDim) and the pass-through tail.
	rot, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: rotaryDim}, flat)
	if err != nil {
		return nil, err
	}
	pass, err := exec1(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: rotaryDim, End: hd}, flat)
	if err != nil {
		return nil, err
	}
	// Rotate the prefix: view as [seq, heads·rotaryDim] and RoPE each head's rotaryDim slice.
	rotWide, err := exec1(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{seq, heads * rotaryDim}}, rot)
	if err != nil {
		return nil, err
	}
	r := rope
	r.Heads = heads
	if rotWide, err = exec1a(ctx, backend.OpRoPE, r, rotWide); err != nil {
		return nil, err
	}
	if rot, err = exec1(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{seq * heads, rotaryDim}}, rotWide); err != nil {
		return nil, err
	}
	// Reassemble rotated prefix + pass-through tail, then restore [seq, heads·headDim].
	merged, err := exec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, rot, pass)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{seq, heads * hd}}, merged)
}
