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
	// FUSED GATHER/SCATTER around the one arithmetic op (ADR-01KYQ9PHNPEFC). The dispatch path
	// below spends eight backend ops — reshape, two slices, reshape, RoPE, reshape, concat,
	// reshape — of which exactly one does arithmetic. The other seven are pure layout, and this
	// helper is called from 31 sites, so it was 51% of a GPTNeoX decode step's allocations.
	//
	// Bit-identical by construction, and the layout equivalence is worth spelling out because
	// it is the whole proof: the dispatch path's rotWide has row s equal to the concatenation
	// over h of x[s, h*hd : h*hd+rotaryDim], which is exactly what the gather below builds; the
	// RoPE call is the same op with the same attrs on the same shape; and the scatter rebuilds
	// row s as the concatenation over h of (rotated prefix, untouched tail), which is what the
	// concat-plus-reshape produced.
	//
	// Gated on ctx.Recorder == nil: under a tape every one of those seven layout ops is a
	// gradient edge, and replacing them with raw copies would detach the graph.
	if ctx.Recorder == nil && x.Dtype() == tensor.F32 {
		xf := x.Contiguous().Storage().F32()
		if len(xf) >= seq*heads*hd {
			rotWide := tensor.New(tensor.F32, tensor.Shape{seq, heads * rotaryDim})
			rw := rotWide.Storage().F32()
			for sIdx := range seq {
				for h := range heads {
					src := sIdx*heads*hd + h*hd
					dst := sIdx*heads*rotaryDim + h*rotaryDim
					copy(rw[dst:dst+rotaryDim], xf[src:src+rotaryDim])
				}
			}
			r := rope
			r.Heads = heads
			rotated, err := exec1a(ctx, backend.OpRoPE, r, rotWide)
			if err != nil {
				return nil, err
			}
			rf := rotated.Contiguous().Storage().F32()
			out := tensor.New(tensor.F32, tensor.Shape{seq, heads * hd})
			of := out.Storage().F32()
			for sIdx := range seq {
				for h := range heads {
					dst := sIdx*heads*hd + h*hd
					srcRot := sIdx*heads*rotaryDim + h*rotaryDim
					copy(of[dst:dst+rotaryDim], rf[srcRot:srcRot+rotaryDim])
					copy(of[dst+rotaryDim:dst+hd], xf[dst+rotaryDim:dst+hd])
				}
			}
			return out, nil
		}
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
