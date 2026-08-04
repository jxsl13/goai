package nn

import (
	"fmt"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Embedding is a lookup table: it maps integer ids (tokens, positions, segment
// types) to dense rows of a trainable matrix W[vocab, dim], so out[i,:] =
// W[ids[i], :]. It is the first layer of essentially every language model.
//
// In plain terms: a vocabulary-sized array of learned vectors, indexed by token
// id. The lookup is differentiable — the backward pass scatter-adds each output
// row's gradient back into the row it came from, so only the rows a batch actually
// used receive an update, and a repeated id accumulates.
//
// IDS ARE PASSED AS A FLOAT TENSOR. The backend's row-gather op takes indices in
// a rank-1 tensor of the SAME DTYPE AS THE TABLE, with the integer value stored
// exactly as a float (§B12 — an f32 holds every integer below 2^24 exactly, and
// vocabularies are far smaller). This matches the six hand-rolled lookups already
// in the tree (nlp/gpt.go, nlp/llama.go, nlp/bert.go, nlp/eagle.go,
// llamagpu/bert.go); [Embedding.Forward] takes that same tensor so the layer is a
// drop-in replacement for them. [Embedding.Lookup] is the convenience form that
// takes a []int and builds the index tensor for you.
//
// The forward pass is exactly one OpEmbed through the dispatch — no extra
// arithmetic — so it runs on whichever backend the context carries (Metal and
// Vulkan have hand-written kernels) and is bit-identical to the hand-rolled sites.
type Embedding struct {
	W *tensor.Tensor // [vocab, dim]; the trainable table

	// PadIdx is the padding row, or -1 for none. See [NewEmbeddingPadded] for
	// what it does and — importantly — what it does not do on its own.
	PadIdx int
}

// NewEmbedding builds a [vocab, dim] embedding table initialised N(0,1) —
// torch.nn.Embedding's default — with no padding row. Deterministic via seed.
// Panics on a non-positive dimension.
func NewEmbedding(dtype tensor.Dtype, vocab, dim int, seed uint64) *Embedding {
	return NewEmbeddingPadded(dtype, vocab, dim, -1, seed)
}

// NewEmbeddingPadded builds an embedding table that reserves padIdx as the
// padding row: the row is zero-initialised and is meant never to train, so that
// padded positions in a batch contribute nothing. Pass padIdx < 0 for no padding
// row. Panics on a bad shape or an out-of-range padIdx.
//
// HOW THE FROZEN GRADIENT ACTUALLY WORKS — read this, it is not automatic.
// torch zeroes the padding row's gradient inside its autograd node. This package
// keeps the forward pass a bare OpEmbed (so it stays bit-identical to the existing
// hand-rolled lookups and to the GPU kernels), and instead exposes the zeroing as
// a GRADIENT HOOK you install on the tape:
//
//	emb := nn.NewEmbeddingPadded(tensor.F32, vocab, dim, 0, 7)
//	tape.RegisterGradHook(emb.W, emb.PadGradHook())
//
// PadGradHook returns a function assignable to autograd.GradHook, which the tape
// calls with W's fully-accumulated gradient; it returns a copy with the padding
// row set to exactly zero, so the optimizer's update for that row is exactly zero
// and the row stays at its zero init forever. Combined with the zero
// initialisation above, that reproduces torch's semantics exactly.
//
// WITHOUT THE HOOK, PadIdx IS INERT: the padding row is merely zero-initialised
// and WILL train like any other row. This layer cannot install the hook itself —
// it has no reference to the tape, and nn does not import autograd — so the one
// line above is on the caller. [Embedding.ZeroPadRow] is the manual alternative
// (call it after each optimizer step); it is more expensive and only approximates
// the hook for optimizers with momentum, since the row's optimizer state still
// moves. Prefer the hook.
func NewEmbeddingPadded(dtype tensor.Dtype, vocab, dim, padIdx int, seed uint64) *Embedding {
	if vocab <= 0 || dim <= 0 {
		panic(fmt.Sprintf("nn: embedding needs vocab>0 and dim>0, got [%d,%d]", vocab, dim))
	}
	if padIdx >= vocab {
		panic(fmt.Sprintf("nn: embedding padding index %d outside vocab %d", padIdx, vocab))
	}
	w := tensor.New(dtype, tensor.Shape{vocab, dim})
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	//perfscan:ignore PS1002 model-construction weight fill, one-time init
	fillGen(w, func() float64 { return rng.NormFloat64() })
	e := &Embedding{W: w, PadIdx: -1}
	if padIdx >= 0 {
		e.PadIdx = padIdx
		e.ZeroPadRow()
	}
	return e
}

// Forward gathers rows of W by index: ids is a rank-1 tensor of length m holding
// the integer ids as floats (see the type doc), and the result is [m, dim]. It
// runs through ctx, so passing a recording context makes the table trainable.
//
// ids must have the same dtype as W and rank 1; every value must be an in-range
// row index. Violations are reported as errors, not panics.
func (e *Embedding) Forward(ctx *backend.Context, ids *tensor.Tensor) (*tensor.Tensor, error) {
	if ids == nil {
		return nil, fmt.Errorf("nn: Embedding needs an index tensor, got nil")
	}
	if e.W.Ndim() != 2 {
		return nil, fmt.Errorf("nn: Embedding table must be [vocab,dim], got %v", e.W.Shape())
	}
	if ids.Ndim() != 1 {
		return nil, fmt.Errorf("nn: Embedding expects ids rank-1 [m], got %v", ids.Shape())
	}
	if ids.Dtype() != e.W.Dtype() {
		return nil, fmt.Errorf("nn: Embedding ids dtype %v != table dtype %v", ids.Dtype(), e.W.Dtype())
	}
	vocab := e.W.Shape()[0]
	//perfscan:ignore PS1001 id validation O(m) dominated by OpEmbed gather O(m*dim)
	for i := range ids.Numel() {
		v := ids.AtF64(i)
		if v != float64(int(v)) {
			return nil, fmt.Errorf("nn: Embedding id %g at position %d is not an integer", v, i)
		}
		if int(v) < 0 || int(v) >= vocab {
			return nil, fmt.Errorf("nn: Embedding id %d at position %d outside vocab %d", int(v), i, vocab)
		}
	}
	//perfscan:ignore PS3038 resource-only 2-elem slice alloc per dispatch
	out, err := backend.Execute(ctx, backend.OpEmbed, []*tensor.Tensor{e.W, ids}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Lookup is [Embedding.Forward] for callers holding plain ids: it builds the
// float index tensor (dtype matching the table) and gathers. Ids out of range are
// reported as an error.
func (e *Embedding) Lookup(ctx *backend.Context, ids []int) (*tensor.Tensor, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("nn: Embedding.Lookup needs at least one id")
	}
	idx := tensor.New(e.W.Dtype(), tensor.Shape{len(ids)})
	for i, id := range ids {
		idx.SetF64(float64(id), i)
	}
	return e.Forward(ctx, idx)
}

// Params returns the table — the layer's only trainable tensor.
func (e *Embedding) Params() []*tensor.Tensor { return []*tensor.Tensor{e.W} }

// ZeroPadRow sets the padding row of W to zero (a no-op when PadIdx < 0). Use it
// after an optimizer step if you are not using [Embedding.PadGradHook]; note that
// this repairs the WEIGHTS but not the optimizer's momentum for that row, so the
// hook remains the faithful option.
func (e *Embedding) ZeroPadRow() {
	if e.PadIdx < 0 {
		return
	}
	for j := range e.W.Shape()[1] {
		e.W.SetF64(0, e.PadIdx, j)
	}
}

// PadGradHook returns a gradient hook that zeroes the padding row of an incoming
// gradient for W, which is what freezes the padding embedding (see
// [NewEmbeddingPadded]). Install it on the tape that records the forward pass:
//
//	tape.RegisterGradHook(emb.W, emb.PadGradHook())
//
// The returned function matches autograd.GradHook. It never mutates its argument
// (hooks must not — the same tensor may be shared): it returns a copy with the
// padding row zeroed, or nil when there is no padding row, which the tape reads as
// "keep the gradient unchanged".
func (e *Embedding) PadGradHook() func(grad *tensor.Tensor) *tensor.Tensor {
	pad := e.PadIdx
	return func(grad *tensor.Tensor) *tensor.Tensor {
		if pad < 0 || grad == nil || grad.Ndim() != 2 || pad >= grad.Shape()[0] {
			return nil // nothing to freeze: observe only
		}
		rows, cols := grad.Shape()[0], grad.Shape()[1]
		out := tensor.New(grad.Dtype(), grad.Shape())
		// Fast path: the whole gradient is copied except the padding row, which
		// stays at its zero init. On a contiguous, offset-0 tensor the rows before
		// and after `pad` are two bulk copies of the typed backing slice — one
		// dtype decision instead of the per-element SetF64(AtF64(i,j)) loop that
		// ran vocab×dim dispatched calls on every backward pass (the whole
		// embedding gradient). Values are bit-identical: the same float bits are
		// copied and the pad row is zero in both paths. F16/BF16 and non-contiguous
		// gradients fall through to the general path.
		if grad.IsContiguous() && grad.Offset() == 0 {
			lo, hi := pad*cols, (pad+1)*cols
			switch grad.Dtype() {
			case tensor.F64:
				src, dst := grad.Storage().F64(), out.Storage().F64()
				copy(dst[:lo], src[:lo])
				copy(dst[hi:], src[hi:])
				return out
			case tensor.F32:
				src, dst := grad.Storage().F32(), out.Storage().F32()
				copy(dst[:lo], src[:lo])
				copy(dst[hi:], src[hi:])
				return out
			}
		}
		for i := range rows {
			if i == pad {
				continue // leave the padding row at its zero init
			}
			for j := range cols {
				out.SetF64(grad.AtF64(i, j), i, j)
			}
		}
		return out
	}
}

var _ Layer = (*Embedding)(nil)
