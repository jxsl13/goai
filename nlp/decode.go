package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// KVCache holds the per-layer key/value tensors accumulated during
// autoregressive decoding (§T35), so each new token attends to cached past
// keys/values instead of recomputing attention over the whole prefix.
//
// It also remembers where it sits on its stream's position axis, which is what makes
// [KVCache.NextPos] answerable after [KVCache.EvictStreaming] has dropped rows. See the
// position contract at the top of kvcache.go: the pos passed to [GPT.DecodeStep] is the
// token's true position in the stream, never its row index here.
type KVCache struct {
	K, V []*tensor.Tensor // per block; nil until the first token
	pos  kvPos            // where the retained rows sit on the stream's position axis

	// Amortized row buffers backing the K/V views (see rowbuf.go). The public
	// K/V fields keep holding the same [t,width] views callers already read;
	// this only changes how the next row gets there. EvictStreaming replaces
	// the entries outright, which rowBuf.owns detects and resynchronizes from.
	bufs kvBufs
}

// NewCache returns an empty cache sized for this model's blocks.
func (g *GPT) NewCache() *KVCache {
	return &KVCache{K: make([]*tensor.Tensor, len(g.Blocks)), V: make([]*tensor.Tensor, len(g.Blocks))}
}

// Len returns the number of tokens currently cached.
//
// After eviction this is a memory measurement, not a position: it counts the rows the
// cache still HOLDS, while [KVCache.NextPos] reports where the stream has got to. The
// two agree only while nothing has been evicted.
func (c *KVCache) Len() int {
	if len(c.K) == 0 || c.K[0] == nil {
		return 0
	}
	return c.K[0].Shape()[0]
}

// concatRows stacks a[ra,d] on top of b[rb,d] → [ra+rb,d] (inference helper, no
// gradient path). b==nil returns a copy of a.
func concatRows(a, b *tensor.Tensor) *tensor.Tensor {
	if a == nil {
		return b
	}
	d := a.Shape()[1]
	ra := a.Shape()[0]
	rb := 0
	if b != nil {
		rb = b.Shape()[0]
	}
	out := tensor.New(a.Dtype(), tensor.Shape{ra + rb, d})
	ac := a.Contiguous()
	// The KV-cache append runs every decode step and grows with context — a typed
	// row-block copy (§base-perf) instead of a per-element AtF64/SetF64 dispatch.
	sameDtype := b == nil || b.Dtype() == a.Dtype()
	switch {
	case sameDtype && a.Dtype() == tensor.F64:
		dst := out.Storage().F64()
		copy(dst[:ra*d], ac.Storage().F64()[:ra*d])
		if b != nil {
			copy(dst[ra*d:], b.Contiguous().Storage().F64()[:rb*d])
		}
	case sameDtype && a.Dtype() == tensor.F32:
		dst := out.Storage().F32()
		copy(dst[:ra*d], ac.Storage().F32()[:ra*d])
		if b != nil {
			copy(dst[ra*d:], b.Contiguous().Storage().F32()[:rb*d])
		}
	default:
		for i := 0; i < ra; i++ {
			for j := 0; j < d; j++ {
				out.SetF64(ac.AtF64(i, j), i, j)
			}
		}
		if b != nil {
			bc := b.Contiguous()
			for i := 0; i < rb; i++ {
				for j := 0; j < d; j++ {
					out.SetF64(bc.AtF64(i, j), ra+i, j)
				}
			}
		}
	}
	return out
}

// StepKV runs attention for a single new token h[1,dmodel] given cached K,V
// (each [t,dmodel] or nil). It appends the token's k,v to the cache and returns
// (out[1,dmodel], Knew, Vnew). The single query attends to all t+1 keys.
func (m *MHA) StepKV(ctx *backend.Context, h, kc, vc *tensor.Tensor) (out, kNew, vNew *tensor.Tensor, err error) {
	return m.stepKV(ctx, h, func(kt, vt *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
		return concatRows(kc, kt), concatRows(vc, vt)
	})
}

// kvAppend appends the token's k,v rows to the cache and returns the refreshed
// views. It is the seam that lets GPT.DecodeStep use the amortized rowBuf append
// while the exported StepKV keeps its allocating concatRows semantics for callers
// that hand in bare tensors and own no buffer.
type kvAppend func(kt, vt *tensor.Tensor) (kNew, vNew *tensor.Tensor)

func (m *MHA) stepKV(ctx *backend.Context, h *tensor.Tensor, appendRows kvAppend) (out, kNew, vNew *tensor.Tensor, err error) {
	q, err := m.exec(ctx, backend.OpMatMul, nil, h, m.Wq)
	if err != nil {
		return nil, nil, nil, err
	}
	kt, err := m.exec(ctx, backend.OpMatMul, nil, h, m.Wk)
	if err != nil {
		return nil, nil, nil, err
	}
	vt, err := m.exec(ctx, backend.OpMatMul, nil, h, m.Wv)
	if err != nil {
		return nil, nil, nil, err
	}
	kNew, vNew = appendRows(kt, vt)
	// single query at the last position attends to all cached keys → no mask
	attn, err := m.exec(ctx, backend.OpMHA, backend.AttnAttrs{Heads: m.Heads, Causal: false}, q, kNew, vNew)
	if err != nil {
		return nil, nil, nil, err
	}
	out, err = m.exec(ctx, backend.OpMatMul, nil, attn, m.Wo)
	return out, kNew, vNew, err
}

// embedAt returns the embedding of a single token at absolute position pos,
// x[1,dim] = TokEmb[token] + PosEmb[pos].
func (g *GPT) embedAt(token, pos int) (*tensor.Tensor, error) {
	if token < 0 || token >= g.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, g.Config.Vocab)
	}
	if pos < 0 || pos >= g.Config.Ctx {
		return nil, fmt.Errorf("nlp: position %d outside context %d", pos, g.Config.Ctx)
	}
	d := g.Config.Dim
	x := tensor.New(g.TokEmb.Dtype(), tensor.Shape{1, d})
	// Per-token embedding lookup: read the two rows directly and add, typed
	// (§base-perf) instead of a per-element AtF64 over the whole dim.
	te, pe := g.TokEmb.Contiguous(), g.PosEmb.Contiguous()
	switch {
	case te.Dtype() == tensor.F32 && pe.Dtype() == tensor.F32:
		ts := te.Storage().F32()[token*d : token*d+d]
		ps := pe.Storage().F32()[pos*d : pos*d+d]
		dst := x.Storage().F32()
		for j := 0; j < d; j++ {
			dst[j] = ts[j] + ps[j]
		}
	case te.Dtype() == tensor.F64 && pe.Dtype() == tensor.F64:
		ts := te.Storage().F64()[token*d : token*d+d]
		ps := pe.Storage().F64()[pos*d : pos*d+d]
		dst := x.Storage().F64()
		for j := 0; j < d; j++ {
			dst[j] = ts[j] + ps[j]
		}
	default:
		for j := 0; j < d; j++ {
			x.SetF64(te.AtF64(token, j)+pe.AtF64(pos, j), 0, j)
		}
	}
	return x, nil
}

// DecodeStep advances the model by one token using the KV-cache and returns the
// next-token logits [1,vocab]. Inference-only (no tape).
//
// pos is the token's TRUE position in the stream — the number of tokens decoded before
// it — and it must equal [KVCache.NextPos], which is the value to pass. On a plain
// append-only cache that is simply the cache length, so an ordinary decode loop counts
// 0, 1, 2, … as before. It stops being the cache length after [KVCache.EvictStreaming],
// which frees rows without rewinding the stream; pos keeps counting the stream, and a
// pos that disagrees with the cache is REJECTED rather than decoded, because the logits
// it would produce look entirely normal while encoding the wrong position (§V29). The
// full reasoning, including why a learned absolute position embedding needs the same
// answer RoPE does, is in the position contract at the top of kvcache.go.
//
// GPT reads pos straight out of its learned PosEmb table, so it is bounded by Ctx: a
// bounded-cache stream on this architecture ends when the stream reaches the trained
// context length, no matter how few rows the cache holds. That limit is real rather than
// a missing feature — see [Llama.StreamStep] for the cache layout that streams without
// one.
func (g *GPT) DecodeStep(ctx *backend.Context, cache *KVCache, token, pos int) (*tensor.Tensor, error) {
	if err := cache.pos.admit(cache.Len(), pos, "KVCache"); err != nil {
		return nil, err
	}
	x, err := g.embedAt(token, pos)
	if err != nil {
		return nil, err
	}
	for l, b := range g.Blocks {
		h, err := b.LN1.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		attnOut, kNew, vNew, err := b.Attn.stepKV(ctx, h, func(kt, vt *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
			return cache.bufs.appendKV(cache.K, cache.V, l, kt, vt)
		})
		if err != nil {
			return nil, err
		}
		cache.K[l], cache.V[l] = kNew, vNew
		if x, err = exec2(ctx, backend.OpAdd, nil, x, attnOut); err != nil {
			return nil, err
		}
		h, err = b.LN2.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		if h, err = exec2(ctx, backend.OpMatMul, nil, h, b.W1); err != nil {
			return nil, err
		}
		if h, err = exec2(ctx, backend.OpAddBias, nil, h, b.B1); err != nil {
			return nil, err
		}
		if h, err = exec1a(ctx, backend.OpGELU, nil, h); err != nil {
			return nil, err
		}
		if h, err = exec2(ctx, backend.OpMatMul, nil, h, b.W2); err != nil {
			return nil, err
		}
		if h, err = exec2(ctx, backend.OpAddBias, nil, h, b.B2); err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, h); err != nil {
			return nil, err
		}
	}
	if x, err = g.LNf.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec2(ctx, backend.OpMatMul, nil, x, g.Head)
}
