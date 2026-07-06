package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// KVCache holds the per-layer key/value tensors accumulated during
// autoregressive decoding (§T35), so each new token attends to cached past
// keys/values instead of recomputing attention over the whole prefix.
type KVCache struct {
	K, V []*tensor.Tensor // per block; nil until the first token
}

// NewCache returns an empty cache sized for this model's blocks.
func (g *GPT) NewCache() *KVCache {
	return &KVCache{K: make([]*tensor.Tensor, len(g.Blocks)), V: make([]*tensor.Tensor, len(g.Blocks))}
}

// Len returns the number of tokens currently cached.
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
	for i := range ra {
		for j := range d {
			out.SetF64(a.AtF64(i, j), i, j)
		}
	}
	for i := range rb {
		for j := range d {
			out.SetF64(b.AtF64(i, j), ra+i, j)
		}
	}
	return out
}

// StepKV runs attention for a single new token h[1,dmodel] given cached K,V
// (each [t,dmodel] or nil). It appends the token's k,v to the cache and returns
// (out[1,dmodel], Knew, Vnew). The single query attends to all t+1 keys.
func (m *MHA) StepKV(ctx *backend.Context, h, kc, vc *tensor.Tensor) (out, kNew, vNew *tensor.Tensor, err error) {
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
	kNew = concatRows(kc, kt)
	vNew = concatRows(vc, vt)
	// single query at the last position attends to all cached keys → no mask
	attn, err := m.exec(ctx, backend.OpMHA, backend.Attrs{"heads": m.Heads, "causal": false}, q, kNew, vNew)
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
	for j := range d {
		x.SetF64(g.TokEmb.AtF64(token, j)+g.PosEmb.AtF64(pos, j), 0, j)
	}
	return x, nil
}

// DecodeStep advances the model by one token using the KV-cache and returns the
// next-token logits [1,vocab]. pos is the token's absolute position (== cache
// length before the call). Inference-only (no tape).
func (g *GPT) DecodeStep(ctx *backend.Context, cache *KVCache, token, pos int) (*tensor.Tensor, error) {
	x, err := g.embedAt(token, pos)
	if err != nil {
		return nil, err
	}
	for l, b := range g.Blocks {
		h, err := b.LN1.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		attnOut, kNew, vNew, err := b.Attn.StepKV(ctx, h, cache.K[l], cache.V[l])
		if err != nil {
			return nil, err
		}
		cache.K[l], cache.V[l] = kNew, vNew
		if x, err = exec1(ctx, backend.OpAdd, nil, x, attnOut); err != nil {
			return nil, err
		}
		h, err = b.LN2.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpMatMul, nil, h, b.W1); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpAddBias, nil, h, b.B1); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpGELU, nil, h); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpMatMul, nil, h, b.W2); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpAddBias, nil, h, b.B2); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, h); err != nil {
			return nil, err
		}
	}
	if x, err = g.LNf.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, g.Head)
}
