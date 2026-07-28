package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// CLA is a decoder-only transformer with Cross-Layer Attention (Brandon et al.
// 2024, "Reducing Transformer Key-Value Cache Size with Cross-Layer Attention",
// arXiv:2405.12981). Adjacent layers are grouped in runs of Config.Share; the
// FIRST layer of each group (the leader, l%Share == 0) projects keys and values
// from its own pre-attention LayerNorm output, and the remaining layers of the
// group (followers) REUSE the leader's k,v unchanged — every layer still
// computes its OWN query and output projection, so the layers stay distinct.
// Only leaders carry Wk/Wv, so both the attention parameter count and — the
// headline win — the KV cache shrink by the group factor: a CLACache holds
// Layers/Share key/value slots instead of one per layer (see NewCache).
//
// Contrast with the other KV-shrinking axes: GQA/MQA reduce the number of
// key/value HEADS within a single layer (many query heads share few kv heads)
// and MLA (DeepSeek-V2) compresses each layer's k,v into a low-rank latent —
// CLA instead shares whole k,v tensors ACROSS layers and is orthogonal to
// (composable with) both. Everything else is the exact GPT recipe from gpt.go
// (pre-LN blocks, causal fused OpMHA, GELU FFN), and Share=1 degenerates to a
// plain GPT: same weights, same logits.
//
// In plain terms: a transformer normally makes a fresh "lookup table" (the
// keys and values) in every layer and must keep all of them around while
// generating, which is what fills GPU memory on long texts. CLA lets a few
// neighboring layers share one table: each layer still asks its own questions
// (queries), but they all look things up in the first layer's table. With
// Share=2 the model remembers half as many tables — almost the same quality,
// half the memory.
type CLA struct {
	Config CLAConfig      // the model hyperparameters, including the KV-sharing factor Share
	TokEmb *tensor.Tensor // token embedding table [vocab, dim]
	PosEmb *tensor.Tensor // learned position embedding table [ctx, dim]
	Blocks []*CLABlock    // the stacked transformer blocks (leaders carry Wk/Wv, followers reuse)
	LNf    *nn.LayerNorm  // final pre-logits LayerNorm
	Head   *tensor.Tensor // LM head [dim, vocab]
}

// CLAConfig fixes the CLA model geometry (GPTConfig plus the sharing factor).
type CLAConfig struct {
	Vocab  int     // vocabulary size
	Ctx    int     // max context length
	Dim    int     // embedding width (must divide by Heads)
	Heads  int     // number of attention heads
	Layers int     // number of transformer blocks (must divide by Share)
	Share  int     // KV-sharing group size: 1 leader + Share−1 followers per group (1 = plain GPT)
	Eps    float64 // LayerNorm epsilon (≤0 → 1e-5)
}

// CLABlock is one pre-LN transformer block of a CLA stack. Every block owns its
// query and output projections and the GELU FFN; only LEADER blocks (layer index
// l with l%Share == 0) own key/value projections — on follower blocks Wk and Wv
// are nil and attention reads the leader's k,v (Brandon et al. 2024).
type CLABlock struct {
	LN1, LN2 *nn.LayerNorm  // LN1 pre-attention, LN2 pre-MLP LayerNorm
	Wq, Wo   *tensor.Tensor // query and output projections [dim, dim] (every layer)
	Wk, Wv   *tensor.Tensor // key/value projections [dim, dim] on leaders; nil on followers
	W1, B1   *tensor.Tensor // FFN up: [dim, 4dim], [4dim]
	W2, B2   *tensor.Tensor // FFN down: [4dim, dim], [dim]
}

// CLACache holds the per-GROUP key/value tensors accumulated during
// autoregressive decoding — len(K) == len(V) == Layers/Share slots, the KV-cache
// reduction CLA exists for. Slot g serves layers g·Share … g·Share+Share−1: the
// group's leader appends to it, its followers read it.
type CLACache struct {
	K, V []*tensor.Tensor // per KV-sharing group; nil until the group's first token

	bufs kvBufs // amortized backing for the K/V views, keyed by group (rowbuf.go)
}

// NewCLA builds a randomly initialized Cross-Layer Attention model.
// Deterministic via seed; every weight matrix is Xavier-uniform
// (nn.XavierUniform, the repo's trained-from-scratch default), LayerNorms 1/0,
// FFN biases 0; parameters are F64 (the repo's training/gradcheck precision).
// Follower blocks allocate no Wk/Wv at all — the parameter saving is real, not
// masked.
func NewCLA(cfg CLAConfig, seed uint64) (*CLA, error) {
	if cfg.Vocab <= 0 || cfg.Ctx <= 0 || cfg.Dim <= 0 || cfg.Layers <= 0 {
		return nil, fmt.Errorf("nlp: NewCLA needs Vocab, Ctx, Dim, Layers > 0 (got %d, %d, %d, %d)",
			cfg.Vocab, cfg.Ctx, cfg.Dim, cfg.Layers)
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: NewCLA dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	if cfg.Share < 1 {
		return nil, fmt.Errorf("nlp: NewCLA needs Share >= 1, got %d", cfg.Share)
	}
	if cfg.Layers%cfg.Share != 0 {
		return nil, fmt.Errorf("nlp: NewCLA layers %d not divisible by share %d", cfg.Layers, cfg.Share)
	}
	if cfg.Eps <= 0 {
		cfg.Eps = 1e-5
	}
	randn := func(s uint64, fanIn, fanOut int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{fanIn, fanOut})
		nn.XavierUniform(t, fanIn, fanOut, s)
		return t
	}
	ln := func() *nn.LayerNorm {
		l := nn.NewLayerNorm(tensor.F64, cfg.Dim)
		l.Eps = cfg.Eps
		return l
	}
	m := &CLA{
		Config: cfg,
		TokEmb: randn(seed, cfg.Vocab, cfg.Dim),
		PosEmb: randn(seed+1, cfg.Ctx, cfg.Dim),
		Head:   randn(seed+2, cfg.Dim, cfg.Vocab),
		LNf:    ln(),
	}
	for l := range cfg.Layers {
		s := seed + 10 + uint64(l)*8
		b := &CLABlock{
			LN1: ln(), LN2: ln(),
			Wq: randn(s, cfg.Dim, cfg.Dim), Wo: randn(s+3, cfg.Dim, cfg.Dim),
			W1: randn(s+4, cfg.Dim, 4*cfg.Dim), B1: tensor.New(tensor.F64, tensor.Shape{4 * cfg.Dim}),
			W2: randn(s+5, 4*cfg.Dim, cfg.Dim), B2: tensor.New(tensor.F64, tensor.Shape{cfg.Dim}),
		}
		if l%cfg.Share == 0 { // leader: the group's single k,v source
			b.Wk, b.Wv = randn(s+1, cfg.Dim, cfg.Dim), randn(s+2, cfg.Dim, cfg.Dim)
		}
		m.Blocks = append(m.Blocks, b)
	}
	return m, nil
}

// Params returns every trainable tensor (token/pos embeddings, all block
// weights, final LN, and the LM head) for optimizers. Follower blocks
// contribute no Wk/Wv — those slots do not exist.
func (c *CLA) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{c.TokEmb, c.PosEmb}
	for _, b := range c.Blocks {
		ps = append(ps, b.LN1.Gamma, b.LN1.Beta, b.Wq)
		if b.Wk != nil {
			ps = append(ps, b.Wk, b.Wv)
		}
		ps = append(ps, b.Wo,
			b.LN2.Gamma, b.LN2.Beta,
			b.W1, b.B1, b.W2, b.B2,
		)
	}
	return append(ps, c.LNf.Gamma, c.LNf.Beta, c.Head)
}

// embed gathers token+position embeddings through the dispatch (mirrors
// GPT.Embed), so gradients flow to TokEmb/PosEmb.
func (c *CLA) embed(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > c.Config.Ctx {
		return nil, fmt.Errorf("nlp: CLA prompt length %d outside (0,%d]", seq, c.Config.Ctx)
	}
	for _, t := range tokens {
		if t < 0 || t >= c.Config.Vocab {
			return nil, fmt.Errorf("nlp: CLA token %d outside vocab %d", t, c.Config.Vocab)
		}
	}
	tokIdx := tensor.New(c.TokEmb.Dtype(), tensor.Shape{seq})
	posIdx := tensor.New(c.PosEmb.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		tokIdx.SetF64(float64(t), i)
		posIdx.SetF64(float64(i), i)
	}
	et, err := exec1(ctx, backend.OpEmbed, nil, c.TokEmb, tokIdx)
	if err != nil {
		return nil, err
	}
	ep, err := exec1(ctx, backend.OpEmbed, nil, c.PosEmb, posIdx)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpAdd, nil, et, ep)
}

// Forward computes logits [seq, vocab] for the prompt tokens. Fully
// differentiable: inside each KV-sharing group the leader's k,v feed Share
// separate OpMHA calls, and on the backward pass the tape sums the gradients at
// that fan-out, so the leader's Wk/Wv receive contributions from every layer of
// its group.
func (c *CLA) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := c.embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	if x, err = c.hiddenFromEmbed(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, c.Head)
}

// hiddenFromEmbed is the CLA block stack + final LayerNorm: per layer,
// x += Wo·MHA(q, kShare, vShare) with q from THIS layer's LN1(x) and kShare/
// vShare stashed by the group's leader, then the standard GELU FFN (identical
// to gpt.go's).
func (c *CLA) hiddenFromEmbed(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	var kShare, vShare *tensor.Tensor // the current group's shared k,v (set by its leader)
	for l, b := range c.Blocks {
		// attention sublayer: x += Wo·MHA(q_l, k_leader, v_leader)
		h1, err := b.LN1.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		if l%c.Config.Share == 0 { // leader: project and stash the group's k,v
			if kShare, err = exec1(ctx, backend.OpMatMul, nil, h1, b.Wk); err != nil {
				return nil, err
			}
			if vShare, err = exec1(ctx, backend.OpMatMul, nil, h1, b.Wv); err != nil {
				return nil, err
			}
		}
		q, err := exec1(ctx, backend.OpMatMul, nil, h1, b.Wq)
		if err != nil {
			return nil, err
		}
		attn, err := exec1(ctx, backend.OpMHA, backend.AttnAttrs{Heads: c.Config.Heads, Causal: true}, q, kShare, vShare)
		if err != nil {
			return nil, err
		}
		if attn, err = exec1(ctx, backend.OpMatMul, nil, attn, b.Wo); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, attn); err != nil {
			return nil, err
		}
		// FFN sublayer: x += W2·gelu(W1·LN2(x)+b1)+b2 (gpt.go's block, verbatim)
		h, err := b.LN2.Forward(ctx, x)
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
	return c.LNf.Forward(ctx, x)
}

// NewCache returns an empty decode cache with Layers/Share key/value slots —
// the KV-cache reduction that motivates CLA: a plain GPT keeps one k,v pair per
// layer, a CLA keeps one per GROUP.
func (c *CLA) NewCache() *CLACache {
	groups := c.Config.Layers / c.Config.Share
	return &CLACache{K: make([]*tensor.Tensor, groups), V: make([]*tensor.Tensor, groups)}
}

// embedAt returns the embedding of a single token at absolute position pos,
// x[1,dim] = TokEmb[token] + PosEmb[pos] (typed row reads, mirrors GPT decode).
func (c *CLA) embedAt(token, pos int) (*tensor.Tensor, error) {
	if token < 0 || token >= c.Config.Vocab {
		return nil, fmt.Errorf("nlp: CLA token %d outside vocab %d", token, c.Config.Vocab)
	}
	if pos < 0 || pos >= c.Config.Ctx {
		return nil, fmt.Errorf("nlp: CLA position %d outside context %d", pos, c.Config.Ctx)
	}
	d := c.Config.Dim
	x := tensor.New(c.TokEmb.Dtype(), tensor.Shape{1, d})
	te, pe := c.TokEmb.Contiguous(), c.PosEmb.Contiguous()
	switch {
	case te.Dtype() == tensor.F64 && pe.Dtype() == tensor.F64:
		ts := te.Storage().F64()[token*d : token*d+d]
		ps := pe.Storage().F64()[pos*d : pos*d+d]
		dst := x.Storage().F64()
		for j := 0; j < d; j++ {
			dst[j] = ts[j] + ps[j]
		}
	case te.Dtype() == tensor.F32 && pe.Dtype() == tensor.F32:
		ts := te.Storage().F32()[token*d : token*d+d]
		ps := pe.Storage().F32()[pos*d : pos*d+d]
		dst := x.Storage().F32()
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

// DecodeStep advances the model by one token using the group-shared KV-cache
// and returns the next-token logits [1,vocab]. pos is the token's absolute
// position (== the number of tokens already cached). Within each group the
// LEADER layer runs first: it projects the token's k,v and appends them to the
// group's cache slot (concatRows), so its followers — and the leader itself —
// attend to the already-updated slot. Inference-only (no tape).
func (c *CLA) DecodeStep(ctx *backend.Context, cache *CLACache, token, pos int) (*tensor.Tensor, error) {
	groups := c.Config.Layers / c.Config.Share
	if cache == nil || len(cache.K) != groups || len(cache.V) != groups {
		return nil, fmt.Errorf("nlp: CLA cache must have %d K/V slots (from NewCache)", groups)
	}
	x, err := c.embedAt(token, pos)
	if err != nil {
		return nil, err
	}
	for l, b := range c.Blocks {
		h1, err := b.LN1.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		g := l / c.Config.Share
		if l%c.Config.Share == 0 { // leader: append this token's k,v to the group slot
			kt, err := exec1(ctx, backend.OpMatMul, nil, h1, b.Wk)
			if err != nil {
				return nil, err
			}
			vt, err := exec1(ctx, backend.OpMatMul, nil, h1, b.Wv)
			if err != nil {
				return nil, err
			}
			// Amortized row append keyed by the GROUP, not the layer: one slot is
			// shared by the group's Share layers (PS2006). Safe against the view
			// semantics because within a step the slot is appended exactly once, by
			// the leader, and every reader re-reads cache.K[g] from the struct rather
			// than holding a local from before the append.
			cache.K[g], cache.V[g] = cache.bufs.appendKV(cache.K, cache.V, g, kt, vt)
		}
		q, err := exec1(ctx, backend.OpMatMul, nil, h1, b.Wq)
		if err != nil {
			return nil, err
		}
		// single query at the last position attends to all cached keys → no mask
		attn, err := exec1(ctx, backend.OpMHA, backend.AttnAttrs{Heads: c.Config.Heads, Causal: false}, q, cache.K[g], cache.V[g])
		if err != nil {
			return nil, err
		}
		if attn, err = exec1(ctx, backend.OpMatMul, nil, attn, b.Wo); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, attn); err != nil {
			return nil, err
		}
		h, err := b.LN2.Forward(ctx, x)
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
	if x, err = c.LNf.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, c.Head)
}
