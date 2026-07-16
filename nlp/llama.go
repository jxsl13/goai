package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Llama is the LLaMA / Llama-2 decoder-only transformer (Touvron et al. 2023,
// arXiv:2302.13971 / 2307.09288, §R92) assembled from GoAI's verified primitives. It
// is the modern-LLM counterpart to the GPT-2-style [GPT]: pre-normalization with
// RMSNorm, rotary position embeddings (RoPE) on the queries and keys, grouped-query
// attention, a SwiGLU feed-forward, and NO biases or learned positional embeddings.
// Per block (pre-norm residual):
//
//	h = x + Wo·Attn( RoPE(x̄·Wq), RoPE(x̄·Wk), x̄·Wv )   with x̄ = RMSNorm(x)
//	x = h + SwiGLU( RMSNorm(h) )
//
// then a final RMSNorm and the untied output projection produce the logits.
type Llama struct {
	Config LlamaConfig    // the model hyperparameters
	TokEmb *tensor.Tensor // [vocab, dim] token embedding (no positional embedding)
	Blocks []*LlamaBlock  // the stacked transformer blocks
	Norm   *nn.RMSNorm    // final pre-logits RMSNorm
	Out    *tensor.Tensor // [dim, vocab] output projection (untied)
}

// LlamaConfig fixes the model geometry.
type LlamaConfig struct {
	Vocab    int     // vocabulary size
	Ctx      int     // max context length
	Dim      int     // embedding width (Heads·headDim)
	Heads    int     // number of query heads
	KVHeads  int     // key/value heads (GQA); 0 → Heads (standard MHA)
	Layers   int     // number of transformer blocks
	Hidden   int     // SwiGLU inner width (Llama uses ≈ (2/3)·4·Dim rounded)
	Eps      float64 // RMSNorm epsilon
	RopeBase float64 // RoPE frequency base θ; 0 → 10000
}

// LlamaBlock is one pre-norm Llama transformer block.
type LlamaBlock struct {
	AttnNorm       *nn.RMSNorm    // RMSNorm before attention
	Wq, Wk, Wv, Wo *tensor.Tensor // attention projections (no bias); Wk/Wv are [dim, KVHeads·headDim]
	Bq, Bk, Bv     *tensor.Tensor // optional q/k/v projection biases (Qwen2/Qwen2.5 family); nil for Llama/Mistral
	FFNNorm        *nn.RMSNorm    // RMSNorm before the FFN
	FFN            *nn.SwiGLU     // SwiGLU feed-forward
}

func (c LlamaConfig) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// NewLlama builds a Llama with randomly initialized weights (Xavier for the
// projections, ones for the RMSNorm gains), for training or as a target to load
// weights into. It errors on an inconsistent geometry.
func NewLlama(cfg LlamaConfig, seed uint64) (*Llama, error) {
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: Llama dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	kv := cfg.kvHeads()
	if cfg.Heads%kv != 0 {
		return nil, fmt.Errorf("nlp: Llama heads %d not divisible by kv_heads %d", cfg.Heads, kv)
	}
	dk := cfg.Dim / cfg.Heads
	if dk%2 != 0 {
		return nil, fmt.Errorf("nlp: Llama head dim %d must be even for RoPE", dk)
	}
	kvDim := kv * dk
	m := &Llama{Config: cfg}
	m.TokEmb = tensor.New(tensor.F64, tensor.Shape{cfg.Vocab, cfg.Dim})
	nn.XavierUniform(m.TokEmb, cfg.Vocab, cfg.Dim, seed)
	s := seed + 1
	ones := func(n int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{n})
		for i := range n {
			t.SetF64(1, i)
		}
		return t
	}
	proj := func(in, out int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{in, out})
		nn.XavierUniform(t, in, out, s)
		s++
		return t
	}
	for range cfg.Layers {
		m.Blocks = append(m.Blocks, &LlamaBlock{
			AttnNorm: &nn.RMSNorm{Gamma: ones(cfg.Dim), Eps: cfg.Eps},
			Wq:       proj(cfg.Dim, cfg.Dim),
			Wk:       proj(cfg.Dim, kvDim),
			Wv:       proj(cfg.Dim, kvDim),
			Wo:       proj(cfg.Dim, cfg.Dim),
			FFNNorm:  &nn.RMSNorm{Gamma: ones(cfg.Dim), Eps: cfg.Eps},
			FFN:      nn.NewSwiGLU(tensor.F64, cfg.Dim, cfg.Hidden, s),
		})
		s += 3
	}
	m.Norm = &nn.RMSNorm{Gamma: ones(cfg.Dim), Eps: cfg.Eps}
	m.Out = proj(cfg.Dim, cfg.Vocab)
	return m, nil
}

// Embed gathers the token embeddings for the prompt (Llama has no positional
// embedding — position enters through RoPE inside attention).
func (m *Llama) Embed(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: prompt length %d outside (0,%d]", seq, m.Config.Ctx)
	}
	idx := tensor.New(m.TokEmb.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= m.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, m.Config.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	return exec1(ctx, backend.OpEmbed, nil, m.TokEmb, idx)
}

// Forward computes logits [seq, vocab] for the prompt tokens.
func (m *Llama) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := m.Embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	return m.ForwardFromEmbed(ctx, x)
}

// ForwardHidden returns the final hidden states [seq, dim] — the residual stream after
// all blocks and the final RMSNorm, i.e. Forward WITHOUT the output projection (§T447,
// the Llama sibling of GPT.ForwardHidden). This is the representation auxiliary decoding
// heads attach to (Medusa): Forward(tokens) ≡ ForwardHidden(tokens)·Out.
func (m *Llama) ForwardHidden(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := m.Embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	return m.hiddenFromEmbed(ctx, x)
}

// ForwardFromEmbed runs the transformer blocks, final RMSNorm and output projection on
// a precomputed embedding x [seq, dim], returning logits [seq, vocab]. Splitting the
// embedding step out lets a training loop inject NEFTune noise (nn.NEFTune) between the
// embedding and the blocks.
func (m *Llama) ForwardFromEmbed(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := m.hiddenFromEmbed(ctx, x)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, h, m.Out)
}

// hiddenFromEmbed is the shared block stack + final RMSNorm.
func (m *Llama) hiddenFromEmbed(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	rope := backend.RoPEAttrs{Base: cfg.RopeBase}
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}

	var err error
	for _, b := range m.Blocks {
		// attention sublayer: x += Wo·Attn(RoPE(Wq·x̄), RoPE(Wk·x̄), Wv·x̄), x̄=RMSNorm(x)
		xb, err := b.AttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := project(ctx, xb, b.Wq)
		if err != nil {
			return nil, err
		}
		k, err := project(ctx, xb, b.Wk)
		if err != nil {
			return nil, err
		}
		v, err := project(ctx, xb, b.Wv)
		if err != nil {
			return nil, err
		}
		// Qwen2-family q/k/v projection biases (added before RoPE); nil for Llama/Mistral.
		if q, err = addBiasIf(ctx, q, b.Bq); err != nil {
			return nil, err
		}
		if k, err = addBiasIf(ctx, k, b.Bk); err != nil {
			return nil, err
		}
		if v, err = addBiasIf(ctx, v, b.Bv); err != nil {
			return nil, err
		}
		if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: rope.Base, Heads: cfg.Heads}, q); err != nil {
			return nil, err
		}
		if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: rope.Base, Heads: kv}, k); err != nil {
			return nil, err
		}
		a, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
		if err != nil {
			return nil, err
		}
		o, err := project(ctx, a, b.Wo)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, o); err != nil {
			return nil, err
		}
		// FFN sublayer: x += SwiGLU(RMSNorm(x))
		xf, err := b.FFNNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		ff, err := b.FFN.Forward(ctx, xf)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return x, nil
}

// project computes x·W (a bias-free linear layer).
func project(ctx *backend.Context, x, w *tensor.Tensor) (*tensor.Tensor, error) {
	return exec1(ctx, backend.OpMatMul, nil, x, w)
}

// addBiasIf adds bias to x when bias is non-nil (Qwen2-family projections), else
// returns x unchanged (the bias-free Llama/Mistral path).
func addBiasIf(ctx *backend.Context, x, bias *tensor.Tensor) (*tensor.Tensor, error) {
	if bias == nil {
		return x, nil
	}
	return exec1(ctx, backend.OpAddBias, nil, x, bias)
}

// Params returns every trainable tensor for optimizers.
func (m *Llama) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.AttnNorm.Gamma, b.Wq, b.Wk, b.Wv, b.Wo, b.FFNNorm.Gamma)
		for _, bb := range []*tensor.Tensor{b.Bq, b.Bk, b.Bv} {
			if bb != nil {
				ps = append(ps, bb)
			}
		}
		ps = append(ps, b.FFN.Params()...)
	}
	ps = append(ps, m.Norm.Gamma, m.Out)
	return ps
}
