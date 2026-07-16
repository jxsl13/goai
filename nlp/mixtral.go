package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Mixtral is the Mixtral sparse-Mixture-of-Experts decoder (Jiang et al. 2024,
// arXiv:2401.04088). Its attention is identical to [Llama] — RMSNorm, RoPE,
// grouped-query attention, no biases — but each block's feed-forward is a sparse
// MoE: a router picks the top-2 of N SwiGLU experts per token and mixes their
// outputs with the renormalized router weights (nn.SparseMoE). This is the
// modern MoE counterpart to the dense Llama FFN; load a Hugging Face checkpoint
// with [MixtralFromHF].
type Mixtral struct {
	Config MixtralConfig
	TokEmb *tensor.Tensor
	Blocks []*MixtralBlock
	Norm   *nn.RMSNorm
	Out    *tensor.Tensor // [dim, vocab] output projection
}

// MixtralConfig fixes the model geometry. Dim, Vocab, Layers, Hidden, Experts are
// inferred from the checkpoint by MixtralFromHF; the rest come from config.json.
type MixtralConfig struct {
	Vocab    int
	Ctx      int
	Dim      int
	Heads    int
	KVHeads  int // GQA; 0 → Heads
	Layers   int
	Hidden   int     // per-expert SwiGLU inner width
	Experts  int     // number of experts (num_local_experts)
	TopK     int     // experts per token (num_experts_per_tok); 0 → 2
	Eps      float64 // RMSNorm epsilon
	RopeBase float64 // RoPE θ; 0 → 10000
}

// MixtralBlock is one pre-norm Mixtral block: Llama-style attention + a sparse-MoE FFN.
type MixtralBlock struct {
	AttnNorm       *nn.RMSNorm
	Wq, Wk, Wv, Wo *tensor.Tensor
	FFNNorm        *nn.RMSNorm
	MoE            *nn.SparseMoE
}

func (c MixtralConfig) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// Forward computes logits [seq, vocab] for the prompt tokens.
func (m *Mixtral) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: Mixtral prompt length %d outside (0,%d]", seq, m.Config.Ctx)
	}
	idx := tensor.New(m.TokEmb.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= m.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, m.Config.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	x, err := exec1(ctx, backend.OpEmbed, nil, m.TokEmb, idx)
	if err != nil {
		return nil, err
	}

	cfg := m.Config
	kv := cfg.kvHeads()
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}
	for _, b := range m.Blocks {
		// attention sublayer (identical to Llama)
		xb, err := b.AttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := exec1(ctx, backend.OpMatMul, nil, xb, b.Wq)
		if err != nil {
			return nil, err
		}
		k, err := exec1(ctx, backend.OpMatMul, nil, xb, b.Wk)
		if err != nil {
			return nil, err
		}
		v, err := exec1(ctx, backend.OpMatMul, nil, xb, b.Wv)
		if err != nil {
			return nil, err
		}
		if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads}, q); err != nil {
			return nil, err
		}
		if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv}, k); err != nil {
			return nil, err
		}
		a, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
		if err != nil {
			return nil, err
		}
		o, err := exec1(ctx, backend.OpMatMul, nil, a, b.Wo)
		if err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, o); err != nil {
			return nil, err
		}
		// sparse-MoE FFN sublayer
		xf, err := b.FFNNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		ff, _, err := b.MoE.Forward(ctx, xf)
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
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// Params returns every trainable tensor for optimizers (router + all experts included).
func (m *Mixtral) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.AttnNorm.Gamma, b.Wq, b.Wk, b.Wv, b.Wo, b.FFNNorm.Gamma, b.MoE.Router.W)
		for _, e := range b.MoE.Experts {
			ps = append(ps, e.Wgate, e.Wup, e.Wdown)
		}
	}
	ps = append(ps, m.Norm.Gamma, m.Out)
	return ps
}
