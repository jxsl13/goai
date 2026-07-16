package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// OLMoEFromHF builds an [OLMoE] from a Hugging Face checkpoint's tensor map (the weights
// of transformers' OlmoeForCausalLM). Like [MixtralFromHF] the geometry that is not
// derivable from the tensors (Heads, KVHeads, TopK, Eps, RopeBase, Ctx) comes from
// config.json and is passed in cfg; the dimensions (Dim, Vocab, Layers, Hidden, Experts,
// HeadDim) are inferred from the tensors. HF stores each projection as torch [out, in];
// GoAI wants [in, out], so every weight is transposed.
//
// The attention weights load like [Llama]/[OLMo2]: q/k/v/o projections (no bias) plus the
// full-width self_attn.q_norm / self_attn.k_norm gains (shapes heads·head_dim and
// kv·head_dim — e.g. 16 and 8 — NOT per-head). Every RMSNorm — the two QK-norms, the two
// block norms (input_layernorm, post_attention_layernorm) and model.norm — is a STANDARD
// RMSNorm (γ·x̂), loaded via rmsFromGGUF (NOT Gemma's (1+γ) variant). The per-block MoE
// loads with the shared [mixtralMoE] fused loader: router mlp.gate.weight [E,dim] and
// fused experts mlp.experts.gate_up_proj [E,2·ffn,dim] / mlp.experts.down_proj [E,dim,ffn],
// with NO shared expert. The LM head is untied by default (tie_word_embeddings=false) and
// read from lm_head.weight; when that tensor is absent the head ties to the embedding table.
//
// Limitation: the MoE combine step always renormalizes the top-k router weights
// (nn.SparseMoE, matching transformers with norm_topk_prob=True). OLMoE's config default
// is norm_topk_prob=False; a norm_topk_prob=False checkpoint would route to the same
// experts but weight them WITHOUT renormalization and so would NOT match this loader. Load
// (and generate goldens for) OLMoE with norm_topk_prob=True — the same caveat as
// [Qwen3MoeFromHF].
func OLMoEFromHF(ts map[string]*tensor.Tensor, cfg OLMoEConfig) (*OLMoE, error) {
	tok, ok := ts["model.embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF OLMoE missing model.embed_tokens.weight")
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: model.embed_tokens.weight must be 2-D, got %v", tok.Shape())
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: OLMoEFromHF needs cfg.Heads")
	}
	if cfg.TopK <= 0 {
		cfg.TopK = 2
	}

	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF OLMoE has no model.layers.*")
	}
	cfg.Layers = layers
	// head_dim is decoupled from dim/heads: infer it from the q_proj output width.
	if wq, ok := ts["model.layers.0.self_attn.q_proj.weight"]; ok {
		cfg.HeadDim = wq.Shape()[0] / cfg.Heads
	}

	// mixtralMoE reads its geometry from a MixtralConfig; OLMoE reuses the identical fused
	// expert layout, so bridge the shared fields.
	mcfg := MixtralConfig{Dim: cfg.Dim, TopK: cfg.TopK}

	m := &OLMoE{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF OLMoE missing %s%s", p, name)
			}
			return t, nil
		}
		wq, err := g("self_attn.q_proj.weight")
		if err != nil {
			return nil, err
		}
		wk, err := g("self_attn.k_proj.weight")
		if err != nil {
			return nil, err
		}
		wv, err := g("self_attn.v_proj.weight")
		if err != nil {
			return nil, err
		}
		wo, err := g("self_attn.o_proj.weight")
		if err != nil {
			return nil, err
		}
		qNorm, err := g("self_attn.q_norm.weight")
		if err != nil {
			return nil, err
		}
		kNorm, err := g("self_attn.k_norm.weight")
		if err != nil {
			return nil, err
		}
		an, err := g("input_layernorm.weight")
		if err != nil {
			return nil, err
		}
		fn, err := g("post_attention_layernorm.weight")
		if err != nil {
			return nil, err
		}
		moe, err := mixtralMoE(ts, p, mcfg)
		if err != nil {
			return nil, err
		}
		if l == 0 {
			cfg.Experts = len(moe.Experts)
			cfg.Hidden = moe.Experts[0].Wgate.Shape()[1]
			m.Config = cfg
		}
		m.Blocks = append(m.Blocks, &OLMoEBlock{
			AttnNorm: rmsFromGGUF(an, cfg.Eps),
			Wq:       transpose2D(wq),
			Wk:       transpose2D(wk),
			Wv:       transpose2D(wv),
			Wo:       transpose2D(wo),
			QNorm:    rmsFromGGUF(qNorm, cfg.Eps),
			KNorm:    rmsFromGGUF(kNorm, cfg.Eps),
			FFNNorm:  rmsFromGGUF(fn, cfg.Eps),
			MoE:      moe,
		})
	}
	norm, ok := ts["model.norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF OLMoE missing model.norm.weight")
	}
	m.Norm = rmsFromGGUF(norm, cfg.Eps)
	head := tok // tied head if lm_head absent
	if o, ok := ts["lm_head.weight"]; ok {
		head = o
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}
