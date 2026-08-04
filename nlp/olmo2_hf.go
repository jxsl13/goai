package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// OLMo2FromHF builds an [OLMo2] from a Hugging Face checkpoint's tensor map (the weights
// of transformers' Olmo2ForCausalLM). Like [LlamaFromHF] the geometry that is not
// derivable from the tensors (Heads, KVHeads, Eps, RopeBase, Ctx) comes from config.json
// and is passed in cfg; the dimensions (Dim, Vocab, Layers, Hidden, HeadDim) are inferred
// from the tensors. HF stores each projection as torch [out, in]; GoAI wants [in, out],
// so every weight is transposed.
//
// Every RMSNorm — the full-width self_attn.q_norm / self_attn.k_norm, the two post-norms
// (post_attention_layernorm, post_feedforward_layernorm) and model.norm — is a STANDARD
// RMSNorm (γ·x̂), loaded as-is via rmsFromGGUF (NOT Gemma's (1+γ) gemmaRMS). The LM head
// is untied by default (tie_word_embeddings=false) and read from lm_head.weight; when
// that tensor is absent the head ties to the embedding table (like [LlamaFromHF]).
func OLMo2FromHF(ts map[string]*tensor.Tensor, cfg OLMo2Config) (*OLMo2, error) {
	tok, ok := ts["model.embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF OLMo2 missing model.embed_tokens.weight")
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: model.embed_tokens.weight must be 2-D, got %v", tok.Shape())
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: OLMo2FromHF needs cfg.Heads")
	}

	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF OLMo2 has no model.layers.*")
	}
	cfg.Layers = layers
	if gate, ok := ts["model.layers.0.mlp.gate_proj.weight"]; ok {
		cfg.Hidden = gate.Shape()[0]
	}
	// head_dim is decoupled from dim/heads: infer it from the q_proj output width.
	if wq, ok := ts["model.layers.0.self_attn.q_proj.weight"]; ok {
		cfg.HeadDim = wq.Shape()[0] / cfg.Heads
	}

	m := &OLMo2{Config: cfg, TokEmb: cloneF64(tok)}
	//perfscan:ignore PS3060 model-load weight transpose, one-time
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF OLMo2 missing %s%s", p, name)
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
		postAttn, err := g("post_attention_layernorm.weight")
		if err != nil {
			return nil, err
		}
		postFFN, err := g("post_feedforward_layernorm.weight")
		if err != nil {
			return nil, err
		}
		gate, err := g("mlp.gate_proj.weight")
		if err != nil {
			return nil, err
		}
		up, err := g("mlp.up_proj.weight")
		if err != nil {
			return nil, err
		}
		down, err := g("mlp.down_proj.weight")
		if err != nil {
			return nil, err
		}
		m.Blocks = append(m.Blocks, &OLMo2Block{
			Wq:           transpose2D(wq),
			Wk:           transpose2D(wk),
			Wv:           transpose2D(wv),
			Wo:           transpose2D(wo),
			QNorm:        rmsFromGGUF(qNorm, cfg.Eps),
			KNorm:        rmsFromGGUF(kNorm, cfg.Eps),
			PostAttnNorm: rmsFromGGUF(postAttn, cfg.Eps),
			PostFFNNorm:  rmsFromGGUF(postFFN, cfg.Eps),
			FFN:          swiGLUFromGGUF(gate, up, down),
		})
	}
	norm, ok := ts["model.norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF OLMo2 missing model.norm.weight")
	}
	m.Norm = rmsFromGGUF(norm, cfg.Eps)
	head := tok // tied head if lm_head absent
	if o, ok := ts["lm_head.weight"]; ok {
		head = o
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}
