package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// LlamaFromHF builds a Llama from a Hugging Face checkpoint's tensor map (the
// weights of transformers' LlamaForCausalLM, as loaded by safetensors.Load or
// pytorch.Load). It is the HF-named counterpart of [LlamaFromGGUF]: the config
// (Heads, KVHeads, Eps, RopeBase, Ctx) comes from the checkpoint's config.json
// and is passed in cfg; the dimensions (Dim, Vocab, Layers, Hidden) are inferred
// from the tensors.
//
// HF stores each projection as torch [out, in]; GoAI wants [in, out], so every
// weight is transposed (as in LlamaFromGGUF). GoAI's rotary embedding pairs
// element i with i+headDim/2 — the same split-half convention as HF's
// rotate_half — so the q/k projections need no row permutation (unlike the GGUF
// path, whose weights llama.cpp pre-permutes; §R93).
func LlamaFromHF(ts map[string]*tensor.Tensor, cfg LlamaConfig) (*Llama, error) {
	tok, ok := ts["model.embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Llama missing model.embed_tokens.weight")
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: model.embed_tokens.weight must be 2-D, got %v", tok.Shape())
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: LlamaFromHF needs cfg.Heads")
	}

	// Count layers by probing model.layers.N.
	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF Llama has no model.layers.*")
	}
	cfg.Layers = layers
	if gate, ok := ts["model.layers.0.mlp.gate_proj.weight"]; ok {
		cfg.Hidden = gate.Shape()[0]
	}

	m := &Llama{Config: cfg, TokEmb: cloneF64(tok)} // embedding [vocab,dim], no transpose
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF Llama missing %s%s", p, name)
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
		an, err := g("input_layernorm.weight")
		if err != nil {
			return nil, err
		}
		fn, err := g("post_attention_layernorm.weight")
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
		// Optional Qwen2-family q/k/v projection biases (o_proj has none).
		bias := func(name string) *tensor.Tensor {
			if t, ok := ts[p+name]; ok {
				return cloneF64(t)
			}
			return nil
		}
		// Optional Qwen3/OLMo2 per-head QK-norm gains.
		qkNorm := func(name string) *nn.RMSNorm {
			if t, ok := ts[p+name]; ok {
				return rmsFromGGUF(t, cfg.Eps)
			}
			return nil
		}
		m.Blocks = append(m.Blocks, &LlamaBlock{
			AttnNorm: rmsFromGGUF(an, cfg.Eps),
			Wq:       transpose2D(wq),
			Wk:       transpose2D(wk),
			Wv:       transpose2D(wv),
			Wo:       transpose2D(wo),
			Bq:       bias("self_attn.q_proj.bias"),
			Bk:       bias("self_attn.k_proj.bias"),
			Bv:       bias("self_attn.v_proj.bias"),
			QNorm:    qkNorm("self_attn.q_norm.weight"),
			KNorm:    qkNorm("self_attn.k_norm.weight"),
			FFNNorm:  rmsFromGGUF(fn, cfg.Eps),
			FFN:      swiGLUFromGGUF(gate, up, down),
		})
	}
	norm, ok := ts["model.norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Llama missing model.norm.weight")
	}
	m.Norm = rmsFromGGUF(norm, cfg.Eps)
	head := tok // tied head if lm_head absent
	if o, ok := ts["lm_head.weight"]; ok {
		head = o
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}
