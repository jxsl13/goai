package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// StableLMFromHF builds a [StableLM] from a Hugging Face checkpoint's tensor map (the
// weights of transformers' StableLmForCausalLM). Like [LlamaFromHF] the geometry not
// derivable from the tensors (Heads, KVHeads, Eps, RopeBase, RotaryPct/RotaryDim, Ctx)
// comes from config.json and is passed in cfg; the dimensions (Dim, Vocab, Layers, Hidden)
// are inferred from the tensors. HF stores each projection as torch [out, in]; GoAI wants
// [in, out], so every weight is transposed.
//
// Two StableLM specifics are handled here:
//
//   - LayerNorm WITH bias. input_layernorm, post_attention_layernorm and model.norm are
//     ordinary LayerNorms; BOTH the weight (γ) and the bias (β) are loaded from the
//     checkpoint (see [layerNormBiasFromHF]) — unlike Cohere's weight-only LayerNorm whose
//     β is forced to zero.
//
//   - Partial split-half RoPE. Nothing is baked into the weights: StableLM's rotary uses
//     the split-half convention (the same as GoAI's OpRoPE), and [StableLM.attention]
//     rotates only the first cfg.rotaryDim() channels per head via [partialRoPE]. No q/k row
//     permutation is needed (that is Cohere's interleaved-rotary case).
//
// The default config has no attention bias (use_qkv_bias=false) and no QK-norm
// (qk_layernorm=false); those tensors are absent from the checkpoint and this loader does
// not read them. The LM head is UNTIED (tie_word_embeddings=false): lm_head.weight is
// present and used directly.
func StableLMFromHF(ts map[string]*tensor.Tensor, cfg StableLMConfig) (*StableLM, error) {
	tok, ok := ts["model.embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF StableLM missing model.embed_tokens.weight")
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: model.embed_tokens.weight must be 2-D, got %v", tok.Shape())
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: StableLMFromHF needs cfg.Heads")
	}

	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF StableLM has no model.layers.*")
	}
	cfg.Layers = layers
	if gate, ok := ts["model.layers.0.mlp.gate_proj.weight"]; ok {
		cfg.Hidden = gate.Shape()[0]
	}
	// head_dim = q_proj output width / heads (StableLM does not decouple it).
	if wq, ok := ts["model.layers.0.self_attn.q_proj.weight"]; ok {
		cfg.HeadDim = wq.Shape()[0] / cfg.Heads
	}
	if cfg.HeadDim <= 0 {
		return nil, fmt.Errorf("nlp: StableLMFromHF could not infer head_dim")
	}

	m := &StableLM{Config: cfg, TokEmb: cloneF64(tok)}
	//perfscan:ignore PS3060 model-load weight transpose, one-time
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF StableLM missing %s%s", p, name)
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
		inNorm, err := layerNormBiasFromHF(ts, p+"input_layernorm", cfg.Eps)
		if err != nil {
			return nil, err
		}
		postNorm, err := layerNormBiasFromHF(ts, p+"post_attention_layernorm", cfg.Eps)
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
		m.Blocks = append(m.Blocks, &StableLMBlock{
			InputNorm:    inNorm,
			PostAttnNorm: postNorm,
			Wq:           transpose2D(wq),
			Wk:           transpose2D(wk),
			Wv:           transpose2D(wv),
			Wo:           transpose2D(wo),
			FFN:          swiGLUFromGGUF(gate, up, down),
		})
	}
	norm, err := layerNormBiasFromHF(ts, "model.norm", cfg.Eps)
	if err != nil {
		return nil, err
	}
	m.Norm = norm
	head, ok := ts["lm_head.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF StableLM missing lm_head.weight (untied head)")
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}

// layerNormBiasFromHF loads a full LayerNorm (weight γ AND bias β) from prefix.weight and
// prefix.bias, widening both to F64. This is StableLM's ordinary LayerNorm — distinct from
// Cohere's weight-only [layerNormFromHF], which forces β to zero.
func layerNormBiasFromHF(ts map[string]*tensor.Tensor, prefix string, eps float64) (*nn.LayerNorm, error) {
	g, ok := ts[prefix+".weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF StableLM missing %s.weight", prefix)
	}
	b, ok := ts[prefix+".bias"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF StableLM missing %s.bias", prefix)
	}
	return &nn.LayerNorm{Gamma: cloneF64(g), Beta: cloneF64(b), Eps: eps}, nil
}
