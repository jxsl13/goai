package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// StarCoder2FromHF builds a [StarCoder2] from a Hugging Face checkpoint's tensor map (the
// weights of transformers' Starcoder2ForCausalLM). Like [StableLMFromHF] the geometry not
// derivable from the tensors (Heads, KVHeads, Eps, RopeBase, Ctx) comes from config.json and
// is passed in cfg; the dimensions (Dim, Vocab, Layers, Hidden) are inferred from the tensors.
// HF stores each projection as torch [out, in]; GoAI wants [in, out], so every weight is
// transposed (biases pass through unchanged).
//
// Three StarCoder2 specifics are handled here:
//
//   - LayerNorm WITH bias. input_layernorm, post_attention_layernorm and model.norm are
//     ordinary LayerNorms; BOTH the weight (γ) and the bias (β) are loaded from the checkpoint
//     via [layerNormBiasFromHF] (shared with [StableLMFromHF]).
//
//   - BIASED attention (config.use_bias=True). q_proj / k_proj / v_proj / o_proj each carry a
//     bias tensor alongside their weight; all eight are loaded.
//
//   - Biased 2-layer GELU MLP. mlp.c_fc and mlp.c_proj each carry a weight and a bias; there
//     is no SwiGLU gate. Nothing is baked into the weights for RoPE — StarCoder2 uses FULL
//     split-half rotary (the same convention as GoAI's OpRoPE), applied inside
//     [StarCoder2.attention].
//
// The LM head is UNTIED (tie_word_embeddings=false): lm_head.weight is present and used
// directly.
func StarCoder2FromHF(ts map[string]*tensor.Tensor, cfg StarCoder2Config) (*StarCoder2, error) {
	tok, ok := ts["model.embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF StarCoder2 missing model.embed_tokens.weight")
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: model.embed_tokens.weight must be 2-D, got %v", tok.Shape())
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: StarCoder2FromHF needs cfg.Heads")
	}

	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF StarCoder2 has no model.layers.*")
	}
	cfg.Layers = layers
	if fc, ok := ts["model.layers.0.mlp.c_fc.weight"]; ok {
		cfg.Hidden = fc.Shape()[0]
	}
	// head_dim = q_proj output width / heads (StarCoder2 does not decouple it).
	if wq, ok := ts["model.layers.0.self_attn.q_proj.weight"]; ok {
		cfg.HeadDim = wq.Shape()[0] / cfg.Heads
	}
	if cfg.HeadDim <= 0 {
		return nil, fmt.Errorf("nlp: StarCoder2FromHF could not infer head_dim")
	}

	m := &StarCoder2{Config: cfg, TokEmb: cloneF64(tok)}
	//perfscan:ignore PS3060 model-load weight transpose, one-time
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF StarCoder2 missing %s%s", p, name)
			}
			return t, nil
		}
		proj := func(name string) (w, b *tensor.Tensor, err error) {
			if w, err = g(name + ".weight"); err != nil {
				return
			}
			b, err = g(name + ".bias")
			return
		}
		wq, bq, err := proj("self_attn.q_proj")
		if err != nil {
			return nil, err
		}
		wk, bk, err := proj("self_attn.k_proj")
		if err != nil {
			return nil, err
		}
		wv, bv, err := proj("self_attn.v_proj")
		if err != nil {
			return nil, err
		}
		wo, bo, err := proj("self_attn.o_proj")
		if err != nil {
			return nil, err
		}
		wfc, bfc, err := proj("mlp.c_fc")
		if err != nil {
			return nil, err
		}
		wproj, bproj, err := proj("mlp.c_proj")
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
		m.Blocks = append(m.Blocks, &StarCoder2Block{
			InputNorm:    inNorm,
			PostAttnNorm: postNorm,
			Wq:           transpose2D(wq), Bq: cloneF64(bq),
			Wk: transpose2D(wk), Bk: cloneF64(bk),
			Wv: transpose2D(wv), Bv: cloneF64(bv),
			Wo: transpose2D(wo), Bo: cloneF64(bo),
			Wfc: transpose2D(wfc), Bfc: cloneF64(bfc),
			Wproj: transpose2D(wproj), Bproj: cloneF64(bproj),
		})
	}
	norm, err := layerNormBiasFromHF(ts, "model.norm", cfg.Eps)
	if err != nil {
		return nil, err
	}
	m.Norm = norm
	head, ok := ts["lm_head.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF StarCoder2 missing lm_head.weight (untied head)")
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}
