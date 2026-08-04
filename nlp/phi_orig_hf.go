package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// PhiFromHF builds a [Phi] (original Phi-1/1.5/2, transformers' PhiForCausalLM) from a Hugging
// Face checkpoint's tensor map. Like [StableLMFromHF] the geometry not derivable from the
// tensors (Heads, KVHeads, Eps, RopeBase, RotaryPct/RotaryDim, Ctx) comes from config.json and
// is passed in cfg; the dimensions (Dim, Vocab, Layers, Hidden) are inferred. HF stores each
// projection as torch [out, in]; GoAI wants [in, out], so every weight is transposed (biases
// stay 1-D).
//
// Three Phi specifics are handled here:
//
//   - SEPARATE, biased q/k/v/dense. Unlike [GPTNeoX]'s packed query_key_value, Phi stores
//     head-contiguous self_attn.{q,k,v}_proj.{weight,bias} and self_attn.dense.{weight,bias},
//     so no de-interleave split is needed — each is transposed and its bias cloned directly.
//
//   - LayerNorm WITH bias. input_layernorm and model.final_layernorm carry both γ and β,
//     loaded via [phiLayerNorm]. The default config has qk_layernorm=false, so the optional
//     q/k norms are absent and this loader does not read them.
//
//   - BIASED, untied LM head. lm_head carries a bias (lm_head.bias) as well as its weight;
//     both are loaded and the bias is added to the final logits.
//
// The MLP is fc1 → gelu_new → fc2 with biases; PhiFromHF loads fc1/fc2 weights and biases.
// Partial split-half RoPE needs nothing baked into the weights — [Phi.attention] rotates only
// the first cfg.rotaryDim() channels per head via [partialRoPE].
func PhiFromHF(ts map[string]*tensor.Tensor, cfg PhiConfig) (*Phi, error) {
	tok, ok := ts["model.embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Phi missing model.embed_tokens.weight")
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: model.embed_tokens.weight must be 2-D, got %v", tok.Shape())
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: PhiFromHF needs cfg.Heads")
	}

	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF Phi has no model.layers.*")
	}
	cfg.Layers = layers
	if fc1, ok := ts["model.layers.0.mlp.fc1.weight"]; ok {
		cfg.Hidden = fc1.Shape()[0]
	}
	// head_dim = q_proj output width / heads (Phi does not decouple it).
	if wq, ok := ts["model.layers.0.self_attn.q_proj.weight"]; ok {
		cfg.HeadDim = wq.Shape()[0] / cfg.Heads
	}
	if cfg.HeadDim <= 0 {
		return nil, fmt.Errorf("nlp: PhiFromHF could not infer head_dim")
	}

	m := &Phi{Config: cfg, TokEmb: cloneF64(tok)}
	//perfscan:ignore PS3060 model-load weight transpose, one-time
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF Phi missing %s%s", p, name)
			}
			return t, nil
		}
		wq, err := g("self_attn.q_proj.weight")
		if err != nil {
			return nil, err
		}
		bq, err := g("self_attn.q_proj.bias")
		if err != nil {
			return nil, err
		}
		wk, err := g("self_attn.k_proj.weight")
		if err != nil {
			return nil, err
		}
		bk, err := g("self_attn.k_proj.bias")
		if err != nil {
			return nil, err
		}
		wv, err := g("self_attn.v_proj.weight")
		if err != nil {
			return nil, err
		}
		bv, err := g("self_attn.v_proj.bias")
		if err != nil {
			return nil, err
		}
		wd, err := g("self_attn.dense.weight")
		if err != nil {
			return nil, err
		}
		bd, err := g("self_attn.dense.bias")
		if err != nil {
			return nil, err
		}
		fc1, err := g("mlp.fc1.weight")
		if err != nil {
			return nil, err
		}
		bfc1, err := g("mlp.fc1.bias")
		if err != nil {
			return nil, err
		}
		fc2, err := g("mlp.fc2.weight")
		if err != nil {
			return nil, err
		}
		bfc2, err := g("mlp.fc2.bias")
		if err != nil {
			return nil, err
		}
		inNorm, err := phiLayerNorm(ts, p+"input_layernorm", cfg.Eps)
		if err != nil {
			return nil, err
		}
		m.Blocks = append(m.Blocks, &PhiBlock{
			InputNorm: inNorm,
			Wq:        transpose2D(wq), Wk: transpose2D(wk), Wv: transpose2D(wv), Wdense: transpose2D(wd),
			Bq: cloneF64(bq), Bk: cloneF64(bk), Bv: cloneF64(bv), Bdense: cloneF64(bd),
			Wfc1: transpose2D(fc1), Bfc1: cloneF64(bfc1),
			Wfc2: transpose2D(fc2), Bfc2: cloneF64(bfc2),
		})
	}
	finalNorm, err := phiLayerNorm(ts, "model.final_layernorm", cfg.Eps)
	if err != nil {
		return nil, err
	}
	m.FinalNorm = finalNorm
	head, ok := ts["lm_head.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Phi missing lm_head.weight (untied head)")
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	headBias, ok := ts["lm_head.bias"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Phi missing lm_head.bias (Phi's LM head is biased)")
	}
	m.OutBias = cloneF64(headBias)
	return m, nil
}

// phiLayerNorm loads a full LayerNorm (weight γ AND bias β) from prefix.weight and
// prefix.bias, widening both to F64 — Phi's ordinary LayerNorm, as in [GPTNeoX] / [StableLM].
func phiLayerNorm(ts map[string]*tensor.Tensor, prefix string, eps float64) (*nn.LayerNorm, error) {
	g, ok := ts[prefix+".weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Phi missing %s.weight", prefix)
	}
	b, ok := ts[prefix+".bias"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Phi missing %s.bias", prefix)
	}
	return &nn.LayerNorm{Gamma: cloneF64(g), Beta: cloneF64(b), Eps: eps}, nil
}
