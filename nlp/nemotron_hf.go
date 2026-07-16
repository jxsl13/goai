package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// NemotronFromHF builds a [Nemotron] from a Hugging Face checkpoint's tensor map (the
// weights of transformers' NemotronForCausalLM). Like [LlamaFromHF] the geometry not
// derivable from the tensors (Heads, KVHeads, Eps, RopeBase, RotaryPct/RotaryDim, Ctx)
// comes from config.json and is passed in cfg; the dimensions (Dim, Vocab, Layers, Hidden)
// are inferred from the tensors. HF stores each projection as torch [out, in]; GoAI wants
// [in, out], so every weight is transposed.
//
// Three Nemotron specifics are handled here:
//
//   - LayerNorm1P. input_layernorm, post_attention_layernorm and model.norm are
//     NemotronLayerNorm1P, whose forward applies F.layer_norm(x, …, weight + 1.0, bias, eps):
//     the checkpoint stores the gain as an OFFSET from 1. [layerNorm1PFromHF] folds the +1
//     into γ (γ = checkpoint_weight + 1) — exactly as [gemmaRMS] does for Gemma's (1+w)
//     RMSNorm — and loads the bias β directly, so the shared nn.LayerNorm reproduces
//     transformers' output. Forgetting the +1 fold or the β moves the logits far off golden.
//
//   - ReLU² MLP (no gate, no bias). Each block has only mlp.up_proj.weight and
//     mlp.down_proj.weight (mlp_bias=false); [Nemotron.hidden] applies down(relu²(up(x))).
//
//   - Partial split-half RoPE. Nothing is baked into the weights: Nemotron's rotary uses the
//     split-half convention (the same as GoAI's OpRoPE), and [Nemotron.attention] rotates
//     only the first cfg.rotaryDim() channels per head via [partialRoPE]. No q/k row
//     permutation is needed (that is Cohere's interleaved-rotary case).
//
// The config has no attention bias (attention_bias=false) and no QK-norm; those tensors are
// absent from the checkpoint. The LM head is UNTIED (tie_word_embeddings=false):
// lm_head.weight is present and used directly.
func NemotronFromHF(ts map[string]*tensor.Tensor, cfg NemotronConfig) (*Nemotron, error) {
	tok, ok := ts["model.embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Nemotron missing model.embed_tokens.weight")
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: model.embed_tokens.weight must be 2-D, got %v", tok.Shape())
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: NemotronFromHF needs cfg.Heads")
	}

	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF Nemotron has no model.layers.*")
	}
	cfg.Layers = layers
	if up, ok := ts["model.layers.0.mlp.up_proj.weight"]; ok {
		cfg.Hidden = up.Shape()[0]
	}
	// head_dim = q_proj output width / heads.
	if wq, ok := ts["model.layers.0.self_attn.q_proj.weight"]; ok {
		cfg.HeadDim = wq.Shape()[0] / cfg.Heads
	}
	if cfg.HeadDim <= 0 {
		return nil, fmt.Errorf("nlp: NemotronFromHF could not infer head_dim")
	}

	m := &Nemotron{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF Nemotron missing %s%s", p, name)
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
		inNorm, err := layerNorm1PFromHF(ts, p+"input_layernorm", cfg.Eps)
		if err != nil {
			return nil, err
		}
		postNorm, err := layerNorm1PFromHF(ts, p+"post_attention_layernorm", cfg.Eps)
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
		m.Blocks = append(m.Blocks, &NemotronBlock{
			InputNorm:    inNorm,
			PostAttnNorm: postNorm,
			Wq:           transpose2D(wq),
			Wk:           transpose2D(wk),
			Wv:           transpose2D(wv),
			Wo:           transpose2D(wo),
			Wup:          transpose2D(up),   // [out,in] → [in,out] = [dim,hidden]
			Wdown:        transpose2D(down), // [out,in] → [in,out] = [hidden,dim]
		})
	}
	norm, err := layerNorm1PFromHF(ts, "model.norm", cfg.Eps)
	if err != nil {
		return nil, err
	}
	m.Norm = norm
	head, ok := ts["lm_head.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Nemotron missing lm_head.weight (untied head)")
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}

// layerNorm1PFromHF loads a NemotronLayerNorm1P from prefix.weight and prefix.bias. Nemotron
// stores the gain as an OFFSET from 1 and its forward uses (weight + 1.0), so the +1 is
// FOLDED into γ here (γ = weight + 1) — mirroring [gemmaRMS] — and the bias β is loaded
// directly. The resulting nn.LayerNorm (which computes (x−μ)/σ·γ + β) reproduces
// transformers' F.layer_norm(x, …, weight + 1.0, bias, eps).
func layerNorm1PFromHF(ts map[string]*tensor.Tensor, prefix string, eps float64) (*nn.LayerNorm, error) {
	w, ok := ts[prefix+".weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Nemotron missing %s.weight", prefix)
	}
	b, ok := ts[prefix+".bias"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Nemotron missing %s.bias", prefix)
	}
	g := cloneF64(w)
	gs := g.Storage().F64()
	for i := range gs {
		gs[i] += 1.0 // fold Nemotron's (1 + weight) gain
	}
	return &nn.LayerNorm{Gamma: g, Beta: cloneF64(b), Eps: eps}, nil
}
