package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// Gemma2FromHF builds a [Gemma2] from a Hugging Face checkpoint's tensor map (the
// weights of transformers' Gemma2ForCausalLM). Like [GemmaFromHF] the geometry that
// is not derivable from the tensors (Heads, KVHeads, Eps, RopeBase, Ctx) and the three
// Gemma 2 scalars (QueryPreAttnScalar, AttnLogitCap, FinalLogitCap) come from
// config.json and are passed in cfg; the dimensions (Dim, Vocab, Layers, FFN, HeadDim)
// are inferred from the tensors. HF stores each projection as torch [out, in]; GoAI
// wants [in, out], so every weight is transposed.
//
// The four sandwich RMSNorms per block (input_layernorm, post_attention_layernorm,
// pre_feedforward_layernorm, post_feedforward_layernorm) and model.norm all use
// Gemma's (1+w) convention, folded into the gains here via gemmaRMS (γ ← γ+1). The
// LM head ties to the (unscaled) embedding and the GeGLU FFN reuses nn.GLU.
func Gemma2FromHF(ts map[string]*tensor.Tensor, cfg Gemma2Config) (*Gemma2, error) {
	tok, ok := ts["model.embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Gemma2 missing model.embed_tokens.weight")
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: model.embed_tokens.weight must be 2-D, got %v", tok.Shape())
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: Gemma2FromHF needs cfg.Heads")
	}

	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF Gemma2 has no model.layers.*")
	}
	cfg.Layers = layers
	if gate, ok := ts["model.layers.0.mlp.gate_proj.weight"]; ok {
		cfg.FFN = gate.Shape()[0]
	}
	// head_dim is decoupled from dim/heads: infer it from the q_proj inner width.
	if wq, ok := ts["model.layers.0.self_attn.q_proj.weight"]; ok {
		cfg.HeadDim = wq.Shape()[0] / cfg.Heads
	}

	m := &Gemma2{Config: cfg, TokEmb: cloneF64(tok)}
	//perfscan:ignore PS3060 HF Gemma2 loader, one-time model load
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF Gemma2 missing %s%s", p, name)
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
		inNorm, err := g("input_layernorm.weight")
		if err != nil {
			return nil, err
		}
		postAttn, err := g("post_attention_layernorm.weight")
		if err != nil {
			return nil, err
		}
		preFFN, err := g("pre_feedforward_layernorm.weight")
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
		m.Blocks = append(m.Blocks, &Gemma2Block{
			InputNorm:    gemmaRMS(inNorm, cfg.Eps),
			Wq:           transpose2D(wq),
			Wk:           transpose2D(wk),
			Wv:           transpose2D(wv),
			Wo:           transpose2D(wo),
			PostAttnNorm: gemmaRMS(postAttn, cfg.Eps),
			PreFFNNorm:   gemmaRMS(preFFN, cfg.Eps),
			FFN:          geGLUFromHF(gate, up, down),
			PostFFNNorm:  gemmaRMS(postFFN, cfg.Eps),
		})
	}
	norm, ok := ts["model.norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Gemma2 missing model.norm.weight")
	}
	m.FinalNorm = gemmaRMS(norm, cfg.Eps)
	// Sliding-window clamp, mirroring [Gemma2FromGGUF] (which sets
	// Ctx = min(context_length, sliding_window)). Real Gemma 2 alternates sliding-window
	// and full-attention layers; GoAI's Gemma2 implements FULL attention only, and the
	// two graphs coincide exactly only while seq ≤ window. Without this clamp an
	// HF-loaded model would accept prompts past the window and every SWA layer would
	// silently attend the whole prefix — wrong logits, plausible text, no error.
	//
	// The asymmetry to keep in mind when editing either loader: the GGUF path reads the
	// window from gemma2.attention.sliding_window and falls back to 4096, while
	// Gemma2Config carries no window field, so the HF path can only apply the same 4096
	// default. Every released Gemma 2 uses 4096; a checkpoint that does not would need
	// the field added to Gemma2Config and both loaders updated together.
	if gemma2DefaultSWA > 0 && gemma2DefaultSWA < m.Config.Ctx {
		m.Config.Ctx = gemma2DefaultSWA
	}
	return m, nil
}
