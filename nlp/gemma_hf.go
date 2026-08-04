package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// GemmaFromHF builds a [Gemma] from a Hugging Face checkpoint's tensor map (the
// weights of transformers' GemmaForCausalLM). Like [LlamaFromHF] the config
// (Heads, KVHeads, Eps, RopeBase, Ctx) comes from config.json and is passed in
// cfg; the dimensions (Dim, Vocab, Layers, FFN, HeadDim) are inferred from the
// tensors. HF stores each projection as torch [out, in]; GoAI wants [in, out], so
// every weight is transposed.
//
// Two Gemma-specific foldings happen here: the RMSNorm (1+w) convention is folded
// into the gains (γ ← γ+1) so the shared nn.RMSNorm applies it directly, and the
// LM head ties to the (unscaled) embedding. The GeGLU FFN reuses nn.GLU.
func GemmaFromHF(ts map[string]*tensor.Tensor, cfg GemmaConfig) (*Gemma, error) {
	tok, ok := ts["model.embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Gemma missing model.embed_tokens.weight")
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: model.embed_tokens.weight must be 2-D, got %v", tok.Shape())
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: GemmaFromHF needs cfg.Heads")
	}

	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF Gemma has no model.layers.*")
	}
	cfg.Layers = layers
	if gate, ok := ts["model.layers.0.mlp.gate_proj.weight"]; ok {
		cfg.FFN = gate.Shape()[0]
	}
	// head_dim is decoupled from dim/heads: infer it from the q_proj inner width.
	if wq, ok := ts["model.layers.0.self_attn.q_proj.weight"]; ok {
		cfg.HeadDim = wq.Shape()[0] / cfg.Heads
	}

	m := &Gemma{Config: cfg, TokEmb: cloneF64(tok)}
	//perfscan:ignore PS3060 HF Gemma loader, one-time model load
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF Gemma missing %s%s", p, name)
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
		m.Blocks = append(m.Blocks, &GemmaBlock{
			AttnNorm: gemmaRMS(an, cfg.Eps),
			Wq:       transpose2D(wq),
			Wk:       transpose2D(wk),
			Wv:       transpose2D(wv),
			Wo:       transpose2D(wo),
			FFNNorm:  gemmaRMS(fn, cfg.Eps),
			FFN:      geGLUFromHF(gate, up, down),
		})
	}
	norm, ok := ts["model.norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Gemma missing model.norm.weight")
	}
	m.FinalNorm = gemmaRMS(norm, cfg.Eps)
	return m, nil
}

// gemmaRMS folds Gemma's (1+weight) RMSNorm convention into the gain: γ ← γ+1, so
// the shared nn.RMSNorm (which multiplies by γ) reproduces output·(1+weight).
func gemmaRMS(gamma *tensor.Tensor, eps float64) *nn.RMSNorm {
	g := cloneF64(gamma)
	gs := g.Storage().F64()
	for i := range gs {
		gs[i] += 1.0
	}
	return &nn.RMSNorm{Gamma: g, Eps: eps}
}

// geGLUFromHF builds a GeGLU nn.GLU from the HF mlp gate/up/down weights, each
// transposed from torch [out,in] into GoAI's [in,out]. It goes through NewGEGLU so
// the (unexported) GELU gate activation is set, then swaps in the loaded weights.
func geGLUFromHF(gate, up, down *tensor.Tensor) *nn.GLU {
	g := nn.NewGEGLU(tensor.F64, 1, 1, 0)
	g.Wgate, g.Wup, g.Wdown = transpose2D(gate), transpose2D(up), transpose2D(down)
	return g
}
