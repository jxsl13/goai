package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// MPTFromHF builds an [MPT] from a Hugging Face MptForCausalLM checkpoint's tensor map. cfg
// supplies the geometry not derivable from the tensors — Heads, Eps and Ctx from config.json;
// Dim, Vocab, Layers and Hidden are inferred. HF stores every dense weight as torch [out, in];
// GoAI wants [in, out], so each projection is transposed.
//
// The one non-obvious step is the Wqkv split. MPT packs attention as attn.Wqkv [3·hidden,
// hidden] and transformers splits the projection output with mixed_qkv.chunk(3, dim=2) — i.e.
// contiguous [all-q; all-k; all-v] row blocks (the Phi-3 layout), NOT the GPT-NeoX per-head
// interleave. [sliceRows] therefore recovers q/k/v as rows [0,hidden), [hidden,2·hidden),
// [2·hidden,3·hidden). No projection carries a bias; the LM head is tied to the token
// embedding (logits = hidden · wteᵀ).
func MPTFromHF(ts map[string]*tensor.Tensor, cfg MPTConfig) (*MPT, error) {
	get := func(name string) (*tensor.Tensor, error) {
		if t, ok := ts[name]; ok {
			return t, nil
		}
		return nil, fmt.Errorf("nlp: HF MPT missing %s", name)
	}
	tok, err := get("transformer.wte.weight")
	if err != nil {
		return nil, err
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: transformer.wte.weight must be 2-D, got %v", tok.Shape())
	}
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: MPTFromHF needs cfg.Heads")
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: MPT dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}

	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("transformer.blocks.%d.attn.Wqkv.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF MPT has no transformer.blocks.*")
	}
	cfg.Layers = layers
	if up, ok := ts["transformer.blocks.0.ffn.up_proj.weight"]; ok {
		cfg.Hidden = up.Shape()[0]
	}

	m := &MPT{Config: cfg, TokEmb: cloneF64(tok)}
	//perfscan:ignore PS3060 model-load weight transpose, one-time
	for l := range layers {
		p := fmt.Sprintf("transformer.blocks.%d.", l)
		g := func(name string) (*tensor.Tensor, error) { return get(p + name) }

		qkv, err := g("attn.Wqkv.weight")
		if err != nil {
			return nil, err
		}
		if got := qkv.Shape()[0]; got != 3*cfg.Dim {
			return nil, fmt.Errorf("nlp: MPT layer %d Wqkv rows %d != 3·hidden = %d", l, got, 3*cfg.Dim)
		}
		// Standard-MHA split: contiguous all-q / all-k / all-v row blocks (Phi-3 layout).
		wq := sliceRows(qkv, 0, cfg.Dim)
		wk := sliceRows(qkv, cfg.Dim, 2*cfg.Dim)
		wv := sliceRows(qkv, 2*cfg.Dim, 3*cfg.Dim)

		outW, err := g("attn.out_proj.weight")
		if err != nil {
			return nil, err
		}
		upW, err := g("ffn.up_proj.weight")
		if err != nil {
			return nil, err
		}
		downW, err := g("ffn.down_proj.weight")
		if err != nil {
			return nil, err
		}
		n1, err := g("norm_1.weight")
		if err != nil {
			return nil, err
		}
		n2, err := g("norm_2.weight")
		if err != nil {
			return nil, err
		}

		m.Blocks = append(m.Blocks, &MPTBlock{
			Norm1: layerNormFromHF(n1, cfg.Eps), Norm2: layerNormFromHF(n2, cfg.Eps),
			Wq: transpose2D(wq), Wk: transpose2D(wk), Wv: transpose2D(wv), Wo: transpose2D(outW),
			Wup: transpose2D(upW), Wdown: transpose2D(downW),
		})
	}

	normF, err := get("transformer.norm_f.weight")
	if err != nil {
		return nil, err
	}
	m.FinalNorm = layerNormFromHF(normF, cfg.Eps)
	// Tied LM head (MptForCausalLM ties lm_head to transformer.wte): logits = hidden · wteᵀ.
	m.Out = transpose2D(tok)
	return m, nil
}
