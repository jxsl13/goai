package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// FalconFromHF builds a [Falcon] from a Hugging Face FalconForCausalLM checkpoint's
// tensor map, targeting the ORIGINAL Falcon-7B architecture (new_decoder_architecture=
// False, parallel_attn=True, multi_query=True, alibi=False). cfg supplies the geometry
// not derivable from the tensors — Heads, Eps, RopeBase and Ctx from config.json; Dim,
// Vocab, Layers and Hidden are inferred. HF stores every dense weight as torch [out,
// in]; GoAI wants [in, out], so each projection is transposed.
//
// The one non-obvious step is the FUSED multi-query query_key_value split. Falcon packs
// a SINGLE weight [num_heads·head_dim + 2·head_dim, dim] whose rows are the concatenation
// of all query heads followed by ONE shared key head and ONE shared value head
// (FalconAttention._split_heads views it as [num_heads+2, head_dim] and returns [:-2],
// [-2], [-1]). [FalconFromHF] slices those contiguous row bands:
//
//	q = rows [0 : num_heads·hd]              (all query heads)
//	k = rows [num_heads·hd : (num_heads+1)·hd]     (the ONE shared key head)
//	v = rows [(num_heads+1)·hd : (num_heads+2)·hd] (the ONE shared value head)
//
// so GoAI's OpMHA runs the MQA with KVHeads=1. Getting this band split wrong (e.g. a
// naive three-way [all-q; all-k; all-v] split, which would give k/v the wrong width) is
// the #1 parity killer here. The query_key_value and dense projections are bias-free
// (config.bias=False); only the LayerNorms carry a bias.
//
// The LM head is UNTIED (tie_word_embeddings=false); lm_head.weight is loaded separately.
func FalconFromHF(ts map[string]*tensor.Tensor, cfg FalconConfig) (*Falcon, error) {
	get := func(name string) (*tensor.Tensor, error) {
		if t, ok := ts[name]; ok {
			return t, nil
		}
		return nil, fmt.Errorf("nlp: HF Falcon missing %s", name)
	}
	tok, err := get("transformer.word_embeddings.weight")
	if err != nil {
		return nil, err
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: transformer.word_embeddings.weight must be 2-D, got %v", tok.Shape())
	}
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: FalconFromHF needs cfg.Heads")
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: Falcon dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	cfg.KVHeads = 1 // Falcon-7B multi-query attention: one shared key/value head.
	headDim := cfg.Dim / cfg.Heads

	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("transformer.h.%d.self_attention.query_key_value.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF Falcon has no transformer.h.*")
	}
	cfg.Layers = layers
	if hto4h, ok := ts["transformer.h.0.mlp.dense_h_to_4h.weight"]; ok {
		cfg.Hidden = hto4h.Shape()[0]
	}

	// ln loads a Falcon LayerNorm (weight AND bias — Falcon norms always carry a bias,
	// independent of config.bias which governs only the linear layers).
	ln := func(prefix string) (*nn.LayerNorm, error) {
		g, err := get(prefix + ".weight")
		if err != nil {
			return nil, err
		}
		b, err := get(prefix + ".bias")
		if err != nil {
			return nil, err
		}
		return &nn.LayerNorm{Gamma: cloneF64(g), Beta: cloneF64(b), Eps: cfg.Eps}, nil
	}

	m := &Falcon{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range layers {
		p := fmt.Sprintf("transformer.h.%d.", l)
		g := func(name string) (*tensor.Tensor, error) { return get(p + name) }

		qkv, err := g("self_attention.query_key_value.weight")
		if err != nil {
			return nil, err
		}
		wantRows := (cfg.Heads + 2) * headDim
		if got := qkv.Shape()[0]; got != wantRows {
			return nil, fmt.Errorf("nlp: Falcon layer %d query_key_value rows %d != (heads+2)·head_dim = %d", l, got, wantRows)
		}
		// Multi-query fused split: q is the first num_heads·hd rows, then ONE key head, then
		// ONE value head. Slice the contiguous row bands (torch [out,in]) before transpose.
		wq := sliceRows(qkv, 0, cfg.Heads*headDim)
		wk := sliceRows(qkv, cfg.Heads*headDim, (cfg.Heads+1)*headDim)
		wv := sliceRows(qkv, (cfg.Heads+1)*headDim, (cfg.Heads+2)*headDim)

		denseW, err := g("self_attention.dense.weight")
		if err != nil {
			return nil, err
		}
		hto4hW, err := g("mlp.dense_h_to_4h.weight")
		if err != nil {
			return nil, err
		}
		fourhtohW, err := g("mlp.dense_4h_to_h.weight")
		if err != nil {
			return nil, err
		}
		inNorm, err := ln(p + "input_layernorm")
		if err != nil {
			return nil, err
		}

		m.Blocks = append(m.Blocks, &FalconBlock{
			InputNorm: inNorm,
			Wq:        transpose2D(wq),
			Wk:        transpose2D(wk),
			Wv:        transpose2D(wv),
			Wo:        transpose2D(denseW),
			Wh:        transpose2D(hto4hW),
			Wout:      transpose2D(fourhtohW),
		})
	}

	finalNorm, err := ln("transformer.ln_f")
	if err != nil {
		return nil, err
	}
	m.FinalNorm = finalNorm
	// Untied LM head (tie_word_embeddings=false).
	head, err := get("lm_head.weight")
	if err != nil {
		return nil, err
	}
	m.Out = transpose2D(head) // [vocab, dim] → [dim, vocab]
	return m, nil
}
