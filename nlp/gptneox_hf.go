package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// GPTNeoXFromHF builds a [GPTNeoX] from a Hugging Face GPTNeoXForCausalLM / Pythia
// checkpoint's tensor map. cfg supplies the geometry not derivable from the tensors — Heads,
// Eps, RopeBase, Ctx and the partial-rotary fraction (RotaryPct or RotaryDim) from
// config.json; Dim, Vocab, Layers and Hidden are inferred. HF stores every dense weight as
// torch [out, in]; GoAI wants [in, out], so each projection is transposed (biases pass
// through unchanged).
//
// The one non-obvious step is the query_key_value split. HF packs it as [3·hidden, hidden]
// with the output viewed as [num_heads, 3·head_size] and chunked into q/k/v — i.e. the rows
// are INTERLEAVED per head (head h occupies rows [h·3·hd, (h+1)·3·hd), its first head_size
// rows being q, the next k, the last v). [splitNeoXQKV] de-interleaves that into
// head-contiguous Wq/Wk/Wv (and the same for the bias) so GoAI's per-head OpRoPE / OpMHA see
// the standard [heads·head_dim] layout. Getting this layout wrong (e.g. a naive [all-q;
// all-k; all-v] split) scrambles the heads and blows up the logit parity.
func GPTNeoXFromHF(ts map[string]*tensor.Tensor, cfg GPTNeoXConfig) (*GPTNeoX, error) {
	get := func(name string) (*tensor.Tensor, error) {
		if t, ok := ts[name]; ok {
			return t, nil
		}
		return nil, fmt.Errorf("nlp: HF GPT-NeoX missing %s", name)
	}
	tok, err := get("gpt_neox.embed_in.weight")
	if err != nil {
		return nil, err
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: gpt_neox.embed_in.weight must be 2-D, got %v", tok.Shape())
	}
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: GPTNeoXFromHF needs cfg.Heads")
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: GPT-NeoX dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	headDim := cfg.Dim / cfg.Heads

	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("gpt_neox.layers.%d.attention.query_key_value.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF GPT-NeoX has no gpt_neox.layers.*")
	}
	cfg.Layers = layers
	if hto4h, ok := ts["gpt_neox.layers.0.mlp.dense_h_to_4h.weight"]; ok {
		cfg.Hidden = hto4h.Shape()[0]
	}

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

	m := &GPTNeoX{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range layers {
		p := fmt.Sprintf("gpt_neox.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) { return get(p + name) }

		qkvW, err := g("attention.query_key_value.weight")
		if err != nil {
			return nil, err
		}
		if got := qkvW.Shape()[0]; got != 3*cfg.Dim {
			return nil, fmt.Errorf("nlp: GPT-NeoX layer %d query_key_value rows %d != 3·hidden = %d", l, got, 3*cfg.Dim)
		}
		qkvB, err := g("attention.query_key_value.bias")
		if err != nil {
			return nil, err
		}
		// De-interleave the packed [num_heads, 3·head_size] rows into per-head-contiguous
		// q/k/v, then transpose each weight to GoAI's [in, out] layout. Biases stay 1-D.
		wq, wk, wv := splitNeoXQKV(qkvW, cfg.Heads, headDim)
		bq, bk, bv := splitNeoXQKV1D(qkvB, cfg.Heads, headDim)

		denseW, err := g("attention.dense.weight")
		if err != nil {
			return nil, err
		}
		denseB, err := g("attention.dense.bias")
		if err != nil {
			return nil, err
		}
		hto4hW, err := g("mlp.dense_h_to_4h.weight")
		if err != nil {
			return nil, err
		}
		hto4hB, err := g("mlp.dense_h_to_4h.bias")
		if err != nil {
			return nil, err
		}
		fourhtohW, err := g("mlp.dense_4h_to_h.weight")
		if err != nil {
			return nil, err
		}
		fourhtohB, err := g("mlp.dense_4h_to_h.bias")
		if err != nil {
			return nil, err
		}
		inNorm, err := ln(p + "input_layernorm")
		if err != nil {
			return nil, err
		}
		postNorm, err := ln(p + "post_attention_layernorm")
		if err != nil {
			return nil, err
		}

		m.Blocks = append(m.Blocks, &GPTNeoXBlock{
			InputNorm: inNorm, PostAttnNorm: postNorm,
			Wq: transpose2D(wq), Wk: transpose2D(wk), Wv: transpose2D(wv), Wo: transpose2D(denseW),
			Bq: bq, Bk: bk, Bv: bv, Bo: cloneF64(denseB),
			Wh: transpose2D(hto4hW), Bh: cloneF64(hto4hB),
			Wout: transpose2D(fourhtohW), Bout: cloneF64(fourhtohB),
		})
	}

	finalNorm, err := ln("gpt_neox.final_layer_norm")
	if err != nil {
		return nil, err
	}
	m.FinalNorm = finalNorm
	// Untied LM head (tie_word_embeddings=false). Canonically embed_out.weight; newer
	// transformers export it as lm_head.weight, so accept either.
	head, ok := ts["embed_out.weight"]
	if !ok {
		if head, ok = ts["lm_head.weight"]; !ok {
			return nil, fmt.Errorf("nlp: HF GPT-NeoX missing embed_out.weight / lm_head.weight")
		}
	}
	m.Out = transpose2D(head) // [vocab, dim] → [dim, vocab]
	return m, nil
}

// splitNeoXQKV de-interleaves a GPT-NeoX packed query_key_value weight [3·hidden, in] into
// three head-contiguous projection weights [hidden, in] (torch [out, in] layout, ready for
// transpose2D). The packing views the output as [num_heads, 3·head_size]: for head h the
// query rows are [h·3·hd, h·3·hd+hd), the key rows the next hd, the value rows the last hd.
// Each output collects, per head, its head_size rows into contiguous [h·hd, (h+1)·hd).
func splitNeoXQKV(qkv *tensor.Tensor, heads, headDim int) (q, k, v *tensor.Tensor) {
	in := qkv.Shape()[1]
	mk := func(which int) *tensor.Tensor {
		out := tensor.New(tensor.F64, tensor.Shape{heads * headDim, in})
		// out is constructed F64 here, so its storage is written directly — one of the two dispatches
		// per element goes away with no dtype fallback (PERF-NO-FALLBACK-FOR-A-FIXED-DTYPE-001). qkv
		// keeps its accessor: its dtype comes from the loaded file.
		of := out.Storage().F64()
		for h := range heads {
			for d := range headDim {
				src := h*3*headDim + which*headDim + d
				dst := h*headDim + d
				row := of[dst*in : (dst+1)*in]
				for c := range in {
					row[c] = qkv.AtF64(src, c)
				}
			}
		}
		return out
	}
	return mk(0), mk(1), mk(2)
}

// splitNeoXQKV1D de-interleaves the packed query_key_value bias [3·hidden] into three
// head-contiguous biases [hidden] with the same per-head layout as [splitNeoXQKV].
func splitNeoXQKV1D(bias *tensor.Tensor, heads, headDim int) (q, k, v *tensor.Tensor) {
	mk := func(which int) *tensor.Tensor {
		out := tensor.New(tensor.F64, tensor.Shape{heads * headDim})
		for h := range heads {
			for d := range headDim {
				out.SetF64(bias.AtF64(h*3*headDim+which*headDim+d), h*headDim+d)
			}
		}
		return out
	}
	return mk(0), mk(1), mk(2)
}
