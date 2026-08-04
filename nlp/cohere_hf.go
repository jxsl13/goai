package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// CohereFromHF builds a [Cohere] from a Hugging Face checkpoint's tensor map (the weights
// of transformers' CohereForCausalLM). Like [LlamaFromHF] the geometry not derivable from
// the tensors (Heads, KVHeads, Eps, RopeBase, LogitScale, Ctx) comes from config.json and
// is passed in cfg; the dimensions (Dim, Vocab, Layers, Hidden, HeadDim) are inferred from
// the tensors. HF stores each projection as torch [out, in]; GoAI wants [in, out], so
// every weight is transposed.
//
// Two Command-R specifics are baked in here:
//
//   - Interleaved→split-half RoPE permutation. Cohere rotates even/odd channel pairs
//     (GPT-J style) while GoAI's OpRoPE rotates split-half pairs. To reuse OpRoPE
//     unchanged, the q_proj and k_proj ROWS are permuted per head BEFORE the transpose:
//     within head h, original output row h·hd+2i moves to h·hd+i, and h·hd+2i+1 moves to
//     h·hd+(i+hd/2), for i in [0,hd/2). Split-half OpRoPE on the permuted weights then
//     produces the same rotated q·k dot products as Cohere's interleaved rotary (a
//     permutation is orthogonal, so the scores are invariant, and the frequency index i
//     lines up for each pair). V and O are NOT permuted. See [permuteInterleaveToSplit].
//
//   - Weight-only LayerNorm. CohereLayerNorm is a mean-centered LayerNorm with a weight
//     but no bias; GoAI models it with [nn.LayerNorm] built from the HF weight as γ and a
//     ZERO β (see [layerNormFromHF]). This covers every block's input_layernorm and the
//     final model.norm.
//
// The LM head is tied to the embedding table (tie_word_embeddings=true); lm_head.weight is
// absent from the checkpoint and the head reads TokEmbᵀ. Cohere v1 has no QK-norm.
func CohereFromHF(ts map[string]*tensor.Tensor, cfg CohereConfig) (*Cohere, error) {
	tok, ok := ts["model.embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Cohere missing model.embed_tokens.weight")
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: model.embed_tokens.weight must be 2-D, got %v", tok.Shape())
	}
	cfg.Vocab, cfg.Dim = tok.Shape()[0], tok.Shape()[1]
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: CohereFromHF needs cfg.Heads")
	}

	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF Cohere has no model.layers.*")
	}
	cfg.Layers = layers
	if gate, ok := ts["model.layers.0.mlp.gate_proj.weight"]; ok {
		cfg.Hidden = gate.Shape()[0]
	}
	// head_dim is decoupled from dim/heads: infer it from the q_proj output width.
	if wq, ok := ts["model.layers.0.self_attn.q_proj.weight"]; ok {
		cfg.HeadDim = wq.Shape()[0] / cfg.Heads
	}
	if cfg.HeadDim <= 0 {
		return nil, fmt.Errorf("nlp: CohereFromHF could not infer head_dim")
	}
	kv := cfg.kvHeads()

	m := &Cohere{Config: cfg, TokEmb: cloneF64(tok)}
	//perfscan:ignore PS3060 per-layer loader loop; cold
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF Cohere missing %s%s", p, name)
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
		// q/k rows are permuted interleaved→split-half so GoAI's split-half OpRoPE
		// reproduces Cohere's interleaved rotary; v/o are untouched. head_dim was
		// inferred from LAYER 0's q_proj, so every later layer's rows still have to be
		// pinned against it — that is exactly the gap permuteInterleaveToSplitChecked
		// closes.
		qPerm, err := permuteInterleaveToSplitChecked(p+"self_attn.q_proj.weight", wq, cfg.Heads, cfg.HeadDim)
		if err != nil {
			return nil, err
		}
		kPerm, err := permuteInterleaveToSplitChecked(p+"self_attn.k_proj.weight", wk, kv, cfg.HeadDim)
		if err != nil {
			return nil, err
		}
		m.Blocks = append(m.Blocks, &CohereBlock{
			InputNorm: layerNormFromHF(inNorm, cfg.Eps),
			Wq:        transpose2D(qPerm),
			Wk:        transpose2D(kPerm),
			Wv:        transpose2D(wv),
			Wo:        transpose2D(wo),
			FFN:       swiGLUFromGGUF(gate, up, down),
		})
	}
	norm, ok := ts["model.norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Cohere missing model.norm.weight")
	}
	m.Norm = layerNormFromHF(norm, cfg.Eps)
	head := tok // tied head (tie_word_embeddings=true; lm_head.weight absent)
	if o, ok := ts["lm_head.weight"]; ok {
		head = o
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}

// permuteInterleaveToSplit reorders the ROWS of a torch [out, in] projection weight so
// that GoAI's split-half OpRoPE reproduces Cohere's interleaved (GPT-J) rotary. out is
// heads·headDim; within each head the interleaved even/odd channels (0,1,2,3,…) are
// regrouped into split-half order (evens first, then odds): new row h·hd+i takes original
// row h·hd+2i, and new row h·hd+(i+hd/2) takes original row h·hd+2i+1, for i in [0,hd/2).
// A fresh [out, in] tensor is returned; the input is left unchanged. headDim must be even.
func permuteInterleaveToSplit(w *tensor.Tensor, heads, headDim int) *tensor.Tensor {
	out, in := w.Shape()[0], w.Shape()[1]
	res := tensor.New(tensor.F64, tensor.Shape{out, in})
	half := headDim / 2
	for h := range heads {
		base := h * headDim
		for i := range half {
			src0 := base + 2*i      // interleaved even channel
			src1 := base + 2*i + 1  // interleaved odd channel
			dst0 := base + i        // split-half: first half
			dst1 := base + i + half // split-half: second half
			//perfscan:ignore PS1001 RoPE row-permute at load; one-time
			for c := range in {
				res.SetF64(w.AtF64(src0, c), dst0, c)
				res.SetF64(w.AtF64(src1, c), dst1, c)
			}
		}
	}
	return res
}

// permuteInterleaveToSplitChecked is [permuteInterleaveToSplit] with the row geometry
// PINNED first, so a malformed file is a clean error instead of a silently mis-permuted
// model. It is the exact command-r analogue of deepseekv2_gguf.go's
// deinterleaveRoPEChecked (T861/§B76) — the same class of bug, closed the same way.
//
// permuteInterleaveToSplit walks heads×headDim rows and never inspects the tensor it
// was handed, so all three disagreements between the declared and the actual geometry
// were reachable from an untrusted file:
//
//   - a rank-1 (or rank-3) tensor died reading Shape()[1];
//   - FEWER rows than heads·headDim indexed out of range;
//   - MORE rows permuted only the prefix and left the tail at its ZERO initial value —
//     the result is a perfectly well-formed tensor, so the model loaded with err == nil
//     and computed wrong attention (§V29's wrong-but-nil-error class).
//
// The quantized twin never had any of the three: [QuantCohereFromGGUF]'s mkQPermuted
// pins len(Shape) == 2 and routes the row count through [ropeUnpermutePerm]. This
// reuses that same predicate via [ropeRowGeom] rather than restating it, and adds the
// one thing the row-map check cannot express — that the rows are exactly heads·headDim
// for the headDim the CONFIG declares, which is what the fused-qkv branch of
// [CohereFromGGUF] already pins for its own slices. Valid files are unaffected:
// command-r's q_proj is square (create_tensor_qkv(n_embd, n_embd, gqa, gqa)), so
// heads·headDim is the row count every conforming file carries.
func permuteInterleaveToSplitChecked(name string, w *tensor.Tensor, heads, headDim int) (*tensor.Tensor, error) {
	if err := ropeRowGeom(name, w, heads); err != nil {
		return nil, err
	}
	if got := w.Shape()[0]; got != heads*headDim {
		return nil, fmt.Errorf("nlp: %s has %d rows, want heads·head_dim = %d·%d = %d", name, got, heads, headDim, heads*headDim)
	}
	return permuteInterleaveToSplit(w, heads, headDim), nil
}

// layerNormFromHF wraps a Cohere weight-only LayerNorm gain tensor as an [nn.LayerNorm]
// with γ = the HF weight and β = a ZERO vector, so the mean-centered LayerNorm reduces to
// CohereLayerNorm (subtract mean, /√(var+eps), ·weight, no bias).
func layerNormFromHF(gamma *tensor.Tensor, eps float64) *nn.LayerNorm {
	g := cloneF64(gamma)
	beta := tensor.New(tensor.F64, tensor.Shape{g.Shape()[0]}) // zero-initialized
	return &nn.LayerNorm{Gamma: g, Beta: beta, Eps: eps}
}
