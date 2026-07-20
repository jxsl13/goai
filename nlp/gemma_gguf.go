package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// GGUF attention key/value width keys (decoupled head_dim, gguf-py
// constants.py Keys.Attention.KEY_LENGTH/VALUE_LENGTH). Arch-prefixed like the
// rest: "gemma.attention.key_length".
const (
	ggufKeyLen = "attention.key_length"
	ggufValLen = "attention.value_length"
)

// GemmaFromGGUF builds a [Gemma] (v1) from the metadata and (dequantized) tensor maps
// of a parsed GGUF file (gguf.File.Metadata / .Tensors) whose general.architecture is
// "gemma" — the llama.cpp convention for Google's Gemma v1 family. The config comes
// from the gemma.* metadata keys (embedding_length, block_count, feed_forward_length,
// attention.head_count, attention.head_count_kv, attention.layer_norm_rms_epsilon,
// context_length, and — Gemma's decoupled per-head width — attention.key_length) and
// the weights from the token_embd / blk.N.* / output_norm tensors. Every projection is
// transposed from GGUF's torch [out, in] layout into GoAI's [in, out].
//
// The four Gemma conventions this loader implements were verified against llama.cpp
// master (conversion/gemma.py GemmaModel + src/models/gemma.cpp + the rope-type switch
// in src/llama-model.cpp):
//
//   - NO q/k row permutation: GemmaModel.modify_tensors is a pure rename (no permute,
//     unlike the llama arch), and LLM_ARCH_GEMMA sits in the LLAMA_ROPE_TYPE_NEOX case
//     — split-half RoPE on the HF weight layout, which is exactly GoAI's OpRoPE. Like
//     [Qwen2FromGGUF], this loader only transposes.
//   - Norm gains are stored PRE-FOLDED: the converter ADDS ONE to every *norm.weight
//     (GemmaModel.modify_tensors: data_torch = data_torch plus 1, mirroring HF's
//     (1+w) RMSNorm), and build_gemma uses the stored gain as-is (no
//     runtime `1.0f +`). So unlike [GemmaFromHF] (which folds γ ← γ+1 via gemmaRMS),
//     this loader copies the gains UNCHANGED — GoAI's in-memory convention (γ carries
//     the +1) coincides with GGUF's on-disk one. Folding again would be off by one.
//   - Embeddings are stored UNSCALED: build_gemma applies the √dim "normalizer" at
//     runtime (`inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd))`), exactly as
//     Gemma.Forward does, so token_embd is copied as-is.
//   - Tied LM head: MODEL_ARCH.GEMMA has no OUTPUT tensor (the converter skips
//     lm_head.weight) — the head is always token_embd, and an unexpected output.weight
//     is rejected rather than silently dropped.
//
// HeadDim comes from gemma.attention.key_length when present (the converter always
// writes it) and falls back to the blk.0.attn_q.weight shape divided by head_count.
// gemma.rope.freq_base is optional (the converter does not emit it; llama.cpp and this
// loader both default to 10000).
func GemmaFromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*Gemma, error) {
	cfg, err := gemmaCfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	layers, heads := cfg.Layers, cfg.Heads

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	cfg.Vocab = tok.Shape()[0]
	if _, ok := tensors["output.weight"]; ok {
		return nil, fmt.Errorf("nlp: Gemma GGUF has unexpected output.weight (the gemma architecture ties the LM head to token_embd)")
	}

	// Embedding stored unscaled — the √dim normalizer is Gemma.Forward's job.
	m := &Gemma{Config: cfg, TokEmb: cloneF64(tok)}
	for l := range layers {
		p := fmt.Sprintf("blk.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return t, nil
		}
		names := []string{"attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
			"attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}
		w := make([]*tensor.Tensor, len(names))
		for i, n := range names {
			if w[i], err = g(n); err != nil {
				return nil, err
			}
		}
		// Rank-guard the projections this block transposes (all but the 1-D norm gains) —
		// the float twin of QuantGemmaFromGGUF's mkQ `len(qt.Shape) != 2`, without which a
		// rank-1 tensor falls through to transpose2D and PANICS (§B77).
		for i, n := range names {
			if isNormWeight(n) {
				continue
			}
			if err := require2D(p+n, w[i]); err != nil {
				return nil, err
			}
		}
		if m.Config.HeadDim == 0 { // no attention.key_length key: derive from the q width
			if m.Config.HeadDim, err = gemmaHeadDimFromQ(p+"attn_q.weight", w[1], heads); err != nil {
				return nil, err
			}
		}
		m.Blocks = append(m.Blocks, &GemmaBlock{
			// Norm gains are pre-folded on disk (converter's +1): copy, don't re-fold.
			AttnNorm: rmsFromGGUF(w[0], cfg.Eps),
			Wq:       transpose2D(w[1]), // GGUF [out,in] → GoAI [in,out]
			Wk:       transpose2D(w[2]),
			Wv:       transpose2D(w[3]),
			Wo:       transpose2D(w[4]),
			FFNNorm:  rmsFromGGUF(w[5], cfg.Eps),
			// GGUF stores the same torch [out,in] layout as HF, so the GeGLU
			// builder is shared with the HF path (transpose only).
			FFN: geGLUFromHF(w[6], w[7], w[8]),
		})
	}
	on, ok := tensors["output_norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output_norm.weight")
	}
	m.FinalNorm = rmsFromGGUF(on, cfg.Eps)
	return m, nil
}

// gemmaCfgFromGGUFMeta parses the gemma.* metadata keys of a GGUF file whose
// general.architecture is "gemma" into a GemmaConfig — the metadata half shared by
// [GemmaFromGGUF] (dequantized tensors) and [QuantGemmaFromGGUF] (still-quantized
// tensors). Vocab and the shape-derived HeadDim fallback are left at zero: they come
// from the tensor maps, which only the callers hold.
//
// Every structural count is validated POSITIVE through [metaPositiveInt] — the same
// predicate llamaCfgFromGGUFMeta uses (T861/§B76). Gemma needs its own call because it
// parses its own metadata: the llama-family choke point never sees a gemma file, so the
// class fix there did NOT reach this architecture. Validating here covers BOTH twins,
// which is the point of doing it in the shared parser rather than in one loader.
//
// A head_count of 0 was the sharp edge: it divided straight through to
// `attn_q rows / heads` in [GemmaFromGGUF]'s key_length fallback, an integer
// divide-by-zero PANIC (the quantized twin guarded `cfg.Heads > 0` and merely left
// HeadDim at 0 — an err == nil model, §V29's harder-to-trace class). An ABSENT key
// still keeps its documented fallback; only a PRESENT non-positive value errors, so
// round-trips are untouched.
func gemmaCfgFromGGUFMeta(meta map[string]any) (GemmaConfig, error) {
	const arch = "gemma"
	if a, _ := meta[ggufArch].(string); a != arch {
		return GemmaConfig{}, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	positive := func(suffix string) (int, error) { return metaPositiveInt(meta, key(suffix)) }
	dim, err := positive(ggufEmbLen)
	if err != nil {
		return GemmaConfig{}, err
	}
	layers, err := positive(ggufBlockCnt)
	if err != nil {
		return GemmaConfig{}, err
	}
	ffn, err := positive(ggufFFLen)
	if err != nil {
		return GemmaConfig{}, err
	}
	heads, err := positive(ggufHeadCnt)
	if err != nil {
		return GemmaConfig{}, err
	}
	cfg := GemmaConfig{
		Dim: dim, Layers: layers, FFN: ffn, Heads: heads, KVHeads: heads,
		Eps:      metaFloat(meta, key(ggufRMSEps), 1e-6),
		RopeBase: metaFloat(meta, key(ggufRopeFreq), 10000),
	}
	// Absent ⇒ the documented fallback (kv = heads, ctx = dim, head_dim from the q
	// width); present-and-non-positive ⇒ a malformed geometry, not a fallback request.
	if cfg.KVHeads, err = metaOptionalPositiveInt(meta, key(ggufHeadKV), heads); err != nil {
		return GemmaConfig{}, err
	}
	if cfg.Ctx, err = metaOptionalPositiveInt(meta, key(ggufCtxLen), dim); err != nil {
		return GemmaConfig{}, err
	}
	if cfg.HeadDim, err = metaOptionalPositiveInt(meta, key(ggufKeyLen), 0); err != nil {
		return GemmaConfig{}, err
	}
	return cfg, nil
}

// gemmaHeadDimFromRows derives Gemma's decoupled per-head width from an attn_q row
// count — the fallback taken when a file carries no attention.key_length, shared by
// [GemmaFromGGUF], [Gemma2FromGGUF] and both their quantized twins so all four accept
// and reject the same files.
//
// The bare `rows / heads` this replaces had two §V29 holes, both reachable from an
// untrusted file: heads = 0 PANICKED with an integer divide-by-zero on the float path,
// and a row count that is not a multiple of heads truncated to a head width the tensor
// does not actually have — loading with err == nil and mis-shaping every head. The
// rank test is here too because the row count alone is meaningless on a non-2-D tensor,
// which the float path would then hand to transpose2D (Shape()[1], another panic) while
// the quantized twin's `len(aq.Shape) == 2` already skipped it.
func gemmaHeadDimFromRows(name string, rows, heads int) (int, error) {
	if heads <= 0 || rows <= 0 || rows%heads != 0 {
		return 0, fmt.Errorf("nlp: GGUF %s has %d rows, want a positive multiple of head_count %d (the attention.key_length fallback)", name, rows, heads)
	}
	return rows / heads, nil
}

// gemmaHeadDimFromQ is [gemmaHeadDimFromRows] for the float loaders, which hold a
// *tensor.Tensor rather than the quantized twins' gguf.QuantTensor and so must state
// the rank test in their own idiom (Ndim, not len(Shape)). The arithmetic — the part
// that can drift — stays in the one shared function.
//
// The rank test is not optional on this path: [GemmaFromGGUF] and [Gemma2FromGGUF]
// hand the very same tensor to transpose2D immediately afterwards, which reads
// Shape()[1] and panics on a rank-1 tensor.
func gemmaHeadDimFromQ(name string, w *tensor.Tensor, heads int) (int, error) {
	if w.Ndim() != 2 {
		return 0, fmt.Errorf("nlp: GGUF %s is %d-D, want a 2-D [out, in] projection", name, w.Ndim())
	}
	return gemmaHeadDimFromRows(name, w.Shape()[0], heads)
}

// GemmaToGGUF is the inverse of [GemmaFromGGUF]: it serializes a Gemma (v1) into GGUF
// metadata + tensor maps under general.architecture "gemma", transposing every
// projection back into torch [out, in]. The norm gains are written as stored — GoAI's
// γ already carries Gemma's +1 (folded at HF load), which IS the GGUF on-disk
// convention (llama.cpp's converter pre-folds the +1) — and the embedding is written
// unscaled. No output.weight is emitted (the gemma architecture's LM head is always
// tied to token_embd, matching llama.cpp's tensor list). The decoupled head_dim goes
// out as gemma.attention.key_length/value_length. gemma.rope.freq_base is written for
// round-trip fidelity even though llama.cpp's converter omits it (llama.cpp reads it
// as an optional key defaulting to 10000, so the file stays conventional). Pass the
// result to gguf.Write via a gguf.File.
func GemmaToGGUF(m *Gemma) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "gemma"
	c := m.Config
	key := func(suffix string) string { return arch + "." + suffix }
	headDim := c.HeadDim
	if headDim <= 0 && c.Heads > 0 {
		headDim = m.Blocks[0].Wq.Shape()[1] / c.Heads
	}
	meta := map[string]any{
		ggufArch:          arch,
		key(ggufEmbLen):   uint32(c.Dim),
		key(ggufBlockCnt): uint32(c.Layers),
		key(ggufFFLen):    uint32(c.FFN),
		key(ggufHeadCnt):  uint32(c.Heads),
		key(ggufHeadKV):   uint32(c.kvHeads()),
		key(ggufKeyLen):   uint32(headDim),
		key(ggufValLen):   uint32(headDim),
		key(ggufCtxLen):   uint32(c.Ctx),
		key(ggufRMSEps):   float32(c.Eps),
		key(ggufRopeFreq): float32(ropeBaseOr(c.RopeBase)),
	}
	ts := map[string]*tensor.Tensor{
		"token_embd.weight":  cloneF64(m.TokEmb),
		"output_norm.weight": cloneF64(m.FinalNorm.Gamma), // pre-folded (+1) on disk, as stored
	}
	for l, b := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", l)
		ts[p+"attn_norm.weight"] = cloneF64(b.AttnNorm.Gamma)
		ts[p+"attn_q.weight"] = transpose2D(b.Wq) // GoAI [in,out] → GGUF [out,in]
		ts[p+"attn_k.weight"] = transpose2D(b.Wk)
		ts[p+"attn_v.weight"] = transpose2D(b.Wv)
		ts[p+"attn_output.weight"] = transpose2D(b.Wo)
		ts[p+"ffn_norm.weight"] = cloneF64(b.FFNNorm.Gamma)
		ts[p+"ffn_gate.weight"] = transpose2D(b.FFN.Wgate)
		ts[p+"ffn_up.weight"] = transpose2D(b.FFN.Wup)
		ts[p+"ffn_down.weight"] = transpose2D(b.FFN.Wdown)
	}
	return meta, ts
}
