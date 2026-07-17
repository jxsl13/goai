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
	const arch = "gemma"
	if a, _ := meta[ggufArch].(string); a != arch {
		return nil, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	dim, err := metaInt(meta, key(ggufEmbLen))
	if err != nil {
		return nil, err
	}
	layers, err := metaInt(meta, key(ggufBlockCnt))
	if err != nil {
		return nil, err
	}
	ffn, err := metaInt(meta, key(ggufFFLen))
	if err != nil {
		return nil, err
	}
	heads, err := metaInt(meta, key(ggufHeadCnt))
	if err != nil {
		return nil, err
	}
	cfg := GemmaConfig{
		Dim: dim, Layers: layers, FFN: ffn, Heads: heads, KVHeads: heads,
		Eps:      metaFloat(meta, key(ggufRMSEps), 1e-6),
		RopeBase: metaFloat(meta, key(ggufRopeFreq), 10000),
		Ctx:      dim, // provisional; overwritten from context_length below
	}
	if kv, e := metaInt(meta, key(ggufHeadKV)); e == nil {
		cfg.KVHeads = kv
	}
	if c, e := metaInt(meta, key(ggufCtxLen)); e == nil {
		cfg.Ctx = c
	}
	if hd, e := metaInt(meta, key(ggufKeyLen)); e == nil {
		cfg.HeadDim = hd
	}

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
		if m.Config.HeadDim == 0 { // no attention.key_length key: derive from the q width
			m.Config.HeadDim = w[1].Shape()[0] / heads
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
