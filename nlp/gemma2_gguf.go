package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// GGUF logit soft-capping metadata keys (gguf-py constants.py
// Keys.LLM.ATTN_LOGIT_SOFTCAPPING / FINAL_LOGIT_SOFTCAPPING, written as float32 by
// gguf_writer.add_attn_logit_softcapping / add_final_logit_softcapping). Arch-prefixed
// like the rest: "gemma2.attn_logit_softcapping". Note these live directly under the
// arch prefix (Keys.LLM), NOT under "attention." like the sliding window.
const (
	ggufAttnSoftcap  = "attn_logit_softcapping"
	ggufFinalSoftcap = "final_logit_softcapping"
)

// llama.cpp's gemma2 fallbacks for metadata a file may omit: the soft-cap magnitudes
// default to 50/30 (llama-hparams.h: f_attn_logit_softcapping = 50.0f,
// f_final_logit_softcapping = 30.0f — the Gemma 2 paper values; src/models/gemma2.cpp
// reads both keys with required=false) and the sliding window to 4096
// (src/models/gemma2.cpp: `hparams.n_swa = 4096; // default value of gemma 2`).
const (
	gemma2DefaultAttnCap  = 50.0
	gemma2DefaultFinalCap = 30.0
	//perfscan:ignore PS6023 test-coverage request for gemma2DefaultSWA; not a code/perf change
	gemma2DefaultSWA = 4096
)

// Gemma2FromGGUF builds a [Gemma2] from the metadata and (dequantized) tensor maps of
// a parsed GGUF file (gguf.File.Metadata / .Tensors) whose general.architecture is
// "gemma2" — the llama.cpp convention for Google's Gemma 2 family
// (Gemma2ForCausalLM). The config comes from the gemma2.* metadata keys and the
// weights from the token_embd / blk.N.* / output_norm tensors; every projection is
// transposed from GGUF's torch [out, in] layout into GoAI's [in, out].
//
// The Gemma 2 conventions this loader implements were verified against llama.cpp
// master (conversion/gemma.py Gemma2Model + src/models/gemma2.cpp + src/llama-arch.cpp
// + the rope-type switch in src/llama-model.cpp + gguf-py constants.py /
// tensor_mapping.py + the defaults in src/llama-hparams.h):
//
//   - Sandwich-norm names on disk: HF input_layernorm → blk.N.attn_norm,
//     post_attention_layernorm → blk.N.post_attention_norm
//     (MODEL_TENSOR.ATTN_POST_NORM), pre_feedforward_layernorm → blk.N.FFN_NORM's
//     name "blk.N.ffn_norm" (MODEL_TENSOR.FFN_PRE_NORM formats to "blk.{bid}.ffn_norm"
//     in TENSOR_NAMES — the pre-FFN norm reuses the plain llama ffn_norm name), and
//     post_feedforward_layernorm → blk.N.post_ffw_norm (FFN_POST_NORM; note the "ffw"
//     contraction, NOT post_feedforward_norm).
//   - Norm gains are stored PRE-FOLDED: Gemma2Model.modify_tensors runs
//     `data_torch = data_torch + 1` on every *norm.weight (same +1 fold as gemma v1),
//     and src/models/gemma2.cpp's build_norm uses the stored gain as-is. GoAI's
//     in-memory convention (γ carries Gemma's +1, folded by gemmaRMS at HF load)
//     coincides with the on-disk one, so all five gain families (the four sandwich
//     norms and output_norm) are copied UNCHANGED. Folding again would be off by one.
//   - Soft-caps: gemma2.attn_logit_softcapping and gemma2.final_logit_softcapping
//     (Keys.LLM, float32). The converter always writes both (required hparams in
//     Gemma2Model.set_gguf_parameters); when a key is absent this loader mirrors
//     llama.cpp's defaults (50.0 / 30.0, llama-hparams.h) rather than disabling the
//     cap. An explicit 0.0 disables that cap in GoAI (llama.cpp always applies the
//     cap for gemma2 and would divide by zero — no converter-produced file carries 0).
//   - Sliding-window handling: gemma2.attention.sliding_window is always written
//     (4096 for every released Gemma 2), and llama.cpp runs an alternating iSWA graph
//     (set_swa_pattern(2): every other layer attends within the window, the rest see
//     the full context). GoAI's [Gemma2] intentionally implements full attention only,
//     which is bit-identical to the alternating graph whenever seq ≤ window — so this
//     loader sets Ctx = min(context_length, sliding_window). Every prompt the loaded
//     model accepts then reproduces llama.cpp's numerics exactly, and longer prompts
//     are rejected by Forward instead of being silently mis-attended.
//   - Query scale: GGUF has NO query_pre_attn_scalar key. llama.cpp derives the Q
//     scale from the model type, keyed on block_count (src/models/gemma2.cpp
//     load_arch_hparams): 46 layers (27B) → 1/√(embedding_length/head_count), every
//     other size → 1/√n_embd_head_k (= key_length). This loader sets
//     QueryPreAttnScalar by the same rule (Layers==46 → Dim/Heads, else HeadDim).
//   - NO q/k row permutation: Gemma2Model.modify_tensors is the +1 norm fold plus a
//     pure rename (no LlamaModel.permute), and LLM_ARCH_GEMMA2 sits in the
//     LLAMA_ROPE_TYPE_NEOX case — split-half RoPE on the HF weight layout, exactly
//     GoAI's OpRoPE. Like [GemmaFromGGUF], this loader only transposes.
//   - Embeddings are stored UNSCALED: src/models/gemma2.cpp applies the √dim
//     normalizer at runtime (`inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd))`), exactly
//     as [Gemma2.Forward] does, so token_embd is copied as-is.
//   - Tied LM head: MODEL_ARCH.GEMMA2 has no OUTPUT tensor, the converter skips
//     lm_head.weight, and gemma2.cpp serves the head from token_embd
//     (TENSOR_DUPLICATED). An unexpected output.weight is rejected rather than
//     silently dropped.
//   - Decoupled head width: gemma2.attention.key_length / value_length both carry
//     head_dim (the converter always writes them); a file where they disagree cannot
//     be represented and is rejected. When absent, HeadDim falls back to the
//     blk.0.attn_q.weight row count divided by head_count.
//   - gemma2.rope.freq_base is NOT written by the converter (Gemma 2's θ is 10000,
//     llama.cpp's default); this loader reads it optionally with the same default.
//   - Attention biases and a fused attn_qkv are rejected: Gemma2ForCausalLM is
//     bias-free and the converter never emits either form (llama.cpp would tolerate
//     them via create_tensor_qkv, but loading them here could only mean a
//     hand-crafted file — reject, don't misload).
func Gemma2FromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*Gemma2, error) {
	cfg, err := gemma2CfgFromGGUFMeta(meta)
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
		return nil, fmt.Errorf("nlp: Gemma2 GGUF has unexpected output.weight (the gemma2 architecture ties the LM head to token_embd)")
	}

	// Embedding stored unscaled — the √dim normalizer is Gemma2.Forward's job.
	m := &Gemma2{Config: cfg, TokEmb: cloneF64(tok)}
	//perfscan:ignore PS3060 model-load transpose (gemma2 FromGGUF); one-time cold
	for l := range layers {
		p := fmt.Sprintf("blk.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return t, nil
		}
		// Gemma2ForCausalLM is bias-free and the converter writes split q/k/v — a
		// file carrying biases or a fused qkv is hand-crafted; reject, don't misload.
		for _, unsupported := range []string{"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias", "attn_qkv.weight"} {
			if _, ok := tensors[p+unsupported]; ok {
				return nil, fmt.Errorf("nlp: Gemma2 GGUF carries %s%s; the gemma2 architecture is bias-free with split q/k/v", p, unsupported)
			}
		}
		names := []string{"attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
			"attn_output.weight", "post_attention_norm.weight", "ffn_norm.weight",
			"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "post_ffw_norm.weight"}
		//perfscan:ignore PS3035 resource-only per-loop alloc in load path; time win nil
		w := make([]*tensor.Tensor, len(names))
		//perfscan:ignore PS3066 load-path sibling loops; cold one-time
		for i, n := range names {
			if w[i], err = g(n); err != nil {
				return nil, err
			}
		}
		// Rank-guard the projections this block transposes (all but the four 1-D
		// sandwich-norm gains) — the float twin of QuantGemma2FromGGUF's mkQ
		// `len(qt.Shape) != 2`; a rank-1 tensor would otherwise PANIC in transpose2D (§B77).
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
		m.Blocks = append(m.Blocks, &Gemma2Block{
			// All four sandwich-norm gains are pre-folded on disk (converter's +1):
			// copy, don't re-fold.
			InputNorm:    rmsFromGGUF(w[0], cfg.Eps),
			Wq:           transpose2D(w[1]), // GGUF [out,in] → GoAI [in,out]
			Wk:           transpose2D(w[2]),
			Wv:           transpose2D(w[3]),
			Wo:           transpose2D(w[4]),
			PostAttnNorm: rmsFromGGUF(w[5], cfg.Eps),
			PreFFNNorm:   rmsFromGGUF(w[6], cfg.Eps),
			// GGUF stores the same torch [out,in] layout as HF, so the GeGLU builder
			// is shared with the HF path (transpose only).
			FFN:         geGLUFromHF(w[7], w[8], w[9]),
			PostFFNNorm: rmsFromGGUF(w[10], cfg.Eps),
		})
	}
	on, ok := tensors["output_norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output_norm.weight")
	}
	m.FinalNorm = rmsFromGGUF(on, cfg.Eps)

	// GGUF has no query_pre_attn_scalar key; mirror llama.cpp's block_count-keyed
	// derivation (src/models/gemma2.cpp): 46 layers = the 27B → dim/heads, else the
	// per-head width (key_length).
	if m.Config.Layers == 46 {
		m.Config.QueryPreAttnScalar = float64(m.Config.Dim) / float64(m.Config.Heads)
	} else {
		m.Config.QueryPreAttnScalar = float64(m.Config.HeadDim)
	}
	return m, nil
}

// gemma2CfgFromGGUFMeta parses the gemma2.* metadata keys of a GGUF file whose
// general.architecture is "gemma2" into a Gemma2Config. Vocab, the shape-derived
// HeadDim fallback and the block_count-derived QueryPreAttnScalar are left to
// [Gemma2FromGGUF], which holds the tensors. Absent soft-cap keys take llama.cpp's
// gemma2 defaults (50/30), and Ctx is clamped to the sliding window (default 4096)
// because GoAI's full-attention [Gemma2] matches llama.cpp's alternating-window graph
// only for prompts within the window.
func gemma2CfgFromGGUFMeta(meta map[string]any) (Gemma2Config, error) {
	const arch = "gemma2"
	if a, _ := meta[ggufArch].(string); a != arch {
		return Gemma2Config{}, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	positive := func(suffix string) (int, error) { return metaPositiveInt(meta, key(suffix)) }
	dim, err := positive(ggufEmbLen)
	if err != nil {
		return Gemma2Config{}, err
	}
	layers, err := positive(ggufBlockCnt)
	if err != nil {
		return Gemma2Config{}, err
	}
	ffn, err := positive(ggufFFLen)
	if err != nil {
		return Gemma2Config{}, err
	}
	heads, err := positive(ggufHeadCnt)
	if err != nil {
		return Gemma2Config{}, err
	}
	cfg := Gemma2Config{
		Dim: dim, Layers: layers, FFN: ffn, Heads: heads, KVHeads: heads,
		Eps:           metaFloat(meta, key(ggufRMSEps), 1e-6),
		RopeBase:      metaFloat(meta, key(ggufRopeFreq), 10000), // converter writes no freq_base; llama.cpp default
		AttnLogitCap:  metaFloat(meta, key(ggufAttnSoftcap), gemma2DefaultAttnCap),
		FinalLogitCap: metaFloat(meta, key(ggufFinalSoftcap), gemma2DefaultFinalCap),
	}
	// Absent ⇒ the documented fallback; present-and-non-positive ⇒ malformed.
	if cfg.KVHeads, err = metaOptionalPositiveInt(meta, key(ggufHeadKV), heads); err != nil {
		return Gemma2Config{}, err
	}
	if cfg.Ctx, err = metaOptionalPositiveInt(meta, key(ggufCtxLen), dim); err != nil {
		return Gemma2Config{}, err
	}
	if cfg.HeadDim, err = metaOptionalPositiveInt(meta, key(ggufKeyLen), 0); err != nil {
		return Gemma2Config{}, err
	}
	if hd := cfg.HeadDim; hd > 0 {
		if vl, e := metaInt(meta, key(ggufValLen)); e == nil && vl != hd {
			return Gemma2Config{}, fmt.Errorf("nlp: Gemma2 GGUF attention.value_length %d != key_length %d (GoAI's Gemma2 has a single per-head width)", vl, hd)
		}
	}
	// Sliding-window clamp: full attention == the alternating-window graph only for
	// seq ≤ window, so cap the usable context at the window (llama.cpp default 4096).
	swa := gemma2DefaultSWA
	if w, e := metaInt(meta, key(ggufSlidingWin)); e == nil {
		swa = w
	}
	if swa > 0 && swa < cfg.Ctx {
		cfg.Ctx = swa
	}
	return cfg, nil
}

// Gemma2ToGGUF is the inverse of [Gemma2FromGGUF]: it serializes a Gemma2 (e.g. from
// [Gemma2FromHF]) into GGUF metadata + tensor maps under general.architecture
// "gemma2", exactly the way llama.cpp's converter lays the file out — the four
// sandwich norms under attn_norm / post_attention_norm / ffn_norm / post_ffw_norm
// with the gains written as stored (GoAI's γ already carries Gemma's +1, which IS the
// pre-folded on-disk convention), the embedding unscaled, no output.weight (the arch
// is tied-head only), the decoupled head width as key_length/value_length, and both
// soft-caps always written (as the converter does; a GoAI model with a cap disabled
// writes 0.0, which round-trips through this package but is outside llama.cpp's
// convention — no converter-produced file carries 0). Two llama.cpp-side caveats:
//
//   - QueryPreAttnScalar is NOT representable in GGUF — llama.cpp re-derives it from
//     block_count (46 → dim/heads, else key_length). A model whose scalar differs
//     from that derivation will not round-trip it; [Gemma2FromGGUF] re-derives by the
//     same rule.
//   - The sliding window is written as Ctx (GoAI's full-attention Gemma2 is exact
//     precisely for prompts ≤ window, so window = Ctx makes the file's alternating
//     iSWA graph coincide with what this model computes for every prompt it accepts).
//
// gemma2.rope.freq_base is written for round-trip fidelity even though llama.cpp's
// converter omits it (it is read as an optional key defaulting to 10000, so the file
// stays conventional). Pass the result to gguf.Write via a gguf.File.
func Gemma2ToGGUF(m *Gemma2) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "gemma2"
	c := m.Config
	key := func(suffix string) string { return arch + "." + suffix }
	headDim := c.HeadDim
	if headDim <= 0 && c.Heads > 0 {
		headDim = m.Blocks[0].Wq.Shape()[1] / c.Heads
	}
	attnCap, finalCap := c.AttnLogitCap, c.FinalLogitCap
	if attnCap < 0 {
		attnCap = 0
	}
	if finalCap < 0 {
		finalCap = 0
	}
	meta := map[string]any{
		ggufArch:              arch,
		key(ggufEmbLen):       uint32(c.Dim),
		key(ggufBlockCnt):     uint32(c.Layers),
		key(ggufFFLen):        uint32(c.FFN),
		key(ggufHeadCnt):      uint32(c.Heads),
		key(ggufHeadKV):       uint32(c.kvHeads()),
		key(ggufKeyLen):       uint32(headDim),
		key(ggufValLen):       uint32(headDim),
		key(ggufCtxLen):       uint32(c.Ctx),
		key(ggufSlidingWin):   uint32(c.Ctx), // window = Ctx: full attention ≡ iSWA for every accepted prompt
		key(ggufRMSEps):       float32(c.Eps),
		key(ggufRopeFreq):     float32(ropeBaseOr(c.RopeBase)),
		key(ggufAttnSoftcap):  float32(attnCap),
		key(ggufFinalSoftcap): float32(finalCap),
	}
	ts := map[string]*tensor.Tensor{
		"token_embd.weight":  cloneF64(m.TokEmb),
		"output_norm.weight": cloneF64(m.FinalNorm.Gamma), // pre-folded (+1) on disk, as stored
	}
	//perfscan:ignore PS3060 ToGGUF export transpose; one-time cold
	for l, b := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", l)
		ts[p+"attn_norm.weight"] = cloneF64(b.InputNorm.Gamma)
		ts[p+"attn_q.weight"] = transpose2D(b.Wq) // GoAI [in,out] → GGUF [out,in]
		ts[p+"attn_k.weight"] = transpose2D(b.Wk)
		ts[p+"attn_v.weight"] = transpose2D(b.Wv)
		ts[p+"attn_output.weight"] = transpose2D(b.Wo)
		ts[p+"post_attention_norm.weight"] = cloneF64(b.PostAttnNorm.Gamma)
		ts[p+"ffn_norm.weight"] = cloneF64(b.PreFFNNorm.Gamma) // the pre-FFN sandwich norm rides the plain ffn_norm name
		ts[p+"ffn_gate.weight"] = transpose2D(b.FFN.Wgate)
		ts[p+"ffn_up.weight"] = transpose2D(b.FFN.Wup)
		ts[p+"ffn_down.weight"] = transpose2D(b.FFN.Wdown)
		ts[p+"post_ffw_norm.weight"] = cloneF64(b.PostFFNNorm.Gamma)
	}
	return meta, ts
}
