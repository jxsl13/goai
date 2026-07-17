package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// GGUF Granite scalar-multiplier and rope-width metadata keys (gguf-py
// constants.py: Keys.LLM.EMBEDDING_SCALE / RESIDUAL_SCALE, Keys.Attention.SCALE,
// Keys.Rope.DIMENSION_COUNT — all float32 except the uint32 dimension count).
// Arch-prefixed like the rest: "granite.embedding_scale",
// "granite.attention.scale". ggufLogitScale (Keys.LLM.LOGIT_SCALE
// "{arch}.logit_scale") is shared with the command-r loader.
const (
	ggufEmbScale  = "embedding_scale"
	ggufResScale  = "residual_scale"
	ggufAttnScale = "attention.scale"
	ggufRopeDims  = "rope.dimension_count"
)

// GraniteFromGGUF builds a [Llama] carrying IBM Granite's four scalar multipliers
// from the metadata and (dequantized) tensor maps of a parsed GGUF file
// (gguf.File.Metadata / .Tensors) whose general.architecture is "granite" — the
// llama.cpp convention for IBM's GraniteForCausalLM. Granite is structurally a plain
// Llama (same blk.N.* tensor set, RMSNorm, SwiGLU, GQA — see [GraniteFromHF]), so the
// weights ride the shared llama-family loader; the Granite specifics are the four
// scalars and the q/k row permutation below.
//
// The Granite conventions this loader implements were verified against llama.cpp
// master (conversion/granite.py GraniteModel + conversion/llama.py LlamaModel +
// src/models/granite.cpp + src/llama-arch.cpp, the rope-type switch in
// src/llama-model.cpp, gguf-py constants.py and the defaults in src/llama-hparams.h):
//
//   - Q/K rows ARE permuted on disk: GraniteModel extends LlamaModel
//     (conversion/granite.py: `class GraniteModel(LlamaModel)`) and does not override
//     modify_tensors, so it inherits undo_permute = True and LlamaModel.permute on
//     q_proj/k_proj — HF's split-half rows are reordered per head into the
//     interleaved pairing that LLM_ARCH_GRANITE consumes with LLAMA_ROPE_TYPE_NORM
//     (the same rope case as plain llama). GoAI's OpRoPE is the split-half
//     (HF rotate_half) convention, so this loader UN-permutes attn_q/attn_k rows back
//     into the HF layout before transposing, exactly like [MixtralFromGGUF] (the
//     other LlamaModel-lineage loader). Skipping the un-permute would show a
//     rotary-sized divergence against the [GraniteFromHF] parity anchor.
//   - The four scalars: granite.embedding_scale, granite.residual_scale and
//     granite.logit_scale (Keys.LLM) plus granite.attention.scale (Keys.Attention) —
//     the converter renames Granite's config *_multiplier / logits_scaling hparams to
//     *_scale and writes each only when non-zero. llama.cpp
//     (src/models/granite.cpp load_arch_hparams) requires logit_scale and reads the
//     other three optionally, defaulting to 0, where 0 means "identity / default":
//     the embedding and residual scales are applied only when non-zero and a zero
//     attention scale falls back to 1/√head_dim, while the logits are ALWAYS divided
//     by f_logit_scale. That maps exactly onto [LlamaConfig]'s Granite fields
//     (0 → identity/default), so absent optional keys load as 0 and a missing
//     granite.logit_scale is rejected.
//   - Standard llama tensor set, optionally tied head: token_embd / blk.N.{attn_norm,
//     attn_q, attn_k, attn_v, attn_output, ffn_norm, ffn_gate, ffn_up, ffn_down} /
//     output_norm, with output.weight TENSOR_NOT_REQUIRED in granite.cpp — absent →
//     the head ties to token_embd (Granite 3's 2B/8B tie; the shared loader already
//     handles both).
//   - Coupled head width: the converter POPS head_dim ("No head_dim support",
//     conversion/granite.py) and LlamaModel.set_gguf_parameters writes
//     granite.rope.dimension_count = hidden_size/num_heads; granite.cpp asserts
//     n_embd_head == n_rot. HeadDim is therefore Dim/Heads, and a present
//     rope.dimension_count that disagrees is rejected rather than misrotated.
//   - RMS epsilon under granite.attention.layer_norm_rms_epsilon (the RMS key) and
//     granite.rope.freq_base from config rope_theta (the TextModel base writes it).
//   - Rejected extras (reject-don't-misload): attention/FFN bias tensors, a fused
//     attn_qkv and Qwen3-style attn_{q,k}_norm gains. Every released Granite
//     checkpoint is bias-free (GraniteConfig attention_bias/mlp_bias default false),
//     the converter emits split projections, and the granite graph has no QK-norm —
//     such tensors could only come from a hand-crafted file whose q/k biases would
//     additionally need un-permuting, so they are refused with a clear error instead
//     of being loaded wrong or silently dropped.
func GraniteFromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*Llama, error) {
	const arch = "granite"
	cfg, err := llamaCfgFromGGUFMeta(arch, meta)
	if err != nil {
		return nil, err
	}
	key := func(suffix string) string { return arch + "." + suffix }

	// granite.logit_scale is required (llama.cpp reads LLM_KV_LOGIT_SCALE for
	// granite without the optional flag and always divides the logits by it).
	if _, ok := meta[key(ggufLogitScale)]; !ok {
		return nil, fmt.Errorf("nlp: Granite GGUF missing %s (required by the granite architecture)", key(ggufLogitScale))
	}
	logitScale := metaFloat(meta, key(ggufLogitScale), 0)
	embMult := metaFloat(meta, key(ggufEmbScale), 0)   // optional; 0 → identity
	resMult := metaFloat(meta, key(ggufResScale), 0)   // optional; 0 → identity
	attnMult := metaFloat(meta, key(ggufAttnScale), 0) // optional; 0 → default 1/√head_dim

	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: Granite GGUF dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	headDim := cfg.Dim / cfg.Heads
	if headDim%2 != 0 {
		return nil, fmt.Errorf("nlp: Granite GGUF head_dim %d must be even (RoPE rotates pairs)", headDim)
	}
	if rd, e := metaInt(meta, key(ggufRopeDims)); e == nil && rd != headDim {
		return nil, fmt.Errorf("nlp: Granite GGUF %s=%d != dim/heads=%d (granite couples n_rot to the head width)", key(ggufRopeDims), rd, headDim)
	}

	// Un-permute q/k into a shallow copy of the tensor map, then ride the shared
	// llama-family loader for everything else (names, transposes, tied-head
	// fallback). The rejects come first so nothing unsupported slips through the
	// shared loader's optional-tensor tolerance.
	mod := make(map[string]*tensor.Tensor, len(tensors))
	for k, v := range tensors {
		mod[k] = v
	}
	kvSize := cfg.kvHeads() * headDim
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		for _, unsupported := range []string{"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias",
			"ffn_gate.bias", "ffn_up.bias", "ffn_down.bias", "attn_qkv.weight",
			"attn_q_norm.weight", "attn_k_norm.weight"} {
			if _, ok := tensors[p+unsupported]; ok {
				return nil, fmt.Errorf("nlp: Granite GGUF carries %s%s; the granite architecture is a bias-free split-projection llama without QK-norm", p, unsupported)
			}
		}
		wq, ok := tensors[p+"attn_q.weight"]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %sattn_q.weight", p)
		}
		wk, ok := tensors[p+"attn_k.weight"]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %sattn_k.weight", p)
		}
		if got := wq.Shape()[0]; got != cfg.Dim {
			return nil, fmt.Errorf("nlp: Granite GGUF %sattn_q.weight rows %d != dim %d", p, got, cfg.Dim)
		}
		if got := wk.Shape()[0]; got != kvSize {
			return nil, fmt.Errorf("nlp: Granite GGUF %sattn_k.weight rows %d != kv_heads·head_dim %d", p, got, kvSize)
		}
		// GGUF llama-arch q/k are stored interleaved-permuted; restore the HF
		// split-half row order for GoAI's OpRoPE.
		mod[p+"attn_q.weight"] = ropeUnpermuteRows(wq, cfg.Heads)
		mod[p+"attn_k.weight"] = ropeUnpermuteRows(wk, cfg.kvHeads())
	}
	m, err := llamaArchFromGGUF(arch, meta, mod)
	if err != nil {
		return nil, err
	}
	m.Config.EmbeddingMult = embMult
	m.Config.AttentionMult = attnMult
	m.Config.ResidualMult = resMult
	m.Config.LogitsScale = logitScale
	return m, nil
}

// GraniteToGGUF is the inverse of [GraniteFromGGUF]: it serializes a Granite-shaped
// [Llama] (e.g. from [GraniteFromHF]) into GGUF metadata + tensor maps under
// general.architecture "granite", exactly the way llama.cpp's converter lays the file
// out — the standard llama tensor set with attn_q/attn_k rows PERMUTED into the
// interleaved llama-arch layout (LlamaModel.permute, inherited by GraniteModel via
// undo_permute = True), granite.rope.dimension_count = dim/heads (the converter's
// coupled-head convention), and the four scalars under their *_scale names. Following
// the converter, the three optional scalars (embedding_scale, residual_scale,
// attention.scale) are written only when non-zero; granite.logit_scale is always
// written because llama.cpp requires it (a model with LogitsScale 0 — the GoAI
// identity — is written as 1.0, the divide-by-one identity). Models carrying
// Qwen2-style q/k/v biases or QK-norm gains are outside the granite lineage; their
// extra tensors are emitted by the shared serializer un-permuted, and
// [GraniteFromGGUF] refuses such files loudly rather than misloading them. Pass the
// result to gguf.Write via a gguf.File.
func GraniteToGGUF(m *Llama) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "granite"
	meta, ts := llamaArchToGGUF(arch, m)
	c := m.Config
	key := func(suffix string) string { return arch + "." + suffix }
	meta[key(ggufRopeDims)] = uint32(c.Dim / c.Heads)
	logitScale := c.LogitsScale
	if logitScale == 0 {
		logitScale = 1 // required key; GoAI's 0-identity is llama.cpp's divide-by-one
	}
	meta[key(ggufLogitScale)] = float32(logitScale)
	if c.EmbeddingMult != 0 {
		meta[key(ggufEmbScale)] = float32(c.EmbeddingMult)
	}
	if c.ResidualMult != 0 {
		meta[key(ggufResScale)] = float32(c.ResidualMult)
	}
	if c.AttentionMult != 0 {
		meta[key(ggufAttnScale)] = float32(c.AttentionMult)
	}
	for l := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", l)
		// GoAI [in,out] was already transposed to torch [out,in] by the shared
		// serializer; apply the converter's HF→interleaved permute on top.
		ts[p+"attn_q.weight"] = ropePermuteRows(ts[p+"attn_q.weight"], c.Heads)
		ts[p+"attn_k.weight"] = ropePermuteRows(ts[p+"attn_k.weight"], c.kvHeads())
	}
	return meta, ts
}
