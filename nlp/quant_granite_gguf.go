package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
)

// QuantGraniteFromGGUF builds a QuantLlama from the metadata and STILL-QUANTIZED tensor
// map of a GGUF file whose general.architecture is "granite" (gguf.ReadRaw) — the
// quantized twin of [GraniteFromGGUF] and [QuantLlamaFromGGUF]. IBM's Granite is a
// bias-free split-projection llama with four scalar multipliers, so this loader rides the
// shared quantized llama-family loader and then stamps the scalars onto the config, where
// QuantLlama.Forward / DecodeStep apply them (they are no-ops on the other quantized
// families, whose scalars are 0/1).
//
// Two Granite specifics beyond [QuantLlamaFromGGUF] (verified against llama.cpp master:
// conversion/granite.py GraniteModel(LlamaModel) + src/models/granite.cpp):
//
//   - The scalars come from granite.embedding_scale, granite.residual_scale,
//     granite.logit_scale (Keys.LLM) and granite.attention.scale (Keys.Attention).
//     logit_scale is REQUIRED (granite.cpp always divides the logits by it); the other
//     three default to 0 = identity / the default 1/√head_dim. They map onto [LlamaConfig]'s
//     EmbeddingMult / ResidualMult / LogitsScale / AttentionMult exactly as the float loader.
//   - q/k are stored in the llama arch's INTERLEAVED rotary row layout (GraniteModel
//     inherits LlamaModel.permute), so the Q-block rows are un-permuted back into GoAI's
//     split-half layout via [quantPermuteRows]/[ropeUnpermutePerm] — a lossless byte-range
//     row permutation, never a dequantize→requantize round-trip (§B67, T802). Granite is
//     bias-free and has no QK-norm; any q/k/v/o bias, QK-norm gain or packed attn_qkv is
//     refused (a hand-crafted file with such tensors would need extra un-permuting and is
//     not a real granite checkpoint).
func QuantGraniteFromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantLlama, error) {
	const arch = "granite"
	cfg, err := llamaCfgFromGGUFMeta(arch, meta)
	if err != nil {
		return nil, err
	}
	key := func(suffix string) string { return arch + "." + suffix }

	if _, ok := meta[key(ggufLogitScale)]; !ok {
		return nil, fmt.Errorf("nlp: Granite GGUF missing %s (required by the granite architecture)", key(ggufLogitScale))
	}
	logitScale := metaFloat(meta, key(ggufLogitScale), 0)
	embMult := metaFloat(meta, key(ggufEmbScale), 0)
	resMult := metaFloat(meta, key(ggufResScale), 0)
	attnMult := metaFloat(meta, key(ggufAttnScale), 0)

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

	// Reject the unsupported tensors first (so nothing slips through the shared
	// loader's optional-tensor tolerance), then un-permute the quantized q/k rows.
	ts := make(map[string]gguf.QuantTensor, len(tensors))
	for k, v := range tensors {
		ts[k] = v
	}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		for _, unsupported := range []string{"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias",
			"ffn_gate.bias", "ffn_up.bias", "ffn_down.bias", "attn_qkv.weight",
			"attn_q_norm.weight", "attn_k_norm.weight"} {
			if _, ok := tensors[p+unsupported]; ok {
				return nil, fmt.Errorf("nlp: Granite GGUF carries %s%s; the granite architecture is a bias-free split-projection llama without QK-norm", p, unsupported)
			}
		}
		if err := unpermuteQuantRows(ts, p+"attn_q.weight", cfg.Heads); err != nil {
			return nil, err
		}
		if err := unpermuteQuantRows(ts, p+"attn_k.weight", cfg.kvHeads()); err != nil {
			return nil, err
		}
	}

	m, err := quantLlamaArchFromGGUF(arch, meta, ts)
	if err != nil {
		return nil, err
	}
	m.Config.EmbeddingMult = embMult
	m.Config.AttentionMult = attnMult
	m.Config.ResidualMult = resMult
	m.Config.LogitsScale = logitScale
	return m, nil
}

// unpermuteQuantRows reorders a 2-D quantized tensor's rows from the llama arch's
// on-disk interleaved rotary layout back into GoAI's split-half layout, in place in ts —
// the quantized twin of [ropeUnpermuteRows], a lossless Q-block byte-range permutation
// (§B67, T802). Missing or non-2-D tensors are left untouched (the caller validates
// presence).
func unpermuteQuantRows(ts map[string]gguf.QuantTensor, name string, heads int) error {
	qt, ok := ts[name]
	if !ok || len(qt.Shape) != 2 {
		return nil
	}
	perm, err := ropeUnpermutePerm(qt.Shape[0], heads)
	if err != nil {
		return fmt.Errorf("nlp: GGUF %s: %w", name, err)
	}
	out, err := quantPermuteRows(qt, perm)
	if err != nil {
		return fmt.Errorf("nlp: GGUF %s: %w", name, err)
	}
	ts[name] = out
	return nil
}
