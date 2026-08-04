package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// GGUF ALiBi / QKV-clamp metadata keys (gguf-py constants.py
// Keys.Attention.MAX_ALIBI_BIAS "{arch}.attention.max_alibi_bias" and
// Keys.Attention.CLAMP_KQV "{arch}.attention.clamp_kqv"), arch-prefixed like the
// rest: "mpt.attention.max_alibi_bias".
const (
	ggufMaxAlibi = "attention.max_alibi_bias"
	ggufClampKQV = "attention.clamp_kqv"
)

// MPTFromGGUF builds an [MPT] from the metadata and (dequantized) tensor maps of a
// parsed GGUF file (gguf.File.Metadata / .Tensors) whose general.architecture is
// "mpt" — the llama.cpp convention for MosaicML's MPT family. The config comes from
// the mpt.* metadata keys (embedding_length, block_count, feed_forward_length,
// attention.head_count, attention.layer_norm_epsilon, context_length,
// attention.max_alibi_bias) and the weights from the token_embd / blk.N.* /
// output_norm tensors. Every projection is transposed from GGUF's torch [out, in]
// layout into GoAI's [in, out].
//
// The MPT conventions this loader implements were verified against llama.cpp master
// (conversion/mpt.py MPTModel + src/models/mpt.cpp + gguf-py/gguf/constants.py and
// tensor_mapping.py):
//
//   - PURE RENAME, HF-order fused qkv: MPTModel.modify_tensors only renames (its one
//     special case rewrites AWQ "scales" names) — no reshape and no de-interleave
//     (contrast gptneox). So blk.N.attn_qkv.weight [3·hidden, hidden] keeps HF's
//     mixed_qkv.chunk(3) layout: contiguous [all-q; all-k; all-v] row thirds, each
//     third head-contiguous — which is exactly the offset scheme llama.cpp's
//     build_qkv slices at runtime (views at 0, n_embd, n_embd + n_embd_gqa). This
//     loader unpacks with the same plain [sliceRows] thirds as [MPTFromHF].
//   - ALiBi is carried as mpt.attention.max_alibi_bias: MPTModel.set_gguf_parameters
//     writes alibi_bias_max (MPT's default 8) when attn_config.alibi is true and 0.0
//     when it is false; src/models/mpt.cpp reads LLM_KV_ATTENTION_MAX_ALIBI_BIAS as
//     optional. GoAI's OpMHA applies the fixed alibi_bias_max=8 slope family
//     2^(−8·(h+1)/n) ([backend.ALiBiSlopes]), so a present key must equal 8 — 0.0
//     (a no-ALiBi, learned-position MPT) or any other bias-max is REJECTED; an
//     absent key falls back to 8, MPT's own alibi_bias_max default. There is NO RoPE
//     anywhere in the mpt graph and no rope.* key is written.
//   - clip_qkv maps to mpt.attention.clamp_kqv (written only when
//     attn_config.clip_qkv is non-null; optional in src/models/mpt.cpp). GoAI's MPT
//     forward does not clamp the qkv activations, so a present non-zero clamp is
//     REJECTED rather than silently ignored.
//   - NO-BIAS convention (no_bias=True, the MPT default): src/models/mpt.cpp loads
//     EVERY bias — attn_norm/ffn_norm/output_norm .bias, attn_qkv/attn_output/
//     ffn_up/ffn_down .bias — as TENSOR_NOT_REQUIRED, and canonical MPT files carry
//     none of them. GoAI's [MPT] type models exactly the no_bias form (weight-only
//     LayerNorms, bias-free projections), so any bias tensor present is REJECTED as
//     an unsupported variant, as are the optional attn_q_norm/attn_k_norm pair
//     (qk_ln checkpoints), the AWQ blk.N.ffn.act.scales activation scales and a
//     position_embd.weight (an alibi=False MPT).
//   - MHA only: the converter writes attention.head_count_kv ONLY when
//     attn_config.kv_n_heads is set (grouped-query MPT variants). GoAI's [MPT] is
//     strictly MHA, so a present head_count_kv that differs from head_count is
//     REJECTED; absent (the canonical file) means KVHeads == Heads.
//   - TIED head: MptForCausalLM has no lm_head tensor, so the converter emits no
//     output.weight and llama.cpp falls back to token_embd (TENSOR_NOT_REQUIRED →
//     TENSOR_DUPLICATED). This loader mirrors that: an absent output.weight ties
//     Out to token_embdᵀ; a present one (llama.cpp tolerance) is used as-is.
//   - Epsilon: the converter hardcodes add_layer_norm_eps(1e-5) under
//     mpt.attention.layer_norm_epsilon (the full-LayerNorm key, required by
//     src/models/mpt.cpp); absent falls back to that same 1e-5.
func MPTFromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*MPT, error) {
	cfg, err := mptCfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: MPT GGUF dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	dim := cfg.Dim

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	cfg.Vocab = tok.Shape()[0]
	if _, ok := tensors["position_embd.weight"]; ok {
		return nil, fmt.Errorf("nlp: MPT GGUF carries position_embd.weight (an alibi=False, learned-position MPT); GoAI's MPT implements only the ALiBi form")
	}

	m := &MPT{Config: cfg, TokEmb: cloneF64(tok)}
	//perfscan:ignore PS3060 GGUF loader per-layer loop, one-time model load
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return t, nil
		}
		// The no_bias gate: canonical MPT files carry NO biases (llama.cpp loads them
		// all as optional); GoAI's MPT type cannot represent a biased variant.
		for _, unsupported := range []string{
			"attn_norm.bias", "ffn_norm.bias", "attn_qkv.bias", "attn_output.bias",
			"ffn_up.bias", "ffn_down.bias", "attn_q_norm.weight", "attn_k_norm.weight",
			"ffn.act.scales",
		} {
			if _, ok := tensors[p+unsupported]; ok {
				return nil, fmt.Errorf("nlp: MPT GGUF carries %s%s; GoAI's MPT implements only the canonical no_bias=True, no-qk_ln, non-AWQ form", p, unsupported)
			}
		}
		qkv, err := g("attn_qkv.weight")
		if err != nil {
			return nil, err
		}
		// Rank-guard before the row check reads Shape()[1] via sliceRows — the order
		// QuantMPTFromGGUF's `len(qkv.Shape) != 2 || …` uses (§B77).
		if err := require2D(p+"attn_qkv.weight", qkv); err != nil {
			return nil, err
		}
		if got := qkv.Shape()[0]; got != 3*dim {
			return nil, fmt.Errorf("nlp: MPT GGUF %sattn_qkv.weight rows %d != 3·hidden = %d", p, got, 3*dim)
		}
		// On-disk layout is HF's chunk(3) order — contiguous [all-q; all-k; all-v]
		// thirds (the converter is a pure rename): the same split as MPTFromHF.
		wq, wk, wv := sliceRows(qkv, 0, dim), sliceRows(qkv, dim, 2*dim), sliceRows(qkv, 2*dim, 3*dim)

		wo, err := g("attn_output.weight")
		if err != nil {
			return nil, err
		}
		wup, err := g("ffn_up.weight") // ffn.up_proj
		if err != nil {
			return nil, err
		}
		wdown, err := g("ffn_down.weight") // ffn.down_proj
		if err != nil {
			return nil, err
		}
		n1, err := g("attn_norm.weight") // norm_1
		if err != nil {
			return nil, err
		}
		n2, err := g("ffn_norm.weight") // norm_2
		if err != nil {
			return nil, err
		}
		// wq/wk/wv are rank-2 by construction (sliced from the guarded qkv); wo and the
		// two MLP weights reach transpose2D unchecked — the float twin of QuantMPTFromGGUF's
		// mkQ `len(qt.Shape) != 2` (§B77).
		if err := require2DEach(
			ggufWeight{p + "attn_output.weight", wo}, ggufWeight{p + "ffn_up.weight", wup}, ggufWeight{p + "ffn_down.weight", wdown},
		); err != nil {
			return nil, err
		}
		m.Blocks = append(m.Blocks, &MPTBlock{
			Norm1: layerNormFromHF(n1, cfg.Eps), Norm2: layerNormFromHF(n2, cfg.Eps), // weight-only (β = 0)
			Wq: transpose2D(wq), Wk: transpose2D(wk), Wv: transpose2D(wv), Wo: transpose2D(wo), // GGUF [out,in] → GoAI [in,out]
			Wup: transpose2D(wup), Wdown: transpose2D(wdown),
		})
	}
	if _, ok := tensors["output_norm.bias"]; ok {
		return nil, fmt.Errorf("nlp: MPT GGUF carries output_norm.bias; GoAI's MPT implements only the no_bias=True form")
	}
	nf, ok := tensors["output_norm.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output_norm.weight")
	}
	m.FinalNorm = layerNormFromHF(nf, cfg.Eps)
	head, headName := tok, "token_embd.weight (tied LM head)" // TENSOR_DUPLICATED fallback when output.weight is absent
	if o, ok := tensors["output.weight"]; ok {
		head, headName = o, "output.weight"
	}
	if err := require2D(headName, head); err != nil { // guards transpose2D below (§B77)
		return nil, err
	}
	m.Out = transpose2D(head) // [vocab,dim] → [dim,vocab]
	return m, nil
}

// mptCfgFromGGUFMeta parses the mpt.* metadata keys of a GGUF file whose
// general.architecture is "mpt" into an MPTConfig. Vocab is left for [MPTFromGGUF]
// (token_embd-derived). The unsupported-variant flags are gated here: a
// head_count_kv that differs from head_count (grouped-query MPT), a non-zero
// attention.clamp_kqv (clip_qkv) and an attention.max_alibi_bias other than 8 (in
// particular 0.0, the converter's alibi=False marker) are all rejected; an absent
// max_alibi_bias falls back to 8, MPT's alibi_bias_max default and the only slope
// family GoAI's OpMHA implements.
func mptCfgFromGGUFMeta(meta map[string]any) (MPTConfig, error) {
	const arch = "mpt"
	if a, _ := meta[ggufArch].(string); a != arch {
		return MPTConfig{}, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	dim, err := metaInt(meta, key(ggufEmbLen))
	if err != nil {
		return MPTConfig{}, err
	}
	layers, err := metaInt(meta, key(ggufBlockCnt))
	if err != nil {
		return MPTConfig{}, err
	}
	hidden, err := metaInt(meta, key(ggufFFLen))
	if err != nil {
		return MPTConfig{}, err
	}
	heads, err := metaInt(meta, key(ggufHeadCnt))
	if err != nil {
		return MPTConfig{}, err
	}
	if kv, e := metaInt(meta, key(ggufHeadKV)); e == nil && kv != heads {
		return MPTConfig{}, fmt.Errorf("nlp: MPT GGUF has %s=%d (a grouped-query kv_n_heads variant); GoAI's MPT implements only standard MHA (heads=%d)", key(ggufHeadKV), kv, heads)
	}
	if ab := metaFloat(meta, key(ggufMaxAlibi), 8); ab != 8 {
		return MPTConfig{}, fmt.Errorf("nlp: MPT GGUF has %s=%g; GoAI's MPT implements only the alibi_bias_max=8 slope family (0 means an alibi=False MPT)", key(ggufMaxAlibi), ab)
	}
	if cl := metaFloat(meta, key(ggufClampKQV), 0); cl != 0 {
		return MPTConfig{}, fmt.Errorf("nlp: MPT GGUF has %s=%g; GoAI's MPT does not implement qkv clamping (clip_qkv)", key(ggufClampKQV), cl)
	}
	cfg := MPTConfig{
		Dim: dim, Layers: layers, Hidden: hidden, Heads: heads,
		Eps: metaFloat(meta, key(ggufLNEps), 1e-5), // the converter hardcodes 1e-5
		Ctx: dim,                                   // provisional; overwritten from context_length below
	}
	if c, e := metaInt(meta, key(ggufCtxLen)); e == nil {
		cfg.Ctx = c
	}
	return cfg, nil
}

// MPTToGGUF is the inverse of [MPTFromGGUF]: it serializes an MPT (e.g. from
// [MPTFromHF]) into GGUF metadata + tensor maps under general.architecture "mpt",
// exactly the way llama.cpp's converter lays the file out — the fused
// blk.N.attn_qkv.weight in HF's [all-q; all-k; all-v] chunk order (the converter is
// a pure rename; contrast [GPTNeoXToGGUF]'s de-interleave), weight-only norms and
// bias-free projections (no .bias tensors anywhere, the no_bias=True form), NO
// output.weight (MPT's LM head is tied; llama.cpp duplicates token_embd), NO
// head_count_kv (MHA) and mpt.attention.max_alibi_bias = 8 (alibi_bias_max, the
// value the converter writes for an ALiBi MPT). Every projection is transposed back
// into torch [out, in]. Pass the result to gguf.Write via a gguf.File.
func MPTToGGUF(m *MPT) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "mpt"
	c := m.Config
	key := func(suffix string) string { return arch + "." + suffix }
	meta := map[string]any{
		ggufArch:          arch,
		key(ggufEmbLen):   uint32(c.Dim),
		key(ggufBlockCnt): uint32(c.Layers),
		key(ggufFFLen):    uint32(c.Hidden),
		key(ggufHeadCnt):  uint32(c.Heads),
		key(ggufCtxLen):   uint32(c.Ctx),
		key(ggufLNEps):    float32(c.Eps),
		key(ggufMaxAlibi): float32(8), // alibi_bias_max: the one slope family this type models
	}
	ts := map[string]*tensor.Tensor{
		"token_embd.weight":  cloneF64(m.TokEmb),
		"output_norm.weight": cloneF64(m.FinalNorm.Gamma),
		// No output.weight: the head is tied (the converter emits none for MPT).
	}
	//perfscan:ignore PS3060 GGUF export per-layer loop, one-time
	for l, b := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", l)
		ts[p+"attn_norm.weight"] = cloneF64(b.Norm1.Gamma)
		ts[p+"ffn_norm.weight"] = cloneF64(b.Norm2.Gamma)
		// Fused qkv in HF chunk order (pure-rename converter): [all-q; all-k; all-v].
		ts[p+"attn_qkv.weight"] = concatRows(concatRows(transpose2D(b.Wq), transpose2D(b.Wk)), transpose2D(b.Wv))
		ts[p+"attn_output.weight"] = transpose2D(b.Wo)
		ts[p+"ffn_up.weight"] = transpose2D(b.Wup)
		ts[p+"ffn_down.weight"] = transpose2D(b.Wdown)
	}
	return meta, ts
}
