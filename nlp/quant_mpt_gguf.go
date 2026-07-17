package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantMPTFromGGUF builds a [QuantMPT] from the metadata and STILL-QUANTIZED tensor map
// of a GGUF file whose general.architecture is "mpt" (gguf.ReadRaw) — the quantized twin
// of [MPTFromGGUF]: a llama.cpp-quantized MPT checkpoint loads straight into QuantLinear
// projections without ever materializing full-precision weights. The config comes from
// the same mpt.* metadata keys as the float loader (shared [mptCfgFromGGUFMeta], so
// every metadata gate is identical) and the weights from the same token_embd / blk.N.* /
// output_norm tensor names.
//
// The mpt conventions documented at [MPTFromGGUF] carry over unchanged to the quantized
// path:
//
//   - ALiBi gate: mpt.attention.max_alibi_bias must be 8 (or absent — MPT's own
//     alibi_bias_max default) — the one slope family [backend.OpMHA] implements. 0.0 is
//     the converter's alibi=False marker (a learned-position MPT) and is REJECTED, as
//     is a present position_embd.weight; a non-zero mpt.attention.clamp_kqv (clip_qkv)
//     and a head_count_kv that differs from head_count (grouped-query MPT) are also
//     rejected. ALiBi is METADATA, not a tensor — nothing positional is loaded.
//   - PURE RENAME, HF-order fused qkv: blk.N.attn_qkv.weight [3·hidden, hidden] keeps
//     HF's mixed_qkv.chunk(3) layout — contiguous [all-q; all-k; all-v] row thirds
//     (contrast gptneox's per-head de-interleave). The quantized weight is unpacked
//     WITHOUT dequantizing via [quantSliceRows] at plain thirds (0, dim, 2·dim) — ggml
//     blocks are row-granular, so each slice is bit-identical to quantizing the split
//     projections directly. GGUF's [out, in] row layout is exactly what QuantLinear
//     consumes: no transpose, no re-quantization.
//   - NO-BIAS convention (no_bias=True): canonical MPT files carry NO bias tensors.
//     Any bias present — norm or projection — is REJECTED as an unsupported variant,
//     as are the qk_ln attn_q_norm/attn_k_norm pair and the AWQ ffn.act.scales, the
//     same reject set as the float loader. The weight-only norms become f32 γ +
//     zero-β [nn.LayerNorm]s (the γ is F32 on disk — 1-D tensors are never
//     block-quantized — and is decoded to the f32 the quantized activations need).
//   - TIED head: canonical mpt files emit no output.weight (llama.cpp's
//     TENSOR_DUPLICATED falls back to token_embd), so token_embd must itself be
//     stored in a quantized-matmul format — the same convention as [QuantLlamaFromGGUF]
//     — and additionally always feeds the f32 lookup table via dequantization. A
//     present output.weight (llama.cpp tolerance) is used as the head instead.
func QuantMPTFromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantMPT, error) {
	cfg, err := mptCfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: MPT GGUF dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	dim := cfg.Dim

	// wrap a GGUF projection (quantized [out, in]) as a QuantLinear — no transpose, no requant.
	mkQ := func(name string, qt gguf.QuantTensor) (*nn.QuantLinear, error) {
		if len(qt.Shape) != 2 {
			return nil, fmt.Errorf("nlp: GGUF %s must be 2-D, got %v", name, qt.Shape)
		}
		if !quantMatMulSupported(qt.GGType) {
			return nil, fmt.Errorf("nlp: GGUF %s ggml type %d is not a quantized-matmul format", name, qt.GGType)
		}
		return &nn.QuantLinear{Weight: qt.Data, QT: gguf.QuantType(qt.GGType), In: qt.Shape[1], Out: qt.Shape[0]}, nil
	}
	// weight-only LayerNorm from a REQUIRED γ tensor (F32 in real files) — the
	// quantized twin of [layerNormFromHF]: f32 γ, zero f32 β (the no_bias form).
	mkLN := func(name string) (*nn.LayerNorm, error) {
		qt, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %s", name)
		}
		g, err := qt.Dequantize() // F32, the dtype the quantized activations need
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		return &nn.LayerNorm{Gamma: g, Beta: tensor.New(tensor.F32, tensor.Shape{g.Shape()[0]}), Eps: cfg.Eps}, nil
	}

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	if len(tok.Shape) != 2 {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight must be 2-D, got %v", tok.Shape)
	}
	cfg.Vocab = tok.Shape[0]
	if _, ok := tensors["position_embd.weight"]; ok {
		return nil, fmt.Errorf("nlp: MPT GGUF carries position_embd.weight (an alibi=False, learned-position MPT); GoAI's MPT implements only the ALiBi form")
	}
	emb, err := tok.Dequantize() // embedding lookup needs a float (f32) table, whatever the storage type
	if err != nil {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight: %w", err)
	}

	q := &QuantMPT{Config: cfg, TokEmb: emb}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		// The no_bias gate: canonical MPT files carry NO biases (llama.cpp loads them
		// all as optional); GoAI's MPT form cannot represent a biased variant.
		for _, unsupported := range []string{
			"attn_norm.bias", "ffn_norm.bias", "attn_qkv.bias", "attn_output.bias",
			"ffn_up.bias", "ffn_down.bias", "attn_q_norm.weight", "attn_k_norm.weight",
			"ffn.act.scales",
		} {
			if _, ok := tensors[p+unsupported]; ok {
				return nil, fmt.Errorf("nlp: MPT GGUF carries %s%s; GoAI's MPT implements only the canonical no_bias=True, no-qk_ln, non-AWQ form", p, unsupported)
			}
		}
		qkv, ok := tensors[p+"attn_qkv.weight"]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %sattn_qkv.weight", p)
		}
		if len(qkv.Shape) != 2 || qkv.Shape[0] != 3*dim {
			return nil, fmt.Errorf("nlp: MPT GGUF %sattn_qkv.weight shape %v, want [%d, in] = [3·hidden, hidden]", p, qkv.Shape, 3*dim)
		}
		qb := &QuantMPTBlock{}
		// HF chunk(3) order on disk — plain [all-q; all-k; all-v] thirds, unpacked
		// losslessly by quantized row-slice (NOT the gptneox per-head de-interleave).
		for _, part := range []struct {
			w      **nn.QuantLinear
			r0, r1 int
		}{
			{&qb.Wq, 0, dim},
			{&qb.Wk, dim, 2 * dim},
			{&qb.Wv, 2 * dim, 3 * dim},
		} {
			s, err := quantSliceRows(qkv, part.r0, part.r1)
			if err != nil {
				return nil, fmt.Errorf("nlp: MPT GGUF %sattn_qkv.weight: %w", p, err)
			}
			if *part.w, err = mkQ(p+"attn_qkv.weight", s); err != nil {
				return nil, err
			}
		}
		proj := func(name string) (*nn.QuantLinear, error) {
			qt, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return mkQ(p+name, qt)
		}
		if qb.Wo, err = proj("attn_output.weight"); err != nil {
			return nil, err
		}
		if qb.Wup, err = proj("ffn_up.weight"); err != nil { // ffn.up_proj
			return nil, err
		}
		if qb.Wdown, err = proj("ffn_down.weight"); err != nil { // ffn.down_proj
			return nil, err
		}
		if qb.Norm1, err = mkLN(p + "attn_norm.weight"); err != nil { // norm_1
			return nil, err
		}
		if qb.Norm2, err = mkLN(p + "ffn_norm.weight"); err != nil { // norm_2
			return nil, err
		}
		q.Blocks = append(q.Blocks, qb)
	}
	if _, ok := tensors["output_norm.bias"]; ok {
		return nil, fmt.Errorf("nlp: MPT GGUF carries output_norm.bias; GoAI's MPT implements only the no_bias=True form")
	}
	if q.FinalNorm, err = mkLN("output_norm.weight"); err != nil {
		return nil, err
	}
	head, headName := tok, "token_embd.weight" // llama.cpp's TENSOR_DUPLICATED fallback: MPT's head is tied
	if o, ok := tensors["output.weight"]; ok {
		head, headName = o, "output.weight"
	}
	if q.Out, err = mkQ(headName, head); err != nil {
		return nil, err
	}
	return q, nil
}
