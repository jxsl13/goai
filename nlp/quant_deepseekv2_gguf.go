package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantDeepSeekV2FromGGUF builds a [QuantDeepSeekV2] from the metadata and
// STILL-QUANTIZED tensor map of a GGUF file whose general.architecture is
// "deepseek2" (gguf.ReadRaw) — the quantized twin of [DeepSeekV2FromGGUF], and
// the MLA capstone of the quantized loaders: a llama.cpp-quantized DeepSeek-V2
// checkpoint loads straight into QuantLinear projections without ever
// materializing full-precision weights. The deepseek2 conventions documented at
// [DeepSeekV2FromGGUF] (verified against llama.cpp master) carry over, split by
// what stays quantized:
//
//   - WHAT IS QUANTIZED mirrors llama-quant.cpp exactly: every ≥2-D *.weight —
//     attn_q_a/attn_q_b, attn_kv_a_mqa, the split attn_k_b/attn_v_b (3-D,
//     quantized like the expert banks) or the legacy fused attn_kv_b,
//     attn_output, the dense ffn_gate/up/down, the fused 3-D ffn_*_exps bank,
//     the fused ffn_*_shexp shared expert, token_embd and output — while the
//     1-D norm gains (attn_norm, ffn_norm, output_norm AND MLA's
//     attn_q_a_norm/attn_kv_a_norm) and the MoE router ffn_gate_inp (excluded
//     by name in llama-quant.cpp) stay F32 and are decoded to f32 here.
//   - The NORM-rope de-interleave survives ON THE QUANTIZED BYTES: the pe rows
//     of attn_q_b (per head, rows QKNope..QKNope+QKRope of each qkHead block)
//     and attn_kv_a_mqa (the shared k_pe block after the KVLoraRank latent
//     rows) are stored interleaved (LLAMA_ROPE_TYPE_NORM, consecutive pairs)
//     and must land in GoAI's split-half OpRoPE order. [deinterleaveRoPE] is a
//     pure ROW permutation — whole rows move, nothing mixes within a row — and
//     ggml block quantization is row-granular (blocks never span rows), so the
//     inverse is applied directly to the Q-block bytes via [quantPermuteRows]:
//     bit-identical to quantizing the de-interleaved float rows, zero added
//     error (the §B67/T802 argument, third instance).
//   - The kv_b FORM picks the attention path (see [QuantDeepSeekV2]): modern
//     files (key/value_length_mla present) carry attn_k_b
//     [heads, KVLoraRank, QKNope] — per-head PRE-TRANSPOSED, i.e. already the
//     ABSORBED operator — and attn_v_b [heads, VHead, KVLoraRank]; each head is
//     un-fused with [quantSliceExpert] (a lossless byte-range copy of the 3-D
//     stack) and wrapped as the per-head KB/VB QuantLinears of the absorbed
//     latent path. Legacy files carry the unsplit attn_kv_b
//     [heads·(QKNope+VHead), KVLoraRank], wrapped whole as the fused WkvB of
//     the reconstructed path. A file carrying both forms is rejected, like
//     llama.cpp's all-tensors-used check.
//   - MoE layers (index ≥ leading_dense_block_count) load the f32 router, the
//     per-expert bank via [quantSliceExpert] and the fused shared expert;
//     routing config (top-k, routed scaling, softmax-only gating, no
//     renormalization, no V3 score-correction bias) is validated by the shared
//     [deepseekV2CfgFromGGUFMeta] rejects plus the exp_probs_b probe below.
//   - REJECTED, mirroring the float loader: no-q-lora files (V2-Lite's plain
//     attn_q, whether flagged by missing deepseek2.attention.q_lora_rank or by
//     a present blk.N.attn_q.weight), YaRN rope scaling, expert_weights_norm,
//     non-softmax gating, inconsistent *_mla keys and wrong-shape tensors (a
//     wrong shape would slice heads at wrong byte offsets and misload
//     silently).
//
// The token embedding is dequantized to f32 (its lookup needs a float table)
// whatever its storage type; an absent output.weight ties the LM head to
// token_embd, which must then itself be stored in a quantized-matmul format.
func QuantDeepSeekV2FromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantDeepSeekV2, error) {
	cfg, isMLA, err := deepseekV2CfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	qkHead := cfg.QKNope + cfg.QKRope

	get := func(name string) (gguf.QuantTensor, error) {
		qt, ok := tensors[name]
		if !ok {
			return gguf.QuantTensor{}, fmt.Errorf("nlp: GGUF missing %s", name)
		}
		return qt, nil
	}
	// wrap a GGUF projection (quantized [out, in]) as a QuantLinear — no
	// transpose, no requant.
	wrapQ := func(name string, qt gguf.QuantTensor) (*nn.QuantLinear, error) {
		if len(qt.Shape) != 2 {
			return nil, fmt.Errorf("nlp: GGUF %s must be 2-D, got %v", name, qt.Shape)
		}
		if !quantMatMulSupported(qt.GGType) {
			return nil, fmt.Errorf("nlp: GGUF %s ggml type %d is not a quantized-matmul format", name, qt.GGType)
		}
		return &nn.QuantLinear{Weight: qt.Data, QT: gguf.QuantType(qt.GGType), In: qt.Shape[1], Out: qt.Shape[0]}, nil
	}
	mkQ := func(name string) (*nn.QuantLinear, error) {
		qt, err := get(name)
		if err != nil {
			return nil, err
		}
		return wrapQ(name, qt)
	}
	// mkQShaped additionally pins the [out, in] geometry — a wrong-shape MLA
	// projection would otherwise slice pe rows or head blocks at wrong offsets.
	mkQShaped := func(name string, out, in int) (*nn.QuantLinear, error) {
		qt, err := get(name)
		if err != nil {
			return nil, err
		}
		if len(qt.Shape) != 2 || qt.Shape[0] != out || qt.Shape[1] != in {
			return nil, fmt.Errorf("nlp: GGUF %s shape %v, want [%d, %d]", name, qt.Shape, out, in)
		}
		return wrapQ(name, qt)
	}
	// de-interleave the NORM-rope pe rows on the quantized bytes, then wrap.
	mkQDeinterleaved := func(name string, out, in, heads, block, peOffset int) (*nn.QuantLinear, error) {
		qt, err := get(name)
		if err != nil {
			return nil, err
		}
		if len(qt.Shape) != 2 || qt.Shape[0] != out || qt.Shape[1] != in {
			return nil, fmt.Errorf("nlp: GGUF %s shape %v, want [%d, %d]", name, qt.Shape, out, in)
		}
		perm, err := deepseekV2DeinterleavePerm(out, heads, block, peOffset, cfg.QKRope)
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		un, err := quantPermuteRows(qt, perm)
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		return wrapQ(name, un)
	}
	// decode a REQUIRED 1-D RMSNorm gain to f32.
	mkNorm := func(name string) (*nn.RMSNorm, error) {
		qt, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %s", name)
		}
		g, err := qt.Dequantize()
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		return &nn.RMSNorm{Gamma: g, Eps: cfg.Eps}, nil
	}

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	if len(tok.Shape) != 2 || tok.Shape[1] != cfg.Dim {
		return nil, fmt.Errorf("nlp: deepseek2 GGUF token_embd.weight %v does not match embedding_length %d", tok.Shape, cfg.Dim)
	}
	cfg.Vocab = tok.Shape[0]
	emb, err := tok.Dequantize() // embedding lookup needs a float (f32) table
	if err != nil {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight: %w", err)
	}

	q := &QuantDeepSeekV2{Config: cfg, TokEmb: emb}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		if _, ok := tensors[p+"attn_q.weight"]; ok {
			return nil, fmt.Errorf("nlp: deepseek2 GGUF carries %sattn_q.weight (a no-q-lora file, e.g. DeepSeek-V2-Lite); GoAI's DeepSeekV2 supports only the compressed attn_q_a/attn_q_b query path", p)
		}
		qb := &QuantDeepSeekV2Block{}
		if qb.InputNorm, err = mkNorm(p + "attn_norm.weight"); err != nil {
			return nil, err
		}
		if qb.QANorm, err = mkNorm(p + "attn_q_a_norm.weight"); err != nil {
			return nil, err
		}
		if qb.KvANorm, err = mkNorm(p + "attn_kv_a_norm.weight"); err != nil {
			return nil, err
		}
		if qb.PostAttnNorm, err = mkNorm(p + "ffn_norm.weight"); err != nil {
			return nil, err
		}
		if qb.WqA, err = mkQShaped(p+"attn_q_a.weight", cfg.QLoraRank, cfg.Dim); err != nil {
			return nil, err
		}
		// q_b: per-head pe rows de-interleaved on the Q-blocks (heads blocks of
		// qkHead rows, pe span at offset QKNope).
		if qb.WqB, err = mkQDeinterleaved(p+"attn_q_b.weight", cfg.Heads*qkHead, cfg.QLoraRank, cfg.Heads, qkHead, cfg.QKNope); err != nil {
			return nil, err
		}
		// kv_a_mqa: ONE shared k_pe block after the KVLoraRank latent rows.
		if qb.WkvA, err = mkQDeinterleaved(p+"attn_kv_a_mqa.weight", cfg.KVLoraRank+cfg.QKRope, cfg.Dim, 1, cfg.KVLoraRank+cfg.QKRope, cfg.KVLoraRank); err != nil {
			return nil, err
		}
		if qb.KB, qb.VB, qb.WkvB, err = quantDeepSeekV2KvB(tensors, p, cfg, isMLA); err != nil {
			return nil, err
		}
		if qb.Wo, err = mkQShaped(p+"attn_output.weight", cfg.Dim, cfg.Heads*cfg.VHead); err != nil {
			return nil, err
		}
		if l < cfg.FirstKDense || cfg.NRoutedExperts <= 0 {
			d := &nn.QuantSwiGLU{}
			if d.Gate, err = mkQ(p + "ffn_gate.weight"); err != nil {
				return nil, err
			}
			if d.Up, err = mkQ(p + "ffn_up.weight"); err != nil {
				return nil, err
			}
			if d.Down, err = mkQ(p + "ffn_down.weight"); err != nil {
				return nil, err
			}
			qb.Dense = d
		} else if qb.MoE, err = quantDeepSeekV2MoEFromGGUF(tensors, p, cfg); err != nil {
			return nil, err
		}
		q.Blocks = append(q.Blocks, qb)
	}
	if q.Norm, err = mkNorm("output_norm.weight"); err != nil {
		return nil, err
	}
	// Untied head with the tied fallback: a present output.weight wins; absent →
	// the quantized token_embd bytes (llama.cpp's TENSOR_DUPLICATED).
	if o, ok := tensors["output.weight"]; ok {
		if len(o.Shape) != 2 || o.Shape[0] != cfg.Vocab || o.Shape[1] != cfg.Dim {
			return nil, fmt.Errorf("nlp: deepseek2 GGUF output.weight %v, want [vocab, dim] = [%d, %d]", o.Shape, cfg.Vocab, cfg.Dim)
		}
		if q.Out, err = wrapQ("output.weight", o); err != nil {
			return nil, err
		}
	} else if q.Out, err = wrapQ("token_embd.weight", tok); err != nil {
		return nil, err
	}
	return q, nil
}

// quantDeepSeekV2KvB loads one block's kv_b decompression in whichever form the
// file carries, ON THE QUANTIZED BYTES — the crux of the quantized deepseek2
// port (see [QuantDeepSeekV2]): the modern split pair wraps per head as the
// ABSORBED operators (attn_k_b is stored per-head pre-transposed, so its bytes
// already ARE q_nope→q_lat; attn_v_b's are c_att→value), each head a lossless
// [quantSliceExpert] byte-range copy of the 3-D stack; the legacy unsplit
// attn_kv_b wraps whole as the fused RECONSTRUCTION operator. Re-fusing the
// split form (or absorbing the legacy form) would need a per-head transpose,
// which cannot be done losslessly on Q-blocks — so each form keeps its own
// path, exactly as llama.cpp does. A file carrying both forms is rejected.
func quantDeepSeekV2KvB(tensors map[string]gguf.QuantTensor, p string, cfg DeepSeekV2Config, isMLA bool) (kb, vb []*nn.QuantLinear, fused *nn.QuantLinear, err error) {
	kbT, hasKb := tensors[p+"attn_k_b.weight"]
	vbT, hasVb := tensors[p+"attn_v_b.weight"]
	kvbT, hasKvb := tensors[p+"attn_kv_b.weight"]
	if (hasKb || hasVb) && hasKvb {
		return nil, nil, nil, fmt.Errorf("nlp: deepseek2 GGUF %s carries both the split attn_k_b/attn_v_b and the legacy attn_kv_b", p)
	}
	if !isMLA {
		if !hasKvb {
			return nil, nil, nil, fmt.Errorf("nlp: legacy deepseek2 GGUF missing %sattn_kv_b.weight", p)
		}
		want := tensor.Shape{cfg.Heads * (cfg.QKNope + cfg.VHead), cfg.KVLoraRank}
		if len(kvbT.Shape) != 2 || kvbT.Shape[0] != want[0] || kvbT.Shape[1] != want[1] {
			return nil, nil, nil, fmt.Errorf("nlp: deepseek2 GGUF %sattn_kv_b.weight shape %v, want %v", p, kvbT.Shape, want)
		}
		if !quantMatMulSupported(kvbT.GGType) {
			return nil, nil, nil, fmt.Errorf("nlp: GGUF %sattn_kv_b.weight ggml type %d is not a quantized-matmul format", p, kvbT.GGType)
		}
		return nil, nil, &nn.QuantLinear{Weight: kvbT.Data, QT: gguf.QuantType(kvbT.GGType), In: kvbT.Shape[1], Out: kvbT.Shape[0]}, nil
	}
	if !hasKb || !hasVb {
		return nil, nil, nil, fmt.Errorf("nlp: deepseek2 GGUF (MLA-split form) missing %sattn_k_b.weight/attn_v_b.weight", p)
	}
	if want := (tensor.Shape{cfg.Heads, cfg.KVLoraRank, cfg.QKNope}); len(kbT.Shape) != 3 || kbT.Shape[0] != want[0] || kbT.Shape[1] != want[1] || kbT.Shape[2] != want[2] {
		return nil, nil, nil, fmt.Errorf("nlp: deepseek2 GGUF %sattn_k_b.weight shape %v, want %v (per-head transposed)", p, kbT.Shape, want)
	}
	if want := (tensor.Shape{cfg.Heads, cfg.VHead, cfg.KVLoraRank}); len(vbT.Shape) != 3 || vbT.Shape[0] != want[0] || vbT.Shape[1] != want[1] || vbT.Shape[2] != want[2] {
		return nil, nil, nil, fmt.Errorf("nlp: deepseek2 GGUF %sattn_v_b.weight shape %v, want %v", p, vbT.Shape, want)
	}
	for _, qt := range []gguf.QuantTensor{kbT, vbT} {
		if !quantMatMulSupported(qt.GGType) {
			return nil, nil, nil, fmt.Errorf("nlp: GGUF %sattn_k_b/attn_v_b ggml type %d is not a quantized-matmul format", p, qt.GGType)
		}
	}
	for h := range cfg.Heads {
		ks, err := quantSliceExpert(kbT, h) // [KVLoraRank, QKNope] — the absorbed layout as stored
		if err != nil {
			return nil, nil, nil, fmt.Errorf("nlp: GGUF %sattn_k_b.weight head %d: %w", p, h, err)
		}
		vs, err := quantSliceExpert(vbT, h) // [VHead, KVLoraRank]
		if err != nil {
			return nil, nil, nil, fmt.Errorf("nlp: GGUF %sattn_v_b.weight head %d: %w", p, h, err)
		}
		kb = append(kb, &nn.QuantLinear{Weight: ks.Data, QT: gguf.QuantType(ks.GGType), In: ks.Shape[1], Out: ks.Shape[0]})
		vb = append(vb, &nn.QuantLinear{Weight: vs.Data, QT: gguf.QuantType(vs.GGType), In: vs.Shape[1], Out: vs.Shape[0]})
	}
	return kb, vb, nil, nil
}

// quantDeepSeekV2MoEFromGGUF builds one block's [QuantDeepSeekMoE] from the GGUF
// router ([E, dim], decoded to f32 — never quantized in real files), the fused
// quantized expert bank (gate/up [E, moe_ffn, dim], down [E, dim, moe_ffn] —
// un-fused per expert with [quantSliceExpert], a lossless byte-range copy) and
// the fused quantized shared expert (ffn_*_shexp, width
// NSharedExperts·MoEHidden). A DeepSeek-V3-style blk.N.exp_probs_b.bias (the
// sigmoid router's score-correction bias) is rejected — V2's gating has no such
// term, mirroring the float loader.
func quantDeepSeekV2MoEFromGGUF(tensors map[string]gguf.QuantTensor, p string, cfg DeepSeekV2Config) (*QuantDeepSeekMoE, error) {
	if _, ok := tensors[p+"exp_probs_b.bias"]; ok {
		return nil, fmt.Errorf("nlp: deepseek2 GGUF carries %sexp_probs_b.bias (a DeepSeek-V3 score-correction bias); GoAI's DeepSeekV2 implements V2's bias-free softmax gating", p)
	}
	router, ok := tensors[p+"ffn_gate_inp.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing %sffn_gate_inp.weight", p)
	}
	e, inter := cfg.NRoutedExperts, cfg.MoEHidden
	if len(router.Shape) != 2 || router.Shape[0] != e || router.Shape[1] != cfg.Dim {
		return nil, fmt.Errorf("nlp: deepseek2 GGUF %sffn_gate_inp.weight shape %v, want [%d, %d]", p, router.Shape, e, cfg.Dim)
	}
	rt, err := router.Dequantize()
	if err != nil {
		return nil, fmt.Errorf("nlp: GGUF %sffn_gate_inp.weight: %w", p, err)
	}
	moe := &QuantDeepSeekMoE{Router: f32Transpose(rt), TopK: cfg.TopK, Scale: cfg.routedScale()} // [E,dim] → [dim,E]

	exps := make([]gguf.QuantTensor, 3)
	for i, name := range []string{"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"} {
		qt, ok := tensors[p+name]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
		}
		if len(qt.Shape) != 3 || qt.Shape[0] != e {
			return nil, fmt.Errorf("nlp: deepseek2 GGUF %s%s shape %v, want fused 3-D with %d experts", p, name, qt.Shape, e)
		}
		if !quantMatMulSupported(qt.GGType) {
			return nil, fmt.Errorf("nlp: GGUF %s%s ggml type %d is not a quantized-matmul format", p, name, qt.GGType)
		}
		exps[i] = qt
	}
	gate, up, down := exps[0], exps[1], exps[2]
	if gate.Shape[1] != inter || up.Shape[1] != inter || down.Shape[2] != inter || gate.Shape[2] != cfg.Dim {
		return nil, fmt.Errorf("nlp: deepseek2 GGUF %sffn_*_exps shapes %v/%v/%v disagree with expert_feed_forward_length %d / dim %d", p, gate.Shape, up.Shape, down.Shape, inter, cfg.Dim)
	}
	for x := range e {
		mk := func(fused gguf.QuantTensor, name string) (*nn.QuantLinear, error) {
			s, err := quantSliceExpert(fused, x)
			if err != nil {
				return nil, fmt.Errorf("nlp: GGUF %s%s expert %d: %w", p, name, x, err)
			}
			return &nn.QuantLinear{Weight: s.Data, QT: gguf.QuantType(s.GGType), In: s.Shape[1], Out: s.Shape[0]}, nil
		}
		g, err := mk(gate, "ffn_gate_exps.weight")
		if err != nil {
			return nil, err
		}
		u, err := mk(up, "ffn_up_exps.weight")
		if err != nil {
			return nil, err
		}
		d, err := mk(down, "ffn_down_exps.weight")
		if err != nil {
			return nil, err
		}
		moe.Experts = append(moe.Experts, &nn.QuantSwiGLU{Gate: g, Up: u, Down: d})
	}

	// The FUSED shared expert (expert_shared_count · expert_feed_forward_length wide).
	shWant := cfg.NSharedExperts * inter
	sh := &nn.QuantSwiGLU{}
	for name, dst := range map[string]**nn.QuantLinear{
		"ffn_gate_shexp.weight": &sh.Gate, "ffn_up_shexp.weight": &sh.Up, "ffn_down_shexp.weight": &sh.Down,
	} {
		qt, ok := tensors[p+name]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
		}
		if len(qt.Shape) != 2 {
			return nil, fmt.Errorf("nlp: deepseek2 GGUF %s%s must be 2-D, got %v", p, name, qt.Shape)
		}
		if !quantMatMulSupported(qt.GGType) {
			return nil, fmt.Errorf("nlp: GGUF %s%s ggml type %d is not a quantized-matmul format", p, name, qt.GGType)
		}
		*dst = &nn.QuantLinear{Weight: qt.Data, QT: gguf.QuantType(qt.GGType), In: qt.Shape[1], Out: qt.Shape[0]}
	}
	if sh.Gate.Out != shWant || sh.Up.Out != shWant || sh.Down.In != shWant {
		return nil, fmt.Errorf("nlp: deepseek2 GGUF %sffn_*_shexp width %d/%d/%d, want expert_shared_count·expert_feed_forward_length = %d", p, sh.Gate.Out, sh.Up.Out, sh.Down.In, shWant)
	}
	moe.Shared = sh
	return moe, nil
}

// deepseekV2DeinterleavePerm builds [deinterleaveRoPE]'s row permutation as an
// explicit index map (perm[dst] = src) for [quantPermuteRows]: rows split into
// `heads` blocks of `block` rows; within each block the `ropeDim` pe rows
// starting at peOffset are regrouped from interleaved (consecutive-pairs,
// LLAMA_ROPE_TYPE_NORM) into split-half order — dst peOffset+i ← src
// peOffset+2i and dst peOffset+i+ropeDim/2 ← src peOffset+2i+1 — while every
// other row maps to itself. Returns an error (not a panic — this runs on
// untrusted file geometry) when the block/pe arithmetic does not tile rows.
func deepseekV2DeinterleavePerm(rows, heads, block, peOffset, ropeDim int) ([]int, error) {
	if heads <= 0 || block <= 0 || heads*block != rows {
		return nil, fmt.Errorf("rope de-interleave needs rows %d = heads %d × block %d", rows, heads, block)
	}
	if ropeDim <= 0 || ropeDim%2 != 0 || peOffset < 0 || peOffset+ropeDim > block {
		return nil, fmt.Errorf("rope de-interleave pe span [%d,%d) does not fit an even rotary width in block %d", peOffset, peOffset+ropeDim, block)
	}
	perm := make([]int, rows)
	for i := range perm {
		perm[i] = i
	}
	half := ropeDim / 2
	for h := range heads {
		base := h*block + peOffset
		for i := range half {
			perm[base+i] = base + 2*i          // split-half first half ← interleaved even
			perm[base+i+half] = base + 2*i + 1 // split-half second half ← interleaved odd
		}
	}
	return perm, nil
}
