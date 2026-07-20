package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// Phi3FromGGUF builds a Phi-3 model (a plain [Llama], exactly as [Phi3FromHF]
// produces) from the metadata and (dequantized) tensor maps of a parsed GGUF file
// (gguf.File.Metadata / .Tensors) whose general.architecture is "phi3" — the
// llama.cpp convention for the Microsoft Phi-3/Phi-3.5/Phi-4 dense family. The
// config comes from the phi3.* metadata keys (the same suffix set as llama.*:
// embedding_length, block_count, feed_forward_length, attention.head_count, …).
//
// The two Phi-3 conventions this loader implements were verified against llama.cpp
// master (conversion/phi.py Phi3MiniModel + src/models/phi3.cpp + the rope-type
// switch in src/llama-model.cpp):
//
//   - Tensors stay PACKED on disk: Phi3MiniModel defines NO modify_tensors — the
//     conversion is a pure name-map, so HF's row-packed projections survive as
//     blk.N.attn_qkv.weight ([heads·hd + 2·kv·hd, dim], rows [q; k; v]) and
//     blk.N.ffn_up.weight ([2·ffn, dim], rows [gate; up]); MODEL_ARCH.PHI3 has no
//     FFN_GATE tensor at all. At runtime llama.cpp slices q/k/v at exactly those
//     row offsets (llm_graph_context::build_qkv views at 0, n_embd_q,
//     n_embd_q+n_embd_kv) and splits the FFN with non-swapped ggml_swiglu
//     (silu(first half)·second half — HF's silu(gate)·up on the HF row order).
//     This loader unpacks both along their rows (the same sliceRows offsets as
//     [Phi3FromHF]) and delegates to the shared llama-family loader. Split
//     blk.N.attn_{q,k,v}.weight / ffn_gate.weight tensors are also accepted
//     (llama.cpp's create_tensor_qkv loads either form), pass-through unchanged.
//   - NO q/k row permutation: with no modify_tensors there is nothing to permute,
//     and LLM_ARCH_PHI3 sits in the LLAMA_ROPE_TYPE_NEOX case ("pairs of head
//     values offset by n_rot/2") — split-half RoPE on the HF weight layout, which
//     is exactly GoAI's OpRoPE. Like [Qwen2FromGGUF] and unlike the llama arch,
//     the loader only transposes; the packed rows are used exactly as stored.
//
// head_dim is Dim/Heads (phi3 writes no attention.key_length; its
// rope.dimension_count is partial_rotary_factor·Dim/Heads, 1.0 for Phi-3). An
// absent output.weight ties the LM head to token_embd (phi3's OUTPUT tensor is
// optional in llama.cpp, TENSOR_DUPLICATED fallback). LongRoPE scaling factors
// (rope_factors_long/short, 128k models) are not modeled — they are ignored if
// present, matching GoAI's plain-RoPE Phi3FromHF.
func Phi3FromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*Llama, error) {
	const arch = "phi3"
	if a, _ := meta[ggufArch].(string); a != arch {
		return nil, fmt.Errorf("nlp: GGUF general.architecture=%q, want %q", a, arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }
	dim, err := metaInt(meta, key(ggufEmbLen))
	if err != nil {
		return nil, err
	}
	heads, err := metaInt(meta, key(ggufHeadCnt))
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
	kv := heads
	if k, e := metaInt(meta, key(ggufHeadKV)); e == nil {
		kv = k
	}
	if heads <= 0 || dim%heads != 0 {
		return nil, fmt.Errorf("nlp: Phi-3 GGUF dim %d not divisible by heads %d", dim, heads)
	}
	headDim := dim / heads // phi3 convention: no decoupled head_dim key
	qSize, kvSize := heads*headDim, kv*headDim

	// Unpack the packed attn_qkv / ffn_up tensors (llama.cpp's on-disk phi3 form)
	// into the standard per-projection llama-family names, then delegate — the
	// same shape the HF converter reaches via Phi3FromHF→LlamaFromHF.
	out := make(map[string]*tensor.Tensor, len(tensors)+4*layers)
	for k, v := range tensors {
		out[k] = v
	}
	for l := range layers {
		p := fmt.Sprintf("blk.%d.", l)
		if qkv, ok := tensors[p+"attn_qkv.weight"]; ok {
			// Rank-guard before the row check reads Shape()[1] via sliceRows — the order
			// QuantPhi3FromGGUF's `len(qkv.Shape) != 2 || …` uses (§B77).
			if err := require2D(p+"attn_qkv.weight", qkv); err != nil {
				return nil, err
			}
			if got := qkv.Shape()[0]; got != qSize+2*kvSize {
				return nil, fmt.Errorf("nlp: Phi-3 GGUF %sattn_qkv.weight rows %d != heads·hd+2·kv·hd = %d", p, got, qSize+2*kvSize)
			}
			delete(out, p+"attn_qkv.weight")
			out[p+"attn_q.weight"] = sliceRows(qkv, 0, qSize)
			out[p+"attn_k.weight"] = sliceRows(qkv, qSize, qSize+kvSize)
			out[p+"attn_v.weight"] = sliceRows(qkv, qSize+kvSize, qSize+2*kvSize)
		}
		_, hasGate := tensors[p+"ffn_gate.weight"]
		if gu, ok := tensors[p+"ffn_up.weight"]; ok && !hasGate {
			if err := require2D(p+"ffn_up.weight", gu); err != nil { // sliceRows below reads Shape()[1] (§B77)
				return nil, err
			}
			if got := gu.Shape()[0]; got != 2*ffn {
				return nil, fmt.Errorf("nlp: Phi-3 GGUF %sffn_up.weight rows %d != 2·feed_forward_length = %d", p, got, 2*ffn)
			}
			delete(out, p+"ffn_up.weight")
			out[p+"ffn_gate.weight"] = sliceRows(gu, 0, ffn)
			out[p+"ffn_up.weight"] = sliceRows(gu, ffn, 2*ffn)
		}
	}
	return llamaArchFromGGUF(arch, meta, out)
}

// Phi3ToGGUF is the inverse of [Phi3FromGGUF]: it serializes a Phi-3 model (a
// [Llama], e.g. from [Phi3FromHF]) into GGUF metadata + tensor maps under
// general.architecture "phi3", RE-PACKING the projections into llama.cpp's
// on-disk phi3 form — blk.N.attn_qkv.weight rows [q; k; v] and
// blk.N.ffn_up.weight rows [gate; up], with no separate attn_{q,k,v} or ffn_gate
// tensors (MODEL_ARCH.PHI3 has no FFN_GATE) — so the file matches what
// llama.cpp's converter emits for this architecture. Rows are written unpermuted
// (phi3 is NEOX rope; the converter never permutes), and every projection is
// transposed back into torch [out, in]. phi3.rope.dimension_count (= head_dim,
// always written by the converter) rides along for fidelity. Pass the result to
// gguf.Write via a gguf.File.
func Phi3ToGGUF(m *Llama) (map[string]any, map[string]*tensor.Tensor) {
	const arch = "phi3"
	meta, ts := llamaArchToGGUF(arch, m)
	if m.Config.Heads > 0 {
		meta[arch+".rope.dimension_count"] = uint32(m.Config.Dim / m.Config.Heads)
	}
	for l := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", l)
		q, k, v := ts[p+"attn_q.weight"], ts[p+"attn_k.weight"], ts[p+"attn_v.weight"]
		delete(ts, p+"attn_q.weight")
		delete(ts, p+"attn_k.weight")
		delete(ts, p+"attn_v.weight")
		ts[p+"attn_qkv.weight"] = concatRows(concatRows(q, k), v)
		g, u := ts[p+"ffn_gate.weight"], ts[p+"ffn_up.weight"]
		delete(ts, p+"ffn_gate.weight")
		ts[p+"ffn_up.weight"] = concatRows(g, u)
	}
	return meta, ts
}
