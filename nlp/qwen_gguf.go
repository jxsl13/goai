package nlp

import "github.com/jxsl13/goai/tensor"

// Qwen2FromGGUF builds a Qwen2/Qwen2.5 model (a [Llama] with q/k/v projection biases)
// from the metadata and tensor maps of a parsed GGUF file (gguf.File.Metadata /
// .Tensors) whose general.architecture is "qwen2" — the llama.cpp convention for the
// whole Qwen2/Qwen2.5 dense family. The config comes from the qwen2.* metadata keys
// (the same suffix set as llama.*: embedding_length, block_count, attention.head_count,
// …) and the weights from the identical token_embd / blk.N.* / output tensor names,
// plus the per-block blk.N.attn_{q,k,v}.bias vectors. An absent output.weight ties the
// LM head to token_embd.
//
// NOTE (§R93, rotary layout): unlike the llama architecture — whose HF→GGUF conversion
// permutes the attn_q/attn_k rows (llama.cpp conversion/llama.py, undo_permute=True,
// rope type NORM) — the qwen2/qwen3 converters (conversion/qwen.py Qwen2Model/
// Qwen3Model) apply NO permutation: llama.cpp runs these architectures with NEOX-style
// (split-half) RoPE directly on the Hugging Face weight layout. GoAI's OpRoPE is that
// same split-half convention (it pairs element i with i+headDim/2, matching HF's
// rotate_half), so this loader — like [LlamaFromHF] — only transposes each projection
// from torch [out, in] into GoAI's [in, out] and never permutes rows; biases are
// per-output-row and are likewise copied as stored.
func Qwen2FromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*Llama, error) {
	return llamaArchFromGGUF("qwen2", meta, tensors)
}

// Qwen2ToGGUF is the inverse of [Qwen2FromGGUF]: it serializes a Qwen2-family model
// into GGUF metadata + tensor maps under general.architecture "qwen2", emitting the
// blk.N.attn_{q,k,v}.bias tensors from each block's Bq/Bk/Bv (skipped when nil). Pass
// the result to gguf.Write via a gguf.File.
func Qwen2ToGGUF(m *Llama) (map[string]any, map[string]*tensor.Tensor) {
	return llamaArchToGGUF("qwen2", m)
}

// Qwen3FromGGUF builds a Qwen3 model (a [Llama] with per-head QK-norm and no attention
// biases) from the metadata and tensor maps of a parsed GGUF file whose
// general.architecture is "qwen3". The config comes from the qwen3.* metadata keys and
// the weights from the llama tensor names plus the per-block
// blk.N.attn_{q,k}_norm.weight gains ([headDim] each, applied per head before RoPE —
// loaded into LlamaBlock.QNorm/KNorm exactly as [LlamaFromHF] loads
// self_attn.{q,k}_norm.weight). Qwen3's decoupled head_dim (attn_q rows ≠
// embedding_length is legal) needs no extra metadata: GoAI derives every width from the
// tensor shapes.
//
// Rotary layout: identical to [Qwen2FromGGUF] — llama.cpp does not permute qwen3 q/k
// rows (NEOX split-half RoPE on the HF layout, §R93), so the loader transposes only.
func Qwen3FromGGUF(meta map[string]any, tensors map[string]*tensor.Tensor) (*Llama, error) {
	return llamaArchFromGGUF("qwen3", meta, tensors)
}

// Qwen3ToGGUF is the inverse of [Qwen3FromGGUF]: it serializes a Qwen3 model into GGUF
// metadata + tensor maps under general.architecture "qwen3", emitting the
// blk.N.attn_{q,k}_norm.weight gains from each block's QNorm/KNorm (skipped when nil).
// Pass the result to gguf.Write via a gguf.File.
func Qwen3ToGGUF(m *Llama) (map[string]any, map[string]*tensor.Tensor) {
	return llamaArchToGGUF("qwen3", m)
}
