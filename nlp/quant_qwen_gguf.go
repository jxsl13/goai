package nlp

import "github.com/jxsl13/goai/format/gguf"

// QuantQwen2FromGGUF builds a QuantLlama from the STILL-QUANTIZED tensor map of a GGUF file
// whose general.architecture is "qwen2" (gguf.ReadRaw) — the quantized twin of [Qwen2FromGGUF],
// covering the whole Qwen2/Qwen2.5 dense family. This is the real community use case for
// quantized GGUF decode: a llama.cpp-quantized Qwen checkpoint loads straight into QuantLinear
// projections without ever materializing full-precision weights.
//
// The layout follows llama.cpp's qwen2 convention exactly (§R93): the config comes from the
// qwen2.* metadata keys (the llama.* suffix set), the 2-D projections stay in their ggml
// Q-block byte form, and the per-block blk.N.attn_{q,k,v}.bias vectors — which llama.cpp keeps
// unquantized (1-D tensors are stored F32 in real quantized files) — are decoded to f32 and
// added after the quantized q/k/v matmuls, before RoPE, exactly where the float pipeline adds
// them. Rotary layout is NEOX/no-permute, identical to [Qwen2FromGGUF]: q/k rows are wrapped
// as stored.
func QuantQwen2FromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantLlama, error) {
	return quantLlamaArchFromGGUF("qwen2", meta, tensors)
}

// QuantQwen3FromGGUF builds a QuantLlama from the STILL-QUANTIZED tensor map of a GGUF file
// whose general.architecture is "qwen3" (gguf.ReadRaw) — the quantized twin of [Qwen3FromGGUF].
//
// Qwen3 has no attention biases; instead the per-block blk.N.attn_{q,k}_norm.weight gains
// ([headDim] each, stored F32 like every 1-D tensor in a real quantized GGUF) are loaded into
// QuantBlock.QNorm/KNorm and applied per head to the projected q/k before RoPE — the same
// applyQKNorm the float pipeline uses. Qwen3's decoupled head_dim (attn_q rows ≠
// embedding_length) needs no extra metadata: every width derives from the quantized tensor
// shapes. Rotary layout is NEOX/no-permute (§R93), identical to [Qwen3FromGGUF].
func QuantQwen3FromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantLlama, error) {
	return quantLlamaArchFromGGUF("qwen3", meta, tensors)
}
