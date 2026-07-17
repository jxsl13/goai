package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// QuantPhi3FromGGUF builds a QuantLlama from the STILL-QUANTIZED tensor map of a GGUF file
// whose general.architecture is "phi3" (gguf.ReadRaw) — the quantized twin of [Phi3FromGGUF],
// covering the Microsoft Phi-3/Phi-3.5/Phi-4 dense family: a llama.cpp-quantized Phi-3
// checkpoint loads straight into QuantLinear projections without ever materializing
// full-precision weights.
//
// Phi-3's on-disk peculiarity (verified against llama.cpp master for [Phi3FromGGUF]:
// Phi3MiniModel defines no modify_tensors) is that the projections stay PACKED: blk.N.attn_qkv.weight carries the rows
// [q; k; v] and blk.N.ffn_up.weight the rows [gate; up] (MODEL_ARCH.PHI3 has no FFN_GATE
// tensor). This loader unpacks both WITHOUT dequantizing, via [quantSliceRows]: ggml block
// quantization is row-granular — every block covers consecutive elements of ONE row (QMatMul
// consumes weights as rows × rowBytes, and a row width that is not a block multiple is
// rejected everywhere in the pipeline) — so slicing a row range of a packed quantized tensor
// is an exact byte-range copy, bit-identical to quantizing the split projections directly.
// No dequantize→split→requantize round-trip, hence no double quantization error.
//
// Split files (blk.N.attn_{q,k,v}.weight / ffn_gate.weight present, llama.cpp's
// create_tensor_qkv fallback form) are also legal and pass through unchanged. Rotary layout
// is NEOX/no-permute exactly as in [Phi3FromGGUF]: rows are wrapped as stored. head_dim is
// Dim/Heads (phi3 writes no attention.key_length); everything else — Q-block wrapping, F32
// 1-D norm gains, the tied LM head — follows the shared llama-family quantized loader.
func QuantPhi3FromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantLlama, error) {
	const arch = "phi3"
	cfg, err := llamaCfgFromGGUFMeta(arch, meta)
	if err != nil {
		return nil, err
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: Phi-3 GGUF dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	headDim := cfg.Dim / cfg.Heads // phi3 convention: no decoupled head_dim key
	qSize, kvSize := cfg.Heads*headDim, cfg.KVHeads*headDim

	// Unpack the packed attn_qkv / ffn_up tensors (llama.cpp's on-disk phi3 form) into the
	// standard per-projection llama-family names — quantized row-slices, the same offsets as
	// the float [Phi3FromGGUF] — then delegate to the shared quantized llama-family loader.
	out := make(map[string]gguf.QuantTensor, len(tensors)+4*cfg.Layers)
	for k, v := range tensors {
		out[k] = v
	}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		if qkv, ok := tensors[p+"attn_qkv.weight"]; ok {
			if len(qkv.Shape) != 2 || qkv.Shape[0] != qSize+2*kvSize {
				return nil, fmt.Errorf("nlp: Phi-3 GGUF %sattn_qkv.weight shape %v, want [%d, in] = [heads·hd+2·kv·hd, in]",
					p, qkv.Shape, qSize+2*kvSize)
			}
			delete(out, p+"attn_qkv.weight")
			for _, part := range []struct {
				name   string
				r0, r1 int
			}{
				{"attn_q.weight", 0, qSize},
				{"attn_k.weight", qSize, qSize + kvSize},
				{"attn_v.weight", qSize + kvSize, qSize + 2*kvSize},
			} {
				s, err := quantSliceRows(qkv, part.r0, part.r1)
				if err != nil {
					return nil, fmt.Errorf("nlp: Phi-3 GGUF %sattn_qkv.weight: %w", p, err)
				}
				out[p+part.name] = s
			}
		}
		_, hasGate := tensors[p+"ffn_gate.weight"]
		if gu, ok := tensors[p+"ffn_up.weight"]; ok && !hasGate {
			if len(gu.Shape) != 2 || gu.Shape[0] != 2*cfg.Hidden {
				return nil, fmt.Errorf("nlp: Phi-3 GGUF %sffn_up.weight shape %v, want [%d, in] = [2·feed_forward_length, in]",
					p, gu.Shape, 2*cfg.Hidden)
			}
			gate, err := quantSliceRows(gu, 0, cfg.Hidden)
			if err != nil {
				return nil, fmt.Errorf("nlp: Phi-3 GGUF %sffn_up.weight: %w", p, err)
			}
			up, err := quantSliceRows(gu, cfg.Hidden, 2*cfg.Hidden)
			if err != nil {
				return nil, fmt.Errorf("nlp: Phi-3 GGUF %sffn_up.weight: %w", p, err)
			}
			out[p+"ffn_gate.weight"], out[p+"ffn_up.weight"] = gate, up
		}
	}
	return quantLlamaArchFromGGUF(arch, meta, out)
}

// ggBlockElems returns the element count of one ggml quantization block for the tensor types
// a quantized llama-family GGUF can carry into a projection: 32 for the classic block quants,
// 256 for the k-quant super-blocks, 1 for the unblocked float types (F32=0, F16=1 — element-
// granular, so any row split is legal). Matches the byteSize geometry in format/gguf.
func ggBlockElems(gg uint32) (int, error) {
	if gg == 0 || gg == 1 { // F32 / F16
		return 1, nil
	}
	switch gguf.QuantType(gg) {
	case gguf.Q8_0, gguf.Q4_0:
		return 32, nil
	case gguf.Q2_K, gguf.Q3_K, gguf.Q4_K, gguf.Q5_K, gguf.Q6_K:
		return 256, nil
	}
	return 0, fmt.Errorf("row-slicing unsupported for ggml type %d", gg)
}

// quantSliceRows slices rows [r0, r1) out of a 2-D QuantTensor WITHOUT dequantizing — the
// primitive that lets [QuantPhi3FromGGUF] unpack packed projections losslessly. It is exact
// because ggml block quantization is row-granular: blocks run along the input dimension and
// never span rows (a row width that is not a block multiple is rejected — the same invariant
// gguf.QMatMul enforces when it consumes the weight as rows × rowBytes), so each row owns a
// disjoint, self-contained byte range of scale+quant blocks. Slicing is therefore a byte-range
// copy: bytes-per-row = cols/blockElems × blockBytes, and the result is bit-identical to
// quantizing the row range on its own. Dequantize→split→requantize would instead compound
// quantization error; it is never needed.
func quantSliceRows(qt gguf.QuantTensor, r0, r1 int) (gguf.QuantTensor, error) {
	if len(qt.Shape) != 2 {
		return gguf.QuantTensor{}, fmt.Errorf("row-slice needs a 2-D tensor, got shape %v", qt.Shape)
	}
	rows, cols := qt.Shape[0], qt.Shape[1]
	if r0 < 0 || r1 > rows || r0 >= r1 {
		return gguf.QuantTensor{}, fmt.Errorf("row-slice [%d,%d) outside [0,%d]", r0, r1, rows)
	}
	be, err := ggBlockElems(qt.GGType)
	if err != nil {
		return gguf.QuantTensor{}, err
	}
	if cols <= 0 || cols%be != 0 {
		return gguf.QuantTensor{}, fmt.Errorf("row width %d not a multiple of the %d-element block — blocks would span rows", cols, be)
	}
	totalBlocks := rows * (cols / be)
	if len(qt.Data)%totalBlocks != 0 {
		return gguf.QuantTensor{}, fmt.Errorf("%d data bytes not divisible into %d blocks", len(qt.Data), totalBlocks)
	}
	rowBytes := cols / be * (len(qt.Data) / totalBlocks)
	data := make([]byte, (r1-r0)*rowBytes)
	copy(data, qt.Data[r0*rowBytes:r1*rowBytes])
	return gguf.QuantTensor{Data: data, GGType: qt.GGType, Shape: tensor.Shape{r1 - r0, cols}}, nil
}
