package nlp

import (
	"fmt"
	"strings"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// LlamaGGUFQuantMix returns the per-tensor quantization map for the llama.cpp
// LLAMA_FTYPE_MOSTLY_Q4_K_M mix (§R101), given the GGUF tensor map from LlamaToGGUF. Pass the
// result to gguf.WriteQuantized to write a real ~4.5-bit Q4_K_M model:
//
//	meta, ts := nlp.LlamaToGGUF(m)
//	gguf.WriteQuantized(w, &gguf.File{Version: 3, Metadata: meta, Tensors: ts}, nlp.LlamaGGUFQuantMix(ts))
//
// The recipe (research-lite CONFIRMED unanimous vs llama-quant.cpp llama_tensor_get_type):
// base type Q4_K for every 2-D projection; output.weight and the use_more_bits layers of
// attn_v.weight / ffn_down.weight are upgraded to the higher-precision Q6_K (§R99); the
// 1-D RMSNorm gains and token_embd stay their natural type (norms F32, token_embd Q4_K).
//
// A k-quant super-block is 256 elements along a row, so a tensor whose GGUF row length
// (shape[1]) is not a multiple of 256 CANNOT be k-quantized; the helper leaves it out of the
// map (written F32) rather than producing an invalid file — real Llama geometries (Dim,
// Hidden multiples of 256) are always covered, tiny/odd ones degrade gracefully.
func LlamaGGUFQuantMix(ts map[string]*tensor.Tensor) map[string]gguf.QuantType {
	nLayer := 0
	for name := range ts {
		if strings.HasPrefix(name, "blk.") {
			if l, ok := blkLayer(name); ok && l+1 > nLayer {
				nLayer = l + 1
			}
		}
	}
	qm := make(map[string]gguf.QuantType)
	for name, t := range ts {
		if t.Ndim() != 2 || t.Shape()[1]%256 != 0 {
			continue // 1-D norms and non-256-aligned rows stay F32
		}
		qt := gguf.Q4_K
		switch {
		case name == "output.weight":
			qt = gguf.Q6_K
		case strings.HasSuffix(name, ".attn_v.weight"), strings.HasSuffix(name, ".ffn_down.weight"):
			if l, ok := blkLayer(name); ok && useMoreBits(l, nLayer) {
				qt = gguf.Q6_K
			}
		}
		qm[name] = qt
	}
	return qm
}

// useMoreBits is llama.cpp's per-layer "spend more bits here" predicate (§R101): the first
// 1/8 of layers, the last 1/8, and every third layer in between get the higher-precision
// type. The i<n/8 guard runs before the modulo, so (i−n/8) is never negative there.
func useMoreBits(i, n int) bool {
	return i < n/8 || i >= 7*n/8 || (i-n/8)%3 == 2
}

// blkLayer parses the layer index L from a "blk.L.<name>" GGUF tensor name.
func blkLayer(name string) (int, bool) {
	var l int
	if _, err := fmt.Sscanf(name, "blk.%d.", &l); err != nil {
		return 0, false
	}
	return l, true
}
