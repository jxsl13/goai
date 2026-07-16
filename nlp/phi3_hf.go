package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// Phi3FromHF builds a [Llama] from a Microsoft Phi-3 Hugging Face checkpoint
// (transformers' Phi3ForCausalLM). Phi-3 IS a Llama — RMSNorm, RoPE, GQA, SwiGLU,
// no biases — except that it stores the attention and FFN input projections
// PACKED: one `qkv_proj` [heads·hd + 2·kv·hd, dim] instead of separate q/k/v, and
// one `gate_up_proj` [2·ffn, dim] instead of separate gate/up. This converter
// unpacks those two tensors along their rows into the standard per-projection
// names and delegates to [LlamaFromHF], so the whole loaded model — including the
// tied/untied head — is a plain Llama afterwards.
//
// cfg supplies Heads, KVHeads, Eps, RopeBase, Ctx (from config.json); Dim, Vocab,
// Layers and Hidden are inferred. Phi-3 uses head_dim = Dim/Heads.
func Phi3FromHF(ts map[string]*tensor.Tensor, cfg LlamaConfig) (*Llama, error) {
	tok, ok := ts["model.embed_tokens.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF Phi-3 missing model.embed_tokens.weight")
	}
	if tok.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: model.embed_tokens.weight must be 2-D, got %v", tok.Shape())
	}
	dim := tok.Shape()[1]
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: Phi3FromHF needs cfg.Heads")
	}
	kv := cfg.kvHeads()
	if dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: Phi-3 dim %d not divisible by heads %d", dim, cfg.Heads)
	}
	headDim := dim / cfg.Heads
	qSize, kvSize := cfg.Heads*headDim, kv*headDim

	// Copy the map so the caller's tensors are untouched, then replace the packed
	// projections with their unpacked halves.
	out := make(map[string]*tensor.Tensor, len(ts)+4*8)
	for k, v := range ts {
		out[k] = v
	}
	for l := 0; ; l++ {
		p := fmt.Sprintf("model.layers.%d.", l)
		qkv, ok := ts[p+"self_attn.qkv_proj.weight"]
		if !ok {
			if l == 0 {
				return nil, fmt.Errorf("nlp: HF Phi-3 has no model.layers.*.self_attn.qkv_proj.weight")
			}
			break
		}
		if got := qkv.Shape()[0]; got != qSize+2*kvSize {
			return nil, fmt.Errorf("nlp: Phi-3 layer %d qkv_proj rows %d != heads·hd+2·kv·hd = %d", l, got, qSize+2*kvSize)
		}
		delete(out, p+"self_attn.qkv_proj.weight")
		out[p+"self_attn.q_proj.weight"] = sliceRows(qkv, 0, qSize)
		out[p+"self_attn.k_proj.weight"] = sliceRows(qkv, qSize, qSize+kvSize)
		out[p+"self_attn.v_proj.weight"] = sliceRows(qkv, qSize+kvSize, qSize+2*kvSize)

		gu, ok := ts[p+"mlp.gate_up_proj.weight"]
		if !ok {
			return nil, fmt.Errorf("nlp: HF Phi-3 layer %d missing mlp.gate_up_proj.weight", l)
		}
		ffn := gu.Shape()[0] / 2
		delete(out, p+"mlp.gate_up_proj.weight")
		out[p+"mlp.gate_proj.weight"] = sliceRows(gu, 0, ffn)
		out[p+"mlp.up_proj.weight"] = sliceRows(gu, ffn, 2*ffn)
	}
	return LlamaFromHF(out, cfg)
}

// sliceRows returns rows [r0,r1) of a 2-D tensor as a fresh F64 tensor
// [r1-r0, cols], widening from any dtype. Used to unpack Phi-3's row-packed
// qkv_proj / gate_up_proj into their component projections.
func sliceRows(t *tensor.Tensor, r0, r1 int) *tensor.Tensor {
	cols := t.Shape()[1]
	src := cloneF64(t).Storage().F64() // [rows, cols] row-major
	out := tensor.New(tensor.F64, tensor.Shape{r1 - r0, cols})
	copy(out.Storage().F64(), src[r0*cols:r1*cols])
	return out
}
