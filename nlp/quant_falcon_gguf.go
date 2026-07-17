package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantFalconFromGGUF builds a [QuantFalcon] from the metadata and STILL-QUANTIZED
// tensor map of a GGUF file whose general.architecture is "falcon" (gguf.ReadRaw) —
// the quantized twin of [FalconFromGGUF]: a llama.cpp-quantized Falcon-7B checkpoint
// loads straight into QuantLinear projections without ever materializing
// full-precision weights. The config comes from the same falcon.* metadata keys as
// the float loader (head_count_kv must resolve to 1 — the multi-query Falcon-7B form
// is the only one this type models; epsilon under attention.layer_norm_epsilon) and
// the weights from the same token_embd / blk.N.* / output_norm / output tensor names.
//
// The falcon conventions documented at [FalconFromGGUF] carry over unchanged to the
// quantized path:
//
//   - The fused attn_qkv is stored in the "jploski" [all-q; all-k; all-v] band layout,
//     which for the MQA case (n_head_kv=1, ONE kv group) is the IDENTITY of HF's fused
//     layout: q = rows [0, heads·hd), k = rows [heads·hd, (heads+1)·hd) (ONE head),
//     v = rows [(heads+1)·hd, (heads+2)·hd) (ONE head). The quantized weight is
//     unpacked WITHOUT dequantizing via [quantSliceRows] at exactly those offsets —
//     ggml blocks are row-granular, so each band is bit-identical to quantizing the
//     split projection directly. There is no qkv bias (query_key_value and dense are
//     both bias-free), and no projection bias anywhere else either — but the
//     LayerNorms DO carry F32 γ+β pairs, decoded to the f32 the quantized forward's
//     f32 activations need (llama.cpp never block-quantizes 1-D tensors).
//   - A file carrying blk.N.attn_norm_2 (the Falcon-40B dual-norm architecture) is
//     REJECTED, exactly like the float loader; attn_norm is the ONE per-block norm
//     feeding both sublayers.
//   - Full NEOX rotary over the whole head_dim; rope.freq_base optional (default
//     10000).
//   - OPTIONAL output.weight with tied fallback: an absent output.weight ties the LM
//     head to token_embd (llama.cpp's TENSOR_DUPLICATED), which must then itself be
//     stored in a quantized-matmul format — the same convention as
//     [QuantLlamaFromGGUF]. token_embd additionally always feeds the f32 lookup table
//     via dequantization, whatever its storage type.
func QuantFalconFromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantFalcon, error) {
	cfg, err := falconCfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: Falcon GGUF dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	headDim := cfg.Dim / cfg.Heads
	qSize, kvSize := cfg.Heads*headDim, cfg.kvHeads()*headDim

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
	// look up a REQUIRED quantized projection weight.
	projW := func(name string) (*nn.QuantLinear, error) {
		qt, ok := tensors[name+".weight"]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %s.weight", name)
		}
		return mkQ(name+".weight", qt)
	}
	// decode a REQUIRED small 1-D tensor (norm γ / norm β — F32 in real files) to f32.
	mkVec := func(name string) (*tensor.Tensor, error) {
		qt, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %s", name)
		}
		v, err := qt.Dequantize()
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		return v, nil
	}
	// full LayerNorm from a weight+bias pair — the quantized twin of [layerNormFromGGUF].
	mkLN := func(prefix string) (*nn.LayerNorm, error) {
		g, err := mkVec(prefix + ".weight")
		if err != nil {
			return nil, err
		}
		b, err := mkVec(prefix + ".bias")
		if err != nil {
			return nil, err
		}
		return &nn.LayerNorm{Gamma: g, Beta: b, Eps: cfg.Eps}, nil
	}

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	if len(tok.Shape) != 2 {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight must be 2-D, got %v", tok.Shape)
	}
	cfg.Vocab = tok.Shape[0]
	emb, err := tok.Dequantize() // embedding lookup needs a float table
	if err != nil {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight: %w", err)
	}

	q := &QuantFalcon{Config: cfg, TokEmb: emb}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		if _, ok := tensors[p+"attn_norm_2.weight"]; ok {
			return nil, fmt.Errorf("nlp: Falcon GGUF carries %sattn_norm_2.weight (the Falcon-40B dual-norm architecture); GoAI's Falcon implements only the single-norm parallel-residual Falcon-7B form", p)
		}
		qkv, ok := tensors[p+"attn_qkv.weight"]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %sattn_qkv.weight", p)
		}
		// "jploski" band layout: [all-q; k; v] (MQA one-group identity) — unpacked
		// losslessly by quantized row-slice at the FromHF band offsets.
		if len(qkv.Shape) != 2 || qkv.Shape[0] != qSize+2*kvSize {
			return nil, fmt.Errorf("nlp: Falcon GGUF %sattn_qkv.weight shape %v, want [%d, in] = [(heads+2)·head_dim, in]", p, qkv.Shape, qSize+2*kvSize)
		}
		qb := &QuantFalconBlock{}
		for _, part := range []struct {
			w      **nn.QuantLinear
			r0, r1 int
		}{
			{&qb.Wq, 0, qSize},
			{&qb.Wk, qSize, qSize + kvSize},
			{&qb.Wv, qSize + kvSize, qSize + 2*kvSize},
		} {
			s, err := quantSliceRows(qkv, part.r0, part.r1)
			if err != nil {
				return nil, fmt.Errorf("nlp: Falcon GGUF %sattn_qkv.weight: %w", p, err)
			}
			if *part.w, err = mkQ(p+"attn_qkv.weight", s); err != nil {
				return nil, err
			}
		}
		if qb.Wo, err = projW(p + "attn_output"); err != nil { // self_attention.dense
			return nil, err
		}
		if qb.Wh, err = projW(p + "ffn_up"); err != nil { // mlp.dense_h_to_4h
			return nil, err
		}
		if qb.Wout, err = projW(p + "ffn_down"); err != nil { // mlp.dense_4h_to_h
			return nil, err
		}
		if qb.InputNorm, err = mkLN(p + "attn_norm"); err != nil { // input_layernorm, γ AND β
			return nil, err
		}
		q.Blocks = append(q.Blocks, qb)
	}
	if q.FinalNorm, err = mkLN("output_norm"); err != nil { // ln_f, γ AND β
		return nil, err
	}
	head, headName := tok, "token_embd.weight" // TENSOR_DUPLICATED fallback: tied head
	if o, ok := tensors["output.weight"]; ok {
		head, headName = o, "output.weight"
	}
	if q.Out, err = mkQ(headName, head); err != nil {
		return nil, err
	}
	return q, nil
}
