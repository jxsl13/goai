package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantGPTNeoXFromGGUF builds a [QuantGPTNeoX] from the metadata and STILL-QUANTIZED
// tensor map of a GGUF file whose general.architecture is "gptneox" (gguf.ReadRaw) —
// the quantized twin of [GPTNeoXFromGGUF]: a llama.cpp-quantized GPT-NeoX / Pythia
// checkpoint loads straight into QuantLinear projections without ever materializing
// full-precision weights. The config comes from the same gptneox.* metadata keys as
// the float loader (rope.dimension_count → RotaryDim, use_parallel_residual=false
// REJECTED, epsilon under attention.layer_norm_epsilon) and the weights from the same
// token_embd / blk.N.* / output_norm / output tensor names.
//
// The gptneox conventions documented at [GPTNeoXFromGGUF] carry over unchanged to the
// quantized path:
//
//   - The fused attn_qkv is the converter's DE-INTERLEAVED layout: on disk the rows
//     are [all-q; all-k; all-v] with each section head-contiguous
//     (GPTNeoXModel.modify_tensors has already undone HF's per-head interleave), so
//     the quantized weight is unpacked WITHOUT dequantizing into plain thirds via
//     [quantSliceRows] at row offsets 0, dim, 2·dim — ggml blocks are row-granular,
//     so each third is bit-identical to quantizing the split projection directly —
//     and the packed F32 bias is sliced at the same offsets (slice1D), mirroring the
//     float loader. NOT [splitNeoXQKV], which would scramble every head.
//   - No q/k row permutation; PARTIAL split-half rotary from
//     gptneox.rope.dimension_count (absent → full rotary, llama.cpp's n_rot default).
//   - Full LayerNorm WITH bias: attn_norm / ffn_norm / output_norm are REQUIRED
//     weight+bias pairs. Both 1-D vectors are F32 in real quantized files (llama.cpp
//     never block-quantizes 1-D tensors) and are decoded to the f32 γ/β the quantized
//     forward's f32 activations need.
//   - EVERY projection is biased: attn_qkv, attn_output, ffn_up and ffn_down each
//     require weight AND bias. The biases stay F32 on disk and are decoded to f32 —
//     the §B quant-bias dtype discipline: the quantized residual stream is f32, so
//     the bias adds must be f32 too.
//   - The LM head is UNTIED and REQUIRED: output.weight must be present and in a
//     quantized-matmul format (no token_embd fallback for this architecture — the
//     same rule as the float loader). token_embd always feeds the f32 lookup table
//     via dequantization, whatever its storage type.
func QuantGPTNeoXFromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantGPTNeoX, error) {
	cfg, err := gptNeoXCfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: GPT-NeoX GGUF dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
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
	// decode a REQUIRED small 1-D tensor (bias / norm γ / norm β — F32 in real files) to f32.
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
	// quantized weight + f32 bias pair (every GPT-NeoX projection is biased).
	proj := func(name string) (*nn.QuantLinear, *tensor.Tensor, error) {
		qt, ok := tensors[name+".weight"]
		if !ok {
			return nil, nil, fmt.Errorf("nlp: GGUF missing %s.weight", name)
		}
		w, err := mkQ(name+".weight", qt)
		if err != nil {
			return nil, nil, err
		}
		b, err := mkVec(name + ".bias")
		if err != nil {
			return nil, nil, err
		}
		return w, b, nil
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

	q := &QuantGPTNeoX{Config: cfg, TokEmb: emb}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		qb := &QuantGPTNeoXBlock{}
		qkv, ok := tensors[p+"attn_qkv.weight"]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %sattn_qkv.weight", p)
		}
		// The converter's de-interleaved layout: rows [all-q; all-k; all-v], each
		// section head-contiguous — plain thirds, unpacked losslessly by quantized
		// row-slice; the F32 bias at the same offsets.
		if len(qkv.Shape) != 2 || qkv.Shape[0] != 3*dim {
			return nil, fmt.Errorf("nlp: GPT-NeoX GGUF %sattn_qkv.weight shape %v, want [%d, in] = [3·hidden, in]", p, qkv.Shape, 3*dim)
		}
		qkvB, err := mkVec(p + "attn_qkv.bias")
		if err != nil {
			return nil, err
		}
		if qkvB.Numel() != 3*dim {
			return nil, fmt.Errorf("nlp: GPT-NeoX GGUF %sattn_qkv.bias length %d != 3·hidden = %d", p, qkvB.Numel(), 3*dim)
		}
		for _, part := range []struct {
			w      **nn.QuantLinear
			b      **tensor.Tensor
			r0, r1 int
		}{
			{&qb.Wq, &qb.Bq, 0, dim},
			{&qb.Wk, &qb.Bk, dim, 2 * dim},
			{&qb.Wv, &qb.Bv, 2 * dim, 3 * dim},
		} {
			s, err := quantSliceRows(qkv, part.r0, part.r1)
			if err != nil {
				return nil, fmt.Errorf("nlp: GPT-NeoX GGUF %sattn_qkv.weight: %w", p, err)
			}
			if *part.w, err = mkQ(p+"attn_qkv.weight", s); err != nil {
				return nil, err
			}
			// f32Clone keeps the sliced bias at the residual stream's f32 dtype
			// (slice1D widens to f64; the values round-trip exactly).
			*part.b = f32Clone(slice1D(qkvB, part.r0, part.r1))
		}
		if qb.Wo, qb.Bo, err = proj(p + "attn_output"); err != nil {
			return nil, err
		}
		if qb.Wh, qb.Bh, err = proj(p + "ffn_up"); err != nil { // mlp.dense_h_to_4h
			return nil, err
		}
		if qb.Wout, qb.Bout, err = proj(p + "ffn_down"); err != nil { // mlp.dense_4h_to_h
			return nil, err
		}
		if qb.InputNorm, err = mkLN(p + "attn_norm"); err != nil {
			return nil, err
		}
		if qb.PostAttnNorm, err = mkLN(p + "ffn_norm"); err != nil {
			return nil, err
		}
		q.Blocks = append(q.Blocks, qb)
	}
	if q.FinalNorm, err = mkLN("output_norm"); err != nil {
		return nil, err
	}
	head, ok := tensors["output.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output.weight (the gptneox architecture's LM head is untied and required)")
	}
	if q.Out, err = mkQ("output.weight", head); err != nil {
		return nil, err
	}
	return q, nil
}
