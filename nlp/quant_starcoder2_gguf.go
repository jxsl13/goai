package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantStarCoder2FromGGUF builds a [QuantStarCoder2] from the metadata and
// STILL-QUANTIZED tensor map of a GGUF file whose general.architecture is "starcoder2"
// (gguf.ReadRaw) — the quantized twin of [StarCoder2FromGGUF]: a llama.cpp-quantized
// StarCoder2 checkpoint loads straight into QuantLinear projections without ever
// materializing full-precision weights. The config comes from the same starcoder2.*
// metadata keys as the float loader (epsilon under attention.layer_norm_epsilon — the
// full-LayerNorm key, NOT the RMS one) and the weights from the same token_embd /
// blk.N.* / output_norm / output tensor names.
//
// The starcoder2 conventions documented at [StarCoder2FromGGUF] carry over unchanged to
// the quantized path:
//
//   - PURE RENAME, NO q/k row permutation (NEOX split-half RoPE on the stored layout):
//     each 2-D projection's Q-block bytes are wrapped directly as an [nn.QuantLinear] —
//     GGUF's [out, in] row layout is exactly what QuantLinear consumes. No transpose,
//     no re-quantization, no [unpermuteQuantRows].
//   - Full LayerNorm WITH bias: every attn_norm / ffn_norm / output_norm is a REQUIRED
//     weight+bias pair. Both 1-D vectors are F32 in real quantized files (llama.cpp
//     never block-quantizes 1-D tensors) and are decoded to the f32 γ/β the quantized
//     forward's f32 activations need.
//   - EVERY projection is biased (config.use_bias=True): attn_{q,k,v,output} and
//     ffn_up/ffn_down each require weight AND bias. The biases stay F32 on disk and are
//     decoded to f32 — the §B quant-bias dtype discipline: the quantized residual
//     stream is f32, so the bias adds must be f32 too.
//   - Packed attn_qkv accepted: like llama.cpp's create_tensor_qkv (and the float
//     loader), a fused blk.N.attn_qkv rows [q; k; v] replaces the split tensors. The
//     quantized weight is unpacked WITHOUT dequantizing via [quantSliceRows] — ggml
//     blocks are row-granular, so the row slice is bit-identical to quantizing the
//     split projections directly — and the packed F32 bias is sliced at the same
//     offsets (0, heads·hd, heads·hd + kv·hd), mirroring the float loader's slice1D.
//   - OPTIONAL output.weight with tied fallback: an absent output.weight ties the LM
//     head to token_embd (llama.cpp's TENSOR_DUPLICATED), which must then itself be
//     stored in a quantized-matmul format — the same convention as [QuantLlamaFromGGUF].
//     token_embd additionally always feeds the f32 lookup table via dequantization,
//     whatever its storage type.
func QuantStarCoder2FromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantStarCoder2, error) {
	cfg, err := starCoder2CfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: StarCoder2 GGUF dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	cfg.HeadDim = cfg.Dim / cfg.Heads // starcoder2 convention: no decoupled head_dim key
	qSize, kvSize := cfg.Heads*cfg.HeadDim, cfg.kvHeads()*cfg.HeadDim

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
	// quantized weight + f32 bias pair (every StarCoder2 projection is biased).
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

	q := &QuantStarCoder2{Config: cfg, TokEmb: emb}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		qb := &QuantStarCoder2Block{}
		if qkv, ok := tensors[p+"attn_qkv.weight"]; ok {
			// Fused form (create_tensor_qkv's alternative layout): rows [q; k; v],
			// unpacked losslessly by quantized row-slice; the F32 bias at the same offsets.
			if len(qkv.Shape) != 2 || qkv.Shape[0] != qSize+2*kvSize {
				return nil, fmt.Errorf("nlp: StarCoder2 GGUF %sattn_qkv.weight shape %v, want [%d, in] = [heads·hd+2·kv·hd, in]",
					p, qkv.Shape, qSize+2*kvSize)
			}
			qkvB, err := mkVec(p + "attn_qkv.bias")
			if err != nil {
				return nil, err
			}
			if qkvB.Numel() != qSize+2*kvSize {
				return nil, fmt.Errorf("nlp: StarCoder2 GGUF %sattn_qkv.bias length %d != heads·hd+2·kv·hd = %d",
					p, qkvB.Numel(), qSize+2*kvSize)
			}
			for _, part := range []struct {
				w      **nn.QuantLinear
				b      **tensor.Tensor
				r0, r1 int
			}{
				{&qb.Wq, &qb.Bq, 0, qSize},
				{&qb.Wk, &qb.Bk, qSize, qSize + kvSize},
				{&qb.Wv, &qb.Bv, qSize + kvSize, qSize + 2*kvSize},
			} {
				s, err := quantSliceRows(qkv, part.r0, part.r1)
				if err != nil {
					return nil, fmt.Errorf("nlp: StarCoder2 GGUF %sattn_qkv.weight: %w", p, err)
				}
				if *part.w, err = mkQ(p+"attn_qkv.weight", s); err != nil {
					return nil, err
				}
				// f32Clone keeps the sliced bias at the residual stream's f32 dtype
				// (slice1D widens to f64; the values round-trip exactly).
				*part.b = f32Clone(slice1D(qkvB, part.r0, part.r1))
			}
		} else {
			if qb.Wq, qb.Bq, err = proj(p + "attn_q"); err != nil {
				return nil, err
			}
			if qb.Wk, qb.Bk, err = proj(p + "attn_k"); err != nil {
				return nil, err
			}
			if qb.Wv, qb.Bv, err = proj(p + "attn_v"); err != nil {
				return nil, err
			}
		}
		if qb.Wo, qb.Bo, err = proj(p + "attn_output"); err != nil {
			return nil, err
		}
		if qb.Wfc, qb.Bfc, err = proj(p + "ffn_up"); err != nil { // mlp.c_fc
			return nil, err
		}
		if qb.Wproj, qb.Bproj, err = proj(p + "ffn_down"); err != nil { // mlp.c_proj
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
	if q.Norm, err = mkLN("output_norm"); err != nil {
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
