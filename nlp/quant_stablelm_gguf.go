package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantStableLMFromGGUF builds a [QuantStableLM] from the metadata and STILL-QUANTIZED
// tensor map of a GGUF file whose general.architecture is "stablelm" (gguf.ReadRaw) —
// the quantized twin of [StableLMFromGGUF]: a llama.cpp-quantized StableLM checkpoint
// loads straight into QuantLinear projections without ever materializing full-precision
// weights. The config comes from the same stablelm.* metadata keys as the float loader
// (shared [stableLMCfgFromGGUFMeta] — epsilon under attention.layer_norm_epsilon, the
// full-LayerNorm key; rope.dimension_count → RotaryDim, the ROTATED CHANNEL COUNT of
// the partial rotary; use_parallel_residual=true rejected) and the weights from the
// same token_embd / blk.N.* / output_norm / output tensor names.
//
// The stablelm conventions documented at [StableLMFromGGUF] carry over unchanged to the
// quantized path:
//
//   - PURE RENAME, NO q/k row permutation (NEOX split-half partial RoPE on the stored
//     layout): each 2-D projection's Q-block bytes are wrapped directly as an
//     [nn.QuantLinear] — GGUF's [out, in] row layout is exactly what QuantLinear
//     consumes. No transpose, no re-quantization, no [unpermuteQuantRows].
//   - Full LayerNorm WITH bias: attn_norm, ffn_norm and output_norm are each a REQUIRED
//     weight+bias pair. Both 1-D vectors are F32 in real quantized files (llama.cpp
//     never block-quantizes 1-D tensors) and decode to the f32 γ/β the quantized
//     forward's f32 activations need.
//   - SEQUENTIAL residual only: an absent blk.N.ffn_norm is llama.cpp's
//     parallel-residual signal (the StableLM 2 12B form — the graph reroutes the FFN
//     input to the attn_norm output), rejected with the same dedicated error as the
//     float loader rather than misloaded; use_parallel_residual=true in the metadata is
//     rejected by the shared cfg helper.
//   - NO qkv biases and NO per-head q/k norms: [QuantStableLM] models the
//     use_qkv_bias=false, qk_layernorm=false default. Any attn_{q,k,v,qkv}.bias
//     (StableLM 2 1.6B) or per-head-stacked attn_{q,k}_norm (StableLM 2 12B) present is
//     REJECTED rather than silently dropped — the float loader's exact reject set.
//   - Packed attn_qkv accepted: like llama.cpp's create_tensor_qkv (and the float
//     loader), a fused blk.N.attn_qkv rows [q; k; v] replaces the split tensors. The
//     quantized weight is unpacked WITHOUT dequantizing via [quantSliceRows] at the
//     float loader's row offsets (0, dim, dim + kv·headDim) — ggml blocks are
//     row-granular, so the row slice is bit-identical to quantizing the split
//     projections directly.
//   - UNTIED head: output.weight is REQUIRED (no tied-embedding fallback in the
//     stablelm architecture) and must be stored in a quantized-matmul format.
//     token_embd feeds the f32 lookup table via dequantization, whatever its storage
//     type (real quantized files may keep it F16/F32 or quantized).
func QuantStableLMFromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantStableLM, error) {
	cfg, err := stableLMCfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: StableLM GGUF dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	cfg.HeadDim = cfg.Dim / cfg.Heads // create_tensor_qkv(n_embd, n_embd, ...): square q_proj, no decoupled head width
	dim, kvSize := cfg.Dim, cfg.kvHeads()*cfg.HeadDim

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
	// full LayerNorm from a REQUIRED weight+bias pair — the quantized twin of [layerNormFromGGUF].
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
	emb, err := tok.Dequantize() // embedding lookup needs a float (f32) table
	if err != nil {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight: %w", err)
	}

	q := &QuantStableLM{Config: cfg, TokEmb: emb}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		// Reject the optional variants GoAI's StableLM cannot represent (llama.cpp
		// loads all of these as TENSOR_NOT_REQUIRED and would apply them) — the float
		// loader's exact reject set.
		for _, unsupported := range []string{
			"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_qkv.bias", // use_qkv_bias=true (StableLM 2 1.6B)
			"attn_q_norm.weight", "attn_k_norm.weight", // per-head QK-LayerNorms (StableLM 2 12B)
		} {
			if _, ok := tensors[p+unsupported]; ok {
				return nil, fmt.Errorf("nlp: StableLM GGUF carries %s%s; GoAI's StableLM implements only the use_qkv_bias=false, qk_layernorm=false form", p, unsupported)
			}
		}
		// An absent ffn_norm is llama.cpp's parallel-residual signal (StableLM 2 12B):
		// the graph reroutes the FFN input to the attn_norm output. Reject, don't misload.
		if _, ok := tensors[p+"ffn_norm.weight"]; !ok {
			return nil, fmt.Errorf("nlp: StableLM GGUF has no %sffn_norm.weight — the parallel-residual StableLM form; GoAI's StableLM implements only the sequential two-norm block", p)
		}
		qb := &QuantStableLMBlock{}
		if qkv, ok := tensors[p+"attn_qkv.weight"]; ok {
			// Fused form (create_tensor_qkv's alternative layout): rows [q; k; v],
			// unpacked losslessly by quantized row-slice at the float loader's offsets.
			if len(qkv.Shape) != 2 || qkv.Shape[0] != dim+2*kvSize {
				return nil, fmt.Errorf("nlp: StableLM GGUF %sattn_qkv.weight shape %v, want [%d, in] = [dim+2·kv·hd, in]",
					p, qkv.Shape, dim+2*kvSize)
			}
			for _, part := range []struct {
				w      **nn.QuantLinear
				r0, r1 int
			}{
				{&qb.Wq, 0, dim},
				{&qb.Wk, dim, dim + kvSize},
				{&qb.Wv, dim + kvSize, dim + 2*kvSize},
			} {
				s, err := quantSliceRows(qkv, part.r0, part.r1)
				if err != nil {
					return nil, fmt.Errorf("nlp: StableLM GGUF %sattn_qkv.weight: %w", p, err)
				}
				if *part.w, err = mkQ(p+"attn_qkv.weight", s); err != nil {
					return nil, err
				}
			}
		} else {
			proj := func(name string) (*nn.QuantLinear, error) {
				qt, ok := tensors[p+name]
				if !ok {
					return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
				}
				return mkQ(p+name, qt)
			}
			if qb.Wq, err = proj("attn_q.weight"); err != nil {
				return nil, err
			}
			if qb.Wk, err = proj("attn_k.weight"); err != nil {
				return nil, err
			}
			if qb.Wv, err = proj("attn_v.weight"); err != nil {
				return nil, err
			}
		}
		proj := func(name string) (*nn.QuantLinear, error) {
			qt, ok := tensors[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: GGUF missing %s%s", p, name)
			}
			return mkQ(p+name, qt)
		}
		if qb.Wo, err = proj("attn_output.weight"); err != nil {
			return nil, err
		}
		gate, err := proj("ffn_gate.weight")
		if err != nil {
			return nil, err
		}
		up, err := proj("ffn_up.weight")
		if err != nil {
			return nil, err
		}
		down, err := proj("ffn_down.weight")
		if err != nil {
			return nil, err
		}
		qb.FFN = &nn.QuantSwiGLU{Gate: gate, Up: up, Down: down}
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
	head, ok := tensors["output.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output.weight (the stablelm architecture's LM head is untied and required)")
	}
	if q.Out, err = mkQ("output.weight", head); err != nil {
		return nil, err
	}
	return q, nil
}
