package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
)

// QuantGemma2FromGGUF builds a [QuantGemma2] from the metadata and STILL-QUANTIZED
// tensor map of a GGUF file whose general.architecture is "gemma2" (gguf.ReadRaw) —
// the quantized twin of [Gemma2FromGGUF]: a llama.cpp-quantized Gemma 2 checkpoint
// loads straight into QuantLinear projections without ever materializing
// full-precision weights. The config comes from the same gemma2.* metadata keys via
// the shared parser, so every float-loader convention is inherited: absent soft-cap
// keys default to llama.cpp's 50/30 (an explicit 0 disables that cap), Ctx is clamped
// to min(context_length, sliding_window) (full attention ≡ the alternating-window
// graph for every accepted prompt), and QueryPreAttnScalar — which has NO GGUF key —
// is re-derived by llama.cpp's block_count rule (46 layers = the 27B → Dim/Heads,
// else the per-head width).
//
// The Gemma 2 tensor conventions documented at [Gemma2FromGGUF] carry over unchanged
// to the quantized path:
//
//   - Sandwich-norm names on disk: attn_norm / post_attention_norm / ffn_norm (the
//     pre-FFN norm rides the plain llama name) / post_ffw_norm (the "ffw"
//     contraction). All four gains plus output_norm are stored PRE-FOLDED (the
//     converter's γ+1) as F32 1-D tensors — llama.cpp never block-quantizes 1-D
//     tensors — and are decoded to f32 and used AS-IS; folding again would be off by
//     one.
//   - NO q/k row permutation (NEOX split-half RoPE on the stored layout): each 2-D
//     projection's Q-block bytes are wrapped directly as an [nn.QuantLinear] — GGUF's
//     [out, in] row layout is exactly what QuantLinear consumes. No transpose, no
//     re-quantization.
//   - Embeddings are stored UNSCALED: the √dim "normalizer" is QuantGemma2.Forward's
//     runtime job, applied to the residual stream only.
//   - Tied LM head: the gemma2 architecture has no output.weight (an unexpected one
//     is rejected, as in the float loader). token_embd.weight serves BOTH roles: its
//     Q-block bytes become the head (QuantLinear, logits = hidden · dequant(table)ᵀ)
//     and their dequantized f32 form the lookup table — so it must itself be stored
//     in a quantized-matmul format, which every llama.cpp-quantized gemma2 file
//     satisfies (the table is 2-D, quantized like any other matrix).
//   - Attention biases and a fused attn_qkv are rejected, mirroring the float loader:
//     Gemma2ForCausalLM is bias-free and the converter never emits either form.
//   - Decoupled head width: gemma2.attention.key_length/value_length carry head_dim
//     (disagreeing lengths are rejected by the shared parser); when absent, HeadDim
//     falls back to the blk.0.attn_q row count divided by head_count.
func QuantGemma2FromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantGemma2, error) {
	cfg, err := gemma2CfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}

	// wrap a GGUF projection (quantized [out, in]) as a QuantLinear — no transpose, no requant.
	mkQ := func(name string) (*nn.QuantLinear, error) {
		qt, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %s", name)
		}
		if len(qt.Shape) != 2 {
			return nil, fmt.Errorf("nlp: GGUF %s must be 2-D, got %v", name, qt.Shape)
		}
		if !quantMatMulSupported(qt.GGType) {
			return nil, fmt.Errorf("nlp: GGUF %s ggml type %d is not a quantized-matmul format", name, qt.GGType)
		}
		return &nn.QuantLinear{Weight: qt.Data, QT: gguf.QuantType(qt.GGType), In: qt.Shape[1], Out: qt.Shape[0]}, nil
	}
	// dequantize a 1-D RMSNorm gain to f32 — used AS-IS (pre-folded +1, see above).
	mkNorm := func(name string) (*nn.RMSNorm, error) {
		qt, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("nlp: GGUF missing %s", name)
		}
		g, err := qt.Dequantize()
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		return &nn.RMSNorm{Gamma: g, Eps: cfg.Eps}, nil
	}

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	if len(tok.Shape) != 2 {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight must be 2-D, got %v", tok.Shape)
	}
	cfg.Vocab = tok.Shape[0]
	if _, ok := tensors["output.weight"]; ok {
		return nil, fmt.Errorf("nlp: Gemma2 GGUF has unexpected output.weight (the gemma2 architecture ties the LM head to token_embd)")
	}
	if !quantMatMulSupported(tok.GGType) {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight ggml type %d is not a quantized-matmul format (Gemma 2's tied LM head needs the quantized table)", tok.GGType)
	}
	// One tensor, two views: the Q-block bytes ARE the tied head; their dequantized
	// f32 form is the (unscaled) lookup table.
	emb, err := tok.Dequantize()
	if err != nil {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight: %w", err)
	}
	if cfg.HeadDim == 0 { // no attention.key_length key: derive from the attn_q row count
		// Same fallback arithmetic as the float twin ([gemmaHeadDimFromRows]), so the
		// two paths accept and reject the same files; only the rank test differs in
		// idiom (len(Shape) here, Ndim there).
		if aq, ok := tensors["blk.0.attn_q.weight"]; ok && len(aq.Shape) == 2 {
			if cfg.HeadDim, err = gemmaHeadDimFromRows("blk.0.attn_q.weight", aq.Shape[0], cfg.Heads); err != nil {
				return nil, err
			}
		}
	}
	// GGUF has no query_pre_attn_scalar key; mirror llama.cpp's block_count-keyed
	// derivation (src/models/gemma2.cpp): 46 layers = the 27B → dim/heads, else the
	// per-head width (key_length).
	if cfg.Layers == 46 {
		cfg.QueryPreAttnScalar = float64(cfg.Dim) / float64(cfg.Heads)
	} else {
		cfg.QueryPreAttnScalar = float64(cfg.HeadDim)
	}

	q := &QuantGemma2{
		Config: cfg,
		TokEmb: emb,
		Out:    &nn.QuantLinear{Weight: tok.Data, QT: gguf.QuantType(tok.GGType), In: tok.Shape[1], Out: tok.Shape[0]},
	}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		// Gemma2ForCausalLM is bias-free and the converter writes split q/k/v — a
		// file carrying biases or a fused qkv is hand-crafted; reject, don't misload.
		for _, unsupported := range []string{"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias", "attn_qkv.weight"} {
			if _, ok := tensors[p+unsupported]; ok {
				return nil, fmt.Errorf("nlp: Gemma2 GGUF carries %s%s; the gemma2 architecture is bias-free with split q/k/v", p, unsupported)
			}
		}
		qb := &QuantGemma2Block{}
		if qb.InputNorm, err = mkNorm(p + "attn_norm.weight"); err != nil {
			return nil, err
		}
		if qb.Wq, err = mkQ(p + "attn_q.weight"); err != nil {
			return nil, err
		}
		if qb.Wk, err = mkQ(p + "attn_k.weight"); err != nil {
			return nil, err
		}
		if qb.Wv, err = mkQ(p + "attn_v.weight"); err != nil {
			return nil, err
		}
		if qb.Wo, err = mkQ(p + "attn_output.weight"); err != nil {
			return nil, err
		}
		if qb.PostAttnNorm, err = mkNorm(p + "post_attention_norm.weight"); err != nil {
			return nil, err
		}
		if qb.PreFFNNorm, err = mkNorm(p + "ffn_norm.weight"); err != nil {
			return nil, err
		}
		gate, err := mkQ(p + "ffn_gate.weight")
		if err != nil {
			return nil, err
		}
		up, err := mkQ(p + "ffn_up.weight")
		if err != nil {
			return nil, err
		}
		down, err := mkQ(p + "ffn_down.weight")
		if err != nil {
			return nil, err
		}
		qb.FFN = &QuantGeGLU{Gate: gate, Up: up, Down: down}
		if qb.PostFFNNorm, err = mkNorm(p + "post_ffw_norm.weight"); err != nil {
			return nil, err
		}
		q.Blocks = append(q.Blocks, qb)
	}
	if q.FinalNorm, err = mkNorm("output_norm.weight"); err != nil {
		return nil, err
	}
	return q, nil
}
