package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
)

// QuantGemmaFromGGUF builds a [QuantGemma] from the metadata and STILL-QUANTIZED tensor
// map of a GGUF file whose general.architecture is "gemma" (gguf.ReadRaw) — the
// quantized twin of [GemmaFromGGUF]: a llama.cpp-quantized Gemma (v1) checkpoint loads
// straight into QuantLinear projections without ever materializing full-precision
// weights. The config comes from the same gemma.* metadata keys as the float loader
// (including the decoupled per-head width, gemma.attention.key_length, with the same
// attn_q shape fallback), and the weights from the same token_embd / blk.N.* /
// output_norm tensor names.
//
// The four Gemma conventions documented at [GemmaFromGGUF] carry over unchanged to the
// quantized path:
//
//   - NO q/k row permutation (NEOX split-half RoPE on the stored layout): each 2-D
//     projection's Q-block bytes are wrapped directly as an [nn.QuantLinear] — GGUF's
//     [out, in] row layout is exactly what QuantLinear consumes. No transpose, no
//     re-quantization.
//   - Norm gains are stored PRE-FOLDED (the converter's γ+1): the 1-D *norm.weight
//     tensors — F32 in real quantized files; llama.cpp never block-quantizes 1-D
//     tensors — are decoded to f32 and used AS-IS. GoAI's in-memory (1+γ)-carrying
//     convention coincides with GGUF's on-disk one; folding again would be off by one.
//   - Embeddings are stored UNSCALED: the √dim "normalizer" is QuantGemma.Forward's
//     runtime job, applied to the residual stream only.
//   - Tied LM head: the gemma architecture has no output.weight (an unexpected one is
//     rejected, as in the float loader). token_embd.weight serves BOTH roles: its
//     Q-block bytes become the head (QuantLinear, logits = hidden · dequant(table)ᵀ)
//     and their dequantized f32 form the lookup table — so it must itself be stored in
//     a quantized-matmul format, which every llama.cpp-quantized gemma file satisfies
//     (the table is 2-D, and llama.cpp quantizes it like any other matrix).
func QuantGemmaFromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantGemma, error) {
	cfg, err := gemmaCfgFromGGUFMeta(meta)
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
		return nil, fmt.Errorf("nlp: Gemma GGUF has unexpected output.weight (the gemma architecture ties the LM head to token_embd)")
	}
	if !quantMatMulSupported(tok.GGType) {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight ggml type %d is not a quantized-matmul format (Gemma's tied LM head needs the quantized table)", tok.GGType)
	}
	// One tensor, two views: the Q-block bytes ARE the tied head; their dequantized
	// f32 form is the (unscaled) lookup table.
	emb, err := tok.Dequantize()
	if err != nil {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight: %w", err)
	}
	if cfg.HeadDim == 0 { // no attention.key_length key: derive from the attn_q row count
		if aq, ok := tensors["blk.0.attn_q.weight"]; ok && len(aq.Shape) == 2 && cfg.Heads > 0 {
			cfg.HeadDim = aq.Shape[0] / cfg.Heads
		}
	}

	q := &QuantGemma{
		Config: cfg,
		TokEmb: emb,
		Out:    &nn.QuantLinear{Weight: tok.Data, QT: gguf.QuantType(tok.GGType), In: tok.Shape[1], Out: tok.Shape[0]},
	}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		qb := &QuantGemmaBlock{}
		if qb.AttnNorm, err = mkNorm(p + "attn_norm.weight"); err != nil {
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
		if qb.FFNNorm, err = mkNorm(p + "ffn_norm.weight"); err != nil {
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
		q.Blocks = append(q.Blocks, qb)
	}
	if q.FinalNorm, err = mkNorm("output_norm.weight"); err != nil {
		return nil, err
	}
	return q, nil
}
