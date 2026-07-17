package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantOLMo2FromGGUF builds a [QuantOLMo2] from the metadata and STILL-QUANTIZED
// tensor map of a GGUF file whose general.architecture is "olmo2" (gguf.ReadRaw) — the
// quantized twin of [OLMo2FromGGUF]: a llama.cpp-quantized OLMo 2 checkpoint loads
// straight into QuantLinear projections without ever materializing full-precision
// weights. The config comes from the same olmo2.* metadata keys as the float loader
// (shared [olmo2CfgFromGGUFMeta] — epsilon under attention.layer_norm_rms_epsilon, the
// RMS key; sliding-window OLMo 3 files rejected) and the weights from the same
// token_embd / blk.N.* / output_norm / output tensor names.
//
// The olmo2 conventions documented at [OLMo2FromGGUF] carry over unchanged to the
// quantized path:
//
//   - PURE RENAME, NO q/k row permutation (NEOX split-half RoPE on the stored layout):
//     each 2-D projection's Q-block bytes are wrapped directly as an [nn.QuantLinear]
//     — GGUF's [out, in] row layout is exactly what QuantLinear consumes. No
//     transpose, no re-quantization, no [unpermuteQuantRows].
//   - POST-norm names on disk: blk.N.post_attention_norm and blk.N.post_ffw_norm (the
//     ffw contraction) are REQUIRED weight-only RMSNorms applied to the sublayer
//     OUTPUT before the residual add. There is NO attn_norm/ffn_norm (no input
//     norms). All norm gains are F32 in real quantized files (llama.cpp never
//     block-quantizes 1-D tensors) and decode to the f32 gains the quantized
//     forward's f32 activations need.
//   - FULL-WIDTH QK-norm before RoPE: blk.N.attn_q_norm.weight {n_embd} and
//     blk.N.attn_k_norm.weight {n_head_kv · n_embd_head} are REQUIRED f32 RMSNorms
//     applied to the WHOLE q/k projection before the head split. Both widths are
//     shape-checked here exactly as in the float loader; a per-head-sized norm (a
//     Qwen3-style file) is rejected rather than broadcast wrong.
//   - NO biases anywhere: Olmo2ForCausalLM has none and the olmo2 graph applies none —
//     a file carrying attn_{q,k,v,qkv,output}.bias (llama.cpp would load-and-drop) is
//     REJECTED rather than misloaded, the float loader's exact reject set.
//   - Packed attn_qkv accepted: like llama.cpp's create_tensor_qkv (and the float
//     loader), a fused blk.N.attn_qkv rows [q; k; v] replaces the split tensors. The
//     quantized weight is unpacked WITHOUT dequantizing via [quantSliceRows] at the
//     float loader's row offsets (0, dim, dim + kv·headDim) — ggml blocks are
//     row-granular, so the row slice is bit-identical to quantizing the split
//     projections directly.
//   - UNTIED head: output.weight is REQUIRED (no tied-embedding fallback in the olmo2
//     architecture; the converter materializes the tied 1B's head) and must be stored
//     in a quantized-matmul format. token_embd feeds the f32 lookup table via
//     dequantization, whatever its storage type (real quantized files may keep it
//     F16/F32 or quantized).
//   - COUPLED head width: like the float loader, HeadDim = Dim/Heads (llama.cpp's
//     olmo2 graph asserts n_rot == n_embd/n_head; the converter writes no
//     rope.dimension_count). The decoupled-HeadDim float form exists only on the HF
//     path and quantizes via [QuantizeOLMo2].
func QuantOLMo2FromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantOLMo2, error) {
	cfg, err := olmo2CfgFromGGUFMeta(meta)
	if err != nil {
		return nil, err
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: OLMo2 GGUF dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	cfg.HeadDim = cfg.Dim / cfg.Heads // create_tensor_qkv(n_embd, n_embd, ...): square q_proj; olmo2 asserts n_rot == n_embd/n_head
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
	// decode a REQUIRED small 1-D tensor (norm gain — F32 in real files) to f32.
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
	// weight-only RMSNorm from a REQUIRED gain — the quantized twin of [rmsFromGGUF].
	mkRMS := func(name string) (*nn.RMSNorm, error) {
		g, err := mkVec(name)
		if err != nil {
			return nil, err
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
	emb, err := tok.Dequantize() // embedding lookup needs a float (f32) table
	if err != nil {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight: %w", err)
	}

	q := &QuantOLMo2{Config: cfg, TokEmb: emb}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		// Olmo2ForCausalLM has no biases and the olmo2 graph applies none — a file
		// carrying them (llama.cpp would load-and-drop) is rejected, not misloaded.
		for _, unsupported := range []string{"attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_qkv.bias", "attn_output.bias"} {
			if _, ok := tensors[p+unsupported]; ok {
				return nil, fmt.Errorf("nlp: OLMo2 GGUF carries %s%s; the olmo2 architecture is bias-free", p, unsupported)
			}
		}
		qb := &QuantOLMo2Block{}
		if qkv, ok := tensors[p+"attn_qkv.weight"]; ok {
			// Fused form (create_tensor_qkv's alternative layout): rows [q; k; v],
			// unpacked losslessly by quantized row-slice at the float loader's offsets.
			if len(qkv.Shape) != 2 || qkv.Shape[0] != dim+2*kvSize {
				return nil, fmt.Errorf("nlp: OLMo2 GGUF %sattn_qkv.weight shape %v, want [%d, in] = [dim+2·kv·hd, in]",
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
					return nil, fmt.Errorf("nlp: OLMo2 GGUF %sattn_qkv.weight: %w", p, err)
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
		if qb.QNorm, err = mkRMS(p + "attn_q_norm.weight"); err != nil {
			return nil, err
		}
		if qb.KNorm, err = mkRMS(p + "attn_k_norm.weight"); err != nil {
			return nil, err
		}
		// Full-width QK-norm shape gate: {n_embd} / {n_head_kv·n_embd_head} exactly
		// (src/models/olmo2.cpp) — a per-head-sized norm would broadcast wrong.
		if got := qb.QNorm.Gamma.Numel(); got != dim {
			return nil, fmt.Errorf("nlp: OLMo2 GGUF %sattn_q_norm.weight has %d elements, want full-width %d (per-head QK-norms are not the olmo2 convention)", p, got, dim)
		}
		if got := qb.KNorm.Gamma.Numel(); got != kvSize {
			return nil, fmt.Errorf("nlp: OLMo2 GGUF %sattn_k_norm.weight has %d elements, want full-width %d (per-head QK-norms are not the olmo2 convention)", p, got, kvSize)
		}
		if qb.PostAttnNorm, err = mkRMS(p + "post_attention_norm.weight"); err != nil {
			return nil, err
		}
		if qb.PostFFNNorm, err = mkRMS(p + "post_ffw_norm.weight"); err != nil {
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
		q.Blocks = append(q.Blocks, qb)
	}
	if q.Norm, err = mkRMS("output_norm.weight"); err != nil {
		return nil, err
	}
	head, ok := tensors["output.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing output.weight (the olmo2 architecture's LM head is untied and required)")
	}
	if q.Out, err = mkQ("output.weight", head); err != nil {
		return nil, err
	}
	return q, nil
}
