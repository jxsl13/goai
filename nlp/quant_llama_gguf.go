package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// quantMatMulSupported reports whether a ggml type code is a block-quantized format that
// QuantLinear can run (gguf.QMatMul + the GPU kernels) — i.e. a genuine quantized projection.
func quantMatMulSupported(gg uint32) bool {
	switch gguf.QuantType(gg) {
	case gguf.Q8_0, gguf.Q4_0, gguf.Q2_K, gguf.Q3_K, gguf.Q4_K, gguf.Q5_K, gguf.Q6_K:
		return true
	}
	return false
}

// QuantLlamaFromGGUF builds a QuantLlama from the metadata and STILL-QUANTIZED tensor map of a
// GGUF file (gguf.ReadRaw, §T151) — loading a real quantized model straight onto the GPU without
// ever materializing full-precision weights. GGUF stores linear weights in the same [out, in]
// block layout QuantLinear expects, so each projection's bytes are wrapped directly — NO
// transpose, NO re-quantization. Only the small, precision-sensitive pieces are dequantized to
// f32: the RMSNorm gains and the token embedding (its lookup needs a float table). An absent
// output.weight ties the LM head to token_embd (which must itself be quantized). The quantized
// twin of [LlamaFromGGUF] — including the §B67 un-permute: the llama arch stores attn_q/attn_k
// in ggml's interleaved rotary row layout (the converter's permute), so the Q-block rows are
// reordered back into GoAI's split-half layout via [quantPermuteRows]/[ropeUnpermutePerm] —
// a lossless byte-range row permutation (T802), never a dequantize→requantize round-trip.
// Optional llama-arch q/k biases (F32 even in quantized files) are un-permuted the same
// way via [ropeUnpermuteVec], since the converter permutes q_proj.bias/k_proj.bias
// identically to the weights (§B67 follow-up).
func QuantLlamaFromGGUF(meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantLlama, error) {
	cfg, err := llamaCfgFromGGUFMeta("llama", meta)
	if err != nil {
		return nil, err
	}
	ts := make(map[string]gguf.QuantTensor, len(tensors))
	for k, v := range tensors {
		ts[k] = v
	}
	unpermute := func(name string, heads int) error {
		qt, ok := ts[name]
		if !ok || len(qt.Shape) != 2 {
			return nil
		}
		rows := qt.Shape[0]
		perm, err := ropeUnpermutePerm(rows, heads)
		if err != nil {
			return fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		out, err := quantPermuteRows(qt, perm)
		if err != nil {
			return fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		ts[name] = out
		return nil
	}
	for l := range cfg.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		if err := unpermute(p+"attn_q.weight", cfg.Heads); err != nil {
			return nil, err
		}
		if err := unpermute(p+"attn_k.weight", cfg.kvHeads()); err != nil {
			return nil, err
		}
	}
	m, err := quantLlamaArchFromGGUF("llama", meta, ts)
	if err != nil {
		return nil, err
	}
	// §B67 follow-up: llama.cpp's converter permutes q_proj.bias/k_proj.bias exactly
	// like the weights, so a bias-carrying llama-arch file (LLaMAfied-Qwen lineage)
	// stores its q/k biases interleaved too. They stay F32 in quantized files, so the
	// un-permute is a pure (lossless) float row reorder; v stays raw.
	unpermuteBias := func(name string, b *tensor.Tensor, heads int) (*tensor.Tensor, error) {
		if b == nil {
			return nil, nil
		}
		n := b.Shape()[0]
		if heads <= 0 || n%heads != 0 || (n/heads)%2 != 0 {
			return nil, fmt.Errorf("nlp: GGUF %s: length %d not heads %d × even head_dim", name, n, heads)
		}
		// Cast back: the quantized forward path applies biases in f32.
		return ropeUnpermuteVec(b, heads).Cast(b.Dtype()), nil
	}
	for l, b := range m.Blocks {
		p := fmt.Sprintf("blk.%d.", l)
		if b.Bq, err = unpermuteBias(p+"attn_q.bias", b.Bq, cfg.Heads); err != nil {
			return nil, err
		}
		if b.Bk, err = unpermuteBias(p+"attn_k.bias", b.Bk, cfg.kvHeads()); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// quantLlamaArchFromGGUF is the shared llama-family quantized-GGUF loader behind
// [QuantLlamaFromGGUF], [QuantQwen2FromGGUF], [QuantQwen3FromGGUF] and (after its row-slice
// unpacking) [QuantPhi3FromGGUF] — the quantized twin of
// llamaArchFromGGUF. The arch string selects the metadata key prefix; the block structure and
// tensor names are identical across the family, plus optional per-block tensors — attn_{q,k,v}.bias
// (Qwen2 family) and attn_{q,k}_norm.weight (Qwen3 per-head QK-norm) — dequantized to f32 when
// present (real quantized GGUFs store these 1-D tensors as F32; llama.cpp never block-quantizes
// them), nil (a forward no-op) otherwise.
func quantLlamaArchFromGGUF(arch string, meta map[string]any, tensors map[string]gguf.QuantTensor) (*QuantLlama, error) {
	cfg, err := llamaCfgFromGGUFMeta(arch, meta)
	if err != nil {
		return nil, err
	}
	layers := cfg.Layers

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
	// dequantize a 1-D RMSNorm gain to f32.
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

	// dequantize an OPTIONAL small 1-D tensor (Qwen bias / QK-norm gain) to f32; absent → nil.
	optF32 := func(name string) (*tensor.Tensor, error) {
		qt, ok := tensors[name]
		if !ok {
			return nil, nil
		}
		t, err := qt.Dequantize()
		if err != nil {
			return nil, fmt.Errorf("nlp: GGUF %s: %w", name, err)
		}
		return t, nil
	}

	tok, ok := tensors["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: GGUF missing token_embd.weight")
	}
	cfg.Vocab = tok.Shape[0]
	emb, err := tok.Dequantize() // embedding lookup needs a float table
	if err != nil {
		return nil, fmt.Errorf("nlp: GGUF token_embd.weight: %w", err)
	}

	q := &QuantLlama{Config: cfg, TokEmb: emb}
	for l := range layers {
		p := fmt.Sprintf("blk.%d.", l)
		qb := &QuantBlock{}
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
		// Optional q/k/v biases (1-D f32). Qwen2/Qwen3 store them unpermuted (§R93);
		// llama-arch files store q/k INTERLEAVED — [QuantLlamaFromGGUF] un-permutes
		// them after this shared build (§B67 follow-up).
		if qb.Bq, err = optF32(p + "attn_q.bias"); err != nil {
			return nil, err
		}
		if qb.Bk, err = optF32(p + "attn_k.bias"); err != nil {
			return nil, err
		}
		if qb.Bv, err = optF32(p + "attn_v.bias"); err != nil {
			return nil, err
		}
		// Optional Qwen3 per-head QK-norm gains ([headDim] each, f32).
		for name, dst := range map[string]**nn.RMSNorm{"attn_q_norm.weight": &qb.QNorm, "attn_k_norm.weight": &qb.KNorm} {
			g, err := optF32(p + name)
			if err != nil {
				return nil, err
			}
			if g != nil {
				*dst = &nn.RMSNorm{Gamma: g, Eps: cfg.Eps}
			}
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
		qb.FFN = &nn.QuantSwiGLU{Gate: gate, Up: up, Down: down}
		q.Blocks = append(q.Blocks, qb)
	}
	if q.Norm, err = mkNorm("output_norm.weight"); err != nil {
		return nil, err
	}
	head := "output.weight"
	if _, ok := tensors[head]; !ok {
		head = "token_embd.weight" // tied LM head
	}
	if q.Out, err = mkQ(head); err != nil {
		return nil, err
	}
	return q, nil
}
