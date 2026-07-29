package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantLlama is a Llama that keeps its projection weights QUANTIZED (nn.QuantLinear) and never
// materializes them as full-precision matrices — the memory-efficient, GPU-accelerated form of
// a quantized model (§T149). Its forward is identical to Llama.ForwardFromEmbed except every
// linear projection (attn q/k/v/o, the FFN, the output head) is an in-kernel dequantized matmul
// that runs on the active accelerator when it accelerates the quant type (else the CPU
// fallback). Everything runs in f32 — quantized-inference activations — which is also what lets
// the GPU ops engage (the accelerators are f32-only). Inference-only: the quantized weights are
// frozen and bypass the tape.
type QuantLlama struct {
	Config LlamaConfig     // model hyperparameters
	TokEmb *tensor.Tensor  // [vocab, dim] f32 token embedding
	Blocks []*QuantBlock   // stacked transformer blocks with quantized projections
	Norm   *nn.RMSNorm     // final pre-logits RMSNorm (f32 gain)
	Out    *nn.QuantLinear // quantized output projection (In=dim, Out=vocab)
}

// QuantBlock is a QuantLlama transformer block: float RMSNorm gains, quantized attention
// projections and a quantized SwiGLU FFN. The optional Qwen extensions mirror LlamaBlock:
// q/k/v projection biases (Qwen2/Qwen2.5) and per-head QK-norm gains (Qwen3) stay in f32 —
// they are tiny 1-D tensors that real quantized GGUFs also keep unquantized (llama.cpp only
// block-quantizes 2-D matrices). All are nil for plain Llama/Mistral, which keeps the
// forward byte-identical to the pre-Qwen quantized path.
type QuantBlock struct {
	AttnNorm       *nn.RMSNorm     // RMSNorm before attention (f32 gain)
	Wq, Wk, Wv, Wo *nn.QuantLinear // quantized attention projections
	Bq, Bk, Bv     *tensor.Tensor  // optional f32 q/k/v projection biases (Qwen2 family); nil otherwise
	QNorm, KNorm   *nn.RMSNorm     // optional per-head q/k RMSNorm before RoPE (Qwen3, f32 gains); nil otherwise
	FFNNorm        *nn.RMSNorm     // RMSNorm before the FFN (f32 gain)
	FFN            *nn.QuantSwiGLU // quantized SwiGLU feed-forward
}

// QuantizeLlama builds a QuantLlama from a float Llama by quantizing every linear projection to
// qt (the ggml block layout) — the projections carry the bulk of the weights and compute, so
// this is where quantization pays off. RMSNorm gains and the token embedding are kept in f32
// (they are tiny and precision-sensitive). Each projection's inner dimension must be a multiple
// of qt's block size (32 for Q8_0/Q4_0, 256 for the k-quants).
func QuantizeLlama(m *Llama, qt gguf.QuantType) (*QuantLlama, error) {
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		in, out := w.Shape()[0], w.Shape()[1] // GoAI [in, out]
		bytes, err := gguf.Quantize(transpose2D(w), qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: bytes, QT: qt, In: in, Out: out}, nil
	}
	q := &QuantLlama{Config: m.Config, TokEmb: f32Clone(m.TokEmb), Norm: f32RMSNorm(m.Norm)}
	for _, b := range m.Blocks {
		qb := &QuantBlock{AttnNorm: f32RMSNorm(b.AttnNorm), FFNNorm: f32RMSNorm(b.FFNNorm)}
		// Optional Qwen extensions ride along in f32 (tiny, precision-sensitive — never quantized).
		qb.Bq, qb.Bk, qb.Bv = f32CloneIf(b.Bq), f32CloneIf(b.Bk), f32CloneIf(b.Bv)
		if b.QNorm != nil {
			qb.QNorm = f32RMSNorm(b.QNorm)
		}
		if b.KNorm != nil {
			qb.KNorm = f32RMSNorm(b.KNorm)
		}
		var err error
		if qb.Wq, err = mkQ(b.Wq); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeLlama Wq: %w", err)
		}
		if qb.Wk, err = mkQ(b.Wk); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeLlama Wk: %w", err)
		}
		if qb.Wv, err = mkQ(b.Wv); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeLlama Wv: %w", err)
		}
		if qb.Wo, err = mkQ(b.Wo); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeLlama Wo: %w", err)
		}
		gate, err := mkQ(b.FFN.Wgate)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeLlama ffn_gate: %w", err)
		}
		up, err := mkQ(b.FFN.Wup)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeLlama ffn_up: %w", err)
		}
		down, err := mkQ(b.FFN.Wdown)
		if err != nil {
			return nil, fmt.Errorf("nlp: QuantizeLlama ffn_down: %w", err)
		}
		qb.FFN = &nn.QuantSwiGLU{Gate: gate, Up: up, Down: down}
		q.Blocks = append(q.Blocks, qb)
	}
	out, err := mkQ(m.Out)
	if err != nil {
		return nil, fmt.Errorf("nlp: QuantizeLlama output: %w", err)
	}
	q.Out = out
	return q, nil
}

// Forward runs the quantized model on the token ids, returning logits [seq, vocab]. It mirrors
// Llama.ForwardFromEmbed exactly — RMSNorm, RoPE, MHA, residual adds and SwiGLU gating — but
// every projection is a quantized in-kernel matmul, all in f32.
func (m *QuantLlama) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	cfg := m.Config
	seq := len(tokens)
	if seq == 0 || seq > cfg.Ctx {
		return nil, fmt.Errorf("nlp: prompt length %d outside (0,%d]", seq, cfg.Ctx)
	}
	idx := tensor.New(m.TokEmb.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= cfg.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, cfg.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	x, err := exec1(ctx, backend.OpEmbed, nil, m.TokEmb, idx)
	if err != nil {
		return nil, err
	}
	// Granite scalars (no-ops for Llama/Qwen/Phi/Gemma where they are 0/1, so this
	// path stays byte-identical for every non-Granite quantized model): the
	// inputs_embeds multiplier, the attention_multiplier replacing 1/√headDim, the
	// residual_multiplier on each sublayer output, and the final logits divisor —
	// applied at the same points and in the same order as Llama.Forward.
	if x, err = scaleScalar(ctx, x, cfg.EmbeddingMult); err != nil {
		return nil, err
	}
	kv := cfg.kvHeads()
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}
	if cfg.AttentionMult != 0 {
		attn.Scale = cfg.AttentionMult * math.Sqrt(float64(cfg.Dim/cfg.Heads))
	}
	// Box these attrs into the Attrs INTERFACE once per token, above the layer loop: the values
	// are layer-independent, and as concrete structs handed to an interface parameter inside the
	// loop each was heap-boxed once per layer per token (escape analysis named every one).
	// exec1a/exec3 also pool their input slices, and only when ctx.Recorder == nil, so a taped
	// training context keeps the fresh-slice path.
	qRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads})
	kRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv})
	for _, b := range m.Blocks {
		xb, err := b.AttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := b.Wq.Forward(ctx, xb)
		if err != nil {
			return nil, err
		}
		k, err := b.Wk.Forward(ctx, xb)
		if err != nil {
			return nil, err
		}
		v, err := b.Wv.Forward(ctx, xb)
		if err != nil {
			return nil, err
		}
		// Qwen2-family q/k/v biases then Qwen3 per-head QK-norm, both BEFORE RoPE — the same
		// order and position as Llama.hiddenFromEmbedCapture; nil fields are byte-identical no-ops.
		if q, k, v, err = applyQwenAttnExtras(ctx, b, q, k, v, cfg.Heads, kv); err != nil {
			return nil, err
		}
		if q, err = exec1a(ctx, backend.OpRoPE, qRoPE, q); err != nil {
			return nil, err
		}
		if k, err = exec1a(ctx, backend.OpRoPE, kRoPE, k); err != nil {
			return nil, err
		}
		a, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
		if err != nil {
			return nil, err
		}
		o, err := b.Wo.Forward(ctx, a)
		if err != nil {
			return nil, err
		}
		if o, err = scaleScalar(ctx, o, cfg.ResidualMult); err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, o); err != nil {
			return nil, err
		}
		xf, err := b.FFNNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		ff, err := b.FFN.Forward(ctx, xf)
		if err != nil {
			return nil, err
		}
		if ff, err = scaleScalar(ctx, ff, cfg.ResidualMult); err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	logits, err := m.Out.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	return divLogits(ctx, logits, cfg.LogitsScale)
}

// Close frees every device-resident weight buffer held by the model's quantized projections
// (attention, FFN, output head). Idempotent; call it when done with the model to release GPU
// memory promptly (otherwise the buffers are reclaimed only at process exit).
func (m *QuantLlama) Close() error {
	var first error
	note := func(err error) {
		if err != nil && first == nil {
			first = err
		}
	}
	for _, b := range m.Blocks {
		for _, l := range []*nn.QuantLinear{b.Wq, b.Wk, b.Wv, b.Wo} {
			if l != nil {
				note(l.Close())
			}
		}
		if b.FFN != nil {
			note(b.FFN.Close())
		}
	}
	if m.Out != nil {
		note(m.Out.Close())
	}
	return first
}

// applyQwenAttnExtras applies a QuantBlock's optional Qwen attention pieces to the projected
// q/k/v: the Qwen2-family biases (addBiasIf) then the Qwen3 per-head QK-norm (applyQKNorm),
// both before RoPE — exactly where and in the order Llama.hiddenFromEmbedCapture applies them,
// so the quantized forward/decode of a Qwen model matches the float pipeline's math. Every nil
// field short-circuits to the input tensor (no ops recorded), keeping a plain quantized Llama
// byte-identical to the pre-Qwen path.
func applyQwenAttnExtras(ctx *backend.Context, b *QuantBlock, q, k, v *tensor.Tensor, heads, kvHeads int) (_, _, _ *tensor.Tensor, err error) {
	if q, err = addBiasIf(ctx, q, b.Bq); err != nil {
		return nil, nil, nil, err
	}
	if k, err = addBiasIf(ctx, k, b.Bk); err != nil {
		return nil, nil, nil, err
	}
	if v, err = addBiasIf(ctx, v, b.Bv); err != nil {
		return nil, nil, nil, err
	}
	if q, err = applyQKNorm(ctx, q, b.QNorm, heads); err != nil {
		return nil, nil, nil, err
	}
	if k, err = applyQKNorm(ctx, k, b.KNorm, kvHeads); err != nil {
		return nil, nil, nil, err
	}
	return q, k, v, nil
}

// f32CloneIf is f32Clone tolerating nil (absent optional tensors stay nil).
func f32CloneIf(t *tensor.Tensor) *tensor.Tensor {
	if t == nil {
		return nil
	}
	return f32Clone(t)
}

// f32Clone copies a tensor into a fresh F32 tensor of the same shape (quantized inference runs
// in f32, which also engages the f32-only accelerators).
//
// A contiguous, offset-0 source reads straight from its typed backing slice instead of the
// per-element AtF64(Unravel(i)…) path (an index-slice allocation plus a strided dispatch for
// EVERY element). This is bit-identical to the accessor path — float32(float64(v)) round-trips
// exactly for an F32 source, and an F64 source narrows exactly as AtF64→float32 does. f32Clone
// converts the (often large) token-embedding table and every RMSNorm/bias gain at quantization
// time, so the loop showed up as model-load latency; this mirrors nn.fillGen's contiguous
// setup fast path. F16/BF16 sources and any non-contiguous view fall through to the general
// accessor (whose widening carries the correct half-float rounding).
func f32Clone(t *tensor.Tensor) *tensor.Tensor {
	out := tensor.New(tensor.F32, t.Shape())
	dst := out.Storage().F32()
	n := t.Numel()
	if t.IsContiguous() && t.Offset() == 0 {
		switch t.Dtype() {
		case tensor.F32:
			copy(dst, t.Storage().F32()[:n])
			return out
		case tensor.F64:
			src := t.Storage().F64()
			for i := range n {
				dst[i] = float32(src[i])
			}
			return out
		}
	}
	for i := range n {
		dst[i] = float32(t.AtF64(tensor.Unravel(i, t.Shape())...))
	}
	return out
}

// f32RMSNorm returns an RMSNorm with an f32 copy of the gain, so it composes with f32 activations.
func f32RMSNorm(n *nn.RMSNorm) *nn.RMSNorm {
	return &nn.RMSNorm{Gamma: f32Clone(n.Gamma), Eps: n.Eps}
}
