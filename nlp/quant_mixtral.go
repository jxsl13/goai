package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantMixtral is a [Mixtral] whose projection weights stay QUANTIZED (nn.QuantLinear)
// and are never materialized as full-precision matrices — the memory-efficient form of a
// llama.cpp-quantized Mixtral checkpoint, the sparse-MoE sibling of [QuantLlama]
// (§T149). Its attention is byte-for-byte the QuantLlama recipe — f32 RMSNorm gains,
// quantized q/k/v/o projections, RoPE, grouped-query attention — while each block's
// feed-forward is a sparse Mixture-of-Experts: a small f32 router scores E experts per
// token, the top-k (2 for Mixtral) are selected with renormalized softmax weights, and
// ONLY those experts' quantized SwiGLU FFNs run ([QuantMoE]). Activations are f32 —
// quantized-inference precision, which also engages the f32-only GPU kernels.
// Inference-only: the quantized weights are frozen and bypass the tape.
//
// For readers new to MoE inference: a Mixtral block stores 8 expert FFNs but each token
// uses only 2 of them, so ~¾ of the FFN weights are idle per token. Keeping the experts
// quantized AND skipping the unrouted ones is exactly where a 47B-parameter Mixtral
// becomes runnable: memory shrinks by the quantization factor and decode compute by
// k/E on the FFNs, the dominant cost.
//
// Load a llama.cpp-quantized checkpoint with [QuantMixtralFromGGUF], or quantize a
// float model with [QuantizeMixtral].
type QuantMixtral struct {
	Config MixtralConfig        // geometry: dims, heads, expert count (see MixtralConfig)
	TokEmb *tensor.Tensor       // [vocab, dim] f32 token embedding (lookup only)
	Blocks []*QuantMixtralBlock // the quantized attention + sparse-MoE blocks
	Norm   *nn.RMSNorm          // final pre-logits RMSNorm (f32 gain)
	Out    *nn.QuantLinear      // quantized output projection (In=dim, Out=vocab)
}

// QuantMixtralBlock is one pre-norm QuantMixtral block: f32 RMSNorm gains, quantized
// Llama-style attention projections, and a sparse-MoE FFN with quantized experts. The
// optional QNorm/KNorm mirror [MixtralBlock]: per-head query/key RMSNorms before RoPE,
// nil for plain Mixtral (a GoAI extension for Qwen3-MoE-style QK-norm models — real
// llama-arch quantized files never carry them).
type QuantMixtralBlock struct {
	AttnNorm       *nn.RMSNorm     // RMSNorm before attention (f32 gain)
	Wq, Wk, Wv, Wo *nn.QuantLinear // quantized attention projections (bias-free)
	QNorm, KNorm   *nn.RMSNorm     // optional per-head QK-norm before RoPE; nil otherwise
	FFNNorm        *nn.RMSNorm     // RMSNorm before the FFN (f32 gain)
	MoE            *QuantMoE       // sparse-MoE FFN with quantized experts
}

// QuantMoE is the quantized twin of [nn.SparseMoE] for inference: an f32 router plus E
// quantized SwiGLU experts, evaluated sparsely — only the experts some token actually
// routes to run.
//
// The router stays in FULL PRECISION (a plain f32 [dim, E] matrix, x·Router → gate
// logits), not a QuantLinear. Two reasons, both matching the wider conventions:
// llama.cpp NEVER quantizes blk.N.ffn_gate_inp.weight (llama-quant.cpp excludes it from
// quantization — real quantized Mixtral GGUFs store it F32/F16), and GoAI's quantized
// models keep every tiny, precision-sensitive tensor in f32 (norm gains, biases,
// QK-norms — see [QuantBlock]). The router is both: E·dim parameters (~0.01% of a
// block) whose output decides the DISCRETE top-k expert choice, where a quantization
// wobble near a routing tie would swap experts rather than nudge a logit.
//
// Forward mirrors [nn.SparseMoE.ForwardDecode]'s exact math and kernel sequence: raw
// router logits, per-token top-k selection via [nn.TopKGating] (softmax over ALL raw
// logits, then the k winners' probabilities renormalized to sum to 1), and
// y = Σ weight_col ⊙ expert_i(x) accumulated with OpMul/OpAdd over the union of
// selected experts. Matching the float path's renormalization order and kernel
// sequence is what makes quantized-vs-float parity a pure quantization error, with no
// routing-algebra residual on top.
type QuantMoE struct {
	Router  *tensor.Tensor    // [dim, E] f32 gate projection — never quantized (see type doc)
	Experts []*nn.QuantSwiGLU // E quantized SwiGLU experts (dim → hidden → dim)
	TopK    int               // experts per token (2 for Mixtral)
}

// Forward routes x [T, dim] through the top-k experts per token and returns the mixed
// output [T, dim]. Only the UNION of experts selected by some row is evaluated — for a
// single decode row that is exactly the top-k, the sparse-decode fast path; for a
// multi-row prefill each used expert runs once over all rows, weighted zero where a row
// did not select it (identical math to per-row dispatch, same as
// [nn.SparseMoE.ForwardDecode]). Inference-only: the routing is discrete and nothing is
// recorded for autograd.
func (m *QuantMoE) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	logits, err := exec1(ctx, backend.OpMatMul, nil, x, m.Router) // [T, E] f32 gate logits
	if err != nil {
		return nil, err
	}
	tks, e := logits.Shape()[0], logits.Shape()[1]

	// Per-token top-k routing. TopKGating's weights ARE the renormalized combine
	// weights (softmax over the raw logits, renormalized over the selected set) —
	// the exact coefficients nn.SparseMoE uses on both its dense and sparse paths.
	weight := tensor.New(x.Dtype(), tensor.Shape{tks, e}) // 0 for unselected
	used := make([]bool, e)
	row := make([]float64, e)
	for t := range tks {
		for i := range e {
			row[i] = logits.AtF64(t, i)
		}
		selected, weights := nn.TopKGating(row, m.TopK)
		for j, i := range selected {
			weight.SetF64(weights[j], t, i)
			used[i] = true
		}
	}

	// Evaluate ONLY the used experts and accumulate y = Σ_i weight[:,i] ⊙ expert_i(x) —
	// the same OpMul-then-OpAdd sequence as nn.SparseMoE.ForwardDecode, so the decode
	// and full-sequence paths of QuantMixtral share one kernel recipe bit for bit.
	var y *tensor.Tensor
	for i := range e {
		if !used[i] {
			continue
		}
		out, err := m.Experts[i].Forward(ctx, x) // [T, dim]
		if err != nil {
			return nil, err
		}
		wcol, err := weight.Slice(1, i, i+1) // [T, 1] column of expert i's weights
		if err != nil {
			return nil, err
		}
		term, err := exec1(ctx, backend.OpMul, nil, out, wcol.Contiguous())
		if err != nil {
			return nil, err
		}
		if y == nil {
			y = term
		} else {
			if y, err = exec2(ctx, backend.OpAdd, nil, y, term); err != nil {
				return nil, err
			}
		}
	}
	return y, nil
}

// Close frees any device-resident weight buffers held by the expert projections. Idempotent.
func (m *QuantMoE) Close() error {
	var first error
	for _, ex := range m.Experts {
		if ex != nil {
			if err := ex.Close(); err != nil && first == nil {
				first = err
			}
		}
	}
	return first
}

// QuantizeMixtral builds a QuantMixtral from a float Mixtral by quantizing every large
// linear projection to qt (the ggml block layout): the attention q/k/v/o, every
// expert's gate/up/down, and the output head — the tensors that carry essentially all
// the weights and compute. Kept in f32, exactly as [QuantizeLlama] keeps them: the
// RMSNorm gains, the token embedding, the optional QK-norm gains, and — the MoE-specific
// addition — the ROUTER (see [QuantMoE] for why the router is never quantized; llama.cpp
// makes the same choice for ffn_gate_inp). Each quantized projection's inner dimension
// must be a multiple of qt's block size (32 for Q8_0/Q4_0, 256 for the k-quants).
func QuantizeMixtral(m *Mixtral, qt gguf.QuantType) (*QuantMixtral, error) {
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		in, out := w.Shape()[0], w.Shape()[1] // GoAI [in, out]
		bytes, err := gguf.Quantize(transpose2D(w), qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: bytes, QT: qt, In: in, Out: out}, nil
	}
	q := &QuantMixtral{Config: m.Config, TokEmb: f32Clone(m.TokEmb), Norm: f32RMSNorm(m.Norm)}
	for _, b := range m.Blocks {
		qb := &QuantMixtralBlock{AttnNorm: f32RMSNorm(b.AttnNorm), FFNNorm: f32RMSNorm(b.FFNNorm)}
		if b.QNorm != nil {
			qb.QNorm = f32RMSNorm(b.QNorm)
		}
		if b.KNorm != nil {
			qb.KNorm = f32RMSNorm(b.KNorm)
		}
		var err error
		if qb.Wq, err = mkQ(b.Wq); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMixtral Wq: %w", err)
		}
		if qb.Wk, err = mkQ(b.Wk); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMixtral Wk: %w", err)
		}
		if qb.Wv, err = mkQ(b.Wv); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMixtral Wv: %w", err)
		}
		if qb.Wo, err = mkQ(b.Wo); err != nil {
			return nil, fmt.Errorf("nlp: QuantizeMixtral Wo: %w", err)
		}
		moe := &QuantMoE{Router: f32Clone(b.MoE.Router.W), TopK: b.MoE.TopK}
		for e, ex := range b.MoE.Experts {
			gate, err := mkQ(ex.Wgate)
			if err != nil {
				return nil, fmt.Errorf("nlp: QuantizeMixtral expert %d gate: %w", e, err)
			}
			up, err := mkQ(ex.Wup)
			if err != nil {
				return nil, fmt.Errorf("nlp: QuantizeMixtral expert %d up: %w", e, err)
			}
			down, err := mkQ(ex.Wdown)
			if err != nil {
				return nil, fmt.Errorf("nlp: QuantizeMixtral expert %d down: %w", e, err)
			}
			moe.Experts = append(moe.Experts, &nn.QuantSwiGLU{Gate: gate, Up: up, Down: down})
		}
		qb.MoE = moe
		q.Blocks = append(q.Blocks, qb)
	}
	out, err := mkQ(m.Out)
	if err != nil {
		return nil, fmt.Errorf("nlp: QuantizeMixtral output: %w", err)
	}
	q.Out = out
	return q, nil
}

// Forward runs the quantized model on the token ids, returning logits [seq, vocab]. It
// mirrors Mixtral's forward exactly — RMSNorm, RoPE, GQA, residual adds — but every
// projection is a quantized in-kernel matmul in f32, and the MoE FFN runs the SAME
// sparse top-k evaluation as [QuantMixtral.DecodeStep] ([QuantMoE.Forward], the
// batched analog of the float model's Prefill sparseFFN path), so the full-sequence
// and KV-cached decode paths share one kernel sequence.
func (m *QuantMixtral) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	cfg := m.Config
	seq := len(tokens)
	if seq == 0 || seq > cfg.Ctx {
		return nil, fmt.Errorf("nlp: Mixtral prompt length %d outside (0,%d]", seq, cfg.Ctx)
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
	kv := cfg.kvHeads()
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}
	// Box these attrs into the Attrs INTERFACE once per token, above the layer loop: the values
	// are layer-independent, and as concrete structs handed to an interface parameter inside the
	// loop each was heap-boxed once per layer per token (escape analysis named every one).
	// exec1a/exec3 also pool their input slices, and only when ctx.Recorder == nil, so a taped
	// training context keeps the fresh-slice path.
	qRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads})
	kRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv})
	for _, b := range m.Blocks {
		// attention sublayer (identical to QuantLlama; bias-free)
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
		// Optional per-head QK-norm before RoPE (nil for plain Mixtral — a no-op).
		if q, err = applyQKNorm(ctx, q, b.QNorm, cfg.Heads); err != nil {
			return nil, err
		}
		if k, err = applyQKNorm(ctx, k, b.KNorm, kv); err != nil {
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
		if x, err = exec2(ctx, backend.OpAdd, nil, x, o); err != nil {
			return nil, err
		}
		// sparse-MoE FFN sublayer: top-k routing, only the routed experts run
		xf, err := b.FFNNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		ff, err := b.MoE.Forward(ctx, xf)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return m.Out.Forward(ctx, x)
}

// Close frees every device-resident weight buffer held by the model's quantized
// projections (attention, every expert, the output head). Idempotent; call it when done
// with the model to release GPU memory promptly (otherwise the buffers are reclaimed
// only at process exit).
func (m *QuantMixtral) Close() error {
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
		if b.MoE != nil {
			note(b.MoE.Close())
		}
	}
	if m.Out != nil {
		note(m.Out.Close())
	}
	return first
}
