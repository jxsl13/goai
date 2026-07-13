// Package nlp is layer L4: transformer/LLM building blocks (§T21, §T23).
package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// MHA is multi-head scaled dot-product attention (Vaswani et al. 2017,
// "Attention Is All You Need") for a single sequence x[seq, dmodel]:
//
//	Q = x·Wq; K = x·Wk; V = x·Wv                     (weights [dmodel,dmodel])
//	per head h: Aₕ = softmax(QₕKₕᵀ/√dₖ)·Vₕ            (dₖ = dmodel/heads)
//	out = concat(A₀..A_{h−1})·Wo
//
// Head splits are zero-copy Slice views; matmul/softmax run through the
// backend dispatch, so accel backends (metal GEMM) apply automatically.
// Inference-focused (§T21); training VJPs for layernorm pending §B22.
type MHA struct {
	Heads          int            // number of attention heads
	Wq, Wk, Wv, Wo *tensor.Tensor // each [dmodel, dmodel]
	// Causal masks position i from attending to j > i (decoder self-attention):
	// masked scores are −∞ before softmax, so their weight is exactly 0.
	Causal bool
	// LoRA holds optional low-rank adapters for the projections, keyed "q","k","v","o"
	// (§T504, attach with nlp.ApplyLoRAGPT or manually). Each adapter wraps the SAME
	// frozen weight tensor; with LoRA's zero-initialized B the forward is numerically
	// identical to the plain projection until the adapter trains. nil = plain paths.
	LoRA map[string]*nn.LoRALinear
	// Bias holds optional per-projection biases [dmodel], keyed "q","k","v","o"
	// (§T505): q = x·Wq + Bias["q"], etc. GPT-2-family checkpoints carry these
	// (c_attn.bias, c_proj.bias); Llama-family models do not. nil map or missing
	// key = no bias, the exact pre-§T505 path.
	Bias map[string]*tensor.Tensor
	// Mask, when set, replaces the fused causal attention with OpMHAMasked using
	// this [seq,seq] additive mask (§T508 — tree attention for Medusa tree
	// verification; −Inf excludes a pair). INFERENCE-ONLY (the masked op has no
	// VJP) and per-call state: set it, run Forward, clear it. nil = the exact
	// fused path.
	Mask *tensor.Tensor
}

// NewMHA builds an MHA block with the given weights. dmodel must divide by heads.
func NewMHA(heads int, wq, wk, wv, wo *tensor.Tensor) (*MHA, error) {
	d := wq.Shape()[0]
	for _, w := range []*tensor.Tensor{wq, wk, wv, wo} {
		if w.Ndim() != 2 || w.Shape()[0] != d || w.Shape()[1] != d {
			return nil, fmt.Errorf("nlp: MHA weights must all be [%d,%d], got %v", d, d, w.Shape())
		}
	}
	if heads <= 0 || d%heads != 0 {
		return nil, fmt.Errorf("nlp: dmodel %d not divisible by heads %d", d, heads)
	}
	return &MHA{Heads: heads, Wq: wq, Wk: wk, Wv: wv, Wo: wo}, nil
}

func (m *MHA) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward computes attention for x[seq, dmodel]. Fully differentiable: the
// projections are matmuls and the multi-head scaled-dot-product core is the
// single fused OpMHA op (head split/concat/mask internal, §T32) — so gradients
// flow to Wq/Wk/Wv/Wo and x through the standard dispatch, with no view ops on
// the tape.
func (m *MHA) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != m.Wq.Shape()[0] {
		return nil, fmt.Errorf("nlp: MHA expects x[seq,%d], got %v", m.Wq.Shape()[0], x.Shape())
	}
	q, err := m.proj(ctx, x, m.Wq, "q")
	if err != nil {
		return nil, err
	}
	k, err := m.proj(ctx, x, m.Wk, "k")
	if err != nil {
		return nil, err
	}
	v, err := m.proj(ctx, x, m.Wv, "v")
	if err != nil {
		return nil, err
	}
	var attn *tensor.Tensor
	if m.Mask != nil { // §T508: mask expresses the structure; Causal is NOT also applied
		attn, err = m.exec(ctx, backend.OpMHAMasked, backend.AttnAttrs{Heads: m.Heads}, q, k, v, m.Mask)
	} else {
		attn, err = m.exec(ctx, backend.OpMHA, backend.AttnAttrs{Heads: m.Heads, Causal: m.Causal}, q, k, v)
	}
	if err != nil {
		return nil, err
	}
	return m.proj(ctx, attn, m.Wo, "o")
}

// proj applies one attention projection — through its LoRA adapter when attached
// (§T504), else the plain matmul — and adds the projection's bias when one is set
// (§T505). With LoRA's zero-initialized B and no bias the path is numerically
// identical to the plain projection.
func (m *MHA) proj(ctx *backend.Context, x, w *tensor.Tensor, name string) (*tensor.Tensor, error) {
	var y *tensor.Tensor
	var err error
	if l, ok := m.LoRA[name]; ok && l != nil {
		y, err = l.Forward(ctx, x)
	} else {
		y, err = m.exec(ctx, backend.OpMatMul, nil, x, w)
	}
	if err != nil {
		return nil, err
	}
	if b, ok := m.Bias[name]; ok && b != nil {
		return m.exec(ctx, backend.OpAddBias, nil, y, b)
	}
	return y, nil
}

// Params returns the projection weights.
func (m *MHA) Params() []*tensor.Tensor {
	return []*tensor.Tensor{m.Wq, m.Wk, m.Wv, m.Wo}
}
