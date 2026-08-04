package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// GatedAttention is standard causal multi-head softmax self-attention with a
// head-specific sigmoid OUTPUT GATE (Qiu et al. 2025, "Gated Attention for
// Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free",
// arXiv:2505.06708). The paper's most effective variant — implemented here —
// gates the SDPA output elementwise, AFTER attention and BEFORE the output
// projection. For X ∈ [T, dim] with H heads (per-head dim d_k = dim/H):
//
//	Q,K,V = split( X · W_in )                       (fused input projection, dim → 3·dim)
//	A     = SDPA(Q, K, V)                           (causal multi-head softmax attention,
//	                                                 per-head outputs concatenated to [T, dim],
//	                                                 NO output projection yet)
//	G     = σ(X · W_θ)                              (data-dependent gate, one scalar per
//	                                                 output channel, i.e. per (head, d_k) slot)
//	Y     = (G ⊙ A) · W_O                           (elementwise gate, THEN the out-proj)
//
// The gate is a query-dependent, per-channel multiplier in (0,1) on the raw
// per-head attention output. The paper reports this single extra sigmoid
// introduces non-linearity between the value map and the out-proj and makes
// the attention output sparse and input-dependent, which improves training
// stability, eliminates attention sinks and massive activations, and enables
// better long-context extrapolation — at negligible parameter/compute cost.
// With WithGatedAttentionHeadwiseGate the gate is the paper's coarser
// HEADWISE variant: W_θ maps dim → H and each head's single sigmoid scalar
// multiplies that head's whole d_k-wide output (fewer gate parameters,
// slightly weaker in the paper's ablations than the elementwise default).
//
// How this differs from the neighboring attention variants in this package:
// FoXBlock (arXiv:2503.02130) biases the PRE-softmax scores with a
// log-cumulative forget decay (a score-bias gate — the softmax itself is
// modified); DiffAttention (arXiv:2410.05258) subtracts TWO softmax maps to
// cancel common-mode noise; GatedDeltaNet (arXiv:2412.06464) gates a LINEAR
// attention delta-rule state, replacing softmax entirely. GatedAttention
// keeps the single ordinary softmax untouched and applies its sigmoid gate
// purely on the attention OUTPUT.
//
// The attention core is the fused OpMHA op (head split/softmax/concat
// internal, VJP-backed), exactly as HymbaBlock uses it: OpMHA returns the
// concatenated per-head outputs WITHOUT an output projection, which is what
// lets the gate sit between attention and W_O. Every op is dispatched and
// VJP-backed, so the whole layer trains end to end — gradients reach the
// fused in-proj, the gate projection W_θ, and the out-proj.
type GatedAttention struct {
	Dim     int // model / embedding width (columns of X)
	Heads   int // number of attention heads
	HeadDim int // per-head dimension d_k = Dim/Heads

	// InProj is the fused input projection [Dim → 3·Dim]: one matmul whose
	// output is sliced into Q, K, V (OpSlice is VJP-backed, so the fused
	// weight accumulates gradients from all three streams).
	InProj *Linear
	// Gate is the gate projection W_θ — [Dim → Dim] for the elementwise
	// default, [Dim → Heads] for the headwise variant; σ of its output
	// multiplies the raw SDPA output.
	Gate *Linear
	// OutProj is the output projection W_O [Dim → Dim], applied AFTER the
	// gate.
	OutProj *Linear

	headwise bool // gate granularity: false = per channel (default), true = per head
}

// GatedAttentionOption configures a GatedAttention layer (functional-options
// idiom, §C12).
type GatedAttentionOption func(*GatedAttention)

// WithGatedAttentionHeadwiseGate switches the gate to the paper's HEADWISE
// granularity (arXiv:2505.06708 §3.1): W_θ maps dim → heads and each head's
// single sigmoid scalar multiplies that head's entire d_k-wide slice of the
// attention output. This uses dim·heads instead of dim² gate parameters; the
// paper finds it effective but slightly weaker than the elementwise default.
func WithGatedAttentionHeadwiseGate() GatedAttentionOption {
	return func(g *GatedAttention) { g.headwise = true }
}

// NewGatedAttention builds a gated attention layer over model width dim with
// heads attention heads (dim must divide by heads). The fused Q/K/V in-proj,
// the gate projection W_θ, and the out-proj W_O are Xavier-uniform Linears
// with zero bias (so at init the gate starts near σ(0) = 0.5, a soft
// half-open gate, as in the paper's from-scratch setting). Deterministic via
// seed.
func NewGatedAttention(dtype tensor.Dtype, dim, heads int, seed uint64, opts ...GatedAttentionOption) (*GatedAttention, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("nn: GatedAttention dim %d must be positive", dim)
	}
	if heads <= 0 || dim%heads != 0 {
		return nil, fmt.Errorf("nn: GatedAttention dim %d not divisible by heads %d", dim, heads)
	}
	g := &GatedAttention{Dim: dim, Heads: heads, HeadDim: dim / heads}
	for _, o := range opts {
		o(g)
	}
	gateOut := dim // elementwise: one gate scalar per output channel
	if g.headwise {
		gateOut = heads // headwise: one gate scalar per head
	}
	g.InProj = NewLinear(dtype, dim, 3*dim, seed)
	g.Gate = NewLinear(dtype, dim, gateOut, seed+1)
	g.OutProj = NewLinear(dtype, dim, dim, seed+2)
	return g, nil
}

func (g *GatedAttention) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward runs gated attention on x[T, Dim] → [T, Dim]: fused Q/K/V
// projection, causal multi-head SDPA (OpMHA, no out-proj), the sigmoid output
// gate σ(x·W_θ) applied elementwise to the raw per-head attention output,
// then the output projection W_O. Fully differentiable — gradients reach the
// in-proj, W_θ, and W_O.
func (g *GatedAttention) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != g.Dim {
		return nil, fmt.Errorf("nn: GatedAttention expects x [T,%d], got %v", g.Dim, x.Shape())
	}
	// Fused input projection, sliced into Q, K, V.
	p, err := g.InProj.Forward(ctx, x) // [T, 3·Dim]
	if err != nil {
		return nil, err
	}
	chunk := func(i int) (*tensor.Tensor, error) {
		//perfscan:ignore PS3024 OpSlice QKV-split dispatch in attention-dominated forward
		return g.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: i * g.Dim, End: (i + 1) * g.Dim}, p)
	}
	q, err := chunk(0)
	if err != nil {
		return nil, err
	}
	k, err := chunk(1)
	if err != nil {
		return nil, err
	}
	v, err := chunk(2)
	if err != nil {
		return nil, err
	}
	// SDPA core: fused causal multi-head attention (head split/softmax/concat
	// internal) — the raw head concat [T, Dim], NO output projection.
	//perfscan:ignore PS3024 single OpMHA dispatch; attention/matmul dominates
	attn, err := g.exec(ctx, backend.OpMHA, backend.AttnAttrs{Heads: g.Heads, Causal: true}, q, k, v)
	if err != nil {
		return nil, err
	}
	// Gate G = σ(x·W_θ): data-dependent, in (0,1).
	gl, err := g.Gate.Forward(ctx, x) // [T, Dim] or [T, Heads]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpSigmoid gate dispatch; matmul-dominated forward
	gate, err := g.exec(ctx, backend.OpSigmoid, nil, gl)
	if err != nil {
		return nil, err
	}
	if g.headwise { // expand [T, Heads] → [T, Dim]: head h's scalar over its d_k channels
		if gate, err = g.expandHeadGate(ctx, gate, x.Shape()[0]); err != nil {
			return nil, err
		}
	}
	// Y = (G ⊙ SDPA) · W_O: gate the raw attention output, THEN project.
	//perfscan:ignore PS3024 OpMul gate dispatch; matmul-dominated forward
	gated, err := g.exec(ctx, backend.OpMul, nil, attn, gate)
	if err != nil {
		return nil, err
	}
	return g.OutProj.Forward(ctx, gated) // [T, Dim]
}

// expandHeadGate widens a headwise gate [T, Heads] to the attention layout
// [T, Dim] by repeating each head's sigmoid scalar across that head's HeadDim
// output channels (columns [h·HeadDim, (h+1)·HeadDim) hold head h, matching
// OpMHA's concat order). Slice/broadcast/concat are all VJP-backed, so the
// headwise gate trains like the elementwise one.
func (g *GatedAttention) expandHeadGate(ctx *backend.Context, gate *tensor.Tensor, t int) (*tensor.Tensor, error) {
	cols := make([]*tensor.Tensor, g.Heads)
	for h := range g.Heads {
		//perfscan:ignore PS3024 OpSlice in per-head loop, low trip-count, headwise-only
		c, err := g.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h, End: h + 1}, gate) // [T,1]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024,PS6016 OpBroadcast per-head loop, low trip-count | per-head broadcast alloc, low trip-count headwise expand
		if cols[h], err = g.exec(ctx, backend.OpBroadcast, backend.BroadcastAttrs{Shape: tensor.Shape{t, g.HeadDim}}, c); err != nil {
			return nil, err
		}
	}
	return g.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, cols...) // [T, Dim]
}

// Params returns every trainable tensor: the fused Q/K/V in-proj, the gate
// projection W_θ, and the out-proj W_O (each Linear's W and bias). Feed this
// to an optimizer.
func (g *GatedAttention) Params() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, l := range []*Linear{g.InProj, g.Gate, g.OutProj} {
		ps = append(ps, l.Params()...)
	}
	return ps
}
