package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// FoXBlock is a Forgetting Transformer (FoX) attention block (Lin, Yang, Su,
// Wang, Xu, Kong & Zhang 2025, "Forgetting Transformer: Softmax Attention with a
// Forget Gate", arXiv:2503.02130, §R244). FoX is ordinary causal softmax
// attention — the full O(T²) score matrix, keeping softmax's expressiveness —
// augmented with a per-head, data-DEPENDENT FORGET GATE that biases the
// pre-softmax logits with a log-cumulative decay, so distant past is down-weighted
// in a content-dependent way. For X ∈ [T, dim] with H heads (per-head dim
// d_k = dim/H):
//
//	forget gate:  f_t^h = σ(w_f^h·x_t + b_f^h) ∈ (0,1)          (a scalar per token t, head h)
//	log-cumsum:   c_t^h = Σ_{l≤t} log f_l^h                     (cumulative sum along time)
//	decay bias:   D_ij^h = c_i^h − c_j^h = Σ_{l=j+1..i} log f_l ≤ 0   (i ≥ j; distant past ↓)
//	attention:    O^h = softmax(Q^h·K^hᵀ·d_k^(−½) + D^h + causalMask)·V^h
//	FoX(X)      = concat_h(O^h)·W_o
//
// The forget gate replaces positional encoding: FoX uses NO RoPE and no other
// positional embedding, because the gate's log-cumulative decay already supplies
// the recency bias (a large forget → f→0 → strongly negative D → attention
// concentrates on recent tokens; f→1 → D→0 → plain softmax attention). When
// D^h = 0 for all heads the block reduces exactly to standard causal softmax
// attention. This implements CORE FoX only — the paper's "Pro" extras (output
// gate, QK-norm, KV-shift) are intentionally omitted.
//
// Every op is dispatched and VJP-backed (the attention is assembled from
// matmul/softmax/add primitives rather than the inference-only fused-MHA op), so
// the whole block trains: gradients reach the Q/K/V/O projections AND the
// forget-gate projection W_f.
type FoXBlock struct {
	Dim     int // model / embedding width (columns of X)
	Heads   int // number of attention heads
	HeadDim int // per-head dimension d_k = Dim/Heads

	Wq []*Linear // per-head query projection Dim→HeadDim
	Wk []*Linear // per-head key   projection Dim→HeadDim
	Wv []*Linear // per-head value projection Dim→HeadDim
	Wo *Linear   // output projection (Heads·HeadDim)→Dim
	Wf *Linear   // forget-gate projection Dim→Heads (one logit per token per head)

	scale float64 // pre-softmax QK scale d_k^(−½)
	eps   float64 // lower floor on the forget gate before log, for numeric stability
}

// FoXOption configures a FoXBlock (functional-options idiom, §C12).
type FoXOption func(*FoXBlock)

// WithFoXEps sets the lower floor applied to the forget gate before taking its log
// (default 1e-9). The gate σ(·) lives in (0,1), so for an extreme forget its log
// can underflow toward −∞; clamping f to [eps,1] keeps log f finite and the decay
// bias well-conditioned. eps must be in (0,1].
func WithFoXEps(eps float64) FoXOption { return func(b *FoXBlock) { b.eps = eps } }

// NewFoXBlock builds a Forgetting Transformer attention block over dim with heads
// heads (dim must be divisible by heads). All projections — per-head Q/K/V, the
// fused output W_o, and the forget-gate W_f (dim→heads) — are Xavier-uniform
// Linears and are trainable. Deterministic via seed.
func NewFoXBlock(dtype tensor.Dtype, dim, heads int, seed uint64, opts ...FoXOption) (*FoXBlock, error) {
	if heads <= 0 || dim%heads != 0 {
		return nil, fmt.Errorf("nn: FoXBlock dim %d not divisible by heads %d", dim, heads)
	}
	hd := dim / heads
	b := &FoXBlock{
		Dim: dim, Heads: heads, HeadDim: hd,
		scale: math.Pow(float64(hd), -0.5), eps: 1e-9,
	}
	var s uint64 = seed
	for range heads {
		b.Wq = append(b.Wq, NewLinear(dtype, dim, hd, s+1))
		b.Wk = append(b.Wk, NewLinear(dtype, dim, hd, s+2))
		b.Wv = append(b.Wv, NewLinear(dtype, dim, hd, s+3))
		s += 4
	}
	b.Wo = NewLinear(dtype, heads*hd, dim, s+1)
	b.Wf = NewLinear(dtype, dim, heads, s+2)
	for _, o := range opts {
		o(b)
	}
	if b.eps <= 0 || b.eps > 1 {
		return nil, fmt.Errorf("nn: FoXBlock eps %g must be in (0,1]", b.eps)
	}
	return b, nil
}

// FoXDecayBias builds the Forgetting Transformer decay-bias matrix D ∈ [T,T] for a
// single head from that head's per-step log-forget-gate values logForget ∈ [T,1]
// (logForget_t = log f_t = log σ(w_f·x_t+b_f)). It forms the log-cumulative decay
// c_t = Σ_{l≤t} log f_l with a time-axis cumulative sum, then the pairwise bias
//
//	D_ij = c_i − c_j = Σ_{l=j+1..i} log f_l          (≤ 0 for i ≥ j),
//
// so a query at i sees key j biased by the total log-forget accumulated between
// them — distant past exponentially down-weighted after softmax. The result is
// added to the pre-softmax logits (a causal mask still zeroes j>i). Differentiable
// w.r.t. logForget (grads flow back through the cumulative sum). This is the core
// FoX mechanism (§R244, arXiv:2503.02130) factored out so the bias math is
// directly testable.
func FoXDecayBias(ctx *backend.Context, logForget *tensor.Tensor) (*tensor.Tensor, error) {
	if logForget.Ndim() != 2 || logForget.Shape()[1] != 1 {
		return nil, fmt.Errorf("nn: FoXDecayBias wants a [T,1] log-forget column, got %v", logForget.Shape())
	}
	t := logForget.Shape()[0]
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	c, err := ex(backend.OpCumsum, backend.CumsumAttrs{Axis: 0}, logForget) // c[T,1], c_i as a column
	if err != nil {
		return nil, err
	}
	cRow, err := ex(backend.OpTranspose, nil, c) // [1,T], c_j as a row
	if err != nil {
		return nil, err
	}
	colB, err := ex(backend.OpBroadcast, backend.BroadcastAttrs{Shape: tensor.Shape{t, t}}, c) // D[i,j]=c_i
	if err != nil {
		return nil, err
	}
	rowB, err := ex(backend.OpBroadcast, backend.BroadcastAttrs{Shape: tensor.Shape{t, t}}, cRow) // D[i,j]=c_j
	if err != nil {
		return nil, err
	}
	return ex(backend.OpSub, nil, colB, rowB) // D_ij = c_i − c_j
}

// Forward runs the FoX attention block on x[T, Dim] → [T, Dim], fully
// differentiable (grads reach the Q/K/V/O projections and the forget-gate W_f).
func (b *FoXBlock) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != b.Dim {
		return nil, fmt.Errorf("nn: FoXBlock expects x [T,%d], got %v", b.Dim, x.Shape())
	}
	t := x.Shape()[0]
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	// Forget gate: logits [T,heads] → f=σ(·) → floor to [eps,1] → log f. The floor
	// keeps log finite for an extreme forget (§R244 uses a stable log-sigmoid; the
	// clamp is our numeric guard, exposed via WithFoXEps).
	fl, err := b.Wf.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	fg, err := ex(backend.OpSigmoid, nil, fl)
	if err != nil {
		return nil, err
	}
	fg, err = ex(backend.OpClip, backend.ClipAttrs{Lo: b.eps, Hi: 1}, fg)
	if err != nil {
		return nil, err
	}
	logf, err := ex(backend.OpLog, nil, fg) // [T,heads]
	if err != nil {
		return nil, err
	}

	invSqrtD := scalarTensor(x.Dtype(), b.scale)
	causal := qkCausalMask(x.Dtype(), t, t)
	heads := make([]*tensor.Tensor, b.Heads)
	for h := range b.Heads {
		lfh, err := ex(backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: h, End: h + 1}, logf) // [T,1]
		if err != nil {
			return nil, err
		}
		bias, err := FoXDecayBias(ctx, lfh) // D^h [T,T]
		if err != nil {
			return nil, err
		}
		q, err := b.Wq[h].Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		k, err := b.Wk[h].Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		v, err := b.Wv[h].Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		kt, err := ex(backend.OpTranspose, nil, k) // [HeadDim,T]
		if err != nil {
			return nil, err
		}
		s, err := ex(backend.OpMatMul, nil, q, kt) // [T,T]
		if err != nil {
			return nil, err
		}
		if s, err = ex(backend.OpMul, nil, s, invSqrtD); err != nil { // ·d_k^(−½)
			return nil, err
		}
		if s, err = ex(backend.OpAdd, nil, s, bias); err != nil { // + D^h
			return nil, err
		}
		if s, err = ex(backend.OpAdd, nil, s, causal); err != nil { // + causal mask
			return nil, err
		}
		if s, err = ex(backend.OpSoftmax, nil, s); err != nil { // softmax over keys
			return nil, err
		}
		if heads[h], err = ex(backend.OpMatMul, nil, s, v); err != nil { // ·V^h → [T,HeadDim]
			return nil, err
		}
	}
	concat, err := ex(backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...) // [T, Heads·HeadDim]
	if err != nil {
		return nil, err
	}
	return b.Wo.Forward(ctx, concat) // [T, Dim]
}

// Params returns every trainable weight: the per-head Q/K/V projections, the fused
// output projection, and the forget-gate projection (each Linear's W and bias).
func (b *FoXBlock) Params() []*tensor.Tensor {
	var p []*tensor.Tensor
	for h := range b.Heads {
		p = append(p, b.Wq[h].Params()...)
		p = append(p, b.Wk[h].Params()...)
		p = append(p, b.Wv[h].Params()...)
	}
	p = append(p, b.Wo.Params()...)
	p = append(p, b.Wf.Params()...)
	return p
}
