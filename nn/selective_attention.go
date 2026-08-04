package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SelectiveAttention is causal multi-head self-attention augmented with the
// parameter-free "selective attention" mechanism of Leviathan, Kalman & Matias
// (2024, "Selective Attention Improves Transformers", arXiv:2410.02703). On top
// of ordinary scaled-dot-product attention it lets each token DECIDE that some
// earlier key should be masked out of FUTURE queries' attention — so a later
// mention can suppress a now-stale earlier one, sharpening retrieval and
// shrinking the effective context.
//
// # The mechanism (paper §2)
//
// Ordinary causal attention forms logits L[i,j] = Qᵢ·Kⱼ/√d for key j ≤ query i.
// Selective attention SUBTRACTS a non-negative SELECTION bias F from the logits
// before the softmax:
//
//	L'[i,j] = L[i,j] − F[i,j]
//	F[i,j]  = Σ_{i'=1}^{i−1} S[i',j]        (cumulative over the QUERY axis, STRICTLY below i)
//	S[i',j] ≥ 0                             (how strongly query i' votes to mask key j out)
//
// O = softmax(L' + causal-mask)·V as usual. Because F is a running sum over the
// queries STRICTLY BEFORE i (the diagonal i'=i is excluded), a token can never
// mask itself at its own step, and F[i,·] depends only on selections made by
// earlier queries — selection is causal, just like attention.
//
// # Where S comes from (paper's default: reuse head 0)
//
// The paper's near-parameter-free default derives the selection score from ONE
// existing head's own pre-softmax logits rather than a new projection: this
// layer uses head 0's scaled logits L₀[i,j] = Q₀ᵢ·K₀ⱼ/√d, rectified to stay
// non-negative and with the diagonal (and the future) removed,
//
//	S = strictly-lower-triangular( ReLU(L₀) )
//
// so S[i',j] > 0 only for a real past key j < i'. ReLU makes "vote to mask" a
// one-sided signal (a token can push a key OUT of future attention, never pull a
// masked one back in). F is then the strictly-below cumulative sum of S over the
// query axis, computed as cumsum(S)−S along axis 0 (cumsum gives Σ_{i'≤i}; minus
// the diagonal term S[i,·] leaves Σ_{i'<i}). This reuse-head-0 choice is the
// paper's §2 default; a dedicated one-parameter projection is the alternative it
// also mentions, not implemented here.
//
// # Contrast with the repo's other attention variants (§C18)
//
// Selective attention changes only the LOGITS (an additive, causally-accumulated
// −F), leaving the softmax itself standard. That is orthogonal to
// DiffAttention (arXiv:2410.05258), which subtracts two softmax MAPS to cancel
// common-mode noise, and to SoftpickAttention (softmax-off-by-one / Softpick,
// arXiv:2504.20966), which changes the NORMALIZER so a query can attend to
// nothing. Selective attention instead lets tokens prune each OTHER from the
// context; the three compose independently.
//
// # Why no new kernel is needed
//
// Every step — OpMatMul, OpMul (scale), OpReLU, an elementwise triangular mask
// (OpMul), OpCumsum, OpSub, the additive causal mask (OpAdd), OpSoftmax, the
// final OpMatMul — is an existing dispatched op with a first-order VJP, so the
// whole layer (Q/K/V/O projections AND the selection path through head 0) trains
// end to end with NO backend change.
type SelectiveAttention struct {
	Dim     int // model / embedding width (columns of X)
	Heads   int // number of attention heads
	HeadDim int // per-head dimension d = Dim/Heads

	QProj *Linear // query projection  [Dim → Dim]
	KProj *Linear // key projection    [Dim → Dim]
	VProj *Linear // value projection  [Dim → Dim]
	OProj *Linear // output projection [Dim → Dim]

	selScale float64 // 1 → selection on (subtract F); 0 → disabled (plain attention)
}

// SelectiveAttentionOption configures a SelectiveAttention layer
// (functional-options idiom, §C12).
type SelectiveAttentionOption func(*SelectiveAttention)

// WithSelectionDisabled turns the selection term OFF, so the layer computes
// ordinary causal softmax attention (L' = L, no −F). It scales the selection
// bias F by 0 rather than branching the forward pass, so the "selection off"
// path is the SAME code with the −F contribution forced to exactly zero — which
// is why the disabled layer is bit-identical to standard attention (the §V16
// collapse anchor). Use it as a baseline to measure what selection buys, or to
// share one layer type between selective and non-selective stacks. Default: OFF
// (i.e. selection is ON by default — a SelectiveAttention with no options
// applies the −F term).
func WithSelectionDisabled() SelectiveAttentionOption {
	return func(s *SelectiveAttention) { s.selScale = 0 }
}

// NewSelectiveAttention builds a selective multi-head attention layer over model
// width dim with heads attention heads (dim must divide by heads). Selection is
// causal by construction — F accumulates over strictly-earlier queries — so the
// layer is ALWAYS causal (query i attends only to keys j ≤ i); there is no
// bidirectional mode, because "an earlier token masks a later query" only makes
// sense left-to-right (paper §2). The Q, K, V, and output projections are
// Xavier-uniform Linears (Glorot 2010) with zero bias — the standard transformer
// init; selection reuses head 0's logits and adds NO parameters of its own.
// Deterministic via seed (the four projections use seed, seed+1, seed+2, seed+3).
func NewSelectiveAttention(dtype tensor.Dtype, dim, heads int, seed uint64, opts ...SelectiveAttentionOption) (*SelectiveAttention, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("nn: SelectiveAttention dim %d must be positive", dim)
	}
	if heads <= 0 || dim%heads != 0 {
		return nil, fmt.Errorf("nn: SelectiveAttention dim %d not divisible by heads %d", dim, heads)
	}
	s := &SelectiveAttention{Dim: dim, Heads: heads, HeadDim: dim / heads, selScale: 1}
	for _, o := range opts {
		o(s)
	}
	s.QProj = NewLinear(dtype, dim, dim, seed)
	s.KProj = NewLinear(dtype, dim, dim, seed+1)
	s.VProj = NewLinear(dtype, dim, dim, seed+2)
	s.OProj = NewLinear(dtype, dim, dim, seed+3)
	return s, nil
}

func (s *SelectiveAttention) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward runs selective causal attention on x[T, Dim] → [T, Dim]: Q/K/V
// projections, head 0's rectified logits accumulated (strictly-below cumsum over
// the query axis) into the selection bias F, per-head scaled-dot-product logits
// with F subtracted and the causal mask added, the softmax weights, the value
// aggregation, head concatenation, then the output projection. Fully
// differentiable — gradients reach all four projections, including the selection
// path back through head 0's Q/K. With WithSelectionDisabled() the −F term is
// scaled to zero and this reduces to ordinary causal attention.
func (s *SelectiveAttention) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	out, _, _, err := s.forward(ctx, x)
	return out, err
}

// forward is Forward exposing the intermediate per-head softmax weights and the
// scaled selection bias F for white-box tests (masking, causality, cumsum, and
// the collapse anchor). Forward returns only the output.
func (s *SelectiveAttention) forward(ctx *backend.Context, x *tensor.Tensor) (out *tensor.Tensor, weights []*tensor.Tensor, fBias *tensor.Tensor, err error) {
	if x.Ndim() != 2 || x.Shape()[1] != s.Dim {
		return nil, nil, nil, fmt.Errorf("nn: SelectiveAttention expects x [T,%d], got %v", s.Dim, x.Shape())
	}
	t := x.Shape()[0]
	q, err := s.QProj.Forward(ctx, x)
	if err != nil {
		return nil, nil, nil, err
	}
	k, err := s.KProj.Forward(ctx, x)
	if err != nil {
		return nil, nil, nil, err
	}
	v, err := s.VProj.Forward(ctx, x)
	if err != nil {
		return nil, nil, nil, err
	}
	invSqrtD := scalarTensor(x.Dtype(), 1/math.Sqrt(float64(s.HeadDim)))
	mask := qkCausalMask(x.Dtype(), t, t)

	slice := func(m *tensor.Tensor, h int) (*tensor.Tensor, error) {
		lo, hi := h*s.HeadDim, (h+1)*s.HeadDim
		//perfscan:ignore PS3024,PS6018 per-head OpSlice graph dispatch; attention matmul-dominated | resource-class on OpSlice dispatch, no wallclock
		return s.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, m)
	}
	// head-h scaled logits Qₕ·Kₕᵀ/√d → [T,T].
	logits := func(h int) (*tensor.Tensor, error) {
		qh, err := slice(q, h)
		if err != nil {
			return nil, err
		}
		kh, err := slice(k, h)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 OpTranspose dispatch, per-head matmul-dominated
		khT, err := s.exec(ctx, backend.OpTranspose, nil, kh) // [HeadDim,T]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 OpMatMul is the compute; dispatch overhead negligible
		sc, err := s.exec(ctx, backend.OpMatMul, nil, qh, khT) // [T,T]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 OpMul scale dispatch, attention-dominated
		return s.exec(ctx, backend.OpMul, nil, sc, invSqrtD)
	}

	// Selection bias F from head 0 (paper §2): F = selScale · (cumsum(S)−S) where
	// S = strictly-lower-triangular(ReLU(L₀)). Scaling by selScale (1 or 0) via
	// OpMul keeps a single code path; selScale=0 forces F≡0 so the layer collapses
	// to standard attention (§V16 collapse, §B60 uses OpMul not OpAXPY).
	l0, err := logits(0)
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 one-time selection ReLU over [T,T], sub-1pct
	sel, err := s.exec(ctx, backend.OpReLU, nil, l0) // ReLU(L₀) ≥ 0
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 one-time selection Mul over [T,T], sub-1pct
	sel, err = s.exec(ctx, backend.OpMul, nil, sel, selectiveStrictLowerMask(x.Dtype(), t)) // zero diagonal + future
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 one-time Cumsum over [T,T], sub-1pct
	cum, err := s.exec(ctx, backend.OpCumsum, backend.CumsumAttrs{Axis: 0}, sel) // Σ_{i'≤i} over queries
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 one-time Sub over [T,T], sub-1pct
	fCausal, err := s.exec(ctx, backend.OpSub, nil, cum, sel) // Σ_{i'<i}: strictly below the diagonal
	if err != nil {
		return nil, nil, nil, err
	}
	//perfscan:ignore PS3024 one-time selScale Mul, sub-1pct
	fBias, err = s.exec(ctx, backend.OpMul, nil, fCausal, scalarTensor(x.Dtype(), s.selScale)) // ·selScale
	if err != nil {
		return nil, nil, nil, err
	}

	weights = make([]*tensor.Tensor, s.Heads)
	headsOut := make([]*tensor.Tensor, s.Heads)
	for h := range s.Heads {
		var lg *tensor.Tensor
		if h == 0 {
			lg = l0 // reuse head 0's logits (already computed for selection)
		} else if lg, err = logits(h); err != nil {
			return nil, nil, nil, err
		}
		//perfscan:ignore PS3024 per-head Sub L-F, softmax/matmul-dominated
		lp, err := s.exec(ctx, backend.OpSub, nil, lg, fBias) // L' = L − F
		if err != nil {
			return nil, nil, nil, err
		}
		//perfscan:ignore PS3024 per-head Add mask, attention-dominated
		if lp, err = s.exec(ctx, backend.OpAdd, nil, lp, mask); err != nil { // + causal mask
			return nil, nil, nil, err
		}
		//perfscan:ignore PS3024 OpSoftmax is the compute; dispatch negligible
		w, err := s.exec(ctx, backend.OpSoftmax, nil, lp) // [T,T]
		if err != nil {
			return nil, nil, nil, err
		}
		weights[h] = w
		vh, err := slice(v, h)
		if err != nil {
			return nil, nil, nil, err
		}
		//perfscan:ignore PS3024 per-head OpMatMul w*v is the compute
		if headsOut[h], err = s.exec(ctx, backend.OpMatMul, nil, w, vh); err != nil { // [T,HeadDim]
			return nil, nil, nil, err
		}
	}
	concat, err := s.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, headsOut...) // [T,Dim]
	if err != nil {
		return nil, nil, nil, err
	}
	out, err = s.OProj.Forward(ctx, concat)
	if err != nil {
		return nil, nil, nil, err
	}
	return out, weights, fBias, nil
}

// Params returns every trainable tensor — the Q, K, V, and output projections
// (each Linear's W and bias, 8 tensors). Selection adds none of its own (it
// reuses head 0's logits), so the selective and non-selective layers have
// identical parameter sets. Feed this to an optimizer.
func (s *SelectiveAttention) Params() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, l := range []*Linear{s.QProj, s.KProj, s.VProj, s.OProj} {
		ps = append(ps, l.Params()...)
	}
	return ps
}

// selectiveStrictLowerMask builds a [T,T] 0/1 matrix that is 1 strictly below the
// diagonal (j < i) and 0 on and above it. Multiplying the rectified head-0 logits
// by it zeroes both the diagonal (a token can't select itself, paper §2) and the
// future (a query can only vote to mask keys it can actually see), so the running
// selection sum accumulates only valid past selections.
func selectiveStrictLowerMask(dtype tensor.Dtype, t int) *tensor.Tensor {
	m := tensor.New(dtype, tensor.Shape{t, t})
	for i := range t {
		//perfscan:ignore PS1005 one-time [T,T] mask build, sub-1pct vs attention
		for j := 0; j < i; j++ {
			m.SetF64(1, i, j)
		}
	}
	return m
}
