package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// StickBreakingAttention is causal multi-head self-attention that REPLACES the
// softmax over keys with a stick-breaking allocation processed from the most
// recent key backward, giving inherent recency bias and strong length
// generalization WITHOUT any explicit position encoding (Tan, Shen, Panda, Chao
// et al. 2024, "Stick-breaking Attention", arXiv:2410.17980).
//
// # The mechanism (paper §3)
//
// Ordinary attention forms A = softmax(QKᵀ/√d) over the keys, which forces the
// weights of every query row to sum to exactly 1. Stick-breaking instead walks
// the keys j ≤ i (causal) for query i from the MOST RECENT key j = i backward to
// the oldest, and hands each key a slice of a unit "stick" of attention mass:
//
//	z_{ij} = q_i·k_j/√d           (+ optional learnable per-head bias)
//	β_{ij} = σ(z_{ij})            (a per-key "break" probability in (0,1))
//	A_{ij} = β_{ij} · ∏_{j < k ≤ i} (1 − β_{ik})
//
// The most-recent key j = i takes A_{ii} = β_{ii} of the whole stick; each older
// key j then takes its own fraction β_{ij} of whatever the more-recent keys k > j
// left behind. Because a fraction may be left unallocated, the row sum obeys
//
//	Σ_j A_{ij} = 1 − ∏_{k ≤ i} (1 − β_{ik}) ≤ 1
//
// — the stick-breaking SIGNATURE: unlike softmax, a query need not spend all its
// mass, and the deficit 1 − Σ_j A_{ij} is a valid "attend to nothing" remainder.
// Then O = Σ_j A_{ij} v_j per head, the heads are concatenated, and W_O mixes.
//
// # Why recency, why no position encoding
//
// The nested product ∏_{k>j}(1−β_{ik}) geometrically discounts a key by every
// break that happened AFTER it, so — all else equal — nearer keys keep more of
// the stick. That decay is intrinsic to the ordering of the keys, not bolted on
// through RoPE / ALiBi / sinusoids, which is precisely why the paper reports the
// pattern transfers to sequences far longer than those seen in training: the
// allocation rule is the same at every length. This layer is therefore inherently
// ordered and inherently causal — there is no bidirectional form and no option to
// disable the causal mask (§C21): a query at position i is defined only over keys
// j ≤ i, and "start the stick at the most recent key" is meaningless without a
// front to the sequence.
//
// # Numerical stability — log-space, reverse cumulative sum (paper, and §V16.3)
//
// Materializing ∏(1−β) directly underflows for long contexts, so the weights are
// built in log space and exponentiated once at the end:
//
//	log A_{ij} = logσ(z_{ij}) + Σ_{j < k ≤ i} logσ(−z_{ik})
//
// using logσ(x) = −softplus(−x). Let L_{ik} = logσ(−z_{ik}) = −softplus(z_{ik}),
// masked to 0 for future keys k > i. The inner term is then a REVERSE cumulative
// sum of L along the key axis, computed as (row total) − (inclusive prefix sum):
//
//	S_{ij} = Σ_{k ≤ i} L_{ik} − Σ_{k ≤ j} L_{ik} = Σ_{j < k ≤ i} L_{ik}
//
// so log A = logσ(z) + S, then A = exp(log A) with the strictly-upper triangle
// (future keys, j > i) zeroed by a multiplicative 0/1 causal mask — which makes
// A_{ij} = 0 bit-exactly there (§V16.4). Every step — OpSoftplus, OpNeg, OpMul
// (the mask; a possibly-zero scale, so OpMul not OpAXPY per §C-B60), OpSum,
// OpCumsum, OpSub, OpAdd, OpExp, OpMatMul — is a dispatched op with a first-order
// VJP, so the whole layer trains end to end with NO new backend kernel and
// gradients flow to Q/K/V/O (and the bias if enabled), §V16.5.
//
// # Contrast with the other non-softmax attentions in this package (§C18)
//
// SoftpickAttention (softpick_attention.go) keeps the softmax shape but appends a
// fixed-0 "sink" logit (softmax-off-by-one), so its rows also sum to < 1 — but the
// deficit is a single global sink, not a per-key recency decay, and its weights
// are still a normalized competition among the keys. SigmoidAttention
// (sigmoid_attention.go) applies an independent σ per key with NO normalization
// and NO cross-key coupling, so it has neither the stick's Σ ≤ 1 bound nor its
// ordering-induced recency. Stick-breaking is the only one whose weights are a
// SEQUENTIAL allocation down the key order, which is what buys the length
// generalization. See also Vaswani et al. 2017 (arXiv:1706.03762) for the softmax
// baseline this departs from.
//
// FURTHER READING (§C18): Tan et al. 2024 (arXiv:2410.17980, the paper);
// Miller 2023 "Attention Is Off By One" and Zuhri et al. 2025 (arXiv:2504.20966,
// Softpick) for the sink/off-by-one relatives contrasted above; Press et al. 2022
// "Train Short, Test Long: Attention with Linear Biases Enables Input Length
// Extrapolation" (ALiBi, arXiv:2108.12409) for the length-generalization goal that
// stick-breaking reaches WITHOUT an explicit bias schedule.
type StickBreakingAttention struct {
	Dim     int // model / embedding width (columns of X)
	Heads   int // number of attention heads
	HeadDim int // per-head dimension d_k = Dim/Heads

	QProj *Linear // query projection  [Dim → Dim]
	KProj *Linear // key projection    [Dim → Dim]
	VProj *Linear // value projection  [Dim → Dim]
	OProj *Linear // output projection [Dim → Dim]

	// biases holds one learnable [1,1] scalar per head, added to that head's
	// pre-sigmoid scores z; nil when the bias option is off. A positive bias
	// raises every break probability β, sharpening the recency decay; a negative
	// bias flattens it. Kept per-head so different heads can learn different
	// decay rates. Initialized to 0 (bias-free stick-breaking).
	biases []*tensor.Tensor
}

// StickBreakingAttentionOption configures a StickBreakingAttention layer
// (functional-options idiom, §C12). Stick-breaking is inherently causal and
// inherently ordered, so — unlike SoftpickAttention — there is deliberately no
// option to make it bidirectional or to disable the causal mask (§C21).
type StickBreakingAttentionOption func(*StickBreakingAttention)

// WithStickBreakingBias adds a learnable per-head scalar bias b_h to the
// pre-sigmoid scores, so β_{ij} = σ(q_i·k_j/√d + b_h). The bias shifts every
// break probability of the head at once: a larger b_h pushes β toward 1 (each key
// grabs more of the remaining stick, i.e. a SHARPER recency decay and shorter
// effective context), a smaller b_h pushes β toward 0 (a FLATTER, longer-range
// allocation). Kept per-head (one scalar per attention head, initialized to 0) so
// heads can specialize to different decay rates — the paper notes such a bias lets
// the model tune how fast the stick is consumed without any position encoding.
// Default OFF: bias-free stick-breaking, whose recency comes purely from the
// ordering of the keys.
func WithStickBreakingBias() StickBreakingAttentionOption {
	return func(s *StickBreakingAttention) {
		s.biases = make([]*tensor.Tensor, 0) // sentinel: allocated per-head in the constructor
	}
}

// NewStickBreakingAttention builds a stick-breaking multi-head attention layer
// over model width dim with heads attention heads (dim must divide by heads). The
// Q, K, V, and output projections are Xavier-uniform Linears (Glorot 2010) with
// zero bias — the standard transformer init; the stick-breaking mechanism itself
// adds no parameters over ordinary attention unless WithStickBreakingBias is
// passed, which adds one zero-initialized scalar per head. Deterministic via seed
// (the four projections use seed, seed+1, seed+2, seed+3). The layer is always
// causal (§C21); there is no bidirectional variant.
func NewStickBreakingAttention(dtype tensor.Dtype, dim, heads int, seed uint64, opts ...StickBreakingAttentionOption) (*StickBreakingAttention, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("nn: StickBreakingAttention dim %d must be positive", dim)
	}
	if heads <= 0 || dim%heads != 0 {
		return nil, fmt.Errorf("nn: StickBreakingAttention dim %d not divisible by heads %d", dim, heads)
	}
	s := &StickBreakingAttention{Dim: dim, Heads: heads, HeadDim: dim / heads}
	for _, o := range opts {
		o(s)
	}
	if s.biases != nil {
		s.biases = make([]*tensor.Tensor, heads)
		for h := range heads {
			s.biases[h] = tensor.New(dtype, tensor.Shape{1, 1}) // zero-initialized scalar
		}
	}
	s.QProj = NewLinear(dtype, dim, dim, seed)
	s.KProj = NewLinear(dtype, dim, dim, seed+1)
	s.VProj = NewLinear(dtype, dim, dim, seed+2)
	s.OProj = NewLinear(dtype, dim, dim, seed+3)
	return s, nil
}

func (s *StickBreakingAttention) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward runs stick-breaking attention on x[T, Dim] → [T, Dim]: Q/K/V
// projections, per-head scaled-dot-product scores, the log-space stick-breaking
// weights over the causal keys (reverse-cumsum of logσ(−z), see the type doc),
// the value aggregation, head concatenation, then the output projection. Fully
// differentiable — gradients reach all four projections and the per-head bias if
// enabled.
func (s *StickBreakingAttention) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != s.Dim {
		return nil, fmt.Errorf("nn: StickBreakingAttention expects x [T,%d], got %v", s.Dim, x.Shape())
	}
	t := x.Shape()[0]
	q, err := s.QProj.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	k, err := s.KProj.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	v, err := s.VProj.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	invSqrtD := scalarTensor(x.Dtype(), 1/math.Sqrt(float64(s.HeadDim)))
	mask := sbCausalMask01(x.Dtype(), t) // 1 where key j ≤ query i, else 0
	heads := make([]*tensor.Tensor, s.Heads)
	for h := range s.Heads {
		lo, hi := h*s.HeadDim, (h+1)*s.HeadDim
		slice := func(m *tensor.Tensor) (*tensor.Tensor, error) {
			return s.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, m)
		}
		qh, err := slice(q)
		if err != nil {
			return nil, err
		}
		kh, err := slice(k)
		if err != nil {
			return nil, err
		}
		vh, err := slice(v)
		if err != nil {
			return nil, err
		}
		khT, err := s.exec(ctx, backend.OpTranspose, nil, kh) // [HeadDim, T]
		if err != nil {
			return nil, err
		}
		z, err := s.exec(ctx, backend.OpMatMul, nil, qh, khT) // [T, T]
		if err != nil {
			return nil, err
		}
		if z, err = s.exec(ctx, backend.OpMul, nil, z, invSqrtD); err != nil {
			return nil, err
		}
		if s.biases != nil {
			if z, err = s.exec(ctx, backend.OpAdd, nil, z, s.biases[h]); err != nil { // broadcast [1,1]
				return nil, err
			}
		}
		w, err := stickBreakingWeights(ctx, z, mask) // [T, T], causal
		if err != nil {
			return nil, err
		}
		if heads[h], err = s.exec(ctx, backend.OpMatMul, nil, w, vh); err != nil { // [T, HeadDim]
			return nil, err
		}
	}
	concat, err := s.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...) // [T, Dim]
	if err != nil {
		return nil, err
	}
	return s.OProj.Forward(ctx, concat)
}

// Params returns every trainable tensor — the Q, K, V, and output projections
// (each Linear's W and bias, 8 tensors), followed by the per-head stick-breaking
// bias scalars when WithStickBreakingBias is enabled (one per head). Feed this to
// an optimizer.
func (s *StickBreakingAttention) Params() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, l := range []*Linear{s.QProj, s.KProj, s.VProj, s.OProj} {
		ps = append(ps, l.Params()...)
	}
	ps = append(ps, s.biases...) // no-op when nil
	return ps
}

// stickBreakingWeights turns a per-head score matrix z[T, T] into stick-breaking
// attention weights A[T, T] in log space (numerically stable, §V16.3):
//
//	L      = logσ(−z) = −softplus(z)        break-COMPLEMENT log-probs
//	Lm     = L ⊙ mask                       zero the future (keys k > i)
//	S      = rowsum(Lm) − cumsum(Lm)        reverse cumulative sum along keys:
//	                                        S_{ij} = Σ_{j < k ≤ i} L_{ik}
//	logA   = logσ(z) + S = −softplus(−z) + S
//	A      = exp(logA) ⊙ mask               causal, A_{ij}=0 bit-exact for j > i
//
// so A_{ij} = β_{ij}·∏_{j<k≤i}(1−β_{ik}) with β = σ(z). mask is the [T,T] 0/1
// causal mask (1 where key j ≤ query i). Every op is VJP-backed, so the result is
// differentiable; the row total minus the inclusive prefix cumsum is how the
// reverse cumulative sum is expressed with the forward-only OpCumsum. Because Lm
// zeroes future keys, S is 0 in the strictly-upper triangle, keeping logA finite
// there (no NaN) before the final mask sets those entries to exactly 0.
func stickBreakingWeights(ctx *backend.Context, z, mask *tensor.Tensor) (*tensor.Tensor, error) {
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	// L = logσ(−z) = −softplus(z), masked to 0 on future keys.
	sp, err := ex(backend.OpSoftplus, nil, z)
	if err != nil {
		return nil, err
	}
	l, err := ex(backend.OpNeg, nil, sp)
	if err != nil {
		return nil, err
	}
	lm, err := ex(backend.OpMul, nil, l, mask) // possibly-zero scale → OpMul (§C-B60)
	if err != nil {
		return nil, err
	}
	// Reverse cumulative sum along keys: S = rowtotal − inclusive-prefix-cumsum.
	total, err := ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{1}, KeepDims: true}, lm) // [T,1]
	if err != nil {
		return nil, err
	}
	pincl, err := ex(backend.OpCumsum, backend.CumsumAttrs{Axis: 1}, lm) // [T,T]
	if err != nil {
		return nil, err
	}
	sfx, err := ex(backend.OpSub, nil, total, pincl) // broadcast [T,1]−[T,T] → [T,T]
	if err != nil {
		return nil, err
	}
	// logβ = logσ(z) = −softplus(−z).
	negz, err := ex(backend.OpNeg, nil, z)
	if err != nil {
		return nil, err
	}
	spNeg, err := ex(backend.OpSoftplus, nil, negz)
	if err != nil {
		return nil, err
	}
	logBeta, err := ex(backend.OpNeg, nil, spNeg)
	if err != nil {
		return nil, err
	}
	logA, err := ex(backend.OpAdd, nil, logBeta, sfx)
	if err != nil {
		return nil, err
	}
	a, err := ex(backend.OpExp, nil, logA)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpMul, nil, a, mask) // enforce A_{ij}=0 for j>i, bit-exact
}

// sbCausalMask01 builds a square [t,t] multiplicative causal mask: 1.0 where key
// j ≤ query i (the lower triangle, inclusive), 0.0 in the strictly-upper triangle
// (future keys). Distinct from qkCausalMask, which is an ADDITIVE −∞ mask for
// softmax; stick-breaking multiplies by this 0/1 mask both to zero the future
// contributions to the reverse cumsum and to force A_{ij}=0 exactly for j>i.
func sbCausalMask01(dtype tensor.Dtype, t int) *tensor.Tensor {
	m := tensor.New(dtype, tensor.Shape{t, t})
	for i := range t {
		for j := 0; j <= i; j++ {
			m.SetF64(1, i, j)
		}
	}
	return m
}
