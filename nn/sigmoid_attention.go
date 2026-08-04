package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SigmoidAttention is causal/bidirectional multi-head self-attention that
// replaces the softmax over the keys with an ELEMENT-WISE logistic sigmoid — no
// normalization across the key axis at all. Each attention weight is
//
//	A_ij = σ(S_ij + b)          σ(z) = 1/(1+e⁻ᶻ),   S = Q·Kᵀ/√d_k
//
// and the head output is O = A·V exactly as usual. Contrast ordinary attention,
// where A = softmax(S) forces every query's weights to sum to 1 (a competition
// across keys): here each query–key score is squashed to (0,1) INDEPENDENTLY, so
// the weights need not sum to anything — a query may attend strongly to many
// keys, to one, or (with a sufficiently negative score+bias) to none. This is
// "sigmoid attention" from Ramapuram et al., Apple 2024, "Theory, Analysis, and
// Best Practices for Sigmoid Attention" (arXiv:2409.04431), which proves it is a
// universal approximator and, with the right bias, matches softmax attention in
// quality while dropping the row-wise normalization (and its running max/sum).
//
// # The bias b and its −log(n) default (§C21)
//
// The scalar bias b is the load-bearing knob. Because the weights are not
// normalized, their MAGNITUDE grows with the number of keys n: a query over n
// keys accumulates up to n contributions of size up to ‖v‖, so without a
// counter-term the pre-activation and the output would scale with sequence
// length and destabilize training. The paper's principled default is
//
//	b = −log(n)          n = sequence length (number of keys)
//
// which pins the sigmoid near its small-activation regime as n grows: at a
// neutral score of 0 each weight is σ(−log n) = 1/(1+n) ≈ 1/n, so the n weights
// sum to ≈ 1 — recovering the same O(1) output scale that softmax's
// normalization guarantees, without normalizing. That is why NewSigmoidAttention
// applies b = −log(T) per forward BY DEFAULT (T is the actual sequence length of
// the input, so the term tracks the sequence a batch actually has).
//
// The default can be overridden two ways, orthogonally:
//   - WithSigmoidAttnBias(b) pins a FIXED scalar b for every forward (turns off
//     the per-sequence −log(T) default), for reproducing a specific setting or
//     ablation.
//   - WithLearnableSigmoidBias() makes b a single trainable scalar parameter,
//     learned by the optimizer (the paper reports a learnable bias as a robust
//     choice). Its initial value is the WithSigmoidAttnBias value if one was
//     given, else 0; to start a learnable bias at the principled default, pass
//     WithSigmoidAttnBias(-math.Log(float64(n))) alongside it.
//
// # Bias boundaries (the mechanism, bracketed)
//
// As b → −∞ every weight σ(S+b) → 0, so A = 0 and O = 0 — the query attends to
// NOTHING. As b → +∞ (with no mask) every weight → 1, so O = Σ_j v_j — the query
// attends to EVERYTHING, its output the plain sum of values. Finite b
// interpolates between these; −log(n) sits in the useful middle.
//
// # Causal masking
//
// With WithSigmoidAttnCausal, query position i may attend only to key positions
// j ≤ i: the strictly-future scores get a large negative additive mask BEFORE the
// sigmoid, so σ(−∞) = 0 and those positions contribute exactly nothing to A·V.
// Default OFF (bidirectional). Because the sigmoid is applied per element, no
// "sink"/renormalization subtlety arises — a masked weight is simply 0.
//
// # Not softpick / softmax-off-by-one
//
// This is a FUNDAMENTALLY different mechanism from the SoftpickAttention layer in
// this package (§C18). Softpick / softmax-off-by-one keep a softmax — a
// normalized competition across keys — and only alter its denominator so a query
// can abstain; their weights still couple all keys through one shared sum.
// Sigmoid attention removes the cross-key normalization entirely: every weight is
// an independent σ of one score, coupled to no other key. Softpick tweaks the
// normalizer; sigmoid attention deletes it.
//
// FURTHER READING (§C18): Ramapuram et al. 2024, arXiv:2409.04431 (this layer);
// Vaswani et al. 2017, arXiv:1706.03762 (the softmax baseline it replaces);
// SoftpickAttention in this package (softpick_attention.go) and Miller 2023
// (attention-off-by-one) for the normalized "abstain" alternatives it is NOT.
//
// # Why no new kernel is needed
//
// Sigmoid attention composes entirely from existing dispatched ops, each with a
// first-order VJP, so the whole layer trains end to end with NO backend change:
//
//	S = OpMatMul(Q, Kᵀ);  S = OpMul(S, 1/√d_k)   (scaled scores, per head)
//	S = OpAdd(S, mask)                            (causal only; −1e30 upper triangle)
//	A = OpSigmoid(OpAdd(S, b))                    (element-wise σ of score + bias)
//	O = OpMatMul(A, V)                            (value aggregation, per head)
//
// then head concatenation (OpConcat) and the output projection. The bias is added
// as a [1,1] tensor that broadcasts over the [T,T] score matrix; when learnable
// it is that same tensor, so its gradient flows through OpAdd.
type SigmoidAttention struct {
	Dim     int // model / embedding width (columns of X)
	Heads   int // number of attention heads
	HeadDim int // per-head dimension d_k = Dim/Heads

	QProj *Linear // query projection  [Dim → Dim]
	KProj *Linear // key projection    [Dim → Dim]
	VProj *Linear // value projection  [Dim → Dim]
	OProj *Linear // output projection [Dim → Dim]

	// Bias is the learnable scalar bias tensor [1,1] — non-nil only when the
	// layer was built WithLearnableSigmoidBias; then it is a trainable Param.
	Bias *tensor.Tensor

	causal    bool    // true → query i attends only to keys j ≤ i
	learnable bool    // true → Bias is a trained parameter
	biasSet   bool    // true → biasVal overrides the −log(T) default
	biasVal   float64 // the fixed bias (when biasSet) / learnable init value
}

// SigmoidAttentionOption configures a SigmoidAttention layer
// (functional-options idiom, §C12).
type SigmoidAttentionOption func(*SigmoidAttention)

// WithSigmoidAttnCausal makes the attention causal (autoregressive): query
// position i may attend only to key positions j ≤ i, via a pre-sigmoid additive
// mask of −1e30 on the strictly-upper triangle (future positions), so σ drives
// those weights to exactly 0.
//
// Default OFF (bidirectional, every query sees every key) — masking is orthogonal
// to the sigmoid mechanism; encoder/embedding stacks want the bidirectional form,
// decoder-style language models want this ON.
func WithSigmoidAttnCausal() SigmoidAttentionOption {
	return func(s *SigmoidAttention) { s.causal = true }
}

// WithSigmoidAttnBias pins a FIXED scalar bias b added to every score before the
// sigmoid, overriding the default per-sequence b = −log(T) (§C21). Use it to
// reproduce a specific setting or run a bias ablation. When combined with
// WithLearnableSigmoidBias it instead supplies the learnable bias's INITIAL
// value — e.g. WithSigmoidAttnBias(-math.Log(float64(n))) starts a trained bias
// at the principled −log(n) default.
func WithSigmoidAttnBias(b float64) SigmoidAttentionOption {
	return func(s *SigmoidAttention) {
		s.biasSet = true
		s.biasVal = b
	}
}

// WithLearnableSigmoidBias makes the scalar bias a single trainable parameter
// (added to Params, learned by the optimizer) rather than a constant. Its initial
// value is the WithSigmoidAttnBias value if one was given, else 0. The paper
// reports a learnable bias as a robust choice across settings (§C21,
// arXiv:2409.04431).
func WithLearnableSigmoidBias() SigmoidAttentionOption {
	return func(s *SigmoidAttention) { s.learnable = true }
}

// NewSigmoidAttention builds a sigmoid multi-head attention layer over model
// width dim with heads attention heads (dim must divide by heads). The Q, K, V
// and output projections are Xavier-uniform Linears (Glorot 2010) with zero bias
// — the standard transformer init. By default the scalar bias is the
// per-sequence b = −log(T) applied at each forward (§C21); WithSigmoidAttnBias
// and WithLearnableSigmoidBias change that. Deterministic via seed (the four
// projections use seed, seed+1, seed+2, seed+3).
func NewSigmoidAttention(dtype tensor.Dtype, dim, heads int, seed uint64, opts ...SigmoidAttentionOption) (*SigmoidAttention, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("nn: SigmoidAttention dim %d must be positive", dim)
	}
	if heads <= 0 || dim%heads != 0 {
		return nil, fmt.Errorf("nn: SigmoidAttention dim %d not divisible by heads %d", dim, heads)
	}
	s := &SigmoidAttention{Dim: dim, Heads: heads, HeadDim: dim / heads}
	for _, o := range opts {
		o(s)
	}
	s.QProj = NewLinear(dtype, dim, dim, seed)
	s.KProj = NewLinear(dtype, dim, dim, seed+1)
	s.VProj = NewLinear(dtype, dim, dim, seed+2)
	s.OProj = NewLinear(dtype, dim, dim, seed+3)
	if s.learnable {
		s.Bias = scalarTensor(dtype, s.biasVal) // init = WithSigmoidAttnBias value, else 0
	}
	return s, nil
}

func (s *SigmoidAttention) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// biasTensor resolves the [1,1] bias to add before the sigmoid, given the actual
// sequence length t: the learnable parameter if any, else the fixed
// WithSigmoidAttnBias value, else the principled per-sequence default −log(t)
// (§C21). For t = 1 the default is −log(1) = 0.
func (s *SigmoidAttention) biasTensor(dtype tensor.Dtype, t int) *tensor.Tensor {
	switch {
	case s.learnable:
		return s.Bias
	case s.biasSet:
		return scalarTensor(dtype, s.biasVal)
	default:
		return scalarTensor(dtype, -math.Log(float64(t)))
	}
}

// Forward runs sigmoid attention on x[T, Dim] → [T, Dim]: Q/K/V projections,
// per-head scaled-dot-product scores, the element-wise sigmoid weights
// σ(score + bias) over the keys (no normalization), the value aggregation, head
// concatenation, then the output projection. Fully differentiable — gradients
// reach all four projections and the bias when learnable.
func (s *SigmoidAttention) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != s.Dim {
		return nil, fmt.Errorf("nn: SigmoidAttention expects x [T,%d], got %v", s.Dim, x.Shape())
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
	bias := s.biasTensor(x.Dtype(), t)
	var mask *tensor.Tensor
	if s.causal {
		mask = qkCausalMask(x.Dtype(), t, t)
	}
	heads := make([]*tensor.Tensor, s.Heads)
	for h := range s.Heads {
		lo, hi := h*s.HeadDim, (h+1)*s.HeadDim
		slice := func(m *tensor.Tensor) (*tensor.Tensor, error) {
			//perfscan:ignore PS3024,PS6018 per-head slice dispatch, matmul-dominated MHA loop | per-head head-slice, matmul-dominated
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
		//perfscan:ignore PS3024 per-head transpose dispatch, matmul-dominated
		khT, err := s.exec(ctx, backend.OpTranspose, nil, kh) // [HeadDim, T]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 the per-head matmul dispatch itself, low head count
		sc, err := s.exec(ctx, backend.OpMatMul, nil, qh, khT) // [T, T]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 per-head scale mul, matmul-dominated
		if sc, err = s.exec(ctx, backend.OpMul, nil, sc, invSqrtD); err != nil { // B60: OpMul, scale may be tiny
			return nil, err
		}
		if s.causal {
			//perfscan:ignore PS3024 per-head mask add, matmul-dominated
			if sc, err = s.exec(ctx, backend.OpAdd, nil, sc, mask); err != nil {
				return nil, err
			}
		}
		w, err := sigmoidWeights(ctx, sc, bias) // σ(scores + bias), element-wise, [T, T]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 per-head output matmul dispatch, matmul-dominated
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

// Params returns every trainable tensor — the Q, K, V and output projections
// (each Linear's W and bias, 8 tensors), plus the scalar bias when the layer was
// built WithLearnableSigmoidBias (a 9th tensor). Feed this to an optimizer.
func (s *SigmoidAttention) Params() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, l := range []*Linear{s.QProj, s.KProj, s.VProj, s.OProj} {
		ps = append(ps, l.Params()...)
	}
	if s.learnable {
		ps = append(ps, s.Bias)
	}
	return ps
}

// sigmoidWeights turns a score matrix scores[T, n_keys] into sigmoid attention
// weights: it adds the [1,1] bias (broadcast over every entry) and applies the
// element-wise logistic sigmoid, giving weights[T, n_keys] each in (0,1) with NO
// normalization across the key axis. Every entry is an independent σ of one
// score+bias; rows need not sum to anything. With bias → −∞ all weights → 0
// (attend to nothing); with bias → +∞ all weights → 1 (attend to everything).
// OpAdd and OpSigmoid are both VJP-backed, so the operation is differentiable in
// both the scores and (when learnable) the bias.
func sigmoidWeights(ctx *backend.Context, scores, bias *tensor.Tensor) (*tensor.Tensor, error) {
	//perfscan:ignore PS3038 add+sigmoid fusion, small fraction vs per-head matmuls
	shifted, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{scores, bias}, nil)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3038 sigmoid dispatch, matmul-dominated attention path
	out, err := backend.Execute(ctx, backend.OpSigmoid, []*tensor.Tensor{shifted[0]}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}
