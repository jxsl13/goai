package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// SoftpickAttention is causal/bidirectional multi-head self-attention that
// replaces the ordinary softmax with softmax-off-by-one (a.k.a. "softmax_1" or
// "quiet attention"): the attention weights over the keys are
//
//	softmax_1(s)_i = exp(s_i) / (1 + Σ_j exp(s_j))
//
// — the SAME as softmax but with a +1 added to the denominator. That extra 1 is
// a "sink" of probability mass a query can dump onto when NO key is relevant.
// In ordinary softmax the weights are forced to sum to exactly 1, so even a
// query whose scores are all very negative (nothing worth attending to) still
// has to distribute a full unit of weight across the keys and picks the
// least-bad one; softmax_1 lets that whole unit drain into the +1 instead, so
// the query can attend to NOTHING and its output collapses toward zero.
//
// In plain terms: normal attention is a committee that must always cast all its
// votes on the candidates present; softmax-off-by-one gives the committee an
// "abstain" option. Empirically this abstain valve is what stops transformers
// from parking huge, meaningless activations on a few "sink" tokens (usually the
// first token / delimiters) — the attention-sink and massive-activation outliers
// that hurt quantization and interpretability. This denominator-+1 form is Evan
// Miller's "Attention Is Off By One" (2023, https://www.evanmiller.org/
// attention-is-off-by-one.html). A closely related, independently proposed
// remedy for the same sink/massive-activation pathology is "Softpick" (Zuhri et
// al. 2025, "Softpick: No Attention Sink, No Massive Activations with Rectified
// Softmax", arXiv:2504.20966), which instead RECTIFIES the softmax (a
// non-sum-to-one ReLU form, not a +1 in the denominator); this layer implements
// the cleaner off-by-one variant, which composes from existing ops.
//
// FURTHER READING (§C18): Miller 2023 (the off-by-one write-up, above); Zuhri et
// al. 2025 arXiv:2504.20966 (Softpick); Xiao et al. 2024 "Efficient Streaming
// Language Models with Attention Sinks" (arXiv:2309.17453) and Sun et al. 2024
// "Massive Activations in Large Language Models" (arXiv:2402.17762) for the
// sink/outlier phenomenon this targets; Vaswani et al. 2017 (arXiv:1706.03762)
// and Jurafsky & Martin, Speech and Language Processing (3rd ed.), ch. on
// self-attention, for the standard-softmax baseline.
//
// # Why no new kernel is needed
//
// softmax_1 over a score vector s is EXACTLY the ordinary softmax over the
// augmented vector [s, 0] — appending one extra logit fixed at 0, whose
// exponential is exp(0) = 1, is precisely the +1 in the denominator. So this
// layer computes attention as:
//
//	S      = Q·Kᵀ / √d_k            (+ additive causal mask, if enabled)
//	S'     = [S, 0]                 (append one zero "sink" column per query row)
//	A'     = softmax(S')            (ordinary softmax over the augmented axis)
//	A      = A'[:, :n_keys]         (drop the sink column — its weight is the
//	                                 "attend to nothing" mass)
//	O      = A·V                    (per head; heads concatenated, then W_O)
//
// Every step — OpMatMul, additive mask (OpAdd), OpConcat (append the zero
// column), OpSoftmax, OpSlice (drop the sink column), OpMatMul — is an existing
// dispatched op with a first-order VJP, so the whole layer trains end to end
// with NO backend change and NO new fused kernel. Dropping the sink column after
// the softmax is what makes A sum to LESS than 1 (the deficit is the sink mass),
// which is the entire point.
type SoftpickAttention struct {
	Dim     int // model / embedding width (columns of X)
	Heads   int // number of attention heads
	HeadDim int // per-head dimension d_k = Dim/Heads

	QProj *Linear // query projection  [Dim → Dim]
	KProj *Linear // key projection    [Dim → Dim]
	VProj *Linear // value projection  [Dim → Dim]
	OProj *Linear // output projection [Dim → Dim]

	causal bool // true → query i attends only to keys j ≤ i
}

// SoftpickAttentionOption configures a SoftpickAttention layer
// (functional-options idiom, §C12).
type SoftpickAttentionOption func(*SoftpickAttention)

// WithSoftpickCausal makes the attention causal (autoregressive): query position
// i may attend only to key positions j ≤ i, via a pre-softmax additive mask of
// −1e30 on the strictly-upper triangle (future positions).
//
// Default OFF (bidirectional, every query sees every key) — the mask is a pure
// masking choice orthogonal to the softmax_1 mechanism, and encoder/embedding
// stacks want the bidirectional form. Turn it ON for decoder-style language
// models, which is where the attention-sink pathology softmax_1 targets is most
// pronounced (Xiao et al. 2024, arXiv:2309.17453). The appended sink column is
// NEVER masked — "attend to nothing" must stay available at every position,
// including the first, whose sole real key is itself.
func WithSoftpickCausal() SoftpickAttentionOption {
	return func(s *SoftpickAttention) { s.causal = true }
}

// NewSoftpickAttention builds a softmax-off-by-one multi-head attention layer
// over model width dim with heads attention heads (dim must divide by heads).
// The Q, K, V, and output projections are Xavier-uniform Linears (Glorot 2010)
// with zero bias — the standard transformer init; the softmax_1 mechanism adds
// no extra parameters over ordinary attention, so there is no init to tune for
// it. Deterministic via seed (the four projections use seed, seed+1, seed+2,
// seed+3).
func NewSoftpickAttention(dtype tensor.Dtype, dim, heads int, seed uint64, opts ...SoftpickAttentionOption) (*SoftpickAttention, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("nn: SoftpickAttention dim %d must be positive", dim)
	}
	if heads <= 0 || dim%heads != 0 {
		return nil, fmt.Errorf("nn: SoftpickAttention dim %d not divisible by heads %d", dim, heads)
	}
	s := &SoftpickAttention{Dim: dim, Heads: heads, HeadDim: dim / heads}
	for _, o := range opts {
		o(s)
	}
	s.QProj = NewLinear(dtype, dim, dim, seed)
	s.KProj = NewLinear(dtype, dim, dim, seed+1)
	s.VProj = NewLinear(dtype, dim, dim, seed+2)
	s.OProj = NewLinear(dtype, dim, dim, seed+3)
	return s, nil
}

func (s *SoftpickAttention) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward runs softmax-off-by-one attention on x[T, Dim] → [T, Dim]: Q/K/V
// projections, per-head scaled-dot-product scores, the augmented-softmax
// (softmax_1) weights over the keys, the value aggregation, head concatenation,
// then the output projection. Fully differentiable — gradients reach all four
// projections. The appended sink column carries the fixed logit 0.
func (s *SoftpickAttention) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	return s.forward(ctx, x, 0)
}

// forward is Forward with a configurable sink logit. Forward pins it at 0 (the
// softmax_1 definition); the collapse test drives it to −∞ (≈ −1e30) to recover
// ordinary softmax attention, proving the augmentation machinery reduces exactly.
func (s *SoftpickAttention) forward(ctx *backend.Context, x *tensor.Tensor, sinkLogit float64) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != s.Dim {
		return nil, fmt.Errorf("nn: SoftpickAttention expects x [T,%d], got %v", s.Dim, x.Shape())
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
	var mask *tensor.Tensor
	if s.causal {
		mask = qkCausalMask(x.Dtype(), t, t)
	}
	heads := make([]*tensor.Tensor, s.Heads)
	for h := range s.Heads {
		lo, hi := h*s.HeadDim, (h+1)*s.HeadDim
		slice := func(m *tensor.Tensor) (*tensor.Tensor, error) {
			//perfscan:ignore PS3024,PS6018 per-head slice dispatch, attention matmul-dominated | slice-closure alloc per head, resource-only trivial
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
		//perfscan:ignore PS3024 per-head transpose, matmul-dominated attention
		khT, err := s.exec(ctx, backend.OpTranspose, nil, kh) // [HeadDim, T]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 this IS the dominant per-head QK matmul
		sc, err := s.exec(ctx, backend.OpMatMul, nil, qh, khT) // [T, T]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 scalar OpMul, <1pct vs per-head matmul
		if sc, err = s.exec(ctx, backend.OpMul, nil, sc, invSqrtD); err != nil {
			return nil, err
		}
		if s.causal {
			//perfscan:ignore PS3024 mask OpAdd, matmul-dominated
			if sc, err = s.exec(ctx, backend.OpAdd, nil, sc, mask); err != nil {
				return nil, err
			}
		}
		w, err := softpickWeights(ctx, sc, sinkLogit) // softmax_1 over keys, [T, T]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 dominant per-head weight-value matmul
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
// (each Linear's W and bias, 8 tensors). Feed this to an optimizer.
func (s *SoftpickAttention) Params() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, l := range []*Linear{s.QProj, s.KProj, s.VProj, s.OProj} {
		ps = append(ps, l.Params()...)
	}
	return ps
}

// softpickWeights turns a score matrix scores[T, n_keys] into softmax-off-by-one
// attention weights: it appends one extra "sink" column fixed at sinkLogit,
// takes an ordinary row-softmax over the [T, n_keys+1] augmented matrix, then
// slices the sink column back off — leaving weights[T, n_keys] whose rows sum to
// LESS than 1 by exactly the sink mass exp(sinkLogit)/(exp(sinkLogit)+Σ_j
// exp(scores_j)). With sinkLogit = 0 (the softmax_1 definition) the sink mass is
// 1/(1+Σ exp(scores_j)); driving sinkLogit → −∞ zeroes the sink column and
// recovers the ordinary softmax (rows sum to 1). OpConcat, OpSoftmax and OpSlice
// are all VJP-backed, so the whole operation is differentiable.
func softpickWeights(ctx *backend.Context, scores *tensor.Tensor, sinkLogit float64) (*tensor.Tensor, error) {
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	t, nKeys := scores.Shape()[0], scores.Shape()[1]
	sink := tensor.New(scores.Dtype(), tensor.Shape{t, 1}) // zero-filled = logit 0
	if sinkLogit != 0 {
		//perfscan:ignore PS1001 O(T) sink fill only when sinkLogit!=0, dwarfed by O(T2) softmax
		for i := range t {
			sink.SetF64(sinkLogit, i, 0)
		}
	}
	aug, err := ex(backend.OpConcat, backend.ConcatAttrs{Axis: 1}, scores, sink) // [T, nKeys+1]
	if err != nil {
		return nil, err
	}
	sm, err := ex(backend.OpSoftmax, nil, aug) // ordinary softmax over the augmented axis
	if err != nil {
		return nil, err
	}
	return ex(backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 0, End: nKeys}, sm) // drop the sink column
}
