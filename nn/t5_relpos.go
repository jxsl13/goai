package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// T5 relative position bias (Raffel, Shazeer, Roberts, Lee, Narang, Matena, Zhou, Li & Liu 2020,
// "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" / T5,
// arXiv:1910.10683 §2.1; the HuggingFace T5Attention._relative_position_bucket). T5 drops absolute
// positional encodings entirely and instead adds a LEARNED per-head scalar to each attention logit
// (before softmax), chosen by a BUCKET of the relative position (key position − query position),
// with the bias parameters SHARED across all layers. It is a content-independent, position-only
// bias — no sinusoids (§R230) and no rotation (RoPE) — so it is the distinct relative scheme behind
// T5/mT5/ByT5 and many encoders. Nearby relative positions each get their own exact bucket while
// distant ones share logarithmically-spaced buckets, so the bias generalizes to longer sequences.

// Default bucketing hyperparameters (T5): 32 buckets over a max distance of 128.
const (
	T5DefaultNumBuckets  = 32
	T5DefaultMaxDistance = 128
)

// T5RelativePositionBucket maps a relative position (key − query) to its bucket index (Raffel et
// al. 2020; the exact HF algorithm). For bidirectional=true (encoder) the sign of the relative
// position is encoded by offsetting into the upper half of the buckets; for bidirectional=false
// (causal decoder) future positions collapse to bucket 0. Small |relative position| (< numBuckets/4
// bidirectional, < numBuckets/2 causal) map to exact buckets; larger ones are placed on a log scale
// up to maxDistance and clamped. numBuckets must be even and ≥ 4 (≥ 2 for causal); maxDistance must
// exceed the exact range.
func T5RelativePositionBucket(relativePosition int, bidirectional bool, numBuckets, maxDistance int) int {
	bucket := 0
	n := numBuckets
	rp := relativePosition
	if bidirectional {
		n /= 2
		if rp > 0 {
			bucket += n
		}
		if rp < 0 {
			rp = -rp // abs
		}
	} else {
		if rp > 0 { // causal: keep only past (relativePosition ≤ 0), future → 0
			rp = 0
		}
		rp = -rp
	}
	// rp is now ≥ 0.
	maxExact := n / 2
	if rp < maxExact {
		return bucket + rp // exact bucket for small distances
	}
	// logarithmically-spaced bucket for larger distances.
	large := maxExact + int(math.Log(float64(rp)/float64(maxExact))/math.Log(float64(maxDistance)/float64(maxExact))*float64(n-maxExact))
	if large > n-1 {
		large = n - 1
	}
	return bucket + large
}

// T5RelativePositionBuckets builds the [queryLen, keyLen] matrix of bucket indices, entry [i][j] =
// T5RelativePositionBucket(j−i, …) — the indices used to gather the learned [numBuckets, numHeads]
// bias for the attention scores. numBuckets ≤ 0 → T5DefaultNumBuckets, maxDistance ≤ 0 →
// T5DefaultMaxDistance.
func T5RelativePositionBuckets(queryLen, keyLen int, bidirectional bool, numBuckets, maxDistance int) ([][]int, error) {
	if numBuckets <= 0 {
		numBuckets = T5DefaultNumBuckets
	}
	if maxDistance <= 0 {
		maxDistance = T5DefaultMaxDistance
	}
	if queryLen <= 0 || keyLen <= 0 {
		return nil, fmt.Errorf("nn: T5RelativePositionBuckets needs positive lengths, got %d/%d", queryLen, keyLen)
	}
	if numBuckets%2 != 0 {
		return nil, fmt.Errorf("nn: T5RelativePositionBuckets numBuckets must be even, got %d", numBuckets)
	}
	half := numBuckets
	if bidirectional {
		half = numBuckets / 2
	}
	if half < 2 || maxDistance <= half/2 {
		return nil, fmt.Errorf("nn: T5RelativePositionBuckets numBuckets=%d/maxDistance=%d too small", numBuckets, maxDistance)
	}
	out := make([][]int, queryLen)
	for i := range queryLen {
		out[i] = make([]int, keyLen)
		for j := range keyLen {
			out[i][j] = T5RelativePositionBucket(j-i, bidirectional, numBuckets, maxDistance)
		}
	}
	return out, nil
}

// T5RelativeBias is the learned relative-position attention bias of T5 (Raffel et al. 2020, §R232):
// a trainable table indexed by the relative-position bucket (T5RelativePositionBucket) that adds a
// per-head scalar to the attention logits, shared across layers. It turns the bucketing into a
// usable trainable module.
type T5RelativeBias struct {
	Table         *tensor.Tensor // [numBuckets, numHeads], trainable (init 0 ⇒ no bias at start)
	NumBuckets    int            // number of relative-position buckets
	MaxDistance   int            // distance beyond which buckets saturate
	Bidirectional bool           // encoder (true) vs causal decoder (false) bucketing
}

// NewT5RelativeBias builds a relative-bias module with a zero-initialized [numBuckets, numHeads]
// table (so it adds nothing until trained). numBuckets must be even and ≥ 4 (≥ 2 causal).
func NewT5RelativeBias(numBuckets, numHeads, maxDistance int, bidirectional bool, dtype tensor.Dtype) (*T5RelativeBias, error) {
	if numBuckets <= 0 {
		numBuckets = T5DefaultNumBuckets
	}
	if maxDistance <= 0 {
		maxDistance = T5DefaultMaxDistance
	}
	if numHeads <= 0 {
		return nil, fmt.Errorf("nn: T5RelativeBias needs positive numHeads, got %d", numHeads)
	}
	if _, err := T5RelativePositionBuckets(1, 1, bidirectional, numBuckets, maxDistance); err != nil {
		return nil, err // reuse the numBuckets/maxDistance validation
	}
	return &T5RelativeBias{
		Table:         tensor.New(dtype, tensor.Shape{numBuckets, numHeads}),
		NumBuckets:    numBuckets,
		MaxDistance:   maxDistance,
		Bidirectional: bidirectional,
	}, nil
}

// Bias returns the [queryLen, keyLen, numHeads] relative-position bias to add to the attention
// logits: Bias[i][j][h] = Table[bucket(j−i)][h]. Built by a one-hot gather (matmul) so it is
// differentiable w.r.t. the Table (each bucket accumulates the gradient of every position pair it
// covers). The [q,k,heads] layout matches attention scores laid out with the head axis last.
func (b *T5RelativeBias) Bias(ctx *backend.Context, queryLen, keyLen int) (*tensor.Tensor, error) {
	buckets, err := T5RelativePositionBuckets(queryLen, keyLen, b.Bidirectional, b.NumBuckets, b.MaxDistance)
	if err != nil {
		return nil, err
	}
	numHeads := b.Table.Shape()[1]
	// one-hot [queryLen·keyLen, numBuckets] selecting each pair's bucket (constant).
	oneHot := tensor.New(b.Table.Dtype(), tensor.Shape{queryLen * keyLen, b.NumBuckets})
	for i := range queryLen {
		for j := range keyLen {
			oneHot.SetF64(1, i*keyLen+j, buckets[i][j])
		}
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	flat, err := ex(backend.OpMatMul, nil, oneHot, b.Table) // [q·k, numHeads]
	if err != nil {
		return nil, err
	}
	return ex(backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{queryLen, keyLen, numHeads}}, flat)
}

// Params returns the trainable table.
func (b *T5RelativeBias) Params() []*tensor.Tensor { return []*tensor.Tensor{b.Table} }
