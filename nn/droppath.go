package nn

import (
	"fmt"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// DropPath is stochastic depth (Huang, Sun, Liu, Sedra & Weinberger 2016, "Deep
// Networks with Stochastic Depth", ECCV, arXiv:1603.09382) in the modern per-sample
// inverted form used by timm/DeiT/Swin/ConvNeXt. Where Dropout zeros individual
// activations, DropPath drops the ENTIRE residual BRANCH for a sample: applied to a
// branch output f(x) inside a residual block y = x + DropPath(f(x)), it randomly
// removes whole layers per example, so the network trains as an ensemble of
// shallower nets and deep transformers train more stably.
//
// During training, for each sample in the batch independently the branch output is
// either zeroed (with drop probability Rate) or kept and divided by keep = 1−Rate
// (inverted dropout, so E[out]=x); at evaluation it is the identity. The mask is one
// value PER SAMPLE (broadcast over every other dimension) — the whole branch output
// for a sample is dropped together, not element-wise.
//
// MODE DEFAULTS TO TRAINING, AND NOTHING SWITCHES IT FOR YOU. A freshly built
// DropPath has Training == true, so running inference without switching modes
// keeps dropping whole residual branches — silently, with no error. Before
// predicting, call [DropPath.Eval] on the layer, or [SetTrain](model, false) /
// [Sequential.Eval] on the model containing it, which propagate to every nested
// mode-dependent layer; [DropPath.Train] (or SetTrain(model, true)) resumes
// training. See [Mode] and [SetTrain].
//
// No learnable parameters; the per-sample mask is multiplied in through the recording
// context, so autograd differentiates it via the Mul VJP (gradient flows only through
// kept, up-scaled samples).
type DropPath struct {
	Rate     float64 // per-sample drop probability, in [0,1)
	Training bool    // when false, Forward is the identity (inference)
	rng      *rand.Rand
}

// NewDropPath builds a stochastic-depth layer with the given per-sample drop rate,
// in training mode. The seed makes the mask sequence deterministic. Panics on an
// out-of-range rate.
func NewDropPath(rate float64, seed uint64) *DropPath {
	if rate < 0 || rate >= 1 {
		panic(fmt.Sprintf("nn: droppath rate %g out of [0,1)", rate))
	}
	return &DropPath{Rate: rate, Training: true, rng: rand.New(rand.NewPCG(seed, 0x24a7))}
}

// Train switches the layer to training mode (path dropping active).
func (d *DropPath) Train() { d.Training = true }

// Eval switches the layer to evaluation mode (Forward is the identity).
func (d *DropPath) Eval() { d.Training = false }

// Forward drops whole per-sample branch outputs in training mode, or returns x
// unchanged in eval mode (or when Rate is 0). x is [batch, ...]; the drop decision
// is made per sample (the first axis) and broadcast over the rest.
func (d *DropPath) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if !d.Training || d.Rate == 0 {
		return x, nil
	}
	if x.Ndim() == 0 {
		return nil, fmt.Errorf("nn: DropPath needs a batch axis, got scalar")
	}
	keep := 1 - d.Rate
	scale := 1 / keep
	batch := x.Shape()[0]
	perSample := make([]float64, batch) // one keep/scale value per sample
	for b := range batch {
		if d.rng.Float64() >= d.Rate {
			perSample[b] = scale
		}
	}
	mask := tensor.New(x.Dtype(), x.Shape())
	n := x.Numel()
	// The mask is freshly allocated by tensor.New, hence contiguous row-major: sample s
	// occupies the flat block [s*rest, (s+1)*rest), so a sample-major slice walk replaces
	// the per-element Unravel+SetF64 dispatch (§T910 fast path) and needs no per-element
	// division. rest is the product of the trailing dims (0-safe: batch==0 ⇒ n==0 ⇒ no
	// iteration). Bit-identical to the generic fallback (F16/BF16/strided) below.
	rest := 1
	for _, s := range x.Shape()[1:] {
		rest *= s
	}
	if mf := flatF64(mask); mf != nil {
		for s := range batch {
			v := perSample[s]
			for j := s * rest; j < (s+1)*rest; j++ {
				mf[j] = v
			}
		}
	} else if mf := flatF32(mask); mf != nil {
		for s := range batch {
			v := float32(perSample[s])
			for j := s * rest; j < (s+1)*rest; j++ {
				mf[j] = v
			}
		}
	} else {
		for i := range n {
			idx := tensor.Unravel(i, x.Shape())
			mask.SetF64(perSample[idx[0]], idx...) // broadcast the per-sample value over the rest
		}
	}
	//perfscan:ignore PS3038 resource-only, time flat (alloc-only)
	out, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{x, mask}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Params returns nil — DropPath has no learnable parameters.
func (d *DropPath) Params() []*tensor.Tensor { return nil }

// StochasticDepthSurvival returns the survival probability p_l for residual block l
// of L under the paper's linear decay rule (Eq. 4): p_l = 1 − (l/L)(1 − pLast), for
// 1-based l ∈ [1, L]. Earlier blocks survive almost always (p≈1); the last block has
// survival pLast (the paper uses 0.5). The corresponding DropPath rate is 1 − p_l.
func StochasticDepthSurvival(l, total int, pLast float64) float64 {
	if total <= 0 {
		return 1
	}
	return 1 - (float64(l)/float64(total))*(1-pLast)
}
