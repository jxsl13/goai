package nn

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Mixup is the input-mixing data-augmentation regularizer of Zhang, Cissé,
// Dauphin & Lopez-Paz 2017 ("mixup: Beyond Empirical Risk Minimization",
// arXiv:1710.09412). It trains on convex combinations of example PAIRS rather
// than raw examples: it draws a mixing weight λ~Beta(α,α) and a random batch
// permutation π, forms the mixed input
//
//	x'[i] = λ·x[i] + (1−λ)·x[π[i]]
//
// and asks the model to predict the correspondingly mixed label. Interpolating
// between examples enforces linear behavior in-between training points, which
// reduces memorization of corrupt labels, sharpens decision boundaries and
// improves generalization and robustness to adversarial examples.
//
// Forward produces the mixed input plus the (π, λ) needed for the label side;
// the LABEL mixing — the differentiable half — is applied by MixupLoss on the
// logits. The input mix is host-side data (no gradient flows through x), so
// Forward takes no context and records nothing on the tape.
//
// §C21 α boundaries: α>0 shapes Beta(α,α) on [0,1]; α→0 pushes λ toward the
// endpoints {0,1} (either the example or its partner, almost no mixing); α=1 is
// uniform on [0,1]; large α concentrates λ near 0.5 (maximal 50/50 blending).
// The paper's default for image classification is α=0.2 (light mixing).
type Mixup struct {
	Alpha     float64 // Beta(α,α) concentration; paper default 0.2 (§C21)
	rng       *rand.Rand
	lambda    float64 // forced λ when hasLambda (test/deterministic seam)
	hasLambda bool
}

// MixupOption configures a Mixup via the functional-options idiom.
type MixupOption func(*Mixup)

// WithMixupLambda pins the mixing weight λ to a fixed value in [0,1] instead of
// sampling Beta(α,α) — a deterministic seam for tests and for schedules that set
// λ externally. λ=1 collapses Mixup to the identity (x'=x, loss=CrossEntropy on
// the original labels). Out-of-range values are ignored (sampling is kept).
func WithMixupLambda(lambda float64) MixupOption {
	return func(m *Mixup) {
		if lambda >= 0 && lambda <= 1 {
			m.lambda, m.hasLambda = lambda, true
		}
	}
}

// NewMixup builds a Mixup augmenter with Beta concentration alpha (paper default
// 0.2). The seed makes the λ and permutation sequence deterministic. Panics when
// alpha≤0.
func NewMixup(alpha float64, seed uint64, opts ...MixupOption) *Mixup {
	if alpha <= 0 {
		panic(fmt.Sprintf("nn: mixup alpha %g must be > 0", alpha))
	}
	m := &Mixup{Alpha: alpha, rng: rand.New(rand.NewPCG(seed, 0x9e3779b9))}
	for _, o := range opts {
		o(m)
	}
	return m
}

// Forward builds the mixed batch x'[i]=λ·x[i]+(1−λ)·x[π[i]] and returns it along
// with the permutation π and the mixing weight λ (feed both to PermuteRows and
// MixupLoss for the label side). x is [batch, ...]; mixing is over the first axis.
// Errors on a scalar (no batch axis).
func (m *Mixup) Forward(x *tensor.Tensor) (mixed *tensor.Tensor, perm []int, lambda float64, err error) {
	if x.Ndim() == 0 {
		return nil, nil, 0, fmt.Errorf("nn: Mixup needs a batch axis, got scalar")
	}
	batch := x.Shape()[0]
	lambda = m.sampleLambda()
	perm = m.rng.Perm(batch)
	mixed = mixupCombine(x, perm, lambda)
	return mixed, perm, lambda, nil
}

// sampleLambda returns the forced λ when set, else a fresh Beta(α,α) draw.
func (m *Mixup) sampleLambda() float64 {
	if m.hasLambda {
		return m.lambda
	}
	return sampleBeta(m.rng, m.Alpha, m.Alpha)
}

// mixupCombine forms out[i]=λ·x[i]+(1−λ)·x[π[i]] over the batch axis. With λ=1
// the result equals x bit-for-bit (1·a+0·b with finite b is a exactly). It uses
// a flat dtype-switched fast path for contiguous F32/F64 and a generic fallback
// otherwise (§base-perf).
func mixupCombine(x *tensor.Tensor, perm []int, lambda float64) *tensor.Tensor {
	out := tensor.New(x.Dtype(), x.Shape())
	batch := x.Shape()[0]
	rowLen := x.Numel() / batch
	comp := 1 - lambda
	switch {
	case flatF64(x) != nil:
		xf, of := flatF64(x), out.Storage().F64()
		for i := range batch {
			base, pbase := i*rowLen, perm[i]*rowLen
			for k := range rowLen {
				of[base+k] = lambda*xf[base+k] + comp*xf[pbase+k]
			}
		}
	case flatF32(x) != nil:
		xf, of := flatF32(x), out.Storage().F32()
		for i := range batch {
			base, pbase := i*rowLen, perm[i]*rowLen
			for k := range rowLen {
				of[base+k] = float32(lambda)*xf[base+k] + float32(comp)*xf[pbase+k]
			}
		}
	default:
		n := x.Numel()
		for i := range n {
			idx := tensor.Unravel(i, x.Shape())
			src := append([]int(nil), idx...)
			src[0] = perm[idx[0]]
			out.SetF64(lambda*x.AtF64(idx...)+comp*x.AtF64(src...), idx...)
		}
	}
	return out
}

// PermuteRows returns a copy of t with its rows reordered along the batch axis so
// that out row i equals t row perm[i] — the label-side companion to a Mixup or
// CutMix input permutation. It works for class-index targets ([batch]) and for
// one-hot / soft targets ([batch, classes]) alike. len(perm) must equal the batch.
func PermuteRows(t *tensor.Tensor, perm []int) *tensor.Tensor {
	out := tensor.New(t.Dtype(), t.Shape())
	if t.Ndim() == 0 {
		out.SetF64(t.AtF64())
		return out
	}
	batch := t.Shape()[0]
	rowLen := t.Numel() / batch
	// The reorder (out row i = t row perm[i]) is a pure value copy, so for a
	// contiguous, offset-0 tensor copy whole rows through the typed backing slice
	// — the general path below allocated a fresh index slice for EVERY element and
	// paid the AtF64/SetF64 dispatch. Bit-identical (an exact copy, same dtype);
	// F16/BF16 and strided/offset views fall through.
	if t.IsContiguous() && t.Offset() == 0 {
		switch t.Dtype() {
		case tensor.F64:
			td, od := t.Storage().F64(), out.Storage().F64()
			for i := range batch {
				copy(od[i*rowLen:(i+1)*rowLen], td[perm[i]*rowLen:perm[i]*rowLen+rowLen])
			}
			return out
		case tensor.F32:
			td, od := t.Storage().F32(), out.Storage().F32()
			for i := range batch {
				copy(od[i*rowLen:(i+1)*rowLen], td[perm[i]*rowLen:perm[i]*rowLen+rowLen])
			}
			return out
		}
	}
	n := t.Numel()
	for i := range n {
		idx := tensor.Unravel(i, t.Shape())
		src := append([]int(nil), idx...)
		src[0] = perm[idx[0]]
		out.SetF64(t.AtF64(src...), idx...)
	}
	return out
}

// MixupLoss applies a mixed-label objective to logits from a Mixup/CutMix batch:
//
//	loss = λ·CrossEntropy(logits, targetsA) + (1−λ)·CrossEntropy(logits, targetsB)
//
// where targetsA are the original labels and targetsB = PermuteRows(targetsA, π)
// are the partner labels. It is exactly the CrossEntropy the mixing implies (the
// loss is linear in the label), and it is format-agnostic — targetsA/B may be
// class indices or one-hot/soft rows, since CrossEntropy accepts both. With λ=1
// it reduces bit-exactly to CrossEntropy(logits, targetsA). Fully differentiable
// w.r.t. logits through the two CrossEntropy VJPs; targets are constants.
func MixupLoss(ctx *backend.Context, logits, targetsA, targetsB *tensor.Tensor, lambda float64) (*tensor.Tensor, error) {
	ceA, err := CrossEntropy(ctx, logits, targetsA)
	if err != nil {
		return nil, err
	}
	ceB, err := CrossEntropy(ctx, logits, targetsB)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3038 resource-only slice-literal alloc per dispatch
	termA, err := backend.Execute(ctx, backend.OpMul,
		[]*tensor.Tensor{ceA, scalarZero(ceA.Dtype(), lambda)}, nil)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3038 resource-only slice-literal alloc per dispatch
	termB, err := backend.Execute(ctx, backend.OpMul,
		[]*tensor.Tensor{ceB, scalarZero(ceB.Dtype(), 1-lambda)}, nil)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3038 resource-only slice-literal alloc per dispatch
	out, err := backend.Execute(ctx, backend.OpAdd,
		[]*tensor.Tensor{termA[0], termB[0]}, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// scalarZero makes a rank-0 constant of the given dtype/value — the scalar shape
// CrossEntropy returns, so it multiplies elementwise against the loss with no
// broadcast.
func scalarZero(dtype tensor.Dtype, v float64) *tensor.Tensor {
	t := tensor.New(dtype, tensor.Shape{})
	t.SetF64(v)
	return t
}

// sampleBeta draws Beta(a,b) as g1/(g1+g2) with g1~Gamma(a), g2~Gamma(b); the
// result lies in [0,1] by construction. The degenerate g1+g2=0 maps to 0.5.
func sampleBeta(rng *rand.Rand, a, b float64) float64 {
	g1, g2 := sampleGamma(rng, a), sampleGamma(rng, b)
	s := g1 + g2
	if s == 0 {
		return 0.5
	}
	return g1 / s
}

// sampleGamma draws Gamma(shape, 1) via Marsaglia & Tsang's method (2000) for
// shape≥1, with the shape<1 boost Gamma(a)=Gamma(a+1)·U^(1/a). math/rand/v2 has
// no Gamma sampler, so this backs the Beta draw Mixup/CutMix need.
func sampleGamma(rng *rand.Rand, shape float64) float64 {
	if shape < 1 {
		u := rng.Float64()
		if u == 0 {
			u = math.SmallestNonzeroFloat64
		}
		return sampleGamma(rng, shape+1) * math.Pow(u, 1/shape)
	}
	d := shape - 1.0/3.0
	c := 1.0 / math.Sqrt(9*d)
	for {
		var x, v float64
		for {
			x = rng.NormFloat64()
			v = 1 + c*x
			if v > 0 {
				break
			}
		}
		v = v * v * v
		u := rng.Float64()
		if u < 1-0.0331*x*x*x*x {
			return d * v
		}
		if math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v)) {
			return d * v
		}
	}
}
