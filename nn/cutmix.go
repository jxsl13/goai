package nn

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/tensor"
)

// CutMix is the region-mixing data-augmentation regularizer of Yun, Han, Oh,
// Chun, Choe & Yoo 2019 ("CutMix: Regularization Strategy to Train Strong
// Classifiers with Localizable Features", arXiv:1905.04899). Instead of blending
// two images pixel-wise like Mixup, it CUTS a rectangular patch out of one image
// and PASTES the same region from a partner image: for a batch x[B,C,H,W] with
// permutation π it samples λ~Beta(α,α), draws a box of area (1−λ)·H·W at a random
// location, and pastes x[π] inside it. Because pixels are replaced (not averaged)
// the composite stays a natural, locally-coherent image, so the model learns from
// the informative parts of both examples rather than an ambiguous overlay.
//
// The label mixing uses the CORRECTED weight
//
//	λ ← 1 − boxArea/(H·W)
//
// computed from the ACTUAL (clamped) pasted-region area, not the sampled λ — the
// box is clipped to the image border and integer-rounded, so its real area rarely
// equals the nominal (1−λ)·H·W. Forward returns this corrected λ; feed it (with π)
// to PermuteRows and MixupLoss for the label side, exactly as for Mixup. The paste
// is host-side data (no gradient through x); the differentiable half is the label
// loss on the logits.
//
// §C21 α boundaries: α>0 shapes Beta(α,α) on [0,1]; α→0 pushes the box area toward
// {0, H·W} (no cut or a full swap); α=1 is uniform; large α concentrates the area
// near ½·H·W. The paper's default for CutMix is α=1.0 (uniform box areas), heavier
// than Mixup's α=0.2. CutMix requires a rank-4 [B,C,H,W] batch.
type CutMix struct {
	Alpha     float64 // Beta(α,α) concentration; paper default 1.0 (§C21)
	rng       *rand.Rand
	lambda    float64 // forced λ when hasLambda (test/deterministic seam)
	hasLambda bool
}

// CutMixOption configures a CutMix via the functional-options idiom.
type CutMixOption func(*CutMix)

// WithCutMixLambda pins the box-sizing weight λ to a fixed value in [0,1] instead
// of sampling Beta(α,α) — a deterministic seam for tests and external schedules.
// λ=1 gives a zero-area box, so x is unchanged and the corrected λ is 1 (loss =
// CrossEntropy on the original labels). Out-of-range values are ignored.
func WithCutMixLambda(lambda float64) CutMixOption {
	return func(c *CutMix) {
		if lambda >= 0 && lambda <= 1 {
			c.lambda, c.hasLambda = lambda, true
		}
	}
}

// NewCutMix builds a CutMix augmenter with Beta concentration alpha (paper default
// 1.0). The seed makes the λ, permutation and box-location sequence deterministic.
// Panics when alpha≤0.
func NewCutMix(alpha float64, seed uint64, opts ...CutMixOption) *CutMix {
	if alpha <= 0 {
		panic(fmt.Sprintf("nn: cutmix alpha %g must be > 0", alpha))
	}
	c := &CutMix{Alpha: alpha, rng: rand.New(rand.NewPCG(seed, 0x85ebca6b))}
	for _, o := range opts {
		o(c)
	}
	return c
}

// Forward pastes a random box from x[π] onto x and returns the composite, the
// permutation π and the CORRECTED mixing weight λ = 1 − boxArea/(H·W) computed
// from the actual clamped box. x must be rank-4 [B,C,H,W]; feed π and λ to
// PermuteRows and MixupLoss for the label side. Errors on a non-rank-4 input.
func (c *CutMix) Forward(x *tensor.Tensor) (mixed *tensor.Tensor, perm []int, lambda float64, err error) {
	if x.Ndim() != 4 {
		return nil, nil, 0, fmt.Errorf("nn: CutMix needs a rank-4 [B,C,H,W] tensor, got rank %d", x.Ndim())
	}
	sh := x.Shape()
	B, C, H, W := sh[0], sh[1], sh[2], sh[3]
	lam := c.sampleLambda()

	// Box side ratio r = √(1−λ): area (r·H)(r·W) = (1−λ)·H·W (paper rand_bbox).
	r := math.Sqrt(1 - lam)
	cutH, cutW := int(float64(H)*r), int(float64(W)*r)
	cy, cx := c.rng.IntN(H), c.rng.IntN(W)
	y1, y2 := clampInt(cy-cutH/2, 0, H), clampInt(cy+cutH/2, 0, H)
	x1, x2 := clampInt(cx-cutW/2, 0, W), clampInt(cx+cutW/2, 0, W)
	boxArea := (y2 - y1) * (x2 - x1)

	// Corrected label weight from the ACTUAL pasted area (§V16 anchor).
	lambda = 1 - float64(boxArea)/float64(H*W)
	perm = c.rng.Perm(B)
	mixed = cutmixPaste(x, perm, C, H, W, y1, y2, x1, x2)
	return mixed, perm, lambda, nil
}

// sampleLambda returns the forced λ when set, else a fresh Beta(α,α) draw.
func (c *CutMix) sampleLambda() float64 {
	if c.hasLambda {
		return c.lambda
	}
	return sampleBeta(c.rng, c.Alpha, c.Alpha)
}

// cutmixPaste copies x and, for every sample b and channel, overwrites the box
// [y1:y2, x1:x2] with the same box from x[perm[b]]. A zero-area box (λ=1) leaves
// the copy bit-identical to x. Contiguous F32/F64 use a flat fast path; other
// layouts/dtypes use the generic index path (§base-perf).
func cutmixPaste(x *tensor.Tensor, perm []int, C, H, W, y1, y2, x1, x2 int) *tensor.Tensor {
	out := tensor.New(x.Dtype(), x.Shape())
	B := x.Shape()[0]
	switch {
	case flatF64(x) != nil:
		xf, of := flatF64(x), out.Storage().F64()
		copy(of, xf)
		for b := range B {
			for ch := range C {
				dstBase := (b*C + ch) * H * W
				srcBase := (perm[b]*C + ch) * H * W
				for yy := y1; yy < y2; yy++ {
					for xx := x1; xx < x2; xx++ {
						of[dstBase+yy*W+xx] = xf[srcBase+yy*W+xx]
					}
				}
			}
		}
	case flatF32(x) != nil:
		xf, of := flatF32(x), out.Storage().F32()
		copy(of, xf)
		for b := range B {
			for ch := range C {
				dstBase := (b*C + ch) * H * W
				srcBase := (perm[b]*C + ch) * H * W
				for yy := y1; yy < y2; yy++ {
					for xx := x1; xx < x2; xx++ {
						of[dstBase+yy*W+xx] = xf[srcBase+yy*W+xx]
					}
				}
			}
		}
	default:
		n := x.Numel()
		for i := range n {
			idx := tensor.Unravel(i, x.Shape())
			out.SetF64(x.AtF64(idx...), idx...)
		}
		for b := range B {
			for ch := range C {
				for yy := y1; yy < y2; yy++ {
					for xx := x1; xx < x2; xx++ {
						out.SetF64(x.AtF64(perm[b], ch, yy, xx), b, ch, yy, xx)
					}
				}
			}
		}
	}
	return out
}

// clampInt clamps v to [lo, hi].
func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
