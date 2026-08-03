package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// The RL surrogate losses were rewritten from math.Min/math.Max onto the min/max builtins,
// which are NOT the same function: math.Max documents +Inf as beating NaN and math.Min
// documents -Inf as beating NaN, while the builtins propagate NaN. The rewrite goes through
// internal/fmath, whose guard restores the difference — but a helper proven correct in
// isolation says nothing about whether these loops COMPOSE it correctly, and the divergence
// is reachable here: a log-probability of -Inf gives a ratio of exactly +0, and +0 times an
// infinite advantage is the NaN that pairs with the -Inf the other surrogate branch produces.
//
// The oracle is the definition the rewrite claims to preserve: the same arithmetic written
// with math.Min and math.Max, evaluated in the test, compared bit for bit.
//
// ONE PLANTED TRIPLE PER CALL, and that is the whole design. A first version swept the entire
// grid in a single batch and was GREEN under a raw-builtin rewrite that this file exists to
// reject — because the kernel reduces to a scalar, one NaN anywhere poisons the sum, and both
// sides then agree on NaN. A reduction hides exactly the disagreement it is summing over.
var rlHostile = []float64{
	math.NaN(), math.Inf(1), math.Inf(-1), 0, math.Copysign(0, -1),
	-745.2, -800, 1, -1, math.MaxFloat64, -math.MaxFloat64, 0.5, -0.5,
}

func rlScalar(v float64) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{1})
	t.Storage().F64()[0] = v
	return t
}

// sameBits compares every bit EXCEPT a NaN payload. Payload is not part of the contract —
// the kernel's NaN comes out of a runtime helper and the oracle's out of inlined arithmetic,
// and they differ — while NaN-versus-not-NaN and the sign of a zero are exactly what this
// file is here to police. Treating all NaNs as equal still catches the divergence under test,
// which is math returning -Inf where the raw builtin returns NaN.
func sameBits(a, b float64) bool {
	if math.IsNaN(a) && math.IsNaN(b) {
		return true
	}
	return math.Float64bits(a) == math.Float64bits(b)
}

// negMean mirrors the kernels' final step exactly, including the accumulator's initial +0.
// Skipping it is not cosmetic: 0 + -0 is +0, so an oracle that negates the term directly
// reports a sign flip on a negative-zero surrogate that the kernel never produces.
func negMean(term float64, n int) float64 {
	var total float64
	total += term
	return -total / float64(n)
}

func TestPPOClipMatchesMathMinMax(t *testing.T) {
	ctx := backend.NewContext()
	for _, eps := range []float64{0.2, 1, 2} {
		for _, ln := range rlHostile {
			for _, lo := range rlHostile {
				for _, adv := range rlHostile {
					out, err := backend.Execute(ctx, backend.OpPPOClip,
						[]*tensor.Tensor{rlScalar(ln), rlScalar(lo), rlScalar(adv)},
						backend.PPOClipAttrs{Epsilon: eps})
					if err != nil {
						t.Fatal(err)
					}
					r := math.Exp(ln - lo)
					want := negMean(math.Min(r*adv, math.Max(1-eps, math.Min(1+eps, r))*adv), 1)
					if got := out[0].AtF64(); !sameBits(got, want) {
						t.Fatalf("eps=%v lpNew=%v lpOld=%v adv=%v (ratio %v): got %v, want %v",
							eps, ln, lo, adv, r, got, want)
					}
				}
			}
		}
	}
}

func TestGRPOMatchesMathMinMax(t *testing.T) {
	ctx := backend.NewContext()
	for _, eps := range []float64{0.2, 2} {
		// Both betas are nonzero: GRPOAttrs.WithDefaults rewrites a zero Beta to 0.04, so a
		// case that passes 0 tests 0.04 against an oracle using 0 and fails on the KL term
		// rather than on anything this file is about.
		for _, beta := range []float64{0.04, 0.5} {
			for _, ln := range rlHostile {
				for _, lo := range rlHostile {
					for _, adv := range rlHostile {
						// A finite reference log-probability keeps the KL term out of the way,
						// so a failure here is the surrogate's and not the penalty's.
						const ref = -0.75
						out, err := backend.Execute(ctx, backend.OpGRPO,
							[]*tensor.Tensor{rlScalar(ln), rlScalar(lo), rlScalar(ref), rlScalar(adv)},
							backend.GRPOAttrs{Epsilon: eps, Beta: beta})
						if err != nil {
							t.Fatal(err)
						}
						r := math.Exp(ln - lo)
						surr := math.Min(r*adv, math.Max(1-eps, math.Min(1+eps, r))*adv)
						d := ref - ln
						want := negMean(surr-beta*(math.Exp(d)-d-1), 1)
						if got := out[0].AtF64(); !sameBits(got, want) {
							t.Fatalf("eps=%v beta=%v lpNew=%v lpOld=%v adv=%v: got %v, want %v",
								eps, beta, ln, lo, adv, got, want)
						}
					}
				}
			}
		}
	}
}

func TestGSPOMatchesMathMinMax(t *testing.T) {
	ctx := backend.NewContext()
	// Length 2 rather than 1: GSPO clips the ratio of a MEAN log-ratio, and a length of one
	// makes the mean the identity, which would leave the divisor untested.
	lengths := []int{2}
	for _, eps := range []float64{3e-4, 0.2, 2} {
		for _, ln := range rlHostile {
			for _, lo := range rlHostile {
				for _, adv := range rlHostile {
					lnT := tensor.New(tensor.F64, tensor.Shape{2})
					loT := tensor.New(tensor.F64, tensor.Shape{2})
					lnT.Storage().F64()[0], lnT.Storage().F64()[1] = ln, -0.25
					loT.Storage().F64()[0], loT.Storage().F64()[1] = lo, -0.5
					out, err := backend.Execute(ctx, backend.OpGSPO,
						[]*tensor.Tensor{lnT, loT, rlScalar(adv)},
						backend.GSPOAttrs{Epsilon: eps, Lengths: lengths})
					if err != nil {
						t.Fatal(err)
					}
					d := (ln - lo) + (-0.25 - -0.5)
					s := math.Exp(d / 2)
					want := negMean(math.Min(s*adv, math.Max(1-eps, math.Min(1+eps, s))*adv), 1)
					if got := out[0].AtF64(); !sameBits(got, want) {
						t.Fatalf("eps=%v lpNew=%v lpOld=%v adv=%v (ratio %v): got %v, want %v",
							eps, ln, lo, adv, s, got, want)
					}
				}
			}
		}
	}
}
