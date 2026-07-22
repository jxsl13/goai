package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// MASImportance estimates the Memory Aware Synapses per-parameter importance (Aljundi, Babiloni,
// Elhoseiny, Rohrbach & Tuytelaars 2018, "Memory Aware Synapses: Learning what (not) to forget",
// ECCV, arXiv:1711.09601, Eq. 2):
//
//	Ω_i = (1/N)·Σ_n |g_i(x_n)| ,   g(x_n) = ∂‖F(x_n;θ)‖₂² / ∂θ
//
// the mean MAGNITUDE (absolute value) of the gradient of the network output's SQUARED L2 norm
// with respect to each parameter. A large Ω_i means small changes to θ_i noticeably change the
// output, so θ_i matters. Unlike EWC's Fisher (nn.EWCFisher — the mean SQUARED gradient of the
// LOG-LIKELIHOOD, which needs labels), MAS is UNSUPERVISED: it looks only at the output, so
// importance can be estimated on unlabeled or even test data, and it takes the L1 (absolute)
// rather than the L2 (square) of the gradient. The resulting Ω plugs into the same quadratic
// anchor as EWC — pass it as the fisher/importance argument of nn.EWCPenalty with the previous
// task's parameters θ* (the surrogate loss L_new + λ·Σ_i Ω_i·(θ_i−θ*_i)²).
//
// gradSamples[n] is the parameter-gradient list of ‖F(x_n)‖₂² for sample n (all samples share the
// parameter layout); the returned importance matches that layout. A pure-f64 statistic (not
// differentiable), computed once after a task.
func MASImportance(gradSamples [][]*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gradSamples) == 0 || len(gradSamples[0]) == 0 {
		return nil, fmt.Errorf("nn: MASImportance needs at least one sample with at least one parameter")
	}
	nS, nP := len(gradSamples), len(gradSamples[0])
	omega := make([]*tensor.Tensor, nP)
	for i := range nP {
		shape := gradSamples[0][i].Shape()
		acc := tensor.New(gradSamples[0][i].Dtype(), shape)
		for s := range nS {
			if !gradSamples[s][i].Shape().Equal(shape) {
				return nil, fmt.Errorf("nn: MASImportance sample %d param %d shape %v != %v", s, i, gradSamples[s][i].Shape(), shape)
			}
		}
		// Typed contiguous fast path (§base-perf; sibling of EWCFisher): acc is freshly
		// allocated (contiguous); when every gradient sample is contiguous too, accumulate
		// ∑|g| by walking the backing []T directly — no per-element Unravel/AtF64/SetF64
		// dispatch. The sum stays in sample order and normalization divides (not reciprocal-
		// multiplies), so the result is bit-identical to the generic walk.
		if af := flatF64(acc); af != nil {
			allFlat := true
			for s := range nS {
				if flatF64(gradSamples[s][i]) == nil {
					allFlat = false
					break
				}
			}
			if allFlat {
				for s := range nS {
					gf := flatF64(gradSamples[s][i])
					for e := range af {
						af[e] += math.Abs(gf[e])
					}
				}
				for e := range af {
					af[e] = af[e] / float64(nS)
				}
				omega[i] = acc
				continue
			}
		}
		if af := flatF32(acc); af != nil {
			allFlat := true
			for s := range nS {
				if flatF32(gradSamples[s][i]) == nil {
					allFlat = false
					break
				}
			}
			if allFlat {
				for s := range nS {
					gf := flatF32(gradSamples[s][i])
					for e := range af {
						af[e] = float32(float64(af[e]) + math.Abs(float64(gf[e])))
					}
				}
				for e := range af {
					af[e] = float32(float64(af[e]) / float64(nS))
				}
				omega[i] = acc
				continue
			}
		}
		// Generic fallback (non-contiguous or mixed-dtype gradients). Shapes were
		// validated above, so accumulate directly.
		for s := range nS {
			g := gradSamples[s][i]
			for e := range g.Numel() {
				c := tensor.Unravel(e, shape)
				acc.SetF64(acc.AtF64(c...)+math.Abs(g.AtF64(c...)), c...)
			}
		}
		for e := range acc.Numel() {
			c := tensor.Unravel(e, shape)
			acc.SetF64(acc.AtF64(c...)/float64(nS), c...)
		}
		omega[i] = acc
	}
	return omega, nil
}
