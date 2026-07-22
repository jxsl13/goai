package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// EWCPenalty computes the Elastic Weight Consolidation penalty (Kirkpatrick, Pascanu,
// Rabinowitz, Veness, Desjardins, Rusu, Milan, Quan, Ramalho, Grabska-Barwinska, Hassabis,
// Clopath, Kumaran & Hadsell 2017, "Overcoming catastrophic forgetting in neural networks",
// PNAS, arXiv:1612.00796, Eq. 3):
//
//	Ω(θ) = Σ_i (λ/2)·F_i·(θ_i − θ*_i)²
//
// a Fisher-weighted quadratic that anchors each parameter to its value θ* at a previous
// task's optimum. Parameters important to the old task (large Fisher F_i) are strongly
// penalized for moving, while unimportant ones (small F_i) stay free to adapt to the new
// task — so training on a new objective L_B with L_B + EWCPenalty resists catastrophic
// forgetting. With a uniform Fisher this reduces to plain (scaled) L2 weight anchoring, and
// at θ = θ* it is exactly 0.
//
// params, refParams (θ*) and fisher (F) are equal-length lists of matching-shape tensors;
// lambda balances the old task against the new. The penalty is a differentiable scalar
// w.r.t. params (refParams and fisher are constants). For several previous tasks, sum an
// EWCPenalty per task.
func EWCPenalty(ctx *backend.Context, params, refParams, fisher []*tensor.Tensor, lambda float64) (*tensor.Tensor, error) {
	if len(params) != len(refParams) || len(params) != len(fisher) {
		return nil, fmt.Errorf("nn: EWCPenalty needs equal-length params/refParams/fisher, got %d/%d/%d", len(params), len(refParams), len(fisher))
	}
	if len(params) == 0 {
		return nil, fmt.Errorf("nn: EWCPenalty needs at least one parameter")
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	var total *tensor.Tensor
	for i := range params {
		if !params[i].Shape().Equal(refParams[i].Shape()) || !params[i].Shape().Equal(fisher[i].Shape()) {
			return nil, fmt.Errorf("nn: EWCPenalty shape mismatch at %d: %v / %v / %v", i, params[i].Shape(), refParams[i].Shape(), fisher[i].Shape())
		}
		diff, err := ex(backend.OpSub, nil, params[i], refParams[i]) // θ − θ*
		if err != nil {
			return nil, err
		}
		sq, err := ex(backend.OpMul, nil, diff, diff)
		if err != nil {
			return nil, err
		}
		weighted, err := ex(backend.OpMul, nil, fisher[i], sq) // F·(θ−θ*)²
		if err != nil {
			return nil, err
		}
		axes := make([]int, weighted.Ndim())
		for a := range axes {
			axes[a] = a
		}
		s, err := ex(backend.OpSum, backend.ReduceAttrs{Axes: axes}, weighted) // → scalar
		if err != nil {
			return nil, err
		}
		if total == nil {
			total = s
		} else if total, err = ex(backend.OpAdd, nil, total, s); err != nil {
			return nil, err
		}
	}
	// scale the scalar total by λ/2 without promoting its rank (AXPY with a rank-0 zero).
	// λ==0 (no regularization) must yield 0, but AXPYAttrs.WithDefaults() rewrites
	// Alpha 0→1 and would return the full unscaled penalty — guard the off case.
	if lambda == 0 {
		return tensor.New(total.Dtype(), total.Shape()), nil
	}
	return ex(backend.OpAXPY, backend.AXPYAttrs{Alpha: lambda / 2}, total, tensor.New(total.Dtype(), total.Shape()))
}

// EWCFisher estimates the diagonal Fisher information at the current parameters as the mean
// of squared per-sample log-likelihood gradients (arXiv:1612.00796 §Methods): F_i =
// mean_s (∂ log p(x_s|θ)/∂θ_i)². gradSamples[s] is the parameter-gradient list for sample s
// (all samples share the parameter layout); the returned Fisher matches that layout.
// Larger F_i marks parameter i as more important to protect. A pure-f64 statistic (not
// differentiable), computed once at a task's optimum.
func EWCFisher(gradSamples [][]*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(gradSamples) == 0 || len(gradSamples[0]) == 0 {
		return nil, fmt.Errorf("nn: EWCFisher needs at least one sample with at least one parameter")
	}
	nS, nP := len(gradSamples), len(gradSamples[0])
	fisher := make([]*tensor.Tensor, nP)
	for i := range nP {
		shape := gradSamples[0][i].Shape()
		acc := tensor.New(gradSamples[0][i].Dtype(), shape)
		for s := range nS {
			if !gradSamples[s][i].Shape().Equal(shape) {
				return nil, fmt.Errorf("nn: EWCFisher sample %d param %d shape %v != %v", s, i, gradSamples[s][i].Shape(), shape)
			}
		}
		// Typed contiguous fast path: acc is freshly allocated (contiguous); when
		// every gradient sample is contiguous too, accumulate ∑ g² by walking the
		// backing []T directly — no per-element Unravel/AtF64/SetF64 dispatch. The
		// sum stays in sample order and normalization divides (not reciprocal-
		// multiplies), so the result is bit-identical to the generic walk (§base-perf).
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
						af[e] += gf[e] * gf[e]
					}
				}
				for e := range af {
					af[e] = af[e] / float64(nS)
				}
				fisher[i] = acc
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
						af[e] = float32(float64(af[e]) + float64(gf[e])*float64(gf[e]))
					}
				}
				for e := range af {
					af[e] = float32(float64(af[e]) / float64(nS))
				}
				fisher[i] = acc
				continue
			}
		}
		// Generic fallback (non-contiguous or mixed-dtype gradients). Shapes were
		// validated above, so accumulate directly.
		for s := range nS {
			g := gradSamples[s][i]
			for e := range g.Numel() {
				c := tensor.Unravel(e, shape)
				v := g.AtF64(c...)
				acc.SetF64(acc.AtF64(c...)+v*v, c...)
			}
		}
		for e := range acc.Numel() {
			c := tensor.Unravel(e, shape)
			acc.SetF64(acc.AtF64(c...)/float64(nS), c...)
		}
		fisher[i] = acc
	}
	return fisher, nil
}
