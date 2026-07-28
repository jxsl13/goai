package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// dpoKernel is the fused Direct Preference Optimization loss (Rafailov et al.
// 2023, arXiv:2305.18290, Eq. 7). Inputs are sequence log-probabilities
// [batch] for the policy on chosen/rejected and the FROZEN reference on
// chosen/rejected:
//
//	Δᵢ = β·((πc − rc) − (πl − rl))          // β·(r̂_w − r̂_l), r̂ = β log πθ/πref
//	loss = mean_i −log σ(Δᵢ) = mean_i softplus(−Δᵢ)
//
// −log σ(z) is evaluated as the numerically stable softplus(−z) = max(−z,0) +
// log(1+exp(−|z|)) so extreme margins never overflow (§V12). Accumulation in f64
// (§V10). β is read from attrs["beta"] (default 0.1). Output is a scalar.
func dpoKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("ref: dpo wants (policyChosen, policyRejected, refChosen, refRejected), got %d inputs", len(in))
	}
	pc, pl, rc, rl := in[0], in[1], in[2], in[3]
	for i, t := range in {
		if t.Ndim() != 1 {
			return nil, fmt.Errorf("ref: dpo input %d must be rank-1 [batch], got %dD", i, t.Ndim())
		}
	}
	b := pc.Shape()[0]
	for i, t := range in {
		if t.Shape()[0] != b {
			return nil, fmt.Errorf("ref: dpo input %d len %d != batch %d", i, t.Shape()[0], b)
		}
	}
	if b == 0 {
		return nil, fmt.Errorf("ref: dpo on empty batch")
	}
	pa, _ := attrs.(backend.DPOAttrs)
	pa = pa.WithDefaults()
	beta := pa.Beta

	var total float64
	// Devirtualised traversal (§base-perf): the generic loop pays FOUR AtF64
	// dispatches per element — an interface hop into storage plus a variadic index
	// — to do three subtractions and one softplus. Exposing each rank-1 input once
	// as a flat row-major []float64 (exact widening for F32) leaves the arithmetic
	// untouched. Same operands, same order, same accumulation into total, so the
	// result is bit-identical; only the loads move.
	pcs, ok0 := f64Data(pc)
	rcs, ok1 := f64Data(rc)
	pls, ok2 := f64Data(pl)
	rls, ok3 := f64Data(rl)
	if ok0 && ok1 && ok2 && ok3 {
		for i := range b {
			delta := beta * ((pcs[i] - rcs[i]) - (pls[i] - rls[i]))
			total += softplus(-delta) // −log σ(delta)
		}
	} else {
		// Generic fallback for dtypes f64Data cannot expose (verbatim original loop).
		for i := range b {
			delta := beta * ((pc.AtF64(i) - rc.AtF64(i)) - (pl.AtF64(i) - rl.AtF64(i)))
			total += softplus(-delta) // −log σ(delta)
		}
	}
	out := tensor.NewOn(ctx.Device(), pc.Dtype(), tensor.Shape{})
	out.SetF64(total / float64(b))
	return []*tensor.Tensor{out}, nil
}

// softplus(u) = log(1+eˣᵖ(u)), computed stably as max(u,0)+log(1+exp(−|u|)).
func softplus(u float64) float64 {
	if u > 0 {
		return u + math.Log1p(math.Exp(-u))
	}
	return math.Log1p(math.Exp(u))
}

func init() {
	std.add(backend.OpDPO, tensor.F32, dpoKernel)
	std.add(backend.OpDPO, tensor.F64, dpoKernel)
}
