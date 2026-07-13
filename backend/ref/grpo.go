package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// grpoKernel is the fused GRPO policy loss (Shao et al. 2024, DeepSeekMath,
// arXiv:2402.03300, Eqs. 3-4). It is the PPO clipped surrogate plus an explicit
// per-token KL-to-reference penalty using Schulman's unbiased k3 estimator. Inputs
// [batch] (one entry per token): the policy log-probs logπθ, the frozen behaviour
// log-probs logπ_old, the reference-model log-probs logπ_ref, and the
// group-normalized advantage Â (constant across the tokens of one output):
//
//	rᵢ    = exp(logπθᵢ − logπ_oldᵢ)               // probability ratio (PPO)
//	surrᵢ = min(rᵢ·Âᵢ, clip(rᵢ,1−ε,1+ε)·Âᵢ)
//	Δᵢ    = logπ_refᵢ − logπθᵢ
//	klᵢ   = exp(Δᵢ) − Δᵢ − 1                       // k3 unbiased KL, ≥ 0
//	loss  = −mean_i( surrᵢ − β·klᵢ )               // minimize −J^GRPO
//
// ε from attrs["epsilon"] (default 0.2), β from attrs["beta"] (default 0.04).
// logπ_old, logπ_ref and Â are rollout-time constants, so gradients flow only
// into logπθ (the k3 term keeps π_θ near π_ref). Accumulation in f64 (§V10).
func grpoKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("ref: grpo wants (logpNew, logpOld, logpRef, advantage), got %d inputs", len(in))
	}
	lpNew, lpOld, lpRef, adv := in[0], in[1], in[2], in[3]
	for i, t := range in {
		if t.Ndim() != 1 {
			return nil, fmt.Errorf("ref: grpo input %d must be rank-1 [batch], got %dD", i, t.Ndim())
		}
	}
	b := lpNew.Shape()[0]
	for i, t := range in {
		if t.Shape()[0] != b {
			return nil, fmt.Errorf("ref: grpo input %d len %d != batch %d", i, t.Shape()[0], b)
		}
	}
	if b == 0 {
		return nil, fmt.Errorf("ref: grpo on empty batch")
	}
	pa, _ := attrs.(backend.GRPOAttrs)
	pa = pa.WithDefaults()
	eps := pa.Epsilon
	beta := pa.Beta

	var total float64
	for i := range b {
		r := math.Exp(lpNew.AtF64(i) - lpOld.AtF64(i))
		a := adv.AtF64(i)
		surr := math.Min(r*a, math.Max(1-eps, math.Min(1+eps, r))*a)
		d := lpRef.AtF64(i) - lpNew.AtF64(i)
		kl := math.Exp(d) - d - 1 // Schulman k3, unbiased, ≥ 0
		total += surr - beta*kl
	}
	out := tensor.NewOn(ctx.Device(), lpNew.Dtype(), tensor.Shape{})
	out.SetF64(-total / float64(b))
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpGRPO, tensor.F32, grpoKernel)
	std.add(backend.OpGRPO, tensor.F64, grpoKernel)
}
