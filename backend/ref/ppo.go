package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// ppoClipKernel is the fused PPO clipped surrogate policy loss (Schulman et al.
// 2017, arXiv:1707.06347, Eq. 7). Inputs [batch]: policy log-probs logπθ(a|s),
// the frozen behaviour log-probs logπ_old, and the advantage estimates Â:
//
//	rᵢ    = exp(logπθᵢ − logπ_oldᵢ)              // probability ratio
//	surr1 = rᵢ·Âᵢ
//	surr2 = clip(rᵢ, 1−ε, 1+ε)·Âᵢ
//	loss  = −mean_i min(surr1, surr2)            // minimize −L^CLIP (paper MAXIMIZES)
//
// ε from attrs["epsilon"] (default 0.2). logπ_old and Â are constants (rollout
// time), so gradients flow only into logπθ (§R49). Accumulation in f64 (§V10).
func ppoClipKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("ref: ppoclip wants (logpNew, logpOld, advantage), got %d inputs", len(in))
	}
	lpNew, lpOld, adv := in[0], in[1], in[2]
	for i, t := range in {
		if t.Ndim() != 1 {
			return nil, fmt.Errorf("ref: ppoclip input %d must be rank-1 [batch], got %dD", i, t.Ndim())
		}
	}
	b := lpNew.Shape()[0]
	for i, t := range in {
		if t.Shape()[0] != b {
			return nil, fmt.Errorf("ref: ppoclip input %d len %d != batch %d", i, t.Shape()[0], b)
		}
	}
	if b == 0 {
		return nil, fmt.Errorf("ref: ppoclip on empty batch")
	}
	pa, _ := attrs.(backend.PPOClipAttrs)
	pa = pa.WithDefaults()
	eps := pa.Epsilon

	var total float64
	// Devirtualised traversal (§base-perf), as in the DPO/KTO twins: three AtF64
	// dispatches per element — an interface hop into storage plus a variadic index —
	// around one Exp and a clamp. Flat row-major []float64 views (exact widening for
	// F32) leave the arithmetic, the clamp and the accumulation order untouched, so
	// the result is bit-identical; only the loads move.
	lns, ok0 := f64Data(lpNew)
	los, ok1 := f64Data(lpOld)
	advs, ok2 := f64Data(adv)
	if ok0 && ok1 && ok2 {
		for i := range b {
			r := math.Exp(lns[i] - los[i])
			a := advs[i]
			surr1 := r * a
			surr2 := math.Max(1-eps, math.Min(1+eps, r)) * a
			total += math.Min(surr1, surr2)
		}
	} else {
		// Generic fallback for dtypes f64Data cannot expose (verbatim original loop).
		for i := range b {
			r := math.Exp(lpNew.AtF64(i) - lpOld.AtF64(i))
			a := adv.AtF64(i)
			surr1 := r * a
			surr2 := math.Max(1-eps, math.Min(1+eps, r)) * a
			total += math.Min(surr1, surr2)
		}
	}
	out := tensor.NewOn(ctx.Device(), lpNew.Dtype(), tensor.Shape{})
	out.SetF64(-total / float64(b))
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpPPOClip, tensor.F32, ppoClipKernel)
	std.add(backend.OpPPOClip, tensor.F64, ppoClipKernel)
}
