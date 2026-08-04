package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// ipoKernel is the fused Identity Preference Optimization loss (Azar et al. 2023,
// arXiv:2310.12036; TRL loss_type="ipo"). Inputs are sequence log-probs [batch]
// for the policy and the FROZEN reference on chosen/rejected. With the same
// log-ratio difference as DPO,
//
//	hᵢ = (πcᵢ − rcᵢ) − (πlᵢ − rlᵢ)
//	loss = mean_i (hᵢ − 1/(2β))²
//
// it is a SQUARED loss toward the finite target margin 1/(2β) (β from
// attrs["beta"], default 0.1), unlike DPO's logistic −log σ — so it regularizes
// to a bounded margin instead of pushing it to infinity (§R58). Accumulation in
// f64 (§V10). Output scalar.
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
func ipoKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("ref: ipo wants (policyChosen, policyRejected, refChosen, refRejected), got %d inputs", len(in))
	}
	pc, pl, rc, rl := in[0], in[1], in[2], in[3]
	for i, t := range in {
		if t.Ndim() != 1 {
			return nil, fmt.Errorf("ref: ipo input %d must be rank-1 [batch], got %dD", i, t.Ndim())
		}
	}
	b := pc.Shape()[0]
	for i, t := range in {
		if t.Shape()[0] != b {
			return nil, fmt.Errorf("ref: ipo input %d len %d != batch %d", i, t.Shape()[0], b)
		}
	}
	if b == 0 {
		return nil, fmt.Errorf("ref: ipo on empty batch")
	}
	pa, _ := attrs.(backend.IPOAttrs)
	pa = pa.WithDefaults()
	beta := pa.Beta
	target := 1.0 / (2.0 * beta)

	var total float64
	// Devirtualised traversal (§base-perf), the IPO member of the preference-loss
	// family: four AtF64 dispatches per element for three subtractions and a square.
	// Bit-identical.
	pcs, ok0 := f64Data(pc)
	rcs, ok1 := f64Data(rc)
	pls, ok2 := f64Data(pl)
	rls, ok3 := f64Data(rl)
	if ok0 && ok1 && ok2 && ok3 {
		for i := range b {
			h := (pcs[i] - rcs[i]) - (pls[i] - rls[i])
			d := h - target
			total += d * d
		}
	} else {
		// Generic fallback for dtypes f64Data cannot expose (verbatim original loop).
		for i := range b {
			h := (pc.AtF64(i) - rc.AtF64(i)) - (pl.AtF64(i) - rl.AtF64(i))
			d := h - target
			total += d * d
		}
	}
	out := tensor.NewOn(ctx.Device(), pc.Dtype(), tensor.Shape{})
	out.SetF64(total / float64(b))
	return []*tensor.Tensor{out}, nil
}

func init() {
	//perfscan:ignore PS3062 reference oracle: intentionally simple, correctness baseline not an optimization target
	std.add(backend.OpIPO, tensor.F32, ipoKernel)
	std.add(backend.OpIPO, tensor.F64, ipoKernel)
}
