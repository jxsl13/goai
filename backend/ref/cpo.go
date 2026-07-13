package ref

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// cpoKernel is the fused CPO loss (Contrastive Preference Optimization, Xu et al.
// 2024, arXiv:2401.08417, Eqs. 3-5) — a reference-free preference objective that
// folds SFT and alignment into one term. Inputs [batch] are the SUMMED sequence
// log-probabilities (DPO-style, NOT length-normalized) of the chosen (y_w) and
// rejected (y_l) responses under the policy — no reference model (π_ref is a
// uniform prior that cancels, Eq. 3):
//
//	z_prefer = β·(logπ_w − logπ_l)
//	L_prefer = −log σ(z_prefer)   = softplus(−z_prefer)   // Eq. 3
//	L_NLL    = −logπ_w                                     // Eq. 4, BC on chosen
//	loss     = mean_i( softplus(−z_prefer) + λ·(−logπ_w) ) // Eq. 5, λ = cpo_alpha
//
// β (reward scale, default 0.1) from attrs; λ (NLL/BC weight, TRL cpo_alpha,
// default 1) weights the behavior-cloning regularizer — λ=0 leaves the bare
// reference-free preference term. Stable via softplus (§V12), f64 accum (§V10).
// Contrast SimPO: length-NORMALIZED reward + target margin γ and NO NLL term.
func cpoKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	b, err := checkPair("cpo", in)
	if err != nil {
		return nil, err
	}
	w, l := in[0], in[1]
	pa, _ := attrs.(backend.CPOAttrs)
	pa = pa.WithDefaults()
	beta, alpha := pa.Beta, pa.Alpha

	var total float64
	for i := range b {
		lw := w.AtF64(i)
		z := beta * (lw - l.AtF64(i))
		total += softplus(-z) + alpha*(-lw)
	}
	out := tensor.NewOn(ctx.Device(), w.Dtype(), tensor.Shape{})
	out.SetF64(total / float64(b))
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpCPO, tensor.F32, cpoKernel)
	std.add(backend.OpCPO, tensor.F64, cpoKernel)
}
