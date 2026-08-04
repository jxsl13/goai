package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// checkPair validates two rank-1 [batch] inputs for the reference-free
// preference losses and returns the batch size.
func checkPair(name string, in []*tensor.Tensor) (int, error) {
	if len(in) != 2 {
		return 0, fmt.Errorf("ref: %s wants (avgChosen, avgRejected), got %d inputs", name, len(in))
	}
	for i, t := range in {
		if t.Ndim() != 1 {
			return 0, fmt.Errorf("ref: %s input %d must be rank-1 [batch], got %dD", name, i, t.Ndim())
		}
	}
	b := in[0].Shape()[0]
	if in[1].Shape()[0] != b {
		return 0, fmt.Errorf("ref: %s chosen/rejected lengths %d != %d", name, b, in[1].Shape()[0])
	}
	if b == 0 {
		return 0, fmt.Errorf("ref: %s on empty batch", name)
	}
	return b, nil
}

// simpoKernel is the fused SimPO loss (Meng et al. 2024, arXiv:2405.14734,
// Eqs. 5-7) — reference-free preference optimization. Inputs [batch] are the
// LENGTH-NORMALIZED average token log-probabilities of the chosen and rejected
// responses (no reference model):
//
//	zᵢ   = β·(avgChosenᵢ − avgRejectedᵢ) − γ      // margin γ subtracted inside σ
//	loss = mean_i −log σ(zᵢ) = mean_i softplus(−zᵢ)
//
// The length-normalized reward β·avglogp removes DPO's length bias and needs no
// reference model. β from attrs["beta"] (default 2), γ from attrs["gamma"]
// (default 1). Stable via softplus (§V12), f64 accumulation (§V10).
func simpoKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	b, err := checkPair("simpo", in)
	if err != nil {
		return nil, err
	}
	w, l := in[0], in[1]
	pa, _ := attrs.(backend.SimPOAttrs)
	pa = pa.WithDefaults()
	beta := pa.Beta
	gamma := pa.Gamma

	var total float64
	for i := range b {
		z := beta*(w.AtF64(i)-l.AtF64(i)) - gamma
		total += softplus(-z)
	}
	out := tensor.NewOn(ctx.Device(), w.Dtype(), tensor.Shape{})
	out.SetF64(total / float64(b))
	return []*tensor.Tensor{out}, nil
}

// orpoKernel is the fused ORPO loss (Hong et al. 2024, arXiv:2403.07691,
// Eqs. 6-7) — a monolithic SFT + odds-ratio preference objective, reference-free
// and with no separate SFT phase. Inputs [batch] are the length-normalized
// average log-probabilities of the chosen and rejected responses:
//
//	Pw, Pl   = exp(avgChosen), exp(avgRejected)                 // seq probabilities
//	logOdds  = (avgChosen − avgRejected) − (log(1−Pw) − log(1−Pl))
//	loss     = mean_i( −avgChosenᵢ + λ·softplus(−logOddsᵢ) )
//
// The first term is the SFT negative-log-likelihood on the chosen response; the
// second is −log σ of the log-odds ratio, weighting correct-over-incorrect odds.
// λ from attrs["lambda"] (default 0.1). log(1−P) is computed as log1p(−P) for
// accuracy near P→1; requires avg log-probs < 0 (P < 1). f64 accumulation (§V10).
func orpoKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	b, err := checkPair("orpo", in)
	if err != nil {
		return nil, err
	}
	w, l := in[0], in[1]
	pa, _ := attrs.(backend.ORPOAttrs)
	pa = pa.WithDefaults()
	lambda := pa.Lambda

	var total float64
	for i := range b {
		aw, al := w.AtF64(i), l.AtF64(i)
		logOdds := (aw - al) - (math.Log1p(-math.Exp(aw)) - math.Log1p(-math.Exp(al)))
		total += -aw + lambda*softplus(-logOdds)
	}
	out := tensor.NewOn(ctx.Device(), w.Dtype(), tensor.Shape{})
	out.SetF64(total / float64(b))
	return []*tensor.Tensor{out}, nil
}

func init() {
	//perfscan:ignore PS3062 reference oracle: intentionally simple, correctness baseline not an optimization target
	std.add(backend.OpSimPO, tensor.F32, simpoKernel)
	std.add(backend.OpSimPO, tensor.F64, simpoKernel)
	//perfscan:ignore PS3062 reference oracle: intentionally simple, correctness baseline not an optimization target
	std.add(backend.OpORPO, tensor.F32, orpoKernel)
	std.add(backend.OpORPO, tensor.F64, orpoKernel)
}
