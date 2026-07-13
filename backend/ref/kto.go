package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// ktoKernel is the fused Kahneman-Tversky Optimization loss (Ethayarajh et al.
// 2024, arXiv:2402.01306). Unlike DPO/IPO it learns from UNPAIRED binary
// feedback: each example carries a label (1 = desirable, 0 = undesirable).
// Inputs [batch]: policy log-probs, FROZEN reference log-probs, labels. The
// implicit reward is r = β·(logπθ − logπref); with a DETACHED reference point
// z_ref in reward units (attrs["z_ref"] = β·KL_batch, the β-scaled batch KL
// estimate — matches TRL's 1−σ(β·(logratio−kl)) = σ(β·kl − r)):
//
//	desirable   (label=1): loss = λ_D·σ(z_ref − r)
//	undesirable (label=0): loss = λ_U·σ(r − z_ref)
//
// so desirable examples push the reward up and undesirable ones push it down
// (the prospect-theory value with reference point z_ref). β=attrs["beta"] (0.1),
// λ_D=attrs["lambda_d"], λ_U=attrs["lambda_u"] (default 1). Mean over batch,
// f64 accumulation (§V10). Output scalar (§R59).
func ktoKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("ref: kto wants (policyLogps, refLogps, labels), got %d inputs", len(in))
	}
	pl, rl, lab := in[0], in[1], in[2]
	for i, t := range in {
		if t.Ndim() != 1 {
			return nil, fmt.Errorf("ref: kto input %d must be rank-1 [batch], got %dD", i, t.Ndim())
		}
	}
	b := pl.Shape()[0]
	for i, t := range in {
		if t.Shape()[0] != b {
			return nil, fmt.Errorf("ref: kto input %d len %d != batch %d", i, t.Shape()[0], b)
		}
	}
	if b == 0 {
		return nil, fmt.Errorf("ref: kto on empty batch")
	}
	pa, _ := attrs.(backend.KTOAttrs)
	pa = pa.WithDefaults()
	beta := pa.Beta
	zRef := pa.ZRef
	lambdaD := pa.LambdaD
	lambdaU := pa.LambdaU

	var total float64
	for i := range b {
		r := beta * (pl.AtF64(i) - rl.AtF64(i))
		if lab.AtF64(i) != 0 { // desirable
			total += lambdaD * logistic(zRef-r)
		} else { // undesirable
			total += lambdaU * logistic(r-zRef)
		}
	}
	out := tensor.NewOn(ctx.Device(), pl.Dtype(), tensor.Shape{})
	out.SetF64(total / float64(b))
	return []*tensor.Tensor{out}, nil
}

// logistic is the numerically stable sigmoid 1/(1+e^{-x}).
func logistic(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	e := math.Exp(x)
	return e / (1 + e)
}

func init() {
	std.add(backend.OpKTO, tensor.F32, ktoKernel)
	std.add(backend.OpKTO, tensor.F64, ktoKernel)
}
