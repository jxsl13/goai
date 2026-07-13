package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// distillKernel is the fused soft-target knowledge-distillation loss (Hinton,
// Vinyals & Dean 2015, arXiv:1503.02531). Inputs are student and teacher logits
// [batch, classes]; the loss is the temperature-softened KL divergence, scaled
// by T² so its gradient magnitude matches the hard-label loss:
//
//	pᵢ = softmax(z_teacherᵢ / T),  qᵢ = softmax(z_studentᵢ / T)
//	loss = mean_b T²·KL(p_b ‖ q_b) = mean_b T²·Σᵢ pᵢ·(log pᵢ − log qᵢ)
//
// Both softmaxes use a stable max-shift; KL accumulates in f64 (§V10). The teacher
// is a constant (frozen). T from attrs["temperature"] (default 1). Output scalar.
func distillKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: distill wants (studentLogits, teacherLogits), got %d inputs", len(in))
	}
	zs, zt := in[0], in[1]
	if zs.Ndim() != 2 || zt.Ndim() != 2 {
		return nil, fmt.Errorf("ref: distill needs rank-2 logits, got %dD, %dD", zs.Ndim(), zt.Ndim())
	}
	b, c := zs.Shape()[0], zs.Shape()[1]
	if zt.Shape()[0] != b || zt.Shape()[1] != c {
		return nil, fmt.Errorf("ref: distill student %v != teacher %v", zs.Shape(), zt.Shape())
	}
	if b == 0 {
		return nil, fmt.Errorf("ref: distill on empty batch")
	}
	pa, _ := attrs.(backend.DistillAttrs)
	pa = pa.WithDefaults()
	temp := pa.Temperature
	if temp <= 0 {
		return nil, fmt.Errorf("ref: distill temperature must be > 0, got %g", temp)
	}

	var total float64
	for i := range b {
		p := softmaxRow(zt, i, c, temp) // teacher soft targets
		q := softmaxRow(zs, i, c, temp) // student soft distribution
		var kl float64
		for j := range c {
			if p[j] > 0 {
				kl += p[j] * (math.Log(p[j]) - math.Log(q[j]))
			}
		}
		total += temp * temp * kl
	}
	out := tensor.NewOn(ctx.Device(), zs.Dtype(), tensor.Shape{})
	out.SetF64(total / float64(b))
	return []*tensor.Tensor{out}, nil
}

// softmaxRow returns the stable softmax of row i of z scaled by 1/temp.
func softmaxRow(z *tensor.Tensor, i, c int, temp float64) []float64 {
	m := math.Inf(-1)
	for j := range c {
		if v := z.AtF64(i, j) / temp; v > m {
			m = v
		}
	}
	out := make([]float64, c)
	var sum float64
	for j := range c {
		e := math.Exp(z.AtF64(i, j)/temp - m)
		out[j] = e
		sum += e
	}
	for j := range c {
		out[j] /= sum
	}
	return out
}

func init() {
	std.add(backend.OpDistill, tensor.F32, distillKernel)
	std.add(backend.OpDistill, tensor.F64, distillKernel)
}
