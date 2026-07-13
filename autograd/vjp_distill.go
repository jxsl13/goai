package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Knowledge-distillation VJP (Hinton et al. 2015). With p = softmax(z_t/T),
// q = softmax(z_s/T) and loss = mean_b T²·KL(p‖q), the gradient of the soft
// cross-entropy w.r.t. the student logits is (1/T)(q−p); the T² prefactor makes
// it T·(q−p), so the soft loss keeps a magnitude comparable to the hard loss
// across temperatures (paper §2, §R51):
//
//	∂loss/∂z_sᵢ = (g/B)·T·(qᵢ − pᵢ)
//
// The teacher logits are frozen → nil gradient.
func init() {
	RegisterVJP(backend.OpDistill, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		zs, zt := in[0], in[1]
		pX, _ := attrs.(backend.DistillAttrs)
		temp := pX.WithDefaults().Temperature
		b, c := zs.Shape()[0], zs.Shape()[1]
		gv := g.AtF64()
		scale := gv * temp / float64(b)

		gs := tensor.New(zs.Dtype(), zs.Shape())
		for i := range b {
			p := softmaxRowT(zt, i, c, temp)
			q := softmaxRowT(zs, i, c, temp)
			for j := range c {
				gs.SetF64(scale*(q[j]-p[j]), i, j)
			}
		}
		return []*tensor.Tensor{gs, nil}, nil // teacher frozen
	})
}

// softmaxRowT returns the stable softmax of row i of z scaled by 1/temp.
func softmaxRowT(z *tensor.Tensor, i, c int, temp float64) []float64 {
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
