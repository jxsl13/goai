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
		// Fast paths: softmaxRowT re-read each logit through z.AtF64 twice (max pass +
		// exp pass) and the result was written back cell-by-cell via gs.SetF64 —
		// interface dispatch dominating an O(b·c) kernel. When teacher and student
		// share a contiguous dtype, walk the row slices directly. Bit-identical: the
		// same v/temp, max, exp(·−m), sum and division, in the same j order.
		switch {
		case zs.Dtype() == tensor.F64 && zt.Dtype() == tensor.F64:
			zss := zs.Contiguous().Storage().F64()
			zts := zt.Contiguous().Storage().F64()
			ds := gs.Storage().F64()
			p := make([]float64, c)
			q := make([]float64, c)
			for i := 0; i < b; i++ {
				base := i * c
				softmaxRowTInto(p, zts[base:base+c], temp)
				softmaxRowTInto(q, zss[base:base+c], temp)
				for j := 0; j < c; j++ {
					ds[base+j] = scale * (q[j] - p[j])
				}
			}
			return []*tensor.Tensor{gs, nil}, nil
		case zs.Dtype() == tensor.F32 && zt.Dtype() == tensor.F32:
			zss := zs.Contiguous().Storage().F32()
			zts := zt.Contiguous().Storage().F32()
			ds := gs.Storage().F32()
			p := make([]float64, c)
			q := make([]float64, c)
			row := make([]float64, c)
			for i := 0; i < b; i++ {
				base := i * c
				for j := 0; j < c; j++ {
					row[j] = float64(zts[base+j])
				}
				softmaxRowTInto(p, row, temp)
				for j := 0; j < c; j++ {
					row[j] = float64(zss[base+j])
				}
				softmaxRowTInto(q, row, temp)
				for j := 0; j < c; j++ {
					ds[base+j] = float32(scale * (q[j] - p[j]))
				}
			}
			return []*tensor.Tensor{gs, nil}, nil
		}

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

// softmaxRowTInto writes the stable softmax of the logit row (scaled by 1/temp)
// into out (len(out) == len(zrow)). Same passes as softmaxRowT on a typed slice.
func softmaxRowTInto(out, zrow []float64, temp float64) {
	m := math.Inf(-1)
	for j := range zrow {
		if v := zrow[j] / temp; v > m {
			m = v
		}
	}
	var sum float64
	for j := range zrow {
		e := math.Exp(zrow[j]/temp - m)
		out[j] = e
		sum += e
	}
	for j := range out {
		out[j] /= sum
	}
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
