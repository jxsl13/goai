package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mhaMaskedBackwardKernel is the backward of mhaMaskedKernel (§T730): given
// (Q[sq,dm], K[sk,kv·dk], V[sk,kv·dk], mask, dO[sq,dm]) it returns
// (dQ, dK, dV, dmask). It supports the shared [sq,sk] and the per-head
// [heads,sq,sk] mask, GQA (kvHeads), and rectangular sq≠sk (cross-attention),
// matching the forward. −Inf mask entries are excluded (zero weight, zero grad).
// Correctness-first (per-element); it is dispatched by OpMHAMasked's VJP for
// training (e.g. fine-tuning T5's relative-position attention), not the decode
// hot path, so it is not devirtualised. dmask lets a trainable bias (T5's
// relative_attention_bias) receive gradients; a constant mask ignores it.
func mhaMaskedBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 5 {
		return nil, fmt.Errorf("ref: mha_masked_backward wants (Q,K,V,mask,dO), got %d", len(in))
	}
	q, k, v, mask, g := in[0], in[1], in[2], in[3], in[4]
	sq, dm := q.Shape()[0], q.Shape()[1]
	sk := k.Shape()[0]
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("ref: mha_masked_backward dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 {
		kvHeads = heads
	}
	rep := heads / kvHeads
	scale := pa.Scale / math.Sqrt(float64(dk))
	perHead := mask.Ndim() == 3

	dQ := tensor.NewOn(ctx.Device(), q.Dtype(), q.Shape())
	dK := tensor.NewOn(ctx.Device(), k.Dtype(), k.Shape())
	dV := tensor.NewOn(ctx.Device(), v.Dtype(), v.Shape())
	dMask := tensor.NewOn(ctx.Device(), mask.Dtype(), mask.Shape())

	maskAt := func(h, i, j int) float64 {
		if perHead {
			return mask.AtF64(h, i, j)
		}
		return mask.AtF64(i, j)
	}
	addMask := func(h, i, j int, d float64) {
		if perHead {
			dMask.SetF64(dMask.AtF64(h, i, j)+d, h, i, j)
		} else {
			dMask.SetF64(dMask.AtF64(i, j)+d, i, j) // 2-D mask is shared → accumulate over heads
		}
	}

	row := make([]float64, sk) // softmax weights for the current (head,row)
	dw := make([]float64, sk)  // dO·V per key
	for h := 0; h < heads; h++ {
		qOff := h * dk
		kvOff := (h / rep) * dk
		for i := 0; i < sq; i++ {
			// recompute the softmax weights for (h, i)
			m := math.Inf(-1)
			for j := 0; j < sk; j++ {
				mv := maskAt(h, i, j)
				if math.IsInf(mv, -1) {
					row[j] = math.Inf(-1)
					continue
				}
				var s float64
				for d := 0; d < dk; d++ {
					s += q.AtF64(i, qOff+d) * k.AtF64(j, kvOff+d)
				}
				s = s*scale + mv
				row[j] = s
				if s > m {
					m = s
				}
			}
			var sum float64
			for j := 0; j < sk; j++ {
				if math.IsInf(row[j], -1) {
					row[j] = 0
					continue
				}
				row[j] = math.Exp(row[j] - m)
				sum += row[j]
			}
			if sum > 0 {
				for j := range row {
					row[j] /= sum
				}
			}
			// dV[j] += weights[j]·dO[i]; dw[j] = dO[i]·V[j]
			var wdot float64
			for j := 0; j < sk; j++ {
				var d float64
				for c := 0; c < dk; c++ {
					gd := g.AtF64(i, qOff+c)
					d += gd * v.AtF64(j, kvOff+c)
					dV.SetF64(dV.AtF64(j, kvOff+c)+row[j]*gd, j, kvOff+c)
				}
				dw[j] = d
				wdot += row[j] * d
			}
			// dscore[j] = weights[j]·(dw[j] − Σ weights·dw); dmask += dscore;
			// dQ/dK carry the scaled dscore into the projections.
			for j := 0; j < sk; j++ {
				dscore := row[j] * (dw[j] - wdot)
				addMask(h, i, j, dscore)
				ds := dscore * scale
				for d := 0; d < dk; d++ {
					dQ.SetF64(dQ.AtF64(i, qOff+d)+ds*k.AtF64(j, kvOff+d), i, qOff+d)
					dK.SetF64(dK.AtF64(j, kvOff+d)+ds*q.AtF64(i, qOff+d), j, kvOff+d)
				}
			}
		}
	}
	return []*tensor.Tensor{dQ, dK, dV, dMask}, nil
}

func init() {
	std.add(backend.OpMHAMaskedBackward, tensor.F32, mhaMaskedBackwardKernel)
	std.add(backend.OpMHAMaskedBackward, tensor.F64, mhaMaskedBackwardKernel)
}
