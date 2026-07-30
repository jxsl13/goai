package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/parallel"
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
	// F64 fast path: the correctness-first loop reads q/k/v/g and RMWs dQ/dK/dV/dMask via
	// AtF64/SetF64 on every element — the score dot, the P·V backward (g re-read per key) and
	// the dQ/dK projections all dispatch. Walk contiguous typed storage, hoisting the query
	// and dO rows per (head,i). Bit-identical: same values, same ascending accumulation and
	// the same h→i→j→d iteration order into every gradient. Original loop kept as fallback.
	qs, qok := f64Data(q)
	ks, kok := f64Data(k)
	vs, vok := f64Data(v)
	gs, gok := f64Data(g)
	masks, mok := f64Data(mask)
	dqs, dqok := f64Data(dQ)
	dks, dkok := f64Data(dK)
	dvs, dvok := f64Data(dV)
	dms, dmok := f64Data(dMask)
	if qok && kok && vok && gok && mok && dqok && dkok && dvok && dmok {
		// Parallel-over-heads path. With rep==1 every head owns DISJOINT query/key/value
		// columns, so dQ/dK/dV are written race-free directly. The only cross-head
		// accumulation is a shared 2-D dMask (perHead==false): each head writes its own
		// [sq,sk] contribution slice, then a serial pass sums them IN HEAD ORDER — the same
		// h=0,1,2… order the serial loop accumulates in, so the result is bit-identical.
		// Falls through to the serial fast path for GQA (rep>1, shared dK/dV), <2 heads, no
		// worker pool, or when the dMask contribution buffer would be too large.
		const maskBufCap = 128 << 20 // bytes
		if rep == 1 && heads >= 2 && parallel.Workers() > 1 && (perHead || heads*sq*sk*8 <= maskBufCap) {
			var maskBuf []float64 // [heads*sq*sk] per-head dMask contributions (perHead==false only)
			if !perHead {
				maskBuf = make([]float64, heads*sq*sk)
			}
			parallel.Rows(heads, func(hlo, hhi int) {
				row := make([]float64, sk)
				dw := make([]float64, sk)
				qi := make([]float64, dk)
				gi := make([]float64, dk)
				for h := hlo; h < hhi; h++ {
					qOff := h * dk
					kvOff := h * dk // rep==1: each head owns its own kv columns
					for i := 0; i < sq; i++ {
						qbase := i * dm
						for d := 0; d < dk; d++ {
							qi[d] = qs[qbase+qOff+d]
							gi[d] = gs[qbase+qOff+d]
						}
						mBase := i * sk
						if perHead {
							mBase = (h*sq + i) * sk
						}
						m := math.Inf(-1)
						for j := 0; j < sk; j++ {
							mv := masks[mBase+j]
							if math.IsInf(mv, -1) {
								row[j] = math.Inf(-1)
								continue
							}
							krow := ks[j*dm+kvOff : j*dm+kvOff+dk : j*dm+kvOff+dk]
							var sc float64
							for d := 0; d < dk; d++ {
								sc += qi[d] * krow[d]
							}
							sc = sc*scale + mv
							row[j] = sc
							if sc > m {
								m = sc
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
						var wdot float64
						for j := 0; j < sk; j++ {
							vrow := vs[j*dm+kvOff : j*dm+kvOff+dk : j*dm+kvOff+dk]
							dvrow := dvs[j*dm+kvOff : j*dm+kvOff+dk : j*dm+kvOff+dk]
							rj := row[j]
							var d float64
							for c := 0; c < dk; c++ {
								d += gi[c] * vrow[c]
								dvrow[c] += rj * gi[c]
							}
							dw[j] = d
							wdot += rj * d
						}
						dqrow := dqs[qbase+qOff : qbase+qOff+dk : qbase+qOff+dk]
						mbBase := h*sq*sk + i*sk
						for j := 0; j < sk; j++ {
							dscore := row[j] * (dw[j] - wdot)
							if perHead {
								dms[mBase+j] += dscore
							} else {
								maskBuf[mbBase+j] = dscore
							}
							ds := dscore * scale
							krow := ks[j*dm+kvOff : j*dm+kvOff+dk : j*dm+kvOff+dk]
							dkrow := dks[j*dm+kvOff : j*dm+kvOff+dk : j*dm+kvOff+dk]
							for d := 0; d < dk; d++ {
								dqrow[d] += ds * krow[d]
								dkrow[d] += ds * qi[d]
							}
						}
					}
				}
			})
			if !perHead { // sum per-head dMask contributions in head order → bit-identical to serial
				plane := sq * sk
				for h := 0; h < heads; h++ {
					base := h * plane
					for idx := 0; idx < plane; idx++ {
						dms[idx] += maskBuf[base+idx]
					}
				}
			}
			return []*tensor.Tensor{dQ, dK, dV, dMask}, nil
		}
		{
			{
				qi := make([]float64, dk)
				gi := make([]float64, dk)
				for h := 0; h < heads; h++ {
					qOff := h * dk
					kvOff := (h / rep) * dk
					for i := 0; i < sq; i++ {
						qbase := i * dm
						for d := 0; d < dk; d++ {
							qi[d] = qs[qbase+qOff+d]
							gi[d] = gs[qbase+qOff+d]
						}
						mBase := i * sk
						if perHead {
							mBase = (h*sq + i) * sk
						}
						m := math.Inf(-1)
						for j := 0; j < sk; j++ {
							mv := masks[mBase+j]
							if math.IsInf(mv, -1) {
								row[j] = math.Inf(-1)
								continue
							}
							krow := ks[j*dm+kvOff : j*dm+kvOff+dk : j*dm+kvOff+dk]
							var sc float64
							for d := 0; d < dk; d++ {
								sc += qi[d] * krow[d]
							}
							sc = sc*scale + mv
							row[j] = sc
							if sc > m {
								m = sc
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
						var wdot float64
						for j := 0; j < sk; j++ {
							vrow := vs[j*dm+kvOff : j*dm+kvOff+dk : j*dm+kvOff+dk]
							dvrow := dvs[j*dm+kvOff : j*dm+kvOff+dk : j*dm+kvOff+dk]
							rj := row[j]
							var d float64
							for c := 0; c < dk; c++ {
								d += gi[c] * vrow[c]
								dvrow[c] += rj * gi[c]
							}
							dw[j] = d
							wdot += rj * d
						}
						dqrow := dqs[qbase+qOff : qbase+qOff+dk : qbase+qOff+dk]
						for j := 0; j < sk; j++ {
							dscore := row[j] * (dw[j] - wdot)
							dms[mBase+j] += dscore
							ds := dscore * scale
							krow := ks[j*dm+kvOff : j*dm+kvOff+dk : j*dm+kvOff+dk]
							dkrow := dks[j*dm+kvOff : j*dm+kvOff+dk : j*dm+kvOff+dk]
							for d := 0; d < dk; d++ {
								dqrow[d] += ds * krow[d]
								dkrow[d] += ds * qi[d]
							}
						}
					}
				}
				return []*tensor.Tensor{dQ, dK, dV, dMask}, nil
			}
		}
	}
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
