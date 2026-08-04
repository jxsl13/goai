package ref

import (
	"fmt"
	"math"
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/parallel"
	"github.com/jxsl13/goai/tensor"
)

// maskBufPool recycles the per-head dMask contribution buffer. See the allocation site for why a
// recycled buffer is safe: every slot is written before it is read.
var maskBufPool sync.Pool

// mhaMaskedBackwardKernel is the backward of mhaMaskedKernel (§T730): given
// (Q[sq,dm], K[sk,kv·dk], V[sk,kv·dk], mask, dO[sq,dm]) it returns
// (dQ, dK, dV, dmask). It supports the shared [sq,sk] and the per-head
// [heads,sq,sk] mask, GQA (kvHeads), and rectangular sq≠sk (cross-attention),
// matching the forward. −Inf mask entries are excluded (zero weight, zero grad).
// Correctness-first (per-element); it is dispatched by OpMHAMasked's VJP for
// training (e.g. fine-tuning T5's relative-position attention), not the decode
// hot path, so it is not devirtualised. dmask lets a trainable bias (T5's
// relative_attention_bias) receive gradients; a constant mask ignores it.
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
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
	// OUTPUTS GO THROUGH outF64, NOT f64Data. f64Data is an INPUT view: on F32 it returns a
	// detached widened copy, so a kernel that accumulates into it writes to a buffer nobody reads
	// and returns all-zero gradients. That is exactly what happened here — every F32 backward
	// through this op silently produced zeros, invisible because the only tests touching it built
	// F64 tensors. outF64 is the output counterpart and carries the flush that narrows the buffer
	// back into F32 storage.
	dqs, dqflush, dqok := outF64(dQ)
	dks, dkflush, dkok := outF64(dK)
	dvs, dvflush, dvok := outF64(dV)
	dms, dmflush, dmok := outF64(dMask)
	flushOut := func() { dqflush(); dkflush(); dvflush(); dmflush() }
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
			// The per-head contribution buffer is the largest allocation this kernel makes —
			// heads*sq*sk float64, 16.7 MB at 8 heads and 512x512 — and it is FULLY OVERWRITTEN
			// before any read: the inner loop stores maskBuf[mbBase+j] for every j, masked entries
			// included, with a plain assignment rather than an accumulate. So recycled memory is
			// indistinguishable from fresh zeroed memory, and the runtime's zeroing of a fresh
			// 16.7 MB is pure waste that scales with sq*sk while the compute scales with sq*sk*dk.
			var maskBuf []float64 // [heads*sq*sk] per-head dMask contributions (perHead==false only)
			if !perHead {
				n := heads * sq * sk
				if b, _ := maskBufPool.Get().(*[]float64); b != nil && cap(*b) >= n {
					maskBuf = (*b)[:n]
					defer func(p *[]float64) { maskBufPool.Put(p) }(b)
				} else {
					maskBuf = make([]float64, n)
					buf := maskBuf
					defer func() { maskBufPool.Put(&buf) }()
				}
			}
			parallel.Rows(heads, func(hlo, hhi int) {
				//perfscan:ignore PS6008 reference oracle: intentionally simple, correctness baseline not an optimization target
				row := make([]float64, sk)
				//perfscan:ignore PS6008 reference oracle: intentionally simple, correctness baseline not an optimization target
				dw := make([]float64, sk)
				//perfscan:ignore PS6008 reference oracle: intentionally simple, correctness baseline not an optimization target
				qi := make([]float64, dk)
				//perfscan:ignore PS6008 reference oracle: intentionally simple, correctness baseline not an optimization target
				gi := make([]float64, dk)
				for h := hlo; h < hhi; h++ {
					qOff := h * dk
					kvOff := h * dk // rep==1: each head owns its own kv columns
					//perfscan:ignore PS3043,PS3046 reference oracle: intentionally simple, correctness baseline not an optimization target
					for i := 0; i < sq; i++ {
						qbase := i * dm
						//perfscan:ignore PS4004 reference oracle: intentionally simple, correctness baseline not an optimization target
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
							//perfscan:ignore PS3010,PS4008,PS4012 reference oracle: intentionally simple, correctness baseline not an optimization target
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
						// FOUR KEYS PER PASS OVER THE dO ROW. gi does not vary with the key, so the
						// key-at-a-time form re-streams it once per key and runs ONE accumulator
						// chain per key. BIT-IDENTICAL: every dw[j] sums over the same ascending
						// c into its own accumulator, every dV element takes the same single
						// addition, and wdot still accumulates in ascending j.
						var wdot float64
						jv := 0
						//perfscan:ignore PS3066,PS3076 reference oracle: intentionally simple, correctness baseline not an optimization target
						for ; jv+3 < sk; jv += 4 {
							b0 := jv*dm + kvOff
							v0 := vs[b0 : b0+dk : b0+dk]
							v1 := vs[b0+dm : b0+dm+dk : b0+dm+dk]
							v2 := vs[b0+2*dm : b0+2*dm+dk : b0+2*dm+dk]
							v3 := vs[b0+3*dm : b0+3*dm+dk : b0+3*dm+dk]
							w0 := dvs[b0 : b0+dk : b0+dk]
							w1 := dvs[b0+dm : b0+dm+dk : b0+dm+dk]
							w2 := dvs[b0+2*dm : b0+2*dm+dk : b0+2*dm+dk]
							w3 := dvs[b0+3*dm : b0+3*dm+dk : b0+3*dm+dk]
							r0, r1 := row[jv], row[jv+1]
							r2, r3 := row[jv+2], row[jv+3]
							var d0, d1, d2, d3 float64
							for c := 0; c < dk; c++ {
								gc := gi[c]
								d0 += gc * v0[c]
								d1 += gc * v1[c]
								d2 += gc * v2[c]
								d3 += gc * v3[c]
								w0[c] += r0 * gc
								w1[c] += r1 * gc
								w2[c] += r2 * gc
								w3[c] += r3 * gc
							}
							//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
							for o, dv := range [4]float64{d0, d1, d2, d3} {
								dw[jv+o] = dv
								wdot += row[jv+o] * dv
							}
						}
						for j := jv; j < sk; j++ {
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
						// FOUR KEYS PER PASS. dqrow and qi are BOTH shared across keys, so the
						// key-at-a-time form makes a full load-store round trip through dqrow for one
						// addition each and re-reads the query element per key. Holding dqrow[d] in a
						// local across four additions stores once. BIT-IDENTICAL: dqrow[d] takes the
						// same four additions in the same ascending j, each dK element takes the same
						// single addition, and the dscore stores stay in ascending j.
						jd := 0
						for ; jd+3 < sk; jd += 4 {
							var sc [4]float64
							for o := range 4 {
								j := jd + o
								dscore := row[j] * (dw[j] - wdot)
								if perHead {
									dms[mBase+j] += dscore
								} else {
									maskBuf[mbBase+j] = dscore
								}
								sc[o] = dscore * scale
							}
							b0 := jd*dm + kvOff
							k0 := ks[b0 : b0+dk : b0+dk]
							k1 := ks[b0+dm : b0+dm+dk : b0+dm+dk]
							k2 := ks[b0+2*dm : b0+2*dm+dk : b0+2*dm+dk]
							k3 := ks[b0+3*dm : b0+3*dm+dk : b0+3*dm+dk]
							y0 := dks[b0 : b0+dk : b0+dk]
							y1 := dks[b0+dm : b0+dm+dk : b0+dm+dk]
							y2 := dks[b0+2*dm : b0+2*dm+dk : b0+2*dm+dk]
							y3 := dks[b0+3*dm : b0+3*dm+dk : b0+3*dm+dk]
							for d := 0; d < dk; d++ {
								t := dqrow[d]
								t += sc[0] * k0[d]
								t += sc[1] * k1[d]
								t += sc[2] * k2[d]
								t += sc[3] * k3[d]
								dqrow[d] = t
								qd := qi[d]
								y0[d] += sc[0] * qd
								y1[d] += sc[1] * qd
								y2[d] += sc[2] * qd
								y3[d] += sc[3] * qd
							}
						}
						//perfscan:ignore PS1007 reference oracle: intentionally simple, correctness baseline not an optimization target
						for j := jd; j < sk; j++ {
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
				//perfscan:ignore PS1007 reference oracle: intentionally simple, correctness baseline not an optimization target
				for h := 0; h < heads; h++ {
					base := h * plane
					for idx := 0; idx < plane; idx++ {
						dms[idx] += maskBuf[base+idx]
					}
				}
			}
			flushOut()
			return []*tensor.Tensor{dQ, dK, dV, dMask}, nil
		}
		{
			{
				qi := make([]float64, dk)
				gi := make([]float64, dk)
				for h := 0; h < heads; h++ {
					qOff := h * dk
					kvOff := (h / rep) * dk
					//perfscan:ignore PS3043,PS3046 reference oracle: intentionally simple, correctness baseline not an optimization target
					for i := 0; i < sq; i++ {
						qbase := i * dm
						//perfscan:ignore PS4004 reference oracle: intentionally simple, correctness baseline not an optimization target
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
							//perfscan:ignore PS3010,PS4008,PS4012 reference oracle: intentionally simple, correctness baseline not an optimization target
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
						// FOUR KEYS PER PASS OVER THE dO ROW. gi does not vary with the key, so the
						// key-at-a-time form re-streams it once per key and runs ONE accumulator
						// chain per key. BIT-IDENTICAL: every dw[j] sums over the same ascending
						// c into its own accumulator, every dV element takes the same single
						// addition, and wdot still accumulates in ascending j.
						var wdot float64
						jv := 0
						//perfscan:ignore PS3076 reference oracle: intentionally simple, correctness baseline not an optimization target
						for ; jv+3 < sk; jv += 4 {
							b0 := jv*dm + kvOff
							v0 := vs[b0 : b0+dk : b0+dk]
							v1 := vs[b0+dm : b0+dm+dk : b0+dm+dk]
							v2 := vs[b0+2*dm : b0+2*dm+dk : b0+2*dm+dk]
							v3 := vs[b0+3*dm : b0+3*dm+dk : b0+3*dm+dk]
							w0 := dvs[b0 : b0+dk : b0+dk]
							w1 := dvs[b0+dm : b0+dm+dk : b0+dm+dk]
							w2 := dvs[b0+2*dm : b0+2*dm+dk : b0+2*dm+dk]
							w3 := dvs[b0+3*dm : b0+3*dm+dk : b0+3*dm+dk]
							r0, r1 := row[jv], row[jv+1]
							r2, r3 := row[jv+2], row[jv+3]
							var d0, d1, d2, d3 float64
							for c := 0; c < dk; c++ {
								gc := gi[c]
								d0 += gc * v0[c]
								d1 += gc * v1[c]
								d2 += gc * v2[c]
								d3 += gc * v3[c]
								w0[c] += r0 * gc
								w1[c] += r1 * gc
								w2[c] += r2 * gc
								w3[c] += r3 * gc
							}
							//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
							for o, dv := range [4]float64{d0, d1, d2, d3} {
								dw[jv+o] = dv
								wdot += row[jv+o] * dv
							}
						}
						for j := jv; j < sk; j++ {
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
						// FOUR KEYS PER PASS. dqrow and qi are BOTH shared across keys, so the
						// key-at-a-time form makes a full load-store round trip through dqrow for one
						// addition each and re-reads the query element per key. Holding dqrow[d] in a
						// local across four additions stores once. BIT-IDENTICAL: dqrow[d] takes the
						// same four additions in the same ascending j, each dK element takes the same
						// single addition, and the dscore stores stay in ascending j.
						jd := 0
						for ; jd+3 < sk; jd += 4 {
							var sc [4]float64
							for o := range 4 {
								j := jd + o
								dscore := row[j] * (dw[j] - wdot)
								dms[mBase+j] += dscore
								sc[o] = dscore * scale
							}
							b0 := jd*dm + kvOff
							k0 := ks[b0 : b0+dk : b0+dk]
							k1 := ks[b0+dm : b0+dm+dk : b0+dm+dk]
							k2 := ks[b0+2*dm : b0+2*dm+dk : b0+2*dm+dk]
							k3 := ks[b0+3*dm : b0+3*dm+dk : b0+3*dm+dk]
							y0 := dks[b0 : b0+dk : b0+dk]
							y1 := dks[b0+dm : b0+dm+dk : b0+dm+dk]
							y2 := dks[b0+2*dm : b0+2*dm+dk : b0+2*dm+dk]
							y3 := dks[b0+3*dm : b0+3*dm+dk : b0+3*dm+dk]
							for d := 0; d < dk; d++ {
								t := dqrow[d]
								t += sc[0] * k0[d]
								t += sc[1] * k1[d]
								t += sc[2] * k2[d]
								t += sc[3] * k3[d]
								dqrow[d] = t
								qd := qi[d]
								y0[d] += sc[0] * qd
								y1[d] += sc[1] * qd
								y2[d] += sc[2] * qd
								y3[d] += sc[3] * qd
							}
						}
						//perfscan:ignore PS1007 reference oracle: intentionally simple, correctness baseline not an optimization target
						for j := jd; j < sk; j++ {
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
				flushOut()
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
	//perfscan:ignore PS3062 reference oracle: intentionally simple, correctness baseline not an optimization target
	std.add(backend.OpMHAMaskedBackward, tensor.F32, mhaMaskedBackwardKernel)
	std.add(backend.OpMHAMaskedBackward, tensor.F64, mhaMaskedBackwardKernel)
}
