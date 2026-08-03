package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mhaMaskedKernelCPU is the parallel F64 masked multi-head attention (Q·Kᵀ scale
// +additive-mask, softmax, ·V). The (head, query-row) iterations are independent —
// each writes a disjoint out[i, h·dk:] slice and computes its scores/softmax/·V in
// the same ascending-j order as the serial ref kernel — so parallelizing over the
// flattened head×query space (per-worker row/obuf scratch) is bit-exact. F64 only;
// F32/exotic fall back to the ref kernel.
func mhaMaskedKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("cpu: mha_masked wants (Q,K,V,mask), got %d inputs", len(in))
	}
	q, k, v, mask := in[0], in[1], in[2], in[3]
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("cpu: mha_masked needs rank-2 Q,K,V")
	}
	sq, dm := q.Shape()[0], q.Shape()[1]
	sk := k.Shape()[0]
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("cpu: mha_masked dmodel %d not divisible by heads %d", dm, heads)
	}
	perHeadMask := mask.Ndim() == 3
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("cpu: mha_masked heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	rep := heads / kvHeads
	if k.Shape()[1] != kvHeads*dk || !v.Shape().Equal(k.Shape()) {
		return nil, fmt.Errorf("cpu: mha_masked K/V must be [sk,%d]", kvHeads*dk)
	}
	// Validate the mask shape (mirrors ref).
	switch {
	case mask.Ndim() == 2 && (mask.Shape()[0] != sq || mask.Shape()[1] != sk):
		return nil, fmt.Errorf("cpu: mha_masked mask must be [%d,%d], got %v", sq, sk, mask.Shape())
	case perHeadMask && (mask.Shape()[0] != heads || mask.Shape()[1] != sq || mask.Shape()[2] != sk):
		return nil, fmt.Errorf("cpu: mha_masked per-head mask must be [%d,%d,%d], got %v", heads, sq, sk, mask.Shape())
	case mask.Ndim() != 2 && mask.Ndim() != 3:
		return nil, fmt.Errorf("cpu: mha_masked mask must be rank 2 or 3, got %d", mask.Ndim())
	}
	scale := pa.Scale / math.Sqrt(float64(dk))

	out := tensor.NewOn(ctx.Device(), q.Dtype(), tensor.Shape{sq, dm})

	// F32 fast path: close the dtype-gap (this kernel was cpu-registered F64-only, so F32
	// fell to backend/ref's masked-attn scan). ref's F32 input deterministically takes its
	// devirtualised path (f64Data widens F32→F64; the generic AtF64 loop is only for dtypes
	// f64Data rejects), computing every score/softmax/·V in F64 and narrowing only on store.
	// We reproduce that F64 arithmetic in the SAME ascending-j / j-outer-d-inner order,
	// widening each F32 read per element (float64(x) is identical whether done up-front like
	// ref or per-access) and narrowing on store → BYTE-IDENTICAL to ref, and we skip ref's
	// up-front Q/K/V/mask F64 materialization. (head,query-row) pairs are independent.
	// EVERY input must be F32, not just q. The guard used to test q alone and then read k, v, mask
	// and out as F32 as well, so a mixed set — an F32 query against an F64 additive mask, which is
	// what a trainable relative-position bias produces — panicked inside Storage().F32(). A dtype
	// fast path has to be guarded on every tensor it reads, not on the one that named it.
	if q.Dtype() == tensor.F32 && k.Dtype() == tensor.F32 && v.Dtype() == tensor.F32 &&
		mask.Dtype() == tensor.F32 && out.Dtype() == tensor.F32 {
		qs := q.Contiguous().Storage().F32()
		ks := k.Contiguous().Storage().F32()
		vs := v.Contiguous().Storage().F32()
		ms := mask.Contiguous().Storage().F32()
		os := out.Storage().F32()
		kdm := kvHeads * dk
		parallelWork(heads*sq, sk*dk, func(lo, hi int) {
			row := make([]float64, sk)
			obuf := make([]float64, dk)
			for idx := lo; idx < hi; idx++ {
				h := idx / sq
				i := idx % sq
				qOff := h * dk
				kvOff := (h / rep) * dk
				qrow := qs[i*dm+qOff : i*dm+qOff+dk]
				mOff := i * sk
				if perHeadMask {
					mOff = (h*sq + i) * sk
				}
				mrow := ms[mOff : mOff+sk]
				m := math.Inf(-1)
				// FOUR KEYS PER PASS OVER THE QUERY ROW, but only where all four are unmasked.
				// Each score is a dk-term reduction into one accumulator, so the loop is bound by
				// the latency of a dependent add chain; four keys give four chains that interleave
				// and load each query element once. A causal or block mask keeps its live region
				// contiguous, so nearly every group is all-in or all-out and the four cheap
				// infinity tests decide which.
				//
				// Bit-identical: every score sums its own terms in ascending d into its own
				// accumulator, and the mask add, the scale and the running maximum are applied to
				// the keys in ascending j exactly as the one-at-a-time form does.
				j := 0
				for ; j+7 < sk; j += 8 {
					m0, m1 := float64(mrow[j+0]), float64(mrow[j+1])
					m2, m3 := float64(mrow[j+2]), float64(mrow[j+3])
					m4, m5 := float64(mrow[j+4]), float64(mrow[j+5])
					m6, m7 := float64(mrow[j+6]), float64(mrow[j+7])
					if math.IsInf(m0, -1) ||
						math.IsInf(m1, -1) ||
						math.IsInf(m2, -1) ||
						math.IsInf(m3, -1) ||
						math.IsInf(m4, -1) ||
						math.IsInf(m5, -1) ||
						math.IsInf(m6, -1) ||
						math.IsInf(m7, -1) {
						// A MIXED GROUP IS HANDLED IN PLACE, not by abandoning the fast path for
						// the rest of the row. Breaking out here was measured and is wrong for any
						// mask whose live region is not a prefix: it sent a whole row to the
						// scalar loop after one mixed group, and cost 2 to 3% on the general-mask
						// cell while the block-masked one gained 30%.
						for o := range 8 {
							mv := float64(mrow[j+o])
							if math.IsInf(mv, -1) {
								row[j+o] = math.Inf(-1)
								continue
							}
							kr := ks[(j+o)*kdm+kvOff : (j+o)*kdm+kvOff+dk]
							var sv float64
							for d, qv := range qrow {
								sv += float64(qv) * float64(kr[d])
							}
							sv = sv*scale + mv
							row[j+o] = sv
							if sv > m {
								m = sv
							}
						}
						continue
					}
					k0 := ks[(j+0)*kdm+kvOff : (j+0)*kdm+kvOff+dk]
					k1 := ks[(j+1)*kdm+kvOff : (j+1)*kdm+kvOff+dk]
					k2 := ks[(j+2)*kdm+kvOff : (j+2)*kdm+kvOff+dk]
					k3 := ks[(j+3)*kdm+kvOff : (j+3)*kdm+kvOff+dk]
					k4 := ks[(j+4)*kdm+kvOff : (j+4)*kdm+kvOff+dk]
					k5 := ks[(j+5)*kdm+kvOff : (j+5)*kdm+kvOff+dk]
					k6 := ks[(j+6)*kdm+kvOff : (j+6)*kdm+kvOff+dk]
					k7 := ks[(j+7)*kdm+kvOff : (j+7)*kdm+kvOff+dk]
					var s0, s1, s2, s3, s4, s5, s6, s7 float64
					for d, qv := range qrow {
						q := float64(qv)
						s0 += q * float64(k0[d])
						s1 += q * float64(k1[d])
						s2 += q * float64(k2[d])
						s3 += q * float64(k3[d])
						s4 += q * float64(k4[d])
						s5 += q * float64(k5[d])
						s6 += q * float64(k6[d])
						s7 += q * float64(k7[d])
					}
					for o, pair := range [8][2]float64{{s0, m0}, {s1, m1}, {s2, m2}, {s3, m3}, {s4, m4}, {s5, m5}, {s6, m6}, {s7, m7}} {
						sv := pair[0]*scale + pair[1]
						row[j+o] = sv
						if sv > m {
							m = sv
						}
					}
				}
				for ; j < sk; j++ {
					mv := float64(mrow[j])
					if math.IsInf(mv, -1) {
						row[j] = math.Inf(-1)
						continue
					}
					krow := ks[j*kdm+kvOff : j*kdm+kvOff+dk]
					var s float64
					for d, qv := range qrow {
						s += float64(qv) * float64(krow[d])
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
				for d := range obuf {
					obuf[d] = 0
				}
				if sum > 0 {
					mhaWeightedSum(obuf, row, vs, sum, sk, kdm, kvOff, dk)
				}
				for d := 0; d < dk; d++ {
					os[i*dm+qOff+d] = float32(obuf[d])
				}
			}
		})
		return []*tensor.Tensor{out}, nil
	}

	// Below here everything is read as F64, so a MIXED set — an F32 query against an F64 additive
	// mask, which is what a trainable relative-position bias produces — belongs to neither arm and
	// used to panic inside Storage().F64(). Hand those to the reference kernel, which reads through
	// the dtype-agnostic accessor and is what this op fell back to before the CPU kernel existed.
	if q.Dtype() != tensor.F64 || k.Dtype() != tensor.F64 || v.Dtype() != tensor.F64 ||
		mask.Dtype() != tensor.F64 || out.Dtype() != tensor.F64 {
		return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpMHAMasked, in, attrs)
	}
	qs := q.Contiguous().Storage().F64()
	ks := k.Contiguous().Storage().F64()
	vs := v.Contiguous().Storage().F64()
	ms := mask.Contiguous().Storage().F64()
	os := out.Storage().F64()
	kdm := kvHeads * dk

	parallelWork(heads*sq, sk*dk, func(lo, hi int) {
		row := make([]float64, sk)
		obuf := make([]float64, dk)
		for idx := lo; idx < hi; idx++ {
			h := idx / sq
			i := idx % sq
			qOff := h * dk
			kvOff := (h / rep) * dk
			qrow := qs[i*dm+qOff : i*dm+qOff+dk]
			mOff := i * sk
			if perHeadMask {
				mOff = (h*sq + i) * sk
			}
			mrow := ms[mOff : mOff+sk]
			m := math.Inf(-1)
			for j, mv := range mrow {
				if math.IsInf(mv, -1) {
					row[j] = math.Inf(-1)
					continue
				}
				krow := ks[j*kdm+kvOff : j*kdm+kvOff+dk]
				var s float64
				for d, qv := range qrow {
					s += qv * krow[d]
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
			for d := range obuf {
				obuf[d] = 0
			}
			if sum > 0 {
				mhaWeightedSum(obuf, row, vs, sum, sk, kdm, kvOff, dk)
			}
			copy(os[i*dm+qOff:i*dm+qOff+dk], obuf)
		}
	})
	return []*tensor.Tensor{out}, nil
}

// mhaWeightedSum accumulates the softmax-weighted value rows into obuf, FOUR KEYS PER PASS. obuf
// does not vary with the key, so the key-at-a-time form makes a full load-store round trip
// through it for a single addition each; holding obuf[d] in a local across four additions stores
// once instead of four times and reads the value rows in one interleaved pass.
//
// BIT-IDENTICAL: obuf[d] takes the same four additions in the same ascending key order, only in
// a register instead of memory. Masked keys carry row[j] == 0 by the time this runs, so no branch
// is needed and the tail is a plain remainder.
//
// MEASURED as 38% of BenchmarkMHAMaskedF32CPU_1024x1024x64x16's profile before the change; the
// same loop in the selective-attention kernel was 26% of its own.
func mhaWeightedSum[T float32 | float64](obuf, row []float64, vs []T, sum float64,
	sk, kdm, kvOff, dk int) {
	j := 0
	for ; j+3 < sk; j += 4 {
		w0, w1 := row[j]/sum, row[j+1]/sum
		w2, w3 := row[j+2]/sum, row[j+3]/sum
		b0 := j*kdm + kvOff
		b1, b2, b3 := b0+kdm, b0+2*kdm, b0+3*kdm
		v0 := vs[b0 : b0+dk : b0+dk]
		v1 := vs[b1 : b1+dk : b1+dk]
		v2 := vs[b2 : b2+dk : b2+dk]
		v3 := vs[b3 : b3+dk : b3+dk]
		for d := 0; d < dk; d++ {
			t := obuf[d]
			t += w0 * float64(v0[d])
			t += w1 * float64(v1[d])
			t += w2 * float64(v2[d])
			t += w3 * float64(v3[d])
			obuf[d] = t
		}
	}
	for ; j < sk; j++ {
		w := row[j] / sum
		vrow := vs[j*kdm+kvOff : j*kdm+kvOff+dk : j*kdm+kvOff+dk]
		for d, vv := range vrow {
			obuf[d] += w * float64(vv)
		}
	}
}

func init() {
	std.add(backend.OpMHAMasked, tensor.F64, mhaMaskedKernelCPU)
	std.add(backend.OpMHAMasked, tensor.F32, mhaMaskedKernelCPU)
}
