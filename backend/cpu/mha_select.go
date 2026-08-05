package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mhaSelectKernelCPU is the parallel F64 selective attention kernel: per (query,key)
// the sel mask picks either the (Q1,K1) or (Q2,K2) score source, then a standard
// softmax over V. Independent (head, query-row) iterations are parallelized over the
// worker pool (per-worker row/obuf scratch); byte-identical to the serial ref kernel
// (same ascending-j scores/softmax/·V, disjoint output slices). F64 only; F32/exotic
// fall back to the ref kernel.
func mhaSelectKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 6 {
		return nil, fmt.Errorf("cpu: mha_select wants (Q1,K1,Q2,K2,V,sel), got %d inputs", len(in))
	}
	q1, k1, q2, k2, v, sel := in[0], in[1], in[2], in[3], in[4], in[5]
	for _, t := range in {
		if t.Ndim() != 2 {
			return nil, fmt.Errorf("cpu: mha_select needs rank-2 inputs")
		}
	}
	sq, dm := q1.Shape()[0], q1.Shape()[1]
	sk := k1.Shape()[0]
	if !q2.Shape().Equal(q1.Shape()) || !k2.Shape().Equal(k1.Shape()) {
		return nil, fmt.Errorf("cpu: mha_select source shapes differ")
	}
	if sel.Shape()[0] != sq || sel.Shape()[1] != sk {
		return nil, fmt.Errorf("cpu: mha_select sel must be [%d,%d], got %v", sq, sk, sel.Shape())
	}
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("cpu: mha_select dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("cpu: mha_select heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	rep := heads / kvHeads
	if k1.Shape()[1] != kvHeads*dk || !v.Shape().Equal(k1.Shape()) {
		return nil, fmt.Errorf("cpu: mha_select K/V must be [sk,%d]", kvHeads*dk)
	}
	scale := pa.Scale / math.Sqrt(float64(dk))

	out := tensor.NewOn(ctx.Device(), q1.Dtype(), tensor.Shape{sq, dm})

	// F32 fast path: close the dtype-gap (cpu-registered F64-only, so F32 fell to
	// backend/ref's serial select scan). ref's F32 input deterministically takes its
	// devirtualised path (f64Data widens F32→F64; the generic AtF64 loop only runs for
	// dtypes f64Data rejects), computing every score/softmax/·V in F64 and narrowing only on
	// store. We reproduce that F64 arithmetic in the SAME ascending-j order, widening each F32
	// read per element and narrowing on store → BYTE-IDENTICAL, and skip ref's up-front F64
	// materialization of all 6 inputs. (head,query-row) pairs are independent.
	if q1.Dtype() == tensor.F32 {
		qs1 := q1.Contiguous().Storage().F32()
		ks1 := k1.Contiguous().Storage().F32()
		qs2 := q2.Contiguous().Storage().F32()
		ks2 := k2.Contiguous().Storage().F32()
		vs := v.Contiguous().Storage().F32()
		sels := sel.Contiguous().Storage().F32()
		os := out.Storage().F32()
		// Perf build: route both score sources (Q1·K1ᵀ, Q2·K2ᵀ) and the P·V through the f32-native
		// SIMD gemm + a select/mask vexp softmax, as unmasked MHA does — the scalar 8-jam path below
		// never used it, leaving selective attention ~10× slower than it need be. Small/decode shapes
		// keep the byte-exact scalar path. Rides the f32 5e-5 tolerance (ADR-0021); the parity test
		// relaxes accordingly on F32NativeKernelsEnabled.
		if f32NativeKernels && sq >= mhaGemmMinSeq {
			geo := mhaGeo{sq: sq, sk: sk, dm: dm, dk: dk, kvDM: kvHeads * dk, heads: heads, rep: rep, scale: scale}
			mhaSelectFwdGemmF32(qs1, ks1, qs2, ks2, vs, sels, os, geo)
			return []*tensor.Tensor{out}, nil
		}
		kdm := kvHeads * dk
		parallelWork(heads*sq, sk*dk, func(lo, hi int) {
			//perfscan:ignore PS6008 per-worker scratch (once per goroutine chunk); correct design
			row := make([]float64, sk)
			//perfscan:ignore PS6008 per-worker scratch (once per goroutine chunk); correct design
			obuf := make([]float64, dk)
			for idx := lo; idx < hi; idx++ {
				h := idx / sq
				i := idx % sq
				qOff := h * dk
				kvOff := (h / rep) * dk
				mhaSelectRow(os, qs1, ks1, qs2, ks2, vs, sels,
					row, obuf, i, dm, qOff, kvOff, kdm, dk, sk, scale)
			}
		})
		return []*tensor.Tensor{out}, nil
	}

	qs1 := q1.Contiguous().Storage().F64()
	ks1 := k1.Contiguous().Storage().F64()
	qs2 := q2.Contiguous().Storage().F64()
	ks2 := k2.Contiguous().Storage().F64()
	vs := v.Contiguous().Storage().F64()
	sels := sel.Contiguous().Storage().F64()
	os := out.Storage().F64()
	kdm := kvHeads * dk

	parallelWork(heads*sq, sk*dk, func(lo, hi int) {
		//perfscan:ignore PS6008 per-worker scratch (once per goroutine chunk); correct design
		row := make([]float64, sk)
		//perfscan:ignore PS6008 per-worker scratch (once per goroutine chunk); correct design
		obuf := make([]float64, dk)
		for idx := lo; idx < hi; idx++ {
			h := idx / sq
			i := idx % sq
			qOff := h * dk
			kvOff := (h / rep) * dk
			mhaSelectRow(os, qs1, ks1, qs2, ks2, vs, sels,
				row, obuf, i, dm, qOff, kvOff, kdm, dk, sk, scale)
		}
	})
	return []*tensor.Tensor{out}, nil
}

// mhaSelectScoreOne is the score of one key for one query row, taking the selector's branch. It
// exists so the jammed loop below can fall back to the original step for a group of four that is
// not uniformly finite, without duplicating the branch.
func mhaSelectScoreOne[T float32 | float64](sv float64, qa, qb, ks1, ks2 []T,
	j, kdm, kvOff, dk int, scale float64) float64 {
	if math.IsInf(sv, -1) {
		return math.Inf(-1)
	}
	qrow, ksrc := qa, ks1
	if sv != 0 {
		qrow, ksrc = qb, ks2
	}
	krow := ksrc[j*kdm+kvOff : j*kdm+kvOff+dk : j*kdm+kvOff+dk]
	var s float64
	//perfscan:ignore PS3010 rare masked-key fallback dot; jammed path handles common case
	for d, qv := range qrow {
		s += float64(qv) * float64(krow[d])
	}
	return s * scale
}

// mhaSelectRow computes one (head, query row) of the selective-attention output. The two dtype
// arms of the kernel were line-for-line duplicates and now share this one body, which is also
// what keeps the jam below from reaching only one of them.
//
// FOUR KEYS PER PASS, TWICE. A profile of the 1024x1024x64x16 F32 cell put 46% of the kernel on
// the score dot and 26% on the weighted sum, and both re-read something that does not vary with
// the key: the score dot re-streams the query row and runs ONE accumulator chain per key, and the
// weighted sum re-loads and re-stores the shared output accumulator once per key. Four keys per
// pass give the first four independent chains over one traversal of the query row, and let the
// second hold obuf[d] in a register across its four additions.
//
// BIT-IDENTICAL: every score still sums over the same ascending d into its own accumulator and is
// scaled once, row and the running maximum are still written in ascending j, and obuf[d] takes
// the same four additions in the same ascending j — only held in a register between them.
func mhaSelectRow[T float32 | float64](os, qs1, ks1, qs2, ks2, vs, sels []T,
	row, obuf []float64, i, dm, qOff, kvOff, kdm, dk, sk int, scale float64) {
	srow := sels[i*sk : i*sk+sk : i*sk+sk]
	qBase := i*dm + qOff
	qa := qs1[qBase : qBase+dk : qBase+dk]
	qb := qs2[qBase : qBase+dk : qBase+dk]
	m := math.Inf(-1)
	j := 0
	//perfscan:ignore PS3066,PS3076 already 4-key hand-jammed hot loop; at-ceiling | already 4-way hand-unrolled score loop; at-ceiling
	for ; j+3 < sk; j += 4 {
		v0, v1 := float64(srow[j]), float64(srow[j+1])
		v2, v3 := float64(srow[j+2]), float64(srow[j+3])
		// A MASKED KEY IN THE GROUP TAKES THE ORIGINAL STEP. Masking is per key, so a group of
		// four can straddle the boundary; running those four one at a time keeps the jam free of
		// a branch it would otherwise carry per element.
		if math.IsInf(v0, -1) || math.IsInf(v1, -1) ||
			math.IsInf(v2, -1) || math.IsInf(v3, -1) {
			for o, sv := range [4]float64{v0, v1, v2, v3} {
				s := mhaSelectScoreOne(sv, qa, qb, ks1, ks2, j+o, kdm, kvOff, dk, scale)
				row[j+o] = s
				if s > m {
					m = s
				}
			}
			continue
		}
		q0, kk0 := qa, ks1
		if v0 != 0 {
			q0, kk0 = qb, ks2
		}
		q1, kk1 := qa, ks1
		if v1 != 0 {
			q1, kk1 = qb, ks2
		}
		q2, kk2 := qa, ks1
		if v2 != 0 {
			q2, kk2 = qb, ks2
		}
		q3, kk3 := qa, ks1
		if v3 != 0 {
			q3, kk3 = qb, ks2
		}
		o0 := j*kdm + kvOff
		o1, o2, o3 := o0+kdm, o0+2*kdm, o0+3*kdm
		k0 := kk0[o0 : o0+dk : o0+dk]
		k1 := kk1[o1 : o1+dk : o1+dk]
		k2 := kk2[o2 : o2+dk : o2+dk]
		k3 := kk3[o3 : o3+dk : o3+dk]
		var a0, a1, a2, a3 float64
		//perfscan:ignore PS3010 inner 4-accumulator score dot; already optimal
		for d := 0; d < dk; d++ {
			a0 += float64(q0[d]) * float64(k0[d])
			a1 += float64(q1[d]) * float64(k1[d])
			a2 += float64(q2[d]) * float64(k2[d])
			a3 += float64(q3[d]) * float64(k3[d])
		}
		for o, av := range [4]float64{a0, a1, a2, a3} {
			s := av * scale
			row[j+o] = s
			if s > m {
				m = s
			}
		}
	}
	for ; j < sk; j++ {
		s := mhaSelectScoreOne(float64(srow[j]), qa, qb, ks1, ks2, j, kdm, kvOff, dk, scale)
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
				//perfscan:ignore PS3017 weighted-sum remainder tail, <4 iters low trip-count
				obuf[d] += w * float64(vv)
			}
		}
	}
	for d := 0; d < dk; d++ {
		os[qBase+d] = T(obuf[d])
	}
}

func init() {
	std.add(backend.OpMHASelect, tensor.F64, mhaSelectKernelCPU)
	std.add(backend.OpMHASelect, tensor.F32, mhaSelectKernelCPU)
}
