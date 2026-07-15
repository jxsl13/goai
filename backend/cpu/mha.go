package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Optimized fused attention (§T599, profile-driven: the GPT cpu training step
// spent 45% of its samples in ref's interface-arithmetic MHA backward and 13%
// in its forward). Same semantics as backend/ref's kernels — GQA/MQA, causal,
// KV-cache offset (forward), ALiBi, sliding window, YaRN-folded scale — on
// contiguous typed slices. Forward parallelizes over (head, query row);
// backward over KV-HEAD GROUPS, whose dQ/dK/dV regions are disjoint (the rep
// query heads of a group share exactly that group's K/V columns), so there is
// no scatter contention and no atomics. All accumulation in f64 (§V10); dK/dV
// accumulate in f64 scratch and store once — more accurate than ref's
// through-the-tensor sums, inside the ulps parity budget (§T596's V9 note).
func mhaKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("cpu: mha wants (Q,K,V), got %d inputs", len(in))
	}
	q, k, v := in[0], in[1], in[2]
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("cpu: mha needs rank-2 [seq,dmodel]")
	}
	sq, dm := q.Shape()[0], q.Shape()[1]
	sk := k.Shape()[0]
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("cpu: mha dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("cpu: mha heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	rep := heads / kvHeads
	kvDM := kvHeads * dk
	if k.Shape()[1] != kvDM || !v.Shape().Equal(k.Shape()) {
		return nil, fmt.Errorf("cpu: mha K/V must be [sk,%d] (kv_heads·dk), got %v/%v", kvDM, k.Shape(), v.Shape())
	}
	if sq > sk {
		return nil, fmt.Errorf("cpu: mha query len %d exceeds key len %d", sq, sk)
	}
	causal := pa.Causal
	scale := pa.Scale / math.Sqrt(float64(dk))
	off := sk - sq
	var slopes []float64
	if pa.ALiBi {
		slopes = backend.ALiBiSlopes(heads)
	}
	window := pa.Window

	geo := mhaGeo{sq: sq, sk: sk, dm: dm, dk: dk, kvDM: kvDM, heads: heads, rep: rep,
		off: off, window: window, causal: causal, scale: scale, slopes: slopes}
	out := tensor.NewOn(ctx.Device(), q.Dtype(), tensor.Shape{sq, dm})
	qc, kc, vc := q.Contiguous(), k.Contiguous(), v.Contiguous()
	// Concrete typed instantiations (§T602): the previous per-element closure
	// reads were 11% of the training profile; generics devirtualize them.
	if q.Dtype() == tensor.F64 {
		mhaFwd(qc.Storage().F64(), kc.Storage().F64(), vc.Storage().F64(), out.Storage().F64(), geo)
	} else {
		mhaFwd(qc.Storage().F32(), kc.Storage().F32(), vc.Storage().F32(), out.Storage().F32(), geo)
	}
	return []*tensor.Tensor{out}, nil
}

// mhaGeo bundles the attention geometry and attributes shared by the typed cores.
type mhaGeo struct {
	sq, sk, dm, dk, kvDM, heads, rep, off, window int
	causal                                        bool
	scale                                         float64
	slopes                                        []float64
}

// dot4 is Σ a[d]·b[d] in f64 with FOUR independent accumulators — the
// single-accumulator form serializes on the FP-add latency (one madd per
// ~4 cycles); four chains keep the FPU pipeline full. Reassociating the dot
// perturbs the pre-softmax score by a few ulps: well inside the f32 parity
// budget (5e-5), but MEASURED to breach the f64 one (~4 ulps) through the
// backward's dS=a·(dA−dot) cancellation — mha-bwd/parallel/f64/dQ differed by
// 1.2e-15 relative. So only the f32 instantiations call this; f64 keeps ref's
// sequential dot.
func dot4[T float32 | float64](a []float64, b []T) float64 {
	var s0, s1, s2, s3 float64
	d := 0
	for ; d+3 < len(a) && d+3 < len(b); d += 4 {
		s0 += a[d] * float64(b[d])
		s1 += a[d+1] * float64(b[d+1])
		s2 += a[d+2] * float64(b[d+2])
		s3 += a[d+3] * float64(b[d+3])
	}
	for ; d < len(a) && d < len(b); d++ {
		s0 += a[d] * float64(b[d])
	}
	return (s0 + s1) + (s2 + s3)
}

// dot4T is dot4 with both operands in their storage type. The pre-widened
// form (dot4 + widen) was A/B-tested in all three attention kernels: it won
// ~28% in the forward (whose k-loop is load-bound on the score pass) but
// consistently REGRESSED ~5-6% in the backward and FlashAttn (p≤0.017, 7
// interleaved rounds), so those keep the two-convert form.
func dot4T[T float32 | float64](a, b []T) float64 {
	var s0, s1, s2, s3 float64
	d := 0
	for ; d+3 < len(a) && d+3 < len(b); d += 4 {
		s0 += float64(a[d]) * float64(b[d])
		s1 += float64(a[d+1]) * float64(b[d+1])
		s2 += float64(a[d+2]) * float64(b[d+2])
		s3 += float64(a[d+3]) * float64(b[d+3])
	}
	for ; d < len(a) && d < len(b); d++ {
		s0 += float64(a[d]) * float64(b[d])
	}
	return (s0 + s1) + (s2 + s3)
}

// widen copies float64(src[d]) into dst — the query row of the current
// (head, i) is reused across every key j, so its f32→f64 conversions are
// hoisted out of the key loop (exact: conversion is value-preserving, so
// results are bit-identical; only worth it for f32 inputs).
func widen[T float32 | float64](dst []float64, src []T) []float64 {
	dst = dst[:len(src)]
	for d, v := range src {
		dst[d] = float64(v)
	}
	return dst
}

// bounds returns the [jmin, jmax) key range for query row i.
func (g mhaGeo) bounds(i int) (int, int) {
	jmax := g.sk
	if g.causal {
		jmax = g.off + i + 1
	}
	jmin := 0
	if g.window > 0 {
		if lo := g.off + i - g.window + 1; lo > 0 {
			jmin = lo
		}
	}
	return jmin, jmax
}

// mhaFwd is the typed attention forward over raw slices (f64 accumulation, §V10).
//
// Two structural changes vs the straight ref transcription, both value-preserving:
//   - the softmax weight row[j]/sum is computed ONCE per key (ref divides inside
//     the d-loop — dk redundant divisions per key, and FP divide doesn't pipeline);
//   - the output accumulation is interchanged to key-outer/d-inner over acc[dk],
//     so V is read along its contiguous rows instead of at stride kvDM. For every
//     output element the per-key contributions still add in ascending-j order with
//     identical operands, so the result is unchanged (§V9 parity intact).
func mhaFwd[T float32 | float64](q, k, v, out []T, g mhaGeo) {
	_, isF32 := any(q).([]float32) // dot4 reassociation fits the f32 budget only
	parallelWork(g.heads*g.sq, g.sk*g.dk, func(lo, hi int) {
		row := make([]float64, g.sk)
		acc := make([]float64, g.dk)
		qf := make([]float64, g.dk)
		dk := g.dk
		for t := lo; t < hi; t++ {
			h, i := t/g.sq, t%g.sq
			qOff := h * dk
			kvOff := (h / g.rep) * dk
			jmin, jmax := g.bounds(i)
			qBase := i*g.dm + qOff
			qr := q[qBase : qBase+dk : qBase+dk]
			m := math.Inf(-1)
			if isF32 {
				qw := widen(qf, qr)
				for j := jmin; j < jmax; j++ {
					kBase := j*g.kvDM + kvOff
					s := dot4(qw, k[kBase:kBase+dk:kBase+dk]) * g.scale
					if g.slopes != nil {
						s += g.slopes[h] * float64(j-(g.off+i))
					}
					row[j] = s
					if s > m {
						m = s
					}
				}
			} else {
				for j := jmin; j < jmax; j++ {
					kBase := j*g.kvDM + kvOff
					kr := k[kBase : kBase+dk : kBase+dk]
					var s float64
					for d, qv := range qr {
						s += float64(qv) * float64(kr[d])
					}
					s *= g.scale
					if g.slopes != nil {
						s += g.slopes[h] * float64(j-(g.off+i))
					}
					row[j] = s
					if s > m {
						m = s
					}
				}
			}
			var sum float64
			for j := jmin; j < jmax; j++ {
				row[j] = math.Exp(row[j] - m)
				sum += row[j]
			}
			ac := acc[:dk]
			for d := range ac {
				ac[d] = 0
			}
			// 2-way unrolled key loop: each ac[d] still adds its j then j+1
			// contribution as two separate roundings in ascending-j order —
			// identical values, half the trips over ac.
			j := jmin
			for ; j+1 < jmax; j += 2 {
				p0 := row[j] / sum
				p1 := row[j+1] / sum
				v0Base := j*g.kvDM + kvOff
				v1Base := (j+1)*g.kvDM + kvOff
				v0 := v[v0Base : v0Base+dk : v0Base+dk]
				v1 := v[v1Base : v1Base+dk : v1Base+dk]
				for d, vv := range v0 {
					s := ac[d] + p0*float64(vv)
					ac[d] = s + p1*float64(v1[d])
				}
			}
			for ; j < jmax; j++ {
				p := row[j] / sum
				vBase := j*g.kvDM + kvOff
				vr := v[vBase : vBase+dk : vBase+dk]
				for d, vv := range vr {
					ac[d] += p * float64(vv)
				}
			}
			or := out[qBase : qBase+dk : qBase+dk]
			for d := range or {
				or[d] = T(ac[d])
			}
		}
	})
}

// mhaBackwardKernelCPU: see the package comment above. Training has sq==sk.
func mhaBackwardKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 4 {
		return nil, fmt.Errorf("cpu: mha-backward wants (Q,K,V,dO), got %d inputs", len(in))
	}
	q, k, v, g := in[0], in[1], in[2], in[3]
	seq, dm := q.Shape()[0], q.Shape()[1]
	if k.Shape()[0] != seq {
		return nil, fmt.Errorf("cpu: mha-backward needs equal Q/K length (got %d/%d); KV-cache is inference-only", seq, k.Shape()[0])
	}
	pa, _ := attrs.(backend.AttnAttrs)
	pa = pa.WithDefaults()
	heads := pa.Heads
	if heads <= 0 || dm%heads != 0 {
		return nil, fmt.Errorf("cpu: mha-backward dmodel %d not divisible by heads %d", dm, heads)
	}
	dk := dm / heads
	kvHeads := pa.KVHeads
	if kvHeads <= 0 || heads%kvHeads != 0 {
		return nil, fmt.Errorf("cpu: mha-backward heads %d not divisible by kv_heads %d", heads, kvHeads)
	}
	rep := heads / kvHeads
	kvDM := kvHeads * dk
	causal := pa.Causal
	scale := pa.Scale / math.Sqrt(float64(dk))
	var slopes []float64
	if pa.ALiBi {
		slopes = backend.ALiBiSlopes(heads)
	}
	window := pa.Window

	dQ := tensor.NewOn(ctx.Device(), q.Dtype(), q.Shape())
	dK := tensor.NewOn(ctx.Device(), k.Dtype(), k.Shape())
	dV := tensor.NewOn(ctx.Device(), v.Dtype(), v.Shape())
	geo := mhaGeo{sq: seq, sk: seq, dm: dm, dk: dk, kvDM: kvDM, heads: heads, rep: rep,
		off: 0, window: window, causal: causal, scale: scale, slopes: slopes}
	qc, kc, vc, gc := q.Contiguous(), k.Contiguous(), v.Contiguous(), g.Contiguous()
	if q.Dtype() == tensor.F64 {
		mhaBwd(qc.Storage().F64(), kc.Storage().F64(), vc.Storage().F64(), gc.Storage().F64(),
			dQ.Storage().F64(), dK.Storage().F64(), dV.Storage().F64(), geo)
	} else {
		mhaBwd(qc.Storage().F32(), kc.Storage().F32(), vc.Storage().F32(), gc.Storage().F32(),
			dQ.Storage().F32(), dK.Storage().F32(), dV.Storage().F32(), geo)
	}
	return []*tensor.Tensor{dQ, dK, dV}, nil
}

// mhaBwdScratch is the per-worker scratch of one backward head pass.
type mhaBwdScratch struct {
	a, dA, dqRow []float64
}

// mhaBwdHead runs the backward for query head h: writes head h's dQ columns
// and ACCUMULATES (+=, ascending i) its dK/dV contributions into the seq×dk
// f64 buffers dkAcc/dvAcc. f64 accumulation throughout (§V10).
func mhaBwdHead[T float32 | float64](q, k, v, g, dQ []T, dkAcc, dvAcc []float64, sc *mhaBwdScratch, geo mhaGeo, h int, isF32 bool) {
	seq, dk, dm, kvDM := geo.sq, geo.dk, geo.dm, geo.kvDM
	causal, window, scale, slopes := geo.causal, geo.window, geo.scale, geo.slopes
	kvOff := (h / geo.rep) * dk
	qOff := h * dk
	a, dA, dqRow := sc.a, sc.dA, sc.dqRow
	for i := range seq {
		jmax := seq
		if causal {
			jmax = i + 1
		}
		jmin := 0
		if window > 0 {
			if lo := i - window + 1; lo > 0 {
				jmin = lo
			}
		}
		qBase := i*dm + qOff
		qr := q[qBase : qBase+dk : qBase+dk]
		gr := g[qBase : qBase+dk : qBase+dk]
		m := math.Inf(-1)
		for j := jmin; j < jmax; j++ {
			kBase := j*kvDM + kvOff
			kr := k[kBase : kBase+dk : kBase+dk]
			var s float64
			if isF32 {
				s = dot4T(qr, kr)
			} else {
				for d, qv := range qr {
					s += float64(qv) * float64(kr[d])
				}
			}
			s *= scale
			if slopes != nil {
				s += slopes[h] * float64(j-i)
			}
			a[j] = s
			if s > m {
				m = s
			}
		}
		var sum float64
		for j := jmin; j < jmax; j++ {
			a[j] = math.Exp(a[j] - m)
			sum += a[j]
		}
		for j := jmin; j < jmax; j++ {
			a[j] /= sum
		}
		var dot float64
		for j := jmin; j < jmax; j++ {
			kvBase := j*kvDM + kvOff
			vr := v[kvBase : kvBase+dk : kvBase+dk]
			dvr := dvAcc[j*dk : j*dk+dk : j*dk+dk]
			aj := a[j]
			var dav float64
			for d, gv := range gr {
				gid := float64(gv)
				dvr[d] += aj * gid
				dav += gid * float64(vr[d])
			}
			dA[j] = dav
			dot += dav * aj
		}
		for d := range dqRow {
			dqRow[d] = 0
		}
		dq := dqRow[:dk]
		for j := jmin; j < jmax; j++ {
			dS := scale * a[j] * (dA[j] - dot)
			kvBase := j*kvDM + kvOff
			kr := k[kvBase : kvBase+dk : kvBase+dk]
			dkr := dkAcc[j*dk : j*dk+dk : j*dk+dk]
			for d, kvv := range kr {
				dq[d] += dS * float64(kvv)
				dkr[d] += dS * float64(qr[d])
			}
		}
		or := dQ[qBase : qBase+dk : qBase+dk]
		for d := range or {
			or[d] = T(dqRow[d])
		}
	}
}

// mhaBwd is the typed attention backward over raw slices (§T602), f64 scratch
// accumulation (§V10). Parallelism (the §T599 kvHeads-task scheme starved a
// 16-thread machine at kv_heads=4 — measured 4.3× slower than the forward):
//
//   - ONE TASK PER QUERY HEAD, each accumulating into its own seq×dk dK/dV
//     partial (pooled, zeroed), then a second parallel pass reduces each KV
//     group's rep partials in ascending-head order. For rep==1 the reduction
//     is a plain copy, so results are IDENTICAL to the old per-group scheme.
//     For rep>1 the per-head partials regroup the (head, i) fold — inside the
//     f32 parity budget, NOT the f64 one, so:
//   - f64 GQA (rep>1) keeps the exact old order: one task per KV group whose
//     rep heads share the group's accumulators sequentially.
func mhaBwd[T float32 | float64](q, k, v, g, dQ, dK, dV []T, geo mhaGeo) {
	seq, dk, kvDM, rep := geo.sq, geo.dk, geo.kvDM, geo.rep
	heads := geo.heads
	kvHeads := heads / rep
	_, isF32 := any(q).([]float32) // dot4 + partial regrouping fit the f32 budget only

	if rep > 1 && !isF32 {
		// f64 GQA: per-group tasks, rep heads accumulate sequentially — ref's order.
		parallelWork(kvHeads, rep*seq*seq*dk, func(glo, ghi int) {
			sc := &mhaBwdScratch{a: make([]float64, seq), dA: make([]float64, seq), dqRow: make([]float64, dk)}
			dkAcc := make([]float64, seq*dk)
			dvAcc := make([]float64, seq*dk)
			for kv := glo; kv < ghi; kv++ {
				kvOff := kv * dk
				for x := range dkAcc {
					dkAcc[x] = 0
					dvAcc[x] = 0
				}
				for hr := range rep {
					mhaBwdHead(q, k, v, g, dQ, dkAcc, dvAcc, sc, geo, kv*rep+hr, false)
				}
				for j := range seq {
					dkr := dkAcc[j*dk : j*dk+dk : j*dk+dk]
					dvr := dvAcc[j*dk : j*dk+dk : j*dk+dk]
					ko := dK[j*kvDM+kvOff : j*kvDM+kvOff+dk : j*kvDM+kvOff+dk]
					vo := dV[j*kvDM+kvOff : j*kvDM+kvOff+dk : j*kvDM+kvOff+dk]
					for d := range dkr {
						ko[d] = T(dkr[d])
						vo[d] = T(dvr[d])
					}
				}
			}
		})
		return
	}

	part := heads * seq * dk
	partK := getF64(2 * part) // zeroed; [h*seq*dk] dK partials, then dV partials
	defer putF64(partK)
	pk, pv := (*partK)[:part], (*partK)[part:]
	parallelWork(heads, seq*seq*dk, func(hlo, hhi int) {
		sc := &mhaBwdScratch{a: make([]float64, seq), dA: make([]float64, seq), dqRow: make([]float64, dk)}
		for h := hlo; h < hhi; h++ {
			mhaBwdHead(q, k, v, g, dQ, pk[h*seq*dk:(h+1)*seq*dk], pv[h*seq*dk:(h+1)*seq*dk], sc, geo, h, isF32)
		}
	})
	// Reduce each KV group's rep head-partials (ascending head — the old
	// accumulation's head order) into dK/dV. Row-parallel; disjoint outputs.
	parallelWork(kvHeads*seq, rep*dk, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			kv, j := t/seq, t%seq
			kvOff := kv * dk
			base := j*kvDM + kvOff
			ko := dK[base : base+dk : base+dk]
			vo := dV[base : base+dk : base+dk]
			h0 := kv * rep
			k0 := pk[h0*seq*dk+j*dk : h0*seq*dk+j*dk+dk : h0*seq*dk+j*dk+dk]
			v0 := pv[h0*seq*dk+j*dk : h0*seq*dk+j*dk+dk : h0*seq*dk+j*dk+dk]
			if rep == 1 {
				for d := range ko {
					ko[d] = T(k0[d])
					vo[d] = T(v0[d])
				}
				continue
			}
			for d := range ko {
				sk, sv := k0[d], v0[d]
				for hr := 1; hr < rep; hr++ {
					off := (h0+hr)*seq*dk + j*dk + d
					sk += pk[off]
					sv += pv[off]
				}
				ko[d] = T(sk)
				vo[d] = T(sv)
			}
		}
	})
}

func init() {
	std.add(backend.OpMHA, tensor.F32, mhaKernelCPU)
	std.add(backend.OpMHA, tensor.F64, mhaKernelCPU)
	std.add(backend.OpMHABackward, tensor.F32, mhaBackwardKernelCPU)
	std.add(backend.OpMHABackward, tensor.F64, mhaBackwardKernelCPU)
}
