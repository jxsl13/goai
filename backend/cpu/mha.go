package cpu

import (
	"fmt"
	"math"
	"runtime"
	"sync/atomic"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/simd"
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
	switch {
	case q.Dtype() == tensor.F64:
		mhaFwd(qc.Storage().F64(), kc.Storage().F64(), vc.Storage().F64(), out.Storage().F64(), geo)
	case f32NativeKernels && sq >= mhaGemmMinSeq:
		// Perf builds: route the two per-head matmuls (QKᵀ, A·V) through the
		// f32-native SIMD gemmF32 (ADR-0021/0026 tolerance, not ulps-exact).
		mhaFwdGemmF32(qc.Storage().F32(), kc.Storage().F32(), vc.Storage().F32(), out.Storage().F32(), geo)
	default:
		mhaFwd(qc.Storage().F32(), kc.Storage().F32(), vc.Storage().F32(), out.Storage().F32(), geo)
	}
	return []*tensor.Tensor{out}, nil
}

// mhaGemmMinSeq gates the gemm-routed attention path: below this query length
// (decode/GEMV shapes) the Kᵀ/V packing traffic is the same order as the
// matmuls themselves, so the scalar row kernel stays faster.
const mhaGemmMinSeq = 16

// mhaFwdGemmF32 is the gemm-routed f32 attention forward of the perf builds
// (f32NativeKernels). Structure — TWO parallelWork barriers total (the first
// per-head-sequential cut of this path spent >80% of its profile in pool
// wake/park churn across ~50 barriers):
//
//  1. one pass packing every KV head's Kᵀ[dk,sk] and V[sk,dk];
//  2. one pass over the heads·sq query rows, each worker running the WHOLE
//     band-local pipeline on its contiguous row band: pack Q rows → S = Q·Kᵀ
//     (gemmF32Rows, the f32-native SIMD band kernel) → stable softmax
//     (scale/ALiBi/causal/window in f64, masked columns zeroed) → O = P·V
//     (gemmF32Rows) → scatter into out. Rows are independent given the packs,
//     so there is no cross-band dependency.
//
// The full [sq,sk] score rectangle is computed even under causal masking — the
// SIMD gemm is far faster than a scalar triangle, masked entries are ignored
// (scores) or zeroed (weights), and uniform per-row work keeps the static
// chunking balanced.
func mhaFwdGemmF32(q, k, v, out []float32, g mhaGeo) {
	sq, sk, dk, kvDM := g.sq, g.sk, g.dk, g.kvDM
	kvHeads := g.heads / g.rep
	ktP, vhP := getF32Raw(kvHeads*dk*sk), getF32Raw(kvHeads*sk*dk) // fully overwritten
	defer putF32(ktP)
	defer putF32(vhP)
	kt, vh := *ktP, *vhP
	parallelWork(kvHeads*sk, 2*dk, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			kv, j := t/sk, t%sk
			base := j*kvDM + kv*dk
			kr := k[base : base+dk : base+dk]
			kth := kt[kv*dk*sk:]
			for d, kvv := range kr {
				kth[d*sk+j] = kvv
			}
			copy(vh[kv*sk*dk+j*dk:kv*sk*dk+(j+1)*dk], v[base:base+dk])
		}
	})
	// Dynamic band scheduling (the NEON gemm entry's scheme): static equal
	// chunks leave the wall clock to the slowest worker — the causal work
	// (score-gemm columns and softmax exps) grows with the row index, and the
	// M-series E-cores lag the P-cores. Each worker pulls (head, row-band)
	// tasks off an atomic counter in DESCENDING-cost order (high row bands
	// first), so the cheap bands fill the tail instead of straggling it.
	rowWork := dk*sk + 4*sk + sk*dk
	bands := (sq + mhaFwdBandRows - 1) / mhaFwdBandRows
	nTasks := g.heads * bands
	workers := max(runtime.GOMAXPROCS(0), 1)
	var next atomic.Int64
	parallelWork(workers, (g.heads*sq*rowWork+workers-1)/workers, func(_, _ int) {
		for {
			t := int(next.Add(1)) - 1
			if t >= nTasks {
				return
			}
			h, b := t%g.heads, bands-1-t/g.heads
			i0 := b * mhaFwdBandRows
			iN := min(mhaFwdBandRows, sq-i0)
			mhaFwdGemmBand(q, kt, vh, out, g, h, i0, iN)
		}
	})
}

// mhaFwdBandRows is the row-band grain of the dynamic forward scheduler:
// small enough for load balance across the causal exp gradient, big enough
// that the two band gemms keep whole 4-row tiles busy.
const mhaFwdBandRows = 30

// mhaFwdGemmBand runs the forward pipeline for rows [i0,i0+iN) of head h.
func mhaFwdGemmBand(q, kt, vh, out []float32, g mhaGeo, h, i0, iN int) {
	sk, dk, dm := g.sk, g.dk, g.dm
	kv := h / g.rep
	kth := kt[kv*dk*sk : (kv+1)*dk*sk]
	vhh := vh[kv*sk*dk : (kv+1)*sk*dk]
	qOff := h * dk
	qbP, sbP, obP := getF32Raw(iN*dk), getF32Raw(iN*sk), getF32Raw(iN*dk)
	defer putF32(qbP)
	defer putF32(sbP)
	defer putF32(obP)
	qb, sb, ob := *qbP, *sbP, *obP
	for r := range iN {
		copy(qb[r*dk:(r+1)*dk], q[(i0+r)*dm+qOff:(i0+r)*dm+qOff+dk])
	}
	// Causal: no row of this band attends beyond key g.off+i0+iN-1, so the
	// score gemm skips the fully-masked column span (rounded up to whole
	// 16-wide tiles). The skipped columns hold stale scratch — the softmax
	// pass never reads past each row's jmax and zeroes them for the P·V gemm.
	jHi := sk
	if g.causal {
		jHi = min((g.off+i0+iN+15)&^15, sk)
	}
	gemmF32RowsCols(qb, kth, sb, 0, iN, dk, sk, 0, jHi)
	mhaSoftmaxBandF32(sb, g, h, i0, iN)
	// P·V. The score gemm above is column-bounded to jHi but the P·V contraction was
	// not — it multiplied the full sk columns even though the softmax zeroed every
	// weight in [jHi, sk) (every band row's jmax ≤ jHi). P·V is the larger of the two
	// band gemms, so for causal shapes ~1/3 of all forward gemm flops were spent on
	// provably-zero weights. Bound the contraction to jHi: compact each row's live
	// [0,jHi) weights to stride jHi (in place — jHi ≤ sk so the forward row copy never
	// clobbers an unread row) and read only the first jHi rows of V. The dropped
	// columns held exactly 0.0, so 0·v=0 adds nothing — the retained terms keep their
	// order, so the result matches the full-width gemm within the f32 MHA parity budget.
	if jHi < sk {
		for r := 1; r < iN; r++ {
			copy(sb[r*jHi:r*jHi+jHi], sb[r*sk:r*sk+jHi])
		}
		gemmF32Rows(sb, vhh, ob, 0, iN, jHi, dk)
	} else {
		gemmF32Rows(sb, vhh, ob, 0, iN, sk, dk)
	}
	for r := range iN {
		copy(out[(i0+r)*dm+qOff:(i0+r)*dm+qOff+dk], ob[r*dk:(r+1)*dk])
	}
}

// mhaSoftmaxBandF32 turns rows [i0,i0+iN) of head h's raw QKᵀ products (sb,
// band-local [iN,sk]) into softmax weights in place: scale (+ALiBi) and the
// stable max-shift exp/sum run in f64, the normalized weight is narrowed to
// f32 once, and every masked column is zeroed so a following full-rectangle
// P·V gemm adds exactly 0 for it.
func mhaSoftmaxBandF32(sb []float32, g mhaGeo, h, i0, iN int) {
	if vexpF32Fast {
		// SIMD perf build: f32-native band softmax with a vectorized exp — 4-wide
		// NEON on arm64, 8-wide AVX2 on amd64 (vexp.go / vexp_amd64.go). After the
		// T657 gemm routing, scalar math.Exp was ~35% of the forward. Compile-time
		// const: the plain (no-simd) build keeps this scalar body bit-for-bit.
		mhaSoftmaxBandVexpF32(sb, g, h, i0, iN)
		return
	}
	sk := g.sk
	rowP := getF64(sk)
	defer putF64(rowP)
	row := *rowP
	for r := range iN {
		i := i0 + r
		jmin, jmax := g.bounds(i)
		sr := sb[r*sk : (r+1)*sk : (r+1)*sk]
		m := math.Inf(-1)
		for j := jmin; j < jmax; j++ {
			x := float64(sr[j]) * g.scale
			if g.slopes != nil {
				x += g.slopes[h] * float64(j-(g.off+i))
			}
			row[j] = x
			if x > m {
				m = x
			}
		}
		var sum float64
		for j := jmin; j < jmax; j++ {
			e := math.Exp(row[j] - m)
			row[j] = e
			sum += e
		}
		inv := 1 / sum
		clear(sr[:jmin])
		for j := jmin; j < jmax; j++ {
			sr[j] = float32(row[j] * inv)
		}
		clear(sr[jmax:])
	}
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
func dot4T[T ~float32 | ~float64](a, b []T) float64 {
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
			if isF32 {
				// f32 rides the 5e-5 parity budget: vectorized exp+sum matches
				// the vexp band softmax the gemm path already uses (§V9). F64
				// stays on scalar math.Exp to keep its ulp budget.
				sum = simd.ExpSumF64(row[jmin:jmax], row[jmin:jmax], m)
			} else {
				for j := jmin; j < jmax; j++ {
					row[j] = math.Exp(row[j] - m)
					sum += row[j]
				}
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
	switch {
	case q.Dtype() == tensor.F64:
		mhaBwd(qc.Storage().F64(), kc.Storage().F64(), vc.Storage().F64(), gc.Storage().F64(),
			dQ.Storage().F64(), dK.Storage().F64(), dV.Storage().F64(), geo)
	case f32NativeKernels && seq >= mhaGemmMinSeq:
		// Perf builds: route the five per-head matmuls (S, dA, dQ, dV, dK)
		// through the f32-native SIMD band kernel (ADR-0021/0026 tolerance).
		mhaBwdGemmF32(qc.Storage().F32(), kc.Storage().F32(), vc.Storage().F32(), gc.Storage().F32(),
			dQ.Storage().F32(), dK.Storage().F32(), dV.Storage().F32(), geo)
	default:
		mhaBwd(qc.Storage().F32(), kc.Storage().F32(), vc.Storage().F32(), gc.Storage().F32(),
			dQ.Storage().F32(), dK.Storage().F32(), dV.Storage().F32(), geo)
	}
	return []*tensor.Tensor{dQ, dK, dV}, nil
}

// mhaBwdBandRows is the query-row band size of the gemm-routed backward: big
// enough that the five band gemms amortize their transposes and packs, small
// enough that heads·(seq/band) tasks fill every core at the training shapes.
const mhaBwdBandRows = 128

// mhaBwdGemmF32 is the gemm-routed f32 attention backward of the perf builds
// (f32NativeKernels) — the same math as mhaBwdHead but with every matmul on
// the f32-native SIMD band kernel, in THREE parallelWork barriers:
//
//  1. pack each KV head's Kᵀ[dk,sk], Vᵀ[dk,sk] and K[sk,dk];
//  2. one pass over (head, query-row band) tasks. Per band: S = Q·Kᵀ →
//     softmax → P; dA = dO·Vᵀ; dS = scale·P∘(dA − rowdot) (rowdot in f64);
//     dQ_band = dS·K (disjoint dQ rows — written directly); and the band's
//     dK/dV PARTIALS dSᵀ·Q_band / Pᵀ·dO_band into its own [sk,dk] slots
//     (full rectangles; masked columns have P = dS = 0 so they add nothing);
//  3. reduce each KV group's rep·bands partials (ascending head-band order)
//     into dK/dV.
//
// The per-band f32 partials regroup the fold vs ref's ascending-i per-group
// accumulation — inside the f32 tolerance this path is gated to (the f64
// backward keeps the exact-order kernels above).
func mhaBwdGemmF32(q, k, v, g, dQ, dK, dV []float32, geo mhaGeo) {
	seq, dk, kvDM := geo.sq, geo.dk, geo.kvDM
	rep, heads := geo.rep, geo.heads
	kvHeads := heads / rep
	// Packs, per KV head: Kᵀ[dk,seq] | Vᵀ[dk,seq] | K[sk,dk].
	packP := getF32Raw(kvHeads * 3 * dk * seq)
	defer putF32(packP)
	pack := *packP
	kvStride := 3 * dk * seq
	parallelWork(kvHeads*seq, 3*dk, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			kv, j := t/seq, t%seq
			base := j*kvDM + kv*dk
			kr := k[base : base+dk : base+dk]
			vr := v[base : base+dk : base+dk]
			kt := pack[kv*kvStride:]
			vt := pack[kv*kvStride+dk*seq:]
			for d := range dk {
				kt[d*seq+j] = kr[d]
				vt[d*seq+j] = vr[d]
			}
			copy(pack[kv*kvStride+2*dk*seq+j*dk:kv*kvStride+2*dk*seq+(j+1)*dk], kr)
		}
	})

	bands := (seq + mhaBwdBandRows - 1) / mhaBwdBandRows
	part := heads * bands * seq * dk
	partP := getF32Raw(2 * part) // store-semantics gemm outputs: fully written
	defer putF32(partP)
	pk, pv := (*partP)[:part], (*partP)[part:]

	// Dynamic (head, band) scheduling in descending-cost order — same
	// rationale as the forward's.
	nTasks := heads * bands
	workers := max(runtime.GOMAXPROCS(0), 1)
	var nextT atomic.Int64
	parallelWork(workers, (nTasks*mhaBwdBandRows*seq*(5*dk+4)/2+workers-1)/workers, func(_, _ int) {
		for {
			t := int(nextT.Add(1)) - 1
			if t >= nTasks {
				return
			}
			h, b := t%heads, bands-1-t/heads
			i0 := b * mhaBwdBandRows
			iN := min(mhaBwdBandRows, seq-i0)
			slot := (h*bands + b) * seq * dk
			mhaBwdGemmBand(q, g, dQ, pack, pk[slot:slot+seq*dk], pv[slot:slot+seq*dk], geo, h, i0, iN)
		}
	})

	// Reduce each KV group's rep·bands partials (ascending head, then band —
	// a fixed fold order) into dK/dV. Row-parallel; disjoint outputs.
	parallelWork(kvHeads*seq, rep*bands*dk, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			kv, j := t/seq, t%seq
			base := j*kvDM + kv*dk
			ko := dK[base : base+dk : base+dk]
			vo := dV[base : base+dk : base+dk]
			for d := range dk {
				var sk, sv float32
				for hb := kv * rep * bands; hb < (kv+1)*rep*bands; hb++ {
					off := hb*seq*dk + j*dk + d
					sk += pk[off]
					sv += pv[off]
				}
				ko[d] = sk
				vo[d] = sv
			}
		}
	})
}

// mhaBwdGemmBand runs the backward pipeline for query rows [i0,i0+iN) of head
// h, writing that band's dQ rows and its dK/dV partials (partK/partV, [sk,dk]).
func mhaBwdGemmBand(q, g, dQ, pack []float32, partK, partV []float32, geo mhaGeo, h, i0, iN int) {
	seq, dk, dm := geo.sq, geo.dk, geo.dm
	kv := h / geo.rep
	kvStride := 3 * dk * seq
	kt := pack[kv*kvStride : kv*kvStride+dk*seq]
	vt := pack[kv*kvStride+dk*seq : kv*kvStride+2*dk*seq]
	kh := pack[kv*kvStride+2*dk*seq : (kv+1)*kvStride]
	qOff := h * dk

	qbP, gbP := getF32Raw(iN*dk), getF32Raw(iN*dk)
	sbP, daP := getF32Raw(iN*seq), getF32Raw(iN*seq)
	ptP, dqbP := getF32Raw(2*seq*iN), getF32Raw(iN*dk)
	defer putF32(qbP)
	defer putF32(gbP)
	defer putF32(sbP)
	defer putF32(daP)
	defer putF32(ptP)
	defer putF32(dqbP)
	qb, gb, sb, da, dqb := *qbP, *gbP, *sbP, *daP, *dqbP
	pt, dat := (*ptP)[:seq*iN], (*ptP)[seq*iN:]

	for r := range iN {
		copy(qb[r*dk:(r+1)*dk], q[(i0+r)*dm+qOff:(i0+r)*dm+qOff+dk])
		copy(gb[r*dk:(r+1)*dk], g[(i0+r)*dm+qOff:(i0+r)*dm+qOff+dk])
	}
	// P (softmax weights, masked columns zero) and dA = dO·Vᵀ. Causal: both
	// gemms skip the band's fully-masked column span (see mhaFwdGemmBand).
	jHi := seq
	if geo.causal {
		jHi = min((i0+iN+15)&^15, seq)
	}
	gemmF32RowsCols(qb, kt, sb, 0, iN, dk, seq, 0, jHi)
	mhaSoftmaxBandF32(sb, geo, h, i0, iN)
	gemmF32RowsCols(gb, vt, da, 0, iN, dk, seq, 0, jHi)
	// dS = scale·P∘(dA − rowdot), rowdot = Σ_j P·dA in f64, on each row's live
	// [jmin,jmax) span; the masked span is explicitly zeroed so the dSᵀ·Q
	// partial gemm adds exactly 0 for it.
	scale := geo.scale
	for r := range iN {
		jmin, jmax := geo.bounds(i0 + r)
		pr := sb[r*seq : (r+1)*seq : (r+1)*seq]
		dar := da[r*seq : (r+1)*seq : (r+1)*seq]
		var dot float64
		for j := jmin; j < jmax; j++ {
			dot += float64(pr[j]) * float64(dar[j])
		}
		clear(dar[:jmin])
		for j := jmin; j < jmax; j++ {
			dar[j] = float32(scale * float64(pr[j]) * (float64(dar[j]) - dot))
		}
		clear(dar[jmax:])
	}
	// dQ band = dS·K → disjoint dQ rows.
	gemmF32Rows(da, kh, dqb, 0, iN, seq, dk)
	for r := range iN {
		copy(dQ[(i0+r)*dm+qOff:(i0+r)*dm+qOff+dk], dqb[r*dk:(r+1)*dk])
	}
	// Band partials: dV += Pᵀ·dO, dK += dSᵀ·Q — as store-semantics gemms into
	// this band's own slots, via the [seq,iN] transposes of P and dS.
	for r := range iN {
		pr := sb[r*seq : (r+1)*seq : (r+1)*seq]
		dar := da[r*seq : (r+1)*seq : (r+1)*seq]
		for j := range seq {
			pt[j*iN+r] = pr[j]
			dat[j*iN+r] = dar[j]
		}
	}
	gemmF32Rows(pt, gb, partV, 0, seq, iN, dk)
	gemmF32Rows(dat, qb, partK, 0, seq, iN, dk)
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
