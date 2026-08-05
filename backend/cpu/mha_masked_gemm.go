package cpu

import (
	"math"
	"runtime"
	"sync/atomic"
)

// mhaMaskedFwdGemmF32 is the gemm-routed f32 masked-attention forward — the f32NativeKernels perf-build
// path for OpMHAMasked. It mirrors mhaFwdGemmF32 (pack Kᵀ/V once, then a dynamic band schedule of the
// two per-head matmuls through the f32-native SIMD gemm) but adds an arbitrary additive mask before the
// softmax: mask is [sq,sk] (perHead=false) or per-head [heads,sq,sk]. This covers the causal / relative-
// position-bias masks the scalar 8-jam kernel handled, but ~8x faster (the scalar path never routed
// through the gemm pipeline even on the perf build — masked attention was 29ms vs unmasked 1.15ms at
// 512×512). Masked entries carry −Inf, which vexpF32 clamps to ~0 (FLT_MIN), well inside the f32 5e-5
// parity budget (ADR-0021). jHi is always sk: an arbitrary mask can place weight on any key, so there
// is no causal column-skip.
func mhaMaskedFwdGemmF32(q, k, v, mask, out []float32, g mhaGeo, perHead bool) {
	sq, sk, dk, kvDM := g.sq, g.sk, g.dk, g.kvDM
	kvHeads := g.heads / g.rep
	ktP, vhP := getF32Raw(kvHeads*dk*sk), getF32Raw(kvHeads*sk*dk)
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
			mhaMaskedFwdGemmBand(q, kt, vh, mask, out, g, perHead, h, i0, iN)
		}
	})
}

// mhaMaskedFwdGemmBand runs the masked forward pipeline for rows [i0,i0+iN) of head h.
func mhaMaskedFwdGemmBand(q, kt, vh, mask, out []float32, g mhaGeo, perHead bool, h, i0, iN int) {
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
	gemmF32RowsCols(qb, kth, sb, 0, iN, dk, sk, 0, sk) // full sk — arbitrary mask, no causal skip
	mhaMaskedSoftmaxBandVexpF32(sb, mask, g, h, i0, iN, perHead)
	gemmF32Rows(sb, vhh, ob, 0, iN, sk, dk)
	for r := range iN {
		copy(out[(i0+r)*dm+qOff:(i0+r)*dm+qOff+dk], ob[r*dk:(r+1)*dk])
	}
}

// mhaMaskedSoftmaxBandVexpF32 turns rows [i0,i0+iN) of head h's raw QKᵀ products (sb, band-local
// [iN,sk]) into softmax weights in place: each score is scaled and the additive mask row is added, then
// the stable max-shift exp/sum runs through the vectorized vexpRowF32 (the perf build's SIMD exp). A
// masked key carries mask=−Inf → score=−Inf → vexpF32 clamps exp(−Inf−m) to ~FLT_MIN, contributing ~0
// to the sum and to the following P·V matmul (within the f32 parity budget). A fully-masked row (every
// key −Inf) is written to all-zeros, matching the scalar kernel's degenerate handling.
func mhaMaskedSoftmaxBandVexpF32(sb, mask []float32, g mhaGeo, h, i0, iN int, perHead bool) {
	sk := g.sk
	scale := float32(g.scale)
	mBase := 0
	if perHead {
		mBase = h * g.sq * sk
	}
	for r := range iN {
		i := i0 + r
		sr := sb[r*sk : (r+1)*sk : (r+1)*sk]
		mrow := mask[mBase+i*sk : mBase+i*sk+sk : mBase+i*sk+sk]
		m := float32(math.Inf(-1))
		for j := range sr {
			x := sr[j]*scale + mrow[j]
			sr[j] = x
			if x > m {
				m = x
			}
		}
		if math.IsInf(float64(m), -1) { // fully-masked row → all zeros (matches the scalar 8-jam kernel)
			for j := range sr {
				sr[j] = 0
			}
			continue
		}
		sum := vexpRowF32(sr, m)
		inv := 1 / sum
		for j := range sr {
			sr[j] *= inv
		}
	}
}

// F32NativeKernelsEnabled reports whether the f32 MHA/Conv gemm-routed perf path is active (the
// goexperiment.simd "perf" build). Exposed so the f32-parity tests can select byte-exact (default
// build, scalar f64-accumulating kernels) vs the ADR-0021 5e-5 tolerance (perf build, f32 gemm+vexp).
func F32NativeKernelsEnabled() bool { return f32NativeKernels }
