package cpu

import (
	"math"
	"runtime"
	"sync/atomic"
)

// mhaSelectFwdGemmF32 is the gemm-routed f32 selective-attention forward — the f32NativeKernels
// perf-build path for OpMHASelect. Selective attention has TWO score sources per (query,key): the
// per-key selector sel[i,j] picks Q1·K1 (sel==0) or Q2·K2 (sel!=0), or masks the key (sel==−Inf).
// The scalar kernel computed one dot per key; this routes BOTH sources through the f32-native SIMD
// gemm (S1=Q1·K1ᵀ, S2=Q2·K2ᵀ), then a per-key select/mask + vexp softmax + the P·V gemm — the same
// pipeline unmasked MHA uses. 2× the score-matmul FLOPs (both sources computed in full) but still
// ~10× the scalar 8-jam path (31.6ms→~3ms at 512×512 on the perf build). Rides the f32 5e-5 budget.
func mhaSelectFwdGemmF32(q1, k1, q2, k2, v, sels, out []float32, g mhaGeo) {
	sq, sk, dk, kvDM := g.sq, g.sk, g.dk, g.kvDM
	kvHeads := g.heads / g.rep
	kt1P, kt2P, vhP := getF32Raw(kvHeads*dk*sk), getF32Raw(kvHeads*dk*sk), getF32Raw(kvHeads*sk*dk)
	defer putF32(kt1P)
	defer putF32(kt2P)
	defer putF32(vhP)
	kt1, kt2, vh := *kt1P, *kt2P, *vhP
	// Pack K1ᵀ, K2ᵀ (kv-head × [dk,sk]) and V (kv-head × [sk,dk]) once.
	parallelWork(kvHeads*sk, 3*dk, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			kv, j := t/sk, t%sk
			base := j*kvDM + kv*dk
			kth1 := kt1[kv*dk*sk:]
			kth2 := kt2[kv*dk*sk:]
			for d := 0; d < dk; d++ {
				kth1[d*sk+j] = k1[base+d]
				kth2[d*sk+j] = k2[base+d]
			}
			copy(vh[kv*sk*dk+j*dk:kv*sk*dk+(j+1)*dk], v[base:base+dk])
		}
	})
	rowWork := 2*dk*sk + 5*sk + sk*dk
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
			mhaSelectFwdGemmBand(q1, kt1, q2, kt2, vh, sels, out, g, h, i0, iN)
		}
	})
}

// mhaSelectFwdGemmBand runs the selective forward pipeline for rows [i0,i0+iN) of head h.
func mhaSelectFwdGemmBand(q1, kt1, q2, kt2, vh, sels, out []float32, g mhaGeo, h, i0, iN int) {
	sk, dk, dm := g.sk, g.dk, g.dm
	kv := h / g.rep
	kth1 := kt1[kv*dk*sk : (kv+1)*dk*sk]
	kth2 := kt2[kv*dk*sk : (kv+1)*dk*sk]
	vhh := vh[kv*sk*dk : (kv+1)*sk*dk]
	qOff := h * dk
	qb1P, qb2P := getF32Raw(iN*dk), getF32Raw(iN*dk)
	sb1P, sb2P, obP := getF32Raw(iN*sk), getF32Raw(iN*sk), getF32Raw(iN*dk)
	defer putF32(qb1P)
	defer putF32(qb2P)
	defer putF32(sb1P)
	defer putF32(sb2P)
	defer putF32(obP)
	qb1, qb2, sb1, sb2, ob := *qb1P, *qb2P, *sb1P, *sb2P, *obP
	for r := range iN {
		copy(qb1[r*dk:(r+1)*dk], q1[(i0+r)*dm+qOff:(i0+r)*dm+qOff+dk])
		copy(qb2[r*dk:(r+1)*dk], q2[(i0+r)*dm+qOff:(i0+r)*dm+qOff+dk])
	}
	gemmF32RowsCols(qb1, kth1, sb1, 0, iN, dk, sk, 0, sk)  // S1 = Q1·K1ᵀ
	gemmF32RowsCols(qb2, kth2, sb2, 0, iN, dk, sk, 0, sk)  // S2 = Q2·K2ᵀ
	mhaSelectSoftmaxBandVexpF32(sb1, sb2, sels, g, i0, iN) // blend by sel → sb1, then softmax
	gemmF32Rows(sb1, vhh, ob, 0, iN, sk, dk)
	for r := range iN {
		copy(out[(i0+r)*dm+qOff:(i0+r)*dm+qOff+dk], ob[r*dk:(r+1)*dk])
	}
}

// mhaSelectSoftmaxBandVexpF32 selects each key's score source and softmaxes in place into sb1. For
// row r (query i0+r), key j: sel[i,j]==−Inf masks the key (−Inf → vexpF32 clamps to ~0), sel==0 keeps
// the S1 score, sel!=0 takes the S2 score; the chosen score is scaled, then the stable max-shift exp/
// sum runs through vexpRowF32. A fully-masked row is written all-zeros (matches the scalar kernel).
func mhaSelectSoftmaxBandVexpF32(sb1, sb2, sels []float32, g mhaGeo, i0, iN int) {
	sk := g.sk
	scale := float32(g.scale)
	for r := range iN {
		i := i0 + r
		s1 := sb1[r*sk : (r+1)*sk : (r+1)*sk]
		s2 := sb2[r*sk : (r+1)*sk : (r+1)*sk]
		srow := sels[i*sk : i*sk+sk : i*sk+sk]
		m := float32(math.Inf(-1))
		for j := range s1 {
			sv := srow[j]
			var x float32
			switch {
			case math.IsInf(float64(sv), -1):
				x = float32(math.Inf(-1))
			case sv == 0:
				x = s1[j] * scale
			default:
				x = s2[j] * scale
			}
			s1[j] = x
			if x > m {
				m = x
			}
		}
		if math.IsInf(float64(m), -1) {
			for j := range s1 {
				s1[j] = 0
			}
			continue
		}
		sum := vexpRowF32(s1, m)
		inv := 1 / sum
		for j := range s1 {
			s1[j] *= inv
		}
	}
}
