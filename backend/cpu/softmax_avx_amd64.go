//go:build goexperiment.simd && amd64

package cpu

import (
	"math"
	"simd/archsimd"
)

// amd64 SIMD perf build: the two remaining scalar passes of the f32 softmax fast path — the row max
// and the ×1/sum scale — run through AVX2 8-wide here (the exp+sum already goes through vexpF32).
// The max/scale primitives are shared by softmaxVexpF32 (row-parallel) and softmaxWideVexpF32
// (intra-row). Other builds keep the scalar loops (softmax_avx_other.go); arm64's SIMD build included
// (this is an amd64-only vectorization), so its behaviour is bit-for-bit unchanged.

// rowMaxF32 returns max(x) with an −Inf start, matching the scalar `m:=-Inf; if v>m {m=v}` reduction
// (NaN-skip identical on finite logit inputs). 8-lane Max accumulate + horizontal max + scalar tail.
func rowMaxF32(x []float32) float32 {
	if !vexpHasAVX || len(x) < 8 {
		m := float32(math.Inf(-1))
		for _, v := range x {
			if v > m {
				m = v
			}
		}
		return m
	}
	acc := vNegInf
	n8 := len(x) &^ 7
	for i := 0; i < n8; i += 8 {
		acc = acc.Max(archsimd.LoadFloat32x8Slice(x[i:]))
	}
	var lanes [8]float32
	acc.Store(&lanes)
	m := lanes[0]
	for _, v := range lanes[1:] {
		if v > m {
			m = v
		}
	}
	for i := n8; i < len(x); i++ {
		if x[i] > m {
			m = x[i]
		}
	}
	return m
}

// scaleRowF32 multiplies x[j] by inv in place, 8-wide. Same arithmetic as the scalar `x[j]*=inv`.
func scaleRowF32(x []float32, inv float32) {
	if !vexpHasAVX {
		for j := range x {
			x[j] *= inv
		}
		return
	}
	iv := archsimd.BroadcastFloat32x8(inv)
	n8 := len(x) &^ 7
	for i := 0; i < n8; i += 8 {
		archsimd.LoadFloat32x8Slice(x[i:]).Mul(iv).StoreSlice(x[i:])
	}
	for i := n8; i < len(x); i++ {
		x[i] *= inv
	}
}
