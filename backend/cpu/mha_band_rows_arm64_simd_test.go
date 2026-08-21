//go:build arm64 && goexperiment.simd

package cpu

import (
	"math"
	"testing"
)

func TestMHAFwdBandScheduleControlDigest(t *testing.T) {
	if mhaFwdBandRows != 32 || mhaFwdBandRows%4 != 0 {
		t.Fatalf("ARM64 MHA band rows = %d, want selected value 32 aligned to 4", mhaFwdBandRows)
	}
	type testCase struct {
		causal  bool
		heads   int
		kvHeads int
	}
	cases := []testCase{
		{causal: true, heads: 4, kvHeads: 4},
		{causal: false, heads: 4, kvHeads: 4},
		{causal: true, heads: 4, kvHeads: 2},
	}
	const (
		sq = 64
		dm = 128
	)
	hash := uint64(14695981039346656037)
	for ci, tc := range cases {
		dk := dm / tc.heads
		kvDM := tc.kvHeads * dk
		q := make([]float32, sq*dm)
		k := make([]float32, sq*kvDM)
		v := make([]float32, sq*kvDM)
		fillMHAControlFixture(q, 17+ci*2)
		fillMHAControlFixture(k, 23+ci*2)
		fillMHAControlFixture(v, 29+ci*2)
		out := make([]float32, sq*dm)
		mhaFwdGemmF32(q, k, v, out, mhaGeo{
			sq: sq, sk: sq, dm: dm, dk: dk, kvDM: kvDM,
			heads: tc.heads, rep: tc.heads / tc.kvHeads,
			causal: tc.causal, scale: 1 / math.Sqrt(float64(dk)),
		})
		for _, x := range out {
			bits := math.Float32bits(x)
			for shift := uint(0); shift < 32; shift += 8 {
				hash ^= uint64(byte(bits >> shift))
				hash *= 1099511628211
			}
		}
	}
	const control = uint64(0x73550b82110bb18f)
	if hash != control {
		t.Fatalf("MHA band schedule digest = %016x, control %016x", hash, control)
	}
}

func fillMHAControlFixture(dst []float32, mul int) {
	for i := range dst {
		dst[i] = float32((i*mul+11)%101-50) / 37
	}
}
