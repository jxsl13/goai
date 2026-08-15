//go:build darwin && cgo

package metal

// TestGEMMColdWeightsMatchWarm tests whether the dtype comparison in TestGEMMDtypeInterleaved is
// flattered by cache residency, and finds that it is not.
//
// The suspicion was reasonable: that probe reuses ONE weight buffer while a real forward pass
// streams 154 distinct weights, and this campaign has already been burned once by a 6.5 MB
// microbenchmark reporting twice the hardware's peak bandwidth. So this rotates 8 weight buffers
// (184 MB total, far past any cache) with the arms alternated:
//
//	M=  64  f16 67.6% / 67.6%   f32 51.4% / 51.6%   -> f16 by ~1.31x
//	M= 128  f16 68.8% / 68.8%   f32 67.2% / 67.4%   -> f16 by ~1.02x
//	M= 512  f16 74.4% / 74.3%   f32 76.2% / 76.2%   -> f32 by ~1.025x
//	M=1024  f16 75.9% / 75.9%   f32 78.3% / 78.2%   -> f32 by ~1.030x
//
// Identical to the warm figures to within 0.2 points. A single 23 MB weight already exceeds this
// machine's cache, so the "warm" probe was never warm — the hypothesis was wrong and the dtype
// result stands unchanged, including f32's ~2.5-3% edge from M=512 up.
//
// It also resolves an apparent contradiction. TestPrefillLeaveOneOut measures 48% of peak at M=64
// while this reads 67.6% — but the chain averages ALL SEVEN matmul shapes in a layer, including the
// narrow k/v projections that TestMPSGemmEfficiencyByShape puts near 50%, whereas this probe uses
// only gate/up, the best-behaved shape. Different aggregates, not a discrepancy: the chain figure is
// what prefill achieves, this one is the ceiling for its best shape.
//
// Reported, not asserted; rates are this machine's.
	"fmt"
	"testing"
)

func TestGEMMColdWeightsMatchWarm(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const K, N = 2048, 5632
	const peak = 6800.0
	fmt.Printf("COLD gate/up K=%d N=%d — 8 rotating weight buffers, arms alternated\n", K, N)
	for _, M := range []int{64, 128, 512, 1024} {
		b16, b32 := 1e18, 1e18
		for i := 0; i < 4; i++ {
			if i%2 == 0 {
				if v := ProbeGEMMDtypeCold(M, K, N, true, 8); v > 0 && v < b16 {
					b16 = v
				}
				if v := ProbeGEMMDtypeCold(M, K, N, false, 8); v > 0 && v < b32 {
					b32 = v
				}
			} else {
				if v := ProbeGEMMDtypeCold(M, K, N, false, 8); v > 0 && v < b32 {
					b32 = v
				}
				if v := ProbeGEMMDtypeCold(M, K, N, true, 8); v > 0 && v < b16 {
					b16 = v
				}
			}
		}
		fl := float64(2*M*K*N) / 1e9
		w := "f16 faster"
		if b32 < b16 {
			w = "f32 faster"
		}
		fmt.Printf("COLD M=%5d  f16 %5.1f%%  f32 %5.1f%%   f32/f16 = %.3f  -> %s\n",
			M, 100*(fl/b16)/peak, 100*(fl/b32)/peak, b32/b16, w)
	}
}
