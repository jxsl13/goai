//go:build darwin && cgo

package metal

// TestGEMMDtypeInterleaved settles f16-vs-f32 for the prefill GEMM by ALTERNATING the two arms,
// after two earlier attempts reached opposite conclusions purely from probe order.
//
// The history is the point. TestMPSGemmEfficiencyByShape probed f32 first and concluded "f32 beats
// f16 at every shape by 2-4%". TestMPSEfficiencyCurveVsM probed f16 first and got the opposite
// ordering at the same shape. Neither interleaved, so both measured warm-up and drift as much as
// dtype. Alternating the arms and taking min-of-6 alternations, gate/up K=2048 N=5632:
//
//	M= 64   f16 67.7% / 67.6%   f32 51.8% / 52.0%   -> f16 by 1.30x
//	M=128   f16 68.9% / 69.0%   f32 67.3% / 67.7%   -> f16 by 1.02x
//	M=512   f16 74.4% / 74.3%   f32 76.3% / 76.2%   -> f32 by 1.025x
//
// Two runs agree to within 0.3 points, against the several-point swings the sequential probes
// showed. So: f16 wins decisively at M=64, marginally at M=128, and loses marginally by M=512.
// The shipped gate (f16 whenever the weight cache is on) is therefore optimal up to ~128 and about
// 2.5% off at M>=512 — and closing that needs the f32 cache, i.e. twice the memory for 2.5%.
//
// It also RETRACTS the M=128 f16 "dip" (42.9%, recorded in TestMPSEfficiencyCurveVsM as a possible
// MPS tiling artifact). Interleaved it reads 68.9%, in line with its neighbours. The dip was probe
// order, not tiling — which is why that test recorded it as a candidate rather than a finding.
//
// Note the absolute levels rise too (M=64 f16: 47.8% sequential, 67.7% interleaved). An isolated
// GEMM with warm buffers is not the same as the same GEMM inside a dependent chain, where
// TestPrefillLeaveOneOut measures 48% at M=64. Use this test for A-vs-B between dtypes, and the
// chain for what prefill actually achieves.
//
// Reported, not asserted; rates are this machine's.
import (
	"fmt"
	"testing"
)

func TestGEMMDtypeInterleaved(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const K, N = 2048, 5632
	const peak = 6800.0
	fmt.Printf("INTER gate/up K=%d N=%d — arms ALTERNATED, min of 6 alternations\n", K, N)
	for _, M := range []int{64, 128, 512} {
		b16, b32 := 1e18, 1e18
		// alternate so thermal drift and warm-up hit both arms equally
		for i := 0; i < 6; i++ {
			if i%2 == 0 {
				if v := ProbeGEMMDtype(M, K, N, true, 3); v > 0 && v < b16 {
					b16 = v
				}
				if v := ProbeGEMMDtype(M, K, N, false, 3); v > 0 && v < b32 {
					b32 = v
				}
			} else {
				if v := ProbeGEMMDtype(M, K, N, false, 3); v > 0 && v < b32 {
					b32 = v
				}
				if v := ProbeGEMMDtype(M, K, N, true, 3); v > 0 && v < b16 {
					b16 = v
				}
			}
		}
		fl := float64(2*M*K*N) / 1e9
		e16, e32 := 100*(fl/b16)/peak, 100*(fl/b32)/peak
		verdict := "f16 faster"
		if b32 < b16 {
			verdict = "f32 faster"
		}
		fmt.Printf("INTER M=%4d  f16 %5.1f%%  f32 %5.1f%%   ratio f32/f16 = %.3f  -> %s\n",
			M, e16, e32, b32/b16, verdict)
	}
}
