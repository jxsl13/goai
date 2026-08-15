//go:build darwin && cgo

package metal

// TestHandwrittenGEMMBaseline prices the last open avenue on this path: replacing MPS with a
// hand-written GEMM, the way llama.cpp does.
//
// Prefill is 91% matmul, so out-executing MPS is the only remaining item with real headroom. The bar
// from TestMPSGemmEfficiencyByShape is ~74-78% of peak on the FLOP-carrying shapes. sg_gemm_f16 in
// metal_bridge.m is the textbook implementation of that bar's challenger — 64x64 output tile per
// threadgroup, 8 simdgroups, K walked in 16-wide chunks staged through threadgroup memory, f16 in
// and f32 accumulate. Arms alternated:
//
//	M=  64  hand-written 1806.1 GFLOP/s (26.6%)   MPS f16 4621.0 (68.0%)   0.39x
//	M= 512  hand-written 2444.9 GFLOP/s (36.0%)   MPS f16 5064.5 (74.5%)   0.48x
//	M=1024  hand-written 2518.9 GFLOP/s (37.0%)   MPS f16 5168.0 (76.0%)   0.49x
//
// A straightforward tiled simdgroup GEMM reaches less than HALF of MPS. Closing that needs roughly
// 2.1x on top of this baseline — double-buffered K stages so loads overlap the matrix ops, wider
// register blocking per simdgroup, async threadgroup copies, and tile sizes tuned per shape — and
// then it still has to clear MPS rather than merely reach it.
//
// TWO STANDARD LEVERS TRIED, BOTH FAILED TO CLOSE IT:
//
//  1. K-chunk size. BK=16/32/64 measured 1850.8 / 1855.0 / 1904.0 GFLOP/s — a 3% spread, so the
//     barrier count per K element is not the constraint. (Absolute figures below those in the table
//     because making BK settable forced a fixed 16 KB threadgroup allocation.)
//  2. Register blocking, the textbook fix for the load:compute ratio. Giving each simdgroup a 32x32
//     patch turns 8 matrix ops from 16 fragment loads into 16 from 8 — and measured 1166.8 GFLOP/s
//     against the baseline's 1850.8, i.e. 0.63x. It halves simdgroups per threadgroup from 8 to 4,
//     and 16 accumulators plus 8 fragments per lane is enough register pressure to lose more to
//     occupancy than the better ratio wins. The same inversion the attention kernels showed, where
//     the variants that "improve" occupancy do it by spilling.
//
// So "beat MPS" is not a near-term option, and that is now measured rather than assumed. The kernel
// is kept, unused by the model path and reachable only through ProbeSGGemm, because a future attempt
// should start from a known 0.39-0.49x baseline instead of from scratch.
//
// Reported, not asserted on absolute rates; peaks are this machine's.
import (
	"fmt"
	"testing"
)

func TestHandwrittenGEMMBaseline(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const K, N = 2048, 5632
	const peak = 6800.0
	for _, M := range []int{64, 512, 1024} {
		var bs, bm float64 = 1e18, 1e18
		for i := 0; i < 3; i++ { // alternate arms
			if v := ProbeSGGemm(M, K, N, 3); v > 0 && v < bs {
				bs = v
			}
			if v := ProbeGEMMDtype(M, K, N, true, 3); v > 0 && v < bm {
				bm = v
			}
			if v := ProbeGEMMDtype(M, K, N, true, 3); v > 0 && v < bm {
				bm = v
			}
			if v := ProbeSGGemm(M, K, N, 3); v > 0 && v < bs {
				bs = v
			}
		}
		if bs > 1e17 {
			t.Fatalf("sg_gemm probe failed at M=%d", M)
		}
		fl := float64(2*M*K*N) / 1e9
		fmt.Printf("SGG M=%5d  hand-written %6.1f GFLOP/s (%4.1f%%)   MPS f16 %6.1f (%4.1f%%)   ratio %.2f\n",
			M, fl/bs, 100*(fl/bs)/peak, fl/bm, 100*(fl/bm)/peak, bm/bs)
	}
}
