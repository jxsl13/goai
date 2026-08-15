//go:build darwin && cgo

package metal

// TestMPSEfficiencyCurveVsM maps MPS efficiency against M for the FLOP-dominant weight shape, which
// is what sets the bar for replacing it — and which retracts an earlier f16-vs-f32 claim.
//
// gate/up K=2048 N=5632, against a 6800 GFLOP/s peak:
//
//	M=  32  f16 33.1%  f32 32.8%
//	M=  64  f16 47.8%  f32 35.1%
//	M=  96  f16 48.6%  f32 50.2%
//	M= 128  f16 42.9%  f32 51.9%
//	M= 192  f16 69.6%  f32 64.1%
//	M= 256  f16 72.4%  f32 65.5%
//	M= 512  f16 73.3%  f32 67.3%
//	M=1024  f16 74.8%  f32 69.9%
//
// The SHAPE of the curve is the robust part and it is steep: efficiency roughly doubles between
// M=32 and M=192 and then flattens. Prefill at short prompts is not slow because of anything we do
// — a thin GEMM simply cannot fill this GPU, and llama.cpp pays it too (its pp64 implies ~51%
// against our 48%).
//
// RETRACTION: TestMPSGemmEfficiencyByShape reports "f32 beats f16 at every shape by 2-4%", measured
// at M=512 as f32 76.8% / f16 74.4%. This test, which probes f16 FIRST, gets the opposite ordering
// at the same shape (f16 73.3% / f32 67.3%). The only difference is probe order, so the f16-vs-f32
// gap is within this machine's drift and neither ordering is established. Anything resting on that
// 2-4% — including the case for a dual-format weight cache — is weaker than it was presented.
//
// The M=128 f16 dip (42.9%, below both M=96 and M=192) reproduces within this run but has the same
// problem: it is a single ordering. It would need interleaved arms with a control before being
// treated as an MPS tiling artifact worth routing around.
//
// Reported, not asserted; peaks and rates are this machine's.
import (
	"fmt"
	"testing"
)

func TestMPSEfficiencyCurveVsM(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const K, N = 2048, 5632 // gate/up: the FLOP-dominant shape
	const peak = 6800.0
	fmt.Printf("CURVE gate/up K=%d N=%d — MPS efficiency vs M\n", K, N)
	for _, M := range []int{32, 64, 96, 128, 192, 256, 512, 1024} {
		f16 := ProbeGEMMDtype(M, K, N, true, 3)
		f32 := ProbeGEMMDtype(M, K, N, false, 3)
		if f16 < 0 || f32 < 0 {
			t.Fatal("probe failed")
		}
		fl := float64(2*M*K*N) / 1e9
		// bytes: weight dominates at small M
		wb16 := float64(K*N*2) / 1e6
		fmt.Printf("CURVE M=%5d  f16 %7.1f GFLOP/s (%4.1f%%)  f32 %7.1f (%4.1f%%)   f16 weight-read %5.1f GB/s\n",
			M, fl/f16, 100*(fl/f16)/peak, fl/f32, 100*(fl/f32)/peak, wb16/f16/1e3)
	}
}
