//go:build darwin && cgo

package metal

// TestMPSGemmEfficiencyByShape maps how close MPS gets to this machine's FLOP peak at the exact
// weight shapes prefill uses, which is what decides whether hand-writing a GEMM is worth it.
//
// Prefill is 91% matmul (TestPrefillLeaveOneOut) and its remaining deficit against llama.cpp is
// ~4.7% of GPU time, so "beat MPS" is the last avenue with real headroom. Measured on an M2 Pro
// against a 6800 GFLOP/s f32 peak:
//
//	M= 512  gate/up 2048x5632  f32 76.8%   f16 74.4%
//	M= 512  down    5632x2048  f32 71.2%   f16 69.4%
//	M= 512  q/o     2048x2048  f32 71.1%   f16 69.0%
//	M= 512  k/v     2048x256   f32 51.9%   f16 47.5%
//	M=1024  gate/up 2048x5632  f32 78.3%   f16 76.0%
//	M=1024  k/v     2048x256   f32 65.8%   f16 60.2%
//
// Two results.
//
// 1. MPS is at 71-78% on the shapes that carry the FLOPs. A hand-written replacement has to clear
//    ~78% to be worth anything, against Apple's own tuned kernel. That is the bar, and it is high —
//    worth knowing before starting rather than after.
//
// 2. f32 beats f16 at EVERY shape by 2-4%, and we currently run f16 at M<=512 because that is what
//    the weight cache holds. That 2-4% is the price of the cache being f16, which halves its memory.
//    Caching f32 instead was tried and reverted (see TestDQGemmCostSplit): f16 1.92 GB + f32 3.88 GB
//    exceeds the 4 GB budget and the two evict each other. An f32-ONLY cache fits, but f16 wins by
//    1.28x at M=64 where the GEMM is bandwidth-bound, so it would trade our weakest shape (pp64,
//    0.87x) for 2-4% at the stronger ones. The optimal format depends on M, not on the weight, which
//    is state this layer does not have.
//
// The narrow k/v shape at 51.9% is the least efficient by far, but k and v are only ~2.4% of a
// layer's FLOPs, so fusing them into one N=512 GEMM would be worth ~0.5% of prefill — measured
// before building it, and not worth building.
//
// Reported, not asserted on absolute rates; the peak is this machine's.
import (
	"fmt"
	"testing"
)

func TestMPSGemmEfficiencyByShape(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	// the four distinct weight shapes of a TinyLlama layer, at prefill M
	shapes := []struct {
		name string
		k, n int
	}{
		{"q/o   2048x2048", 2048, 2048},
		{"k/v   2048x256 ", 2048, 256},
		{"gate/up 2048x5632", 2048, 5632},
		{"down  5632x2048", 5632, 2048},
	}
	const peak = 6800.0
	for _, M := range []int{512, 1024} {
		for _, s := range shapes {
			f32 := ProbeGEMMDtype(M, s.k, s.n, false, 3)
			f16 := ProbeGEMMDtype(M, s.k, s.n, true, 3)
			if f32 < 0 || f16 < 0 {
				t.Fatal("probe failed")
			}
			fl := float64(2*M*s.k*s.n) / 1e9
			fmt.Printf("MPS M=%4d %-18s f32 %7.1f GFLOP/s (%4.1f%%)  f16 %7.1f (%4.1f%%)\n",
				M, s.name, fl/f32, 100*(fl/f32)/peak, fl/f16, 100*(fl/f16)/peak)
		}
	}
}
