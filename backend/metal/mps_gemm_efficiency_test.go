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
// 2. RETRACTED — see TestMPSEfficiencyCurveVsM, which probes f16 first and gets the OPPOSITE
//    ordering at the same shape (f16 73.3% / f32 67.3% at M=512, against 76.8/74.4 here). The gap
//    is within this machine's drift and depends on probe order, so neither ordering is established
//    and the dual-format cache case that rested on it is weaker than presented below.
//
//    (original text) f32 beats f16 at EVERY shape by 2-4%, and we currently run f16 at M<=512 because that is what
//    the weight cache holds. That 2-4% is the price of the cache being f16, which halves its memory.
//    Caching f32 instead was tried and reverted (see TestDQGemmCostSplit): f16 1.92 GB + f32 3.88 GB
//    exceeds the 4 GB budget and the two evict each other. An f32-ONLY cache fits, but f16 wins by
//    1.28x at M=64 where the GEMM is bandwidth-bound, so it would trade our weakest shape (pp64,
//    0.87x) for 2-4% at the stronger ones. The optimal format depends on M, not on the weight, which
//    is state this layer does not have.
//
// MEASURED FOLLOW-UP on point 2. Caching BOTH formats and letting each M regime pick its optimum
// does work, but only with a budget the default does not grant. Interleaved against the shipped
// f16-only cache, two runs averaged:
//
//	          shipped (1.94 GB)   dual @8 GB (5.75 GB)
//	pp64          1541.3               1541.2      —
//	pp256         1962.7               2030.1      +3.4%
//	pp512         1986.5               2041.9      +2.8%
//	pp1024        1942.5               2091.1      +7.7%
//
// pp1024 gains most because M>512 previously fell off the f16 path to an UNCACHED f32 expansion,
// paying it every pass. But at the default 4 GB budget the same code REGRESSES (pp256 1806 vs
// 1962): the two caches together need ~5.8 GB, so f32 entries consume budget the f16 entries then
// cannot get, and weights that miss fall back to per-pass expansion. A default that is worse than
// what it replaces is not shippable, and making it safe needs reservation logic — deciding a
// per-format budget split before either fills — which is more machinery than 3-8% justifies here.
//
// So the 2-4% f32 advantage is reachable, at ~3x the cache memory, and is left unimplemented
// deliberately rather than for want of trying.
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
