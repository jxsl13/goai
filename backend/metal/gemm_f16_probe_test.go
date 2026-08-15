//go:build darwin && cgo

package metal

// TestGEMMF16OnlyWinsWhenBandwidthBound answers, before any f16 plumbing exists, whether an f16 MPS
// GEMM is faster than f32 at the prefill weight shape — and finds that the answer depends entirely
// on M, which is what decides how the f16 path must be gated.
//
// TestDQGemmCostSplit showed the M=64 GEMM reads its 46 MB f32 weight at ~99 GB/s against ~180 GB/s
// sustained, reaching only 47% of FLOP peak, while at M=1024 it is compute-bound at 78%. If that
// reading is right, halving the weight bytes must help at M=64 and do nothing at M=1024. Measured on
// an M2 Pro, K=2048 N=5632:
//
//	M=  64  f32  415.5us (3554 GFLOP/s)  f16  323.8us (4560 GFLOP/s)  1.28x
//	M= 256  f32 1142.7us (5168 GFLOP/s)  f16 1180.1us (5004 GFLOP/s)  0.97x
//	M=1024  f32 4438.1us (5323 GFLOP/s)  f16 4574.0us (5164 GFLOP/s)  0.97x
//
// Exactly that. f16 buys 1.28x at M=64 and COSTS ~3% from M=256 up: this GPU's matrix units run f16
// at the same rate as f32, so once the GEMM is compute-bound there is no arithmetic advantage left
// to collect, only the conversion overhead.
//
// The consequence for the design: an f16 weight path must be gated on small M rather than switched
// on globally, or it makes long-prompt prefill — the shape already near parity with llama.cpp —
// slightly worse to speed up the short one.
//
// Reported, not asserted on absolute timings; the probe allocates its own buffers so it stays valid
// independently of how the real path is wired.
import (
	"fmt"
	"testing"
)

func TestGEMMF16OnlyWinsWhenBandwidthBound(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const K, N = 2048, 5632
	for _, M := range []int{64, 256, 1024} {
		f32 := ProbeGEMMDtype(M, K, N, false, 3)
		f16 := ProbeGEMMDtype(M, K, N, true, 3)
		if f32 < 0 || f16 < 0 {
			t.Fatalf("probe failed: f32=%v f16=%v", f32, f16)
		}
		fl := float64(2*M*K*N) / 1e9
		fmt.Printf("F16 M=%4d f32=%8.1fus (%5.0f GFLOP/s) f16=%8.1fus (%5.0f GFLOP/s)  %.2fx\n",
			M, f32*1e6, fl/f32, f16*1e6, fl/f16, f32/f16)
	}
}
