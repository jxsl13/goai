//go:build darwin && cgo

package metal

// TestEncoderOverheadIsBelowNoise records a REJECTED optimisation: packing the recorder's dispatches
// into one compute encoder instead of one encoder per op.
//
// Every recorded op opens its own MTLComputeCommandEncoder, and a decode step issues ~330 dispatches,
// so encoder-switch cost looked like a plausible slice of the ~2.5 ms/token that is not weight
// streaming. The first measurement seemed to confirm it: 330 trivial dispatches cost 827.5us in
// separate encoders against 438.0us packed into one — 1.89x, implying ~390us/token, ~7% of a 5.74ms
// token, which would have put decode past llama.cpp.
//
// That reading was thermal noise. Repeated on a cooled machine the same comparison gives:
//
//	per=1  597.5us | 1108.8us | 691.1us | 827.5us | 428.3us
//	per=n  446.8us |  435.6us | 512.3us | 438.0us | 377.3us
//	ratio    1.34x |    2.55x |   1.35x |   1.89x |   1.14x
//
// Five ratios between 1.14x and 2.55x for identical code. The minimum of each arm is the least
// contaminated estimate — 428.3 vs 377.3 — giving ~0.155us per encoder switch, so ~51us across 330
// encoders: about 0.9% of a token, comfortably under this machine's ~4% noise floor.
//
// The refactor was implemented far enough to price the risk before being reverted. Sharing one
// encoder means dispatches inside it may overlap, so every op needs an explicit
// memoryBarrierWithScope to restore the ordering separate encoders got free from Metal's hazard
// tracking, and every MPS encode, every blit, every raw encoder and every commit has to flush the
// shared one first. Missing any of those is not a slow path but a hard assertion:
// "commit command buffer with uncommitted encoder", then "A command encoder is already encoding to
// this command buffer" — two failures across 58 encoder sites before the third class appeared.
// That is a lot of new invariant for 0.9%.
//
// Kept as a probe, not a benchmark: it prices the idea so it does not get re-attempted from the
// same plausible reasoning, and it will show a real gap if future hardware or a much larger dispatch
// count changes the arithmetic.
import (
	"fmt"
	"testing"
)

func TestEncoderOverheadIsBelowNoise(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const n = 330 // roughly one Llama decode step
	for _, per := range []int{1, 2, 4, 8, 16, 330} {
		d := ProbeEncoderCost(n, per, 3)
		if d < 0 {
			t.Fatalf("probe failed: %v", d)
		}
		fmt.Printf("ENC n=%d per-encoder=%3d  %8.1fus total  %6.2fus/dispatch\n",
			n, per, d*1e6, d*1e6/float64(n))
	}
}
