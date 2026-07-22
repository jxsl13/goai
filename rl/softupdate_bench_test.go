package rl

import (
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkSoftUpdate measures the Polyak target-network update over a critic-sized MLP
// — the per-step hot path of DDPG/SAC/TD3. The typed contiguous fast path removes the
// per-element tensor.Unravel allocation + AtF64/SetF64 dispatch the generic walk paid
// on every one of the ~130k parameters.
func BenchmarkSoftUpdate(b *testing.B) {
	mk := func(seed uint64) *nn.Sequential {
		return nn.NewSequential(
			nn.NewLinear(tensor.F64, 256, 256, seed),
			nn.NewLinear(tensor.F64, 256, 256, seed+1),
			nn.NewLinear(tensor.F64, 256, 1, seed+2),
		)
	}
	online, target := mk(1), mk(2)
	b.ReportAllocs()
	for b.Loop() {
		softUpdate(online, target, 0.005)
	}
}
