package rl

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkRLForward measures the shared forward() helper over a PPO/A2C-update-sized
// batch (256 states × 64 dims) — the regime where filling the [n,d] input tensor
// dominates. The typed contiguous fill replaces n·d per-element SetF64 dispatches with
// n row copies.
func BenchmarkRLForward(b *testing.B) {
	const n, d = 256, 64
	net := nn.NewSequential(
		nn.NewLinear(tensor.F64, d, 64, 1),
		nn.NewLinear(tensor.F64, 64, 4, 2),
	)
	states := make([][]float64, n)
	for i := range states {
		states[i] = make([]float64, d)
		for j := range states[i] {
			states[i][j] = 0.01 * float64((i+j)%7)
		}
	}
	ctx := backend.NewContext()
	b.ReportAllocs()
	for b.Loop() {
		if _, err := forward(ctx, net, states); err != nil {
			b.Fatal(err)
		}
	}
}
