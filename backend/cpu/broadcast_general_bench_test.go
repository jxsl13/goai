package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// General-broadcast benchmarks: shapes deliberately chosen so the operand is NOT a
// contiguous trailing block of the output, which is what routes the kernel past the
// bcastBlockApply fast path into the materializing broadcastContig. Entry into that
// path is asserted separately (see the runs-transform research item) — a benchmark
// whose name matches a feature is not proof it reaches the branch.

// Middle-axis broadcast: the innermost axis survives (effective stride 1), so a run
// is a contiguous read.
func BenchmarkBroadcastMidAxisF32_32x64x256_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpAdd,
		bench.RandF32(tensor.Shape{32, 1, 256}, 1), bench.RandF32(tensor.Shape{32, 64, 256}, 2))
}

// Innermost-axis broadcast: effective stride 0, so a run is one repeated value.
func BenchmarkBroadcastInnerF32_32x64x256_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpAdd,
		bench.RandF32(tensor.Shape{32, 64, 1}, 1), bench.RandF32(tensor.Shape{32, 64, 256}, 2))
}

func BenchmarkBroadcastMidAxisF64_32x64x256_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpAdd,
		bench.RandF64(tensor.Shape{32, 1, 256}, 1), bench.RandF64(tensor.Shape{32, 64, 256}, 2))
}
