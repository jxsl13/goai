package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// reduce-ALL to scalar on the CPU backend (where ops.Sum/Mean/Max run in model forward
// paths) — the ref reduceKernel's per-element combine closure + odometer vs the CPU
// devirtualised inlined loop.
func BenchmarkSumAllF64_256K_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpSum, bench.RandF64(tensor.Shape{262144}, 3))
}
func BenchmarkMeanAllF64_256K_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpMean, bench.RandF64(tensor.Shape{262144}, 3))
}
func BenchmarkMaxAllF64_256K_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpMax, bench.RandF64(tensor.Shape{262144}, 3))
}
