package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

func BenchmarkSoftmaxF32_512x512_cpu(b *testing.B) { // attention-matrix scale
	benchOn(b, backend.CPU, backend.OpSoftmax, bench.RandF32(tensor.Shape{512, 512}, 1))
}
func BenchmarkSoftmaxF32_2048sq_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpSoftmax, bench.RandF32(tensor.Shape{2048, 2048}, 1))
}
func BenchmarkSoftmaxF64_2048sq_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpSoftmax, bench.RandF64(tensor.Shape{2048, 2048}, 1))
}
