package nn

import (
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// benchFill times weight initialization, which every model construction pays and which showed up
// as 21.75% of a five-benchmark nn profile — all of it setup, and none of it measured before.
func benchFill(b *testing.B, n int, dt tensor.Dtype, f func(*tensor.Tensor)) {
	t := tensor.New(dt, tensor.Shape{n})
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		f(t)
	}
}

func BenchmarkFillUniformF64(b *testing.B) {
	benchFill(b, 1<<20, tensor.F64, func(t *tensor.Tensor) { fillUniform(t, -0.5, 0.5, 7) })
}

func BenchmarkFillUniformF32(b *testing.B) {
	benchFill(b, 1<<20, tensor.F32, func(t *tensor.Tensor) { fillUniform(t, -0.5, 0.5, 7) })
}

func BenchmarkKaimingNormalF32(b *testing.B) {
	benchFill(b, 1<<20, tensor.F32, func(t *tensor.Tensor) { KaimingNormal(t, 128, 7) })
}
