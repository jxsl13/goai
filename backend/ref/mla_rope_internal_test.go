package ref

import (
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// BenchmarkMLARoPE covers the decoupled-RoPE rotation at DeepSeek-V2 query sizes
// (seq=512, nheads=128, dR=64). theta=base^(-2e/dR) is invariant across position and
// head, and cos/sin(p·theta) across head — this benchmark guards that hoist.
func BenchmarkMLARoPE(b *testing.B) {
	const seq, nheads, dR = 512, 128, 64
	const base = 10000.0
	run := func(b *testing.B, dt tensor.Dtype) {
		src := tensor.New(dt, tensor.Shape{seq, nheads * dR})
		dst := make([]float64, seq*nheads*dR)
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			mlaRoPE(src, nheads, dR, base, dst)
		}
	}
	b.Run("f64", func(b *testing.B) { run(b, tensor.F64) })
	b.Run("f32", func(b *testing.B) { run(b, tensor.F32) })
}
