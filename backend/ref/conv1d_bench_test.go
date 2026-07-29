package ref

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkConv1D_ref covers the causal depthwise mixing conv at a Mamba/Jamba
// prefill shape (L=2048 tokens, D=1024 channels, K=4 taps, with bias). The per-tap
// left-pad bounds guard is invariant to the channel loop, so this guards the kStart
// hoist that drops it.
func BenchmarkConv1D_ref(b *testing.B) {
	const L, D, K = 2048, 1024, 4
	run := func(b *testing.B, dt tensor.Dtype) {
		x := mkF32Parity(dt == tensor.F32, tensor.Shape{L, D}, genF32Parity(L*D, 11))
		w := mkF32Parity(dt == tensor.F32, tensor.Shape{D, K}, genF32Parity(D*K, 13))
		bs := mkF32Parity(dt == tensor.F32, tensor.Shape{D}, genF32Parity(D, 17))
		in := []*tensor.Tensor{x, w, bs}
		ctx := backend.NewContext()
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			if _, err := backend.Execute(ctx, backend.OpConv1D, in, nil); err != nil {
				b.Fatal(err)
			}
		}
	}
	b.Run("f64", func(b *testing.B) { run(b, tensor.F64) })
	b.Run("f32", func(b *testing.B) { run(b, tensor.F32) })
}
