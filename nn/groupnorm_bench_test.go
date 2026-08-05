package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
)

// BenchmarkGroupNormForward at a vision/diffusion shape (spatial folded into batch): rows=N·H·W,
// C channels, G groups. The fused path routes the group normalize through OpLayerNorm (one dispatch,
// row-parallel f64-accum AVX2) instead of the 7-op mean/sub/mul/mean/add/sqrt/div chain + 6 allocs.
func benchGroupNorm(b *testing.B, rows, c, groups int) {
	gn := nn.NewGroupNorm(tensor.F32, groups, c)
	x := tensor.New(tensor.F32, tensor.Shape{rows, c})
	xf := x.Storage().F32()
	for i := range xf {
		xf[i] = float32((i*131+7)%1000)/500.0 - 1
	}
	ctx := backend.NewContext()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := gn.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGroupNormForward_4096x512_g32(b *testing.B) { benchGroupNorm(b, 4096, 512, 32) }
