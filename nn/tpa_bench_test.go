package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func benchTPAForward(b *testing.B, T, d, heads, dh int) {
	m, err := nn.NewTPA(tensor.F64, d, heads, dh, true, 42)
	if err != nil {
		b.Fatal(err)
	}
	x := tensor.New(tensor.F64, tensor.Shape{T, d})
	xs := x.Storage().F64()
	for i := range xs {
		xs[i] = float64((i%97))*0.01 - 0.5
	}
	ctx := backend.NewContext() // inference: Recorder == nil
	b.ResetTimer()
	for range b.N {
		if _, err := m.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTPAForward_512(b *testing.B)  { benchTPAForward(b, 512, 512, 8, 64) }
func BenchmarkTPAForward_1024(b *testing.B) { benchTPAForward(b, 1024, 512, 8, 64) }
