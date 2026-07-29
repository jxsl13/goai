package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// NeuralMemory.Forward is a per-TIMESTEP recurrence that dispatches roughly twenty backend
// ops per step, each allocating a tiny tensor (PS4011). It had no benchmark, so the size of
// that overhead was unmeasured.
func benchTitansMem(b *testing.B, seq, dim int, linear bool) {
	var opts []nn.TitansOption
	if linear {
		opts = append(opts, nn.WithTitansLinearMemory())
	}
	hidden := 0
	if !linear {
		hidden = dim
	}
	m, err := nn.NewNeuralMemory(tensor.F64, dim, hidden, 3, opts...)
	if err != nil {
		b.Fatal(err)
	}
	x := tensor.New(tensor.F64, tensor.Shape{seq, dim})
	s := x.Storage().F64()
	for i := range s {
		s[i] = math.Sin(float64(i) * 0.013)
	}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := m.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTitansMemLinear_seq128(b *testing.B) { benchTitansMem(b, 128, 64, true) }
func BenchmarkTitansMemLinear_seq256(b *testing.B) { benchTitansMem(b, 256, 64, true) }
func BenchmarkTitansMemDeep_seq128(b *testing.B)   { benchTitansMem(b, 128, 64, false) }
