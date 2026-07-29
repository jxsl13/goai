package cpu

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchConv1DCPU(b *testing.B, dt tensor.Dtype) {
	const L, D, K = 2048, 1024, 4
	rng := rand.New(rand.NewSource(1))
	fill := func(t *tensor.Tensor) *tensor.Tensor {
		if dt == tensor.F64 {
			s := t.Storage().F64()
			for i := range s {
				s[i] = rng.Float64() - 0.5
			}
		} else {
			s := t.Storage().F32()
			for i := range s {
				s[i] = rng.Float32() - 0.5
			}
		}
		return t
	}
	x := fill(tensor.New(dt, tensor.Shape{L, D}))
	w := fill(tensor.New(dt, tensor.Shape{D, K}))
	bs := fill(tensor.New(dt, tensor.Shape{D}))
	ctx := backend.NewContext().WithBackend(std)
	in := []*tensor.Tensor{x, w, bs}
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpConv1D, in, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConv1D_cpu_f64(b *testing.B) { benchConv1DCPU(b, tensor.F64) }
func BenchmarkConv1D_cpu_f32(b *testing.B) { benchConv1DCPU(b, tensor.F32) }
