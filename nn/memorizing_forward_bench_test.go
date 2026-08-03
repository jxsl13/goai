package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// benchMemForward measures MemorizingAttention.Forward with a POPULATED memory bank so the
// memoryAttention branch (the 2 per-head einsums) actually runs. Inference (Recorder == nil).
func benchMemForward(b *testing.B, T, dim, heads, memN, topK int) {
	m, err := nn.NewMemorizingAttention(tensor.F64, dim, heads, 42,
		nn.WithMemorizingMemorySize(memN), nn.WithMemorizingTopK(topK))
	if err != nil {
		b.Fatal(err)
	}
	mk := func(rows int, s float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{rows, dim})
		st := t.Storage().F64()
		for i := range st {
			st[i] = math.Sin(float64(i)*0.001 + s)
		}
		return t
	}
	if err := m.Memory.AddSegment(mk(memN, 0.3), mk(memN, 0.7)); err != nil {
		b.Fatal(err)
	}
	x := mk(T, 0.1)
	ctx := backend.NewContext() // inference: Recorder == nil
	b.ResetTimer()
	for range b.N {
		if _, err := m.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMemForward_128(b *testing.B) { benchMemForward(b, 128, 512, 8, 2048, 32) }
func BenchmarkMemForward_512(b *testing.B) { benchMemForward(b, 512, 512, 8, 4096, 32) }
