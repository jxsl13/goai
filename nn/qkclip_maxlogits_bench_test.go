package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// MaxAttentionLogits is the QK-Clip probe: per head, the max attention logit over all (query,
// key) pairs. It is O(heads*seq^2*dk) and independent per head. Realistic MHA probe shape.
func benchMaxAttentionLogits(b *testing.B, seq, dm, heads int, causal bool) {
	mk := func(f func(i int) float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{seq, dm})
		s := t.Storage().F64()
		for i := range s {
			s[i] = f(i)
		}
		return t
	}
	q := mk(func(i int) float64 { return math.Sin(float64(i) * 0.01) })
	k := mk(func(i int) float64 { return math.Cos(float64(i) * 0.013) })
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if out, err := nn.MaxAttentionLogits(q, k, heads, 0.125, causal); err != nil || len(out) != heads {
			b.Fatalf("err=%v", err)
		}
	}
}

func BenchmarkMaxAttentionLogits_512x512x8(b *testing.B) {
	benchMaxAttentionLogits(b, 512, 512, 8, true)
}
func BenchmarkMaxAttentionLogits_256x256x4(b *testing.B) {
	benchMaxAttentionLogits(b, 256, 256, 4, false)
}
