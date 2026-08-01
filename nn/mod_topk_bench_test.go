package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// topKIndices is MoD's per-layer router selection: full-column score sort → top-k
// positions. Benched at sequence lengths where the O(seq·log seq) sort matters.
func benchTopKIndices(b *testing.B, seq, k int) {
	col := tensor.New(tensor.F64, tensor.Shape{seq, 1})
	s := col.Storage().F64()
	for i := range s {
		s[i] = math.Sin(float64(i)*0.7) + 0.001*float64(i%7) // varied scores, occasional ties
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = topKIndices(col, k)
	}
}

func BenchmarkTopKIndices_512x256(b *testing.B)  { benchTopKIndices(b, 512, 256) }
func BenchmarkTopKIndices_2048x512(b *testing.B) { benchTopKIndices(b, 2048, 512) }
func BenchmarkTopKIndices_4096x256(b *testing.B) { benchTopKIndices(b, 4096, 256) }
