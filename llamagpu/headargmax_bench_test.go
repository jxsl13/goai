package llamagpu

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// benchHeadArgmax times the host-side Medusa head projection. It is called once per head per
// speculative round (K heads), and a round emits only a handful of tokens, so its cost lands
// directly on decode latency.
//
// The weight is [dim,vocab] at realistic model sizes. That matters for what the benchmark can see:
// at 4096x32000 the matrix is 512 MB, so a walk that touches one cache line per element pulls far
// more than the data it uses, and a smaller cell would hide that entirely
// (§SIZE-THE-CELL-PAST-L1-BEFORE-JUDGING-LAYOUT).
func benchHeadArgmax(b *testing.B, dim, vocab int) {
	w := tensor.New(tensor.F32, tensor.Shape{dim, vocab})
	ws := w.Storage().F32()
	for i := range ws {
		ws[i] = float32(math.Sin(float64(i)*0.001)) * 0.05
	}
	hidden := make([]float32, dim)
	for i := range hidden {
		hidden[i] = float32(math.Cos(float64(i)*0.01)) * 0.1
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if headArgmax(hidden, w) < 0 {
			b.Fatal("negative index")
		}
	}
}

func BenchmarkHeadArgmax_2048x8000(b *testing.B) { benchHeadArgmax(b, 2048, 8000) }
func BenchmarkHeadArgmax_512x2000(b *testing.B)  { benchHeadArgmax(b, 512, 2000) }
