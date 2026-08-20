package cpu

import (
	"strconv"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkReLUF32CPU measures the complete CPU operation boundary, including
// output allocation and the production parallel policy. Keeping that boundary
// in the benchmark prevents a fast private leaf from hiding allocation or
// dispatch costs that determine the Metal host-route crossover.
func BenchmarkReLUF32CPU(b *testing.B) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("CPU backend unavailable")
	}
	ctx := backend.NewContext().WithBackend(be)
	for _, n := range []int{2048, 65536, 131072, 349440, 524288, 2097152, 4194304} {
		b.Run("n"+strconv.Itoa(n), func(b *testing.B) {
			x := bench.RandF32(tensor.Shape{n}, 31)
			in := []*tensor.Tensor{x}
			for range 20 {
				if _, err := backend.Execute(ctx, backend.OpReLU, in, nil); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := backend.Execute(ctx, backend.OpReLU, in, nil); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(2*n*4)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GB/s")
		})
	}
}
