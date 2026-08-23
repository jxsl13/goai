package cpu

import (
	"strconv"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkAbsF32CPU measures the complete CPU operation boundary, including
// output allocation and the production parallel policy. These are the costs
// seen by the Metal host-route selector and by CPU model workloads.
func BenchmarkAbsF32CPU(b *testing.B) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("CPU backend unavailable")
	}
	ctx := backend.NewContext().WithBackend(be)
	for _, n := range []int{2048, 65536, 131072, 349440, 524288, 2097152, 4194304, 8388608} {
		b.Run("n"+strconv.Itoa(n), func(b *testing.B) {
			x := bench.RandF32(tensor.Shape{n}, 37)
			in := []*tensor.Tensor{x}
			for range 20 {
				if _, err := backend.Execute(ctx, backend.OpAbs, in, nil); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := backend.Execute(ctx, backend.OpAbs, in, nil); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(2*n*4)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GB/s")
		})
	}
}

// BenchmarkAbsF32Leaf isolates the preallocated kernel from output allocation,
// dispatch, and the production parallel policy measured by BenchmarkAbsF32CPU.
func BenchmarkAbsF32Leaf(b *testing.B) {
	for _, n := range []int{2048, 65536, 349440, 8388608} {
		b.Run("n"+strconv.Itoa(n), func(b *testing.B) {
			src := make([]float32, n)
			dst := make([]float32, n)
			for i := range src {
				src[i] = float32(i&1023) - 512
			}
			b.SetBytes(int64(2 * n * 4))
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				absF32(dst, src)
			}
		})
	}
}
