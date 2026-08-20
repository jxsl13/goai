package cpu

import (
	"strconv"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkNegF32CPU measures the complete CPU operation boundary, including
// output allocation and the production parallel policy. These are the costs
// seen by the Metal host-route selector and by CPU model workloads.
func BenchmarkNegF32CPU(b *testing.B) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("CPU backend unavailable")
	}
	ctx := backend.NewContext().WithBackend(be)
	for _, n := range []int{2048, 65536, 131072, 349440, 524288, 2097152, 4194304, 8388608} {
		b.Run("n"+strconv.Itoa(n), func(b *testing.B) {
			x := bench.RandF32(tensor.Shape{n}, 43)
			in := []*tensor.Tensor{x}
			for range 20 {
				if _, err := backend.Execute(ctx, backend.OpNeg, in, nil); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := backend.Execute(ctx, backend.OpNeg, in, nil); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(2*n*4)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GB/s")
		})
	}
}

// BenchmarkNegF32ExecutionPolicy isolates the serial NEON stream versus the
// worker-pool path with the same preallocated output. It freezes the M2
// crossover independently from allocation and the different-cost Abs/ReLU
// kernels.
func BenchmarkNegF32ExecutionPolicy(b *testing.B) {
	for _, n := range []int{65536, 131072, 262144, 349440, 524288, 1048576} {
		b.Run("n"+strconv.Itoa(n), func(b *testing.B) {
			x := bench.RandF32(tensor.Shape{n}, 47)
			src := x.Storage().F32()
			for _, arm := range []struct {
				name string
				run  func([]float32)
			}{
				{name: "serial", run: func(dst []float32) { negF32(dst, src) }},
				{name: "parallel", run: func(dst []float32) {
					parallel(len(dst), func(lo, hi int) { negF32(dst[lo:hi], src[lo:hi]) })
				}},
			} {
				b.Run(arm.name, func(b *testing.B) {
					dst := make([]float32, n)
					for range 20 {
						arm.run(dst)
					}
					b.ReportAllocs()
					b.ResetTimer()
					for range b.N {
						arm.run(dst)
					}
					b.ReportMetric(float64(2*n*4)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GB/s")
				})
			}
		})
	}
}
