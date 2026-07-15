package cpu_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkConv2DTorchShapes times the cpu conv at the two torch-compared f32
// shapes from internal/benchcompare (python_compare's Conv2D rows), reporting
// GFLOP/s so the ratio vs torch-cpu (~606 GFLOP/s) reads directly.
func BenchmarkConv2DTorchShapes(b *testing.B) {
	cpuB, _ := backend.Get(backend.CPU)
	cases := []struct {
		name           string
		n, c, hw, f, k int
	}{
		{"n8c16hw32", 8, 16, 32, 32, 3},
		{"n8c64hw56", 8, 64, 56, 64, 3},
	}
	for _, cs := range cases {
		x := bench.RandF32(tensor.Shape{cs.n, cs.c, cs.hw, cs.hw}, 1)
		w := bench.RandF32(tensor.Shape{cs.f, cs.c, cs.k, cs.k}, 2)
		attrs := backend.ConvAttrs{Stride: 1, Pad: 1}
		ins := []*tensor.Tensor{x, w}
		flop := 2 * float64(cs.n) * float64(cs.f) * float64(cs.hw) * float64(cs.hw) *
			float64(cs.c) * float64(cs.k) * float64(cs.k)
		b.Run(fmt.Sprintf("%s/cpu", cs.name), func(b *testing.B) {
			ctx := backend.NewContext().WithBackend(cpuB)
			for b.Loop() {
				if _, err := backend.Execute(ctx, backend.OpConv2D, ins, attrs); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(flop*float64(b.N)/float64(b.Elapsed().Nanoseconds()), "GFLOP/s")
		})
	}
}
