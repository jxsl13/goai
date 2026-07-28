package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

func TestDistillCPUMatchesRef(t *testing.T) {
	cpuBe, _ := backend.Get(backend.CPU)
	refBe, _ := backend.Get(backend.Ref)
	for _, shape := range []tensor.Shape{{8, 100}, {64, 1000}, {3, 7}, {1, 2}} {
		for _, temp := range []float64{1, 2, 0.5} {
			zs := bench.RandF64(shape, 1)
			zt := bench.RandF64(shape, 2)
			attrs := backend.DistillAttrs{Temperature: temp}
			gt, err := backend.Execute(backend.NewContext().WithBackend(cpuBe), backend.OpDistill, []*tensor.Tensor{zs, zt}, attrs)
			if err != nil {
				t.Fatal(err)
			}
			wt, err := backend.Execute(backend.NewContext().WithBackend(refBe), backend.OpDistill, []*tensor.Tensor{zs, zt}, attrs)
			if err != nil {
				t.Fatal(err)
			}
			got, want := gt[0].AtF64(), wt[0].AtF64()
			rel := math.Abs(got-want) / math.Max(1, math.Abs(want))
			if rel > 1e-9 {
				t.Fatalf("shape=%v temp=%g: cpu %.17g != ref %.17g (rel %.2e)", shape, temp, got, want, rel)
			}
		}
	}
}

func BenchmarkDistillF64_64x1K_cpu(b *testing.B) {
	benchOn(b, backend.CPU, backend.OpDistill, bench.RandF64(tensor.Shape{64, 1000}, 1), bench.RandF64(tensor.Shape{64, 1000}, 2))
}
