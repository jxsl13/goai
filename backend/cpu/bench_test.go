package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// cpu benchmarks, to compare against the ref baselines (§V5). Same shapes/seeds
// as backend/ref/bench_test.go so the delta is apples-to-apples.

func benchOn(b *testing.B, name string, op backend.Op, ins ...*tensor.Tensor) {
	be, _ := backend.Get(name)
	ctx := backend.NewContext().WithBackend(be)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, op, ins, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAddF64_4K_cpu(b *testing.B) {
	benchOn(b, "cpu", backend.OpAdd, bench.RandF64(tensor.Shape{4096}, 1), bench.RandF64(tensor.Shape{4096}, 2))
}
func BenchmarkAddF32_4K_cpu(b *testing.B) {
	benchOn(b, "cpu", backend.OpAdd, bench.RandF32(tensor.Shape{4096}, 1), bench.RandF32(tensor.Shape{4096}, 2))
}
func BenchmarkExpF64_64K_cpu(b *testing.B) {
	benchOn(b, "cpu", backend.OpExp, bench.RandF64(tensor.Shape{65536}, 3))
}
func BenchmarkMulF64_256K_cpu(b *testing.B) { // exercises the parallel path
	benchOn(b, "cpu", backend.OpMul, bench.RandF64(tensor.Shape{262144}, 1), bench.RandF64(tensor.Shape{262144}, 2))
}
