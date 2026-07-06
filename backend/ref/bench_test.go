package ref_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref"
)

// Baseline benchmarks for the Pure-Go reference kernels (§V5). These establish
// the numbers the optimized SIMD/blocked kernels (§T11/§T12) must beat by the
// §C3 threshold before any cgo path is considered. Run: `make bench`.

func benchOp(b *testing.B, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) {
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, op, ins, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAddF64_4K(b *testing.B) {
	x := bench.RandF64(tensor.Shape{4096}, 1)
	y := bench.RandF64(tensor.Shape{4096}, 2)
	benchOp(b, backend.OpAdd, nil, x, y)
}

func BenchmarkAddF32_4K(b *testing.B) {
	x := bench.RandF32(tensor.Shape{4096}, 1)
	y := bench.RandF32(tensor.Shape{4096}, 2)
	benchOp(b, backend.OpAdd, nil, x, y)
}

func BenchmarkSumF64_64K(b *testing.B) {
	x := bench.RandF64(tensor.Shape{65536}, 3)
	benchOp(b, backend.OpSum, nil, x)
}

func BenchmarkMatMulF64_128(b *testing.B) {
	a := bench.RandF64(tensor.Shape{128, 128}, 1)
	c := bench.RandF64(tensor.Shape{128, 128}, 2)
	benchOp(b, backend.OpMatMul, nil, a, c)
}

func BenchmarkMatMulF32_128(b *testing.B) {
	a := bench.RandF32(tensor.Shape{128, 128}, 1)
	c := bench.RandF32(tensor.Shape{128, 128}, 2)
	benchOp(b, backend.OpMatMul, nil, a, c)
}
