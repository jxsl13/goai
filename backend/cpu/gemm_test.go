package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V3/§V11 CROSS: ikj-order GEMM keeps the reference's k accumulation order, so
// the optimized result is bit-identical to backend/ref (tolerance 0) across
// square, non-square, and parallel (large-M) shapes and both dtypes.
func TestGemmCrossReferenceExact(t *testing.T) {
	cpu, _ := backend.Get(backend.CPU)
	ref, _ := backend.Get(backend.Ref)

	shapes := []struct{ m, k, n int }{
		{1, 1, 1}, {2, 3, 4}, {5, 7, 3}, {64, 64, 64}, {128, 96, 32}, {200, 10, 7},
	}
	for _, s := range shapes {
		for _, dtype := range []tensor.Dtype{tensor.F64, tensor.F32} {
			var a, b *tensor.Tensor
			if dtype == tensor.F64 {
				a = bench.RandF64(tensor.Shape{s.m, s.k}, 1)
				b = bench.RandF64(tensor.Shape{s.k, s.n}, 2)
			} else {
				a = bench.RandF32(tensor.Shape{s.m, s.k}, 1)
				b = bench.RandF32(tensor.Shape{s.k, s.n}, 2)
			}
			gc := run(t, cpu, backend.OpMatMul, a, b)
			gr := run(t, ref, backend.OpMatMul, a, b)
			assertEqualExact(t, gc, gr, "matmul")
		}
	}
}

// Transposed-view operand: materialized then bit-identical to ref.
func TestGemmTransposedViewCross(t *testing.T) {
	cpu, _ := backend.Get(backend.CPU)
	ref, _ := backend.Get(backend.Ref)
	a := bench.RandF64(tensor.Shape{8, 5}, 3)
	bt := bench.RandF64(tensor.Shape{7, 5}, 4)
	btv, _ := bt.Transpose(0, 1) // (5,7)
	gc := run(t, cpu, backend.OpMatMul, a, btv)
	gr := run(t, ref, backend.OpMatMul, a, btv)
	assertEqualExact(t, gc, gr, "matmul-transposed")
}

func benchMatMul(b *testing.B, name backend.Name, dtype tensor.Dtype, sz int) {
	be, _ := backend.Get(name)
	ctx := backend.NewContext().WithBackend(be)
	var a, c *tensor.Tensor
	if dtype == tensor.F64 {
		a, c = bench.RandF64(tensor.Shape{sz, sz}, 1), bench.RandF64(tensor.Shape{sz, sz}, 2)
	} else {
		a, c = bench.RandF32(tensor.Shape{sz, sz}, 1), bench.RandF32(tensor.Shape{sz, sz}, 2)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{a, c}, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMulF64_128_cpu(b *testing.B)  { benchMatMul(b, backend.CPU, tensor.F64, 128) }
func BenchmarkMatMulF64_256_cpu(b *testing.B)  { benchMatMul(b, backend.CPU, tensor.F64, 256) }
func BenchmarkMatMulF64_512_cpu(b *testing.B)  { benchMatMul(b, backend.CPU, tensor.F64, 512) }
func BenchmarkMatMulF64_1024_cpu(b *testing.B) { benchMatMul(b, backend.CPU, tensor.F64, 1024) }
func BenchmarkMatMulF32_128_cpu(b *testing.B)  { benchMatMul(b, backend.CPU, tensor.F32, 128) }
func BenchmarkMatMulF32_1024_cpu(b *testing.B) { benchMatMul(b, backend.CPU, tensor.F32, 1024) }
