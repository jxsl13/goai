package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// F64 matmul with a TRANSPOSED left operand (the backward dW=Xᵀ·dO / dX=dO·Wᵀ shape): the operand is a
// strided view that matmul must materialize before the GEMM. Measures the transpose-gather cost in
// front of the parallel F64 GEMM.
func benchMatMulF64Transposed(b *testing.B, m, k, n int) {
	be, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(be)
	a := bench.RandF64(tensor.Shape{k, m}, 1) // [k,m]; transpose → [m,k]
	at, _ := a.Transpose(0, 1)
	bb := bench.RandF64(tensor.Shape{k, n}, 2)
	ins := []*tensor.Tensor{at, bb}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMatMul, ins, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMulF64T_2048x2048x256(b *testing.B) { benchMatMulF64Transposed(b, 2048, 2048, 256) }
func BenchmarkMatMulF64T_4096x4096x128(b *testing.B) { benchMatMulF64Transposed(b, 4096, 4096, 128) }
