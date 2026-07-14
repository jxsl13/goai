package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// The archsimd GEMM band kernels (gemm_simd.go) vectorize the free dimension j
// in 4-wide f64 lanes with a scalar column-tail for n%4 columns and a scalar
// row-remainder for the rows past the last 4-row block. This locks EVERY column
// residue (0..3) and row residue against the reference, bit-exactly (tol 0) —
// under GOEXPERIMENT=simd it exercises the SIMD tails; otherwise the scalar
// kernel. Complements TestGemmCrossReferenceExact with residues chosen to hit
// each vector-body/tail split (n=5,6,7,9 and non-4-multiple m), and a larger K
// where the ascending-p accumulation order is what makes tol-0 possible.
func TestGemmSimdTailResiduesExact(t *testing.T) {
	cpu, _ := backend.Get(backend.CPU)
	ref, _ := backend.Get(backend.Ref)

	shapes := []struct{ m, k, n int }{
		{1, 5, 5}, {3, 8, 6}, {6, 33, 7}, {7, 4, 9}, {9, 17, 13},
		{10, 128, 11}, {17, 3, 4}, {4, 1, 15},
	}
	for _, s := range shapes {
		for _, dtype := range []tensor.Dtype{tensor.F64, tensor.F32} {
			var a, b *tensor.Tensor
			if dtype == tensor.F64 {
				a = bench.RandF64(tensor.Shape{s.m, s.k}, 11)
				b = bench.RandF64(tensor.Shape{s.k, s.n}, 22)
			} else {
				a = bench.RandF32(tensor.Shape{s.m, s.k}, 11)
				b = bench.RandF32(tensor.Shape{s.k, s.n}, 22)
			}
			gc := run(t, cpu, backend.OpMatMul, a, b)
			gr := run(t, ref, backend.OpMatMul, a, b)
			assertMatMul(t, gc, gr, "matmul-simd-tail")
		}
	}
}
