package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// matmulKernel is the scalar reference GEMM (§T9): C[M,N] = A[M,K] · B[K,N],
// row-major, with the inner product accumulated in float64 (§V10). It reads
// inputs through AtF64, so transposed/sliced views work unchanged (no trans
// flags needed). This is the truth against which the SIMD/blocked GEMM (§T12) is
// validated (§V3, §V9) — clarity over speed.
func matmulKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: matmul wants 2 inputs, got %d", len(in))
	}
	a, b := in[0], in[1]
	if a.Ndim() != 2 || b.Ndim() != 2 {
		return nil, fmt.Errorf("ref: matmul needs rank-2 inputs, got %dD and %dD", a.Ndim(), b.Ndim())
	}
	if a.Dtype() != b.Dtype() {
		return nil, fmt.Errorf("ref: matmul dtype mismatch %v vs %v", a.Dtype(), b.Dtype())
	}
	m, k := a.Shape()[0], a.Shape()[1]
	k2, n := b.Shape()[0], b.Shape()[1]
	if k != k2 {
		return nil, fmt.Errorf("ref: matmul inner dim mismatch %v · %v", a.Shape(), b.Shape())
	}

	out := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{m, n})
	for i := range m {
		for j := range n {
			var acc float64
			for p := range k {
				acc += a.AtF64(i, p) * b.AtF64(p, j)
			}
			out.SetF64(acc, i, j)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpMatMul, tensor.F32, matmulKernel)
	std.add(backend.OpMatMul, tensor.F64, matmulKernel)
}
