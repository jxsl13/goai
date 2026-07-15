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
	// Devirtualised typed core (§T646 follow-up): expose A and B once as flat
	// row-major []float64 (exact widening for F32) instead of paying the AtF64
	// dtype dispatch per element of the O(m·n·k) triple loop. B is pre-transposed
	// so the p-inner dot product runs over two contiguous rows (pure data
	// movement — each C[i,j] still accumulates a[i,p]·b[p,j] in the SAME
	// ascending-p order, so results are bit-identical to the generic path).
	if as, aok := f64Data(a); aok {
		if bs, bok := f64Data(b); bok {
			if os, flush, ook := outF64(out); ook {
				bt := make([]float64, n*k)
				for p := 0; p < k; p++ {
					prow := bs[p*n : p*n+n]
					for j, v := range prow {
						bt[j*k+p] = v
					}
				}
				for i := 0; i < m; i++ {
					arow := as[i*k : i*k+k]
					orow := os[i*n : i*n+n]
					for j := 0; j < n; j++ {
						brow := bt[j*k : j*k+k]
						var acc float64
						for p, av := range arow {
							acc += av * brow[p]
						}
						orow[j] = acc
					}
				}
				flush()
				return []*tensor.Tensor{out}, nil
			}
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
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
