package cpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Optimized GEMM (§T12). C[M,N] = A[M,K]·B[K,N] on raw contiguous slices using
// the ikj loop order: for fixed (i,j) the k updates run 0..K-1 in the same order
// as the reference, so the result is bit-identical to backend/ref (§V3, §V11
// tol 0) — no summation reorder. ikj also gives row-wise access to B and C (the
// inner j-loop is a vectorizable SAXPY) and parallelizes cleanly over disjoint C
// row-bands. f32 accumulates in an f64 scratch matrix, narrowing once (§V10).
//
// §T74/§B28: BLIS-style kc×nc cache blocking with a packed B-panel was built and
// measured against this kernel on the arm64 (Apple M-series) host and DISCARDED —
// it tied on f64-512, regressed f32-1024 (~-9%), and its one marginal f64-1024
// median (~12%) sat inside the run-to-run noise while costing ~40% more memory.
// The M-series memory subsystem already feeds this streaming kernel near-optimally,
// so panel packing is net overhead here. The real remaining headroom is wider-FMA
// SIMD, which is host-blocked on arm64 (cf §T11b/§B13). Resume blocking only on a
// host whose cache hierarchy actually bottlenecks the unblocked stream (large-cache
// x86 server), re-measuring before merge.

func matmulKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("cpu: matmul wants 2 inputs, got %d", len(in))
	}
	a, b := in[0], in[1]
	if a.Ndim() != 2 || b.Ndim() != 2 {
		return nil, fmt.Errorf("cpu: matmul needs rank-2, got %dD and %dD", a.Ndim(), b.Ndim())
	}
	if a.Dtype() != b.Dtype() {
		return nil, fmt.Errorf("cpu: matmul dtype mismatch %v vs %v", a.Dtype(), b.Dtype())
	}
	m, k := a.Shape()[0], a.Shape()[1]
	k2, n := b.Shape()[0], b.Shape()[1]
	if k != k2 {
		return nil, fmt.Errorf("cpu: matmul inner dim mismatch %v · %v", a.Shape(), b.Shape())
	}
	ac, bc := a.Contiguous(), b.Contiguous()
	out := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{m, n})

	switch a.Dtype() {
	case tensor.F64:
		A, B, C := ac.Storage().F64(), bc.Storage().F64(), out.Storage().F64()
		parallelWork(m, k*n, func(loRow, hiRow int) {
			gemmF64Band(A, B, C, loRow, hiRow, k, n)
		})
	case tensor.F32:
		A, B := ac.Storage().F32(), bc.Storage().F32()
		C := out.Storage().F32()
		gemmF32(A, B, C, m, k, n)
	default:
		return nil, fmt.Errorf("cpu: unsupported dtype %v", a.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpMatMul, tensor.F32, matmulKernel)
	std.add(backend.OpMatMul, tensor.F64, matmulKernel)
}
