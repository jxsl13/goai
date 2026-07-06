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
		acc := make([]float64, m*n) // f64 accumulation (§V10)
		parallelWork(m, k*n, func(loRow, hiRow int) {
			gemmF32Band(A, B, acc, loRow, hiRow, k, n)
		})
		C := out.Storage().F32()
		for i := range C {
			C[i] = float32(acc[i])
		}
	default:
		return nil, fmt.Errorf("cpu: unsupported dtype %v", a.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

// gemmF64Band computes rows [loRow,hiRow) of C with 4-row register blocking
// (§T12b): each B row is loaded once and reused for four C rows, quartering
// B-traffic. Every C element still accumulates its k-products in ascending p
// order, so results stay bit-identical to the reference (§V3, §V11 tol 0).
func gemmF64Band(A, B, C []float64, loRow, hiRow, k, n int) {
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		c0 := C[(i+0)*n : (i+1)*n]
		c1 := C[(i+1)*n : (i+2)*n]
		c2 := C[(i+2)*n : (i+3)*n]
		c3 := C[(i+3)*n : (i+4)*n]
		for p := range k {
			bp := B[p*n : (p+1)*n]
			a0 := A[(i+0)*k+p]
			a1 := A[(i+1)*k+p]
			a2 := A[(i+2)*k+p]
			a3 := A[(i+3)*k+p]
			for j, bv := range bp {
				c0[j] += a0 * bv
				c1[j] += a1 * bv
				c2[j] += a2 * bv
				c3[j] += a3 * bv
			}
		}
	}
	for ; i < hiRow; i++ { // remainder rows
		ci := C[i*n : (i+1)*n]
		for p := range k {
			aip := A[i*k+p]
			bp := B[p*n : (p+1)*n]
			for j, bv := range bp {
				ci[j] += aip * bv
			}
		}
	}
}

// gemmF32Band is the F32 twin accumulating into an f64 scratch (§V10).
func gemmF32Band(A, B []float32, acc []float64, loRow, hiRow, k, n int) {
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		c0 := acc[(i+0)*n : (i+1)*n]
		c1 := acc[(i+1)*n : (i+2)*n]
		c2 := acc[(i+2)*n : (i+3)*n]
		c3 := acc[(i+3)*n : (i+4)*n]
		for p := range k {
			bp := B[p*n : (p+1)*n]
			a0 := float64(A[(i+0)*k+p])
			a1 := float64(A[(i+1)*k+p])
			a2 := float64(A[(i+2)*k+p])
			a3 := float64(A[(i+3)*k+p])
			for j, bv := range bp {
				bf := float64(bv)
				c0[j] += a0 * bf
				c1[j] += a1 * bf
				c2[j] += a2 * bf
				c3[j] += a3 * bf
			}
		}
	}
	for ; i < hiRow; i++ {
		ci := acc[i*n : (i+1)*n]
		for p := range k {
			aip := float64(A[i*k+p])
			bp := B[p*n : (p+1)*n]
			for j, bv := range bp {
				ci[j] += aip * float64(bv)
			}
		}
	}
}

func init() {
	std.add(backend.OpMatMul, tensor.F32, matmulKernel)
	std.add(backend.OpMatMul, tensor.F64, matmulKernel)
}
