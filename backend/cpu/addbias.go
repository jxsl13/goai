package cpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/simd"
	"github.com/jxsl13/goai/tensor"
)

// Optimized row-broadcast bias add: out[i,j] = x[i,j] + b[j]. Each row is one
// simd.Add over contiguous slices (AVX 8/4-lane under GOEXPERIMENT=simd, plain
// loop otherwise), parallel over rows. Bit-identical to ref (§V3): a SIMD lane
// add is the same IEEE-754 add, and every out[i,j] depends on exactly one
// (x[i,j], b[j]) pair, so lane order cannot change results.
func addBiasKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("cpu: addbias wants 2 inputs, got %d", len(in))
	}
	x, b := in[0], in[1]
	if x.Ndim() != 2 || b.Ndim() != 1 {
		return nil, fmt.Errorf("cpu: addbias needs x rank-2 and b rank-1, got %dD, %dD", x.Ndim(), b.Ndim())
	}
	if x.Dtype() != b.Dtype() {
		return nil, fmt.Errorf("cpu: addbias dtype mismatch %v vs %v", x.Dtype(), b.Dtype())
	}
	m, n := x.Shape()[0], x.Shape()[1]
	if b.Shape()[0] != n {
		return nil, fmt.Errorf("cpu: addbias bias len %d != cols %d", b.Shape()[0], n)
	}
	xc, bc := x.Contiguous(), b.Contiguous()
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())

	switch x.Dtype() {
	case tensor.F64:
		X, B, O := xc.Storage().F64(), bc.Storage().F64(), out.Storage().F64()
		parallelWork(m, n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				simd.AddF64(O[i*n:(i+1)*n], X[i*n:(i+1)*n], B)
			}
		})
	case tensor.F32:
		X, B, O := xc.Storage().F32(), bc.Storage().F32(), out.Storage().F32()
		parallelWork(m, n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				simd.AddF32(O[i*n:(i+1)*n], X[i*n:(i+1)*n], B)
			}
		})
	default:
		return nil, fmt.Errorf("cpu: unsupported dtype %v", x.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpAddBias, tensor.F32, addBiasKernel)
	std.add(backend.OpAddBias, tensor.F64, addBiasKernel)
}
