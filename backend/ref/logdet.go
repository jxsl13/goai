package ref

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// logdetKernel computes the natural log of the determinant of a symmetric
// positive-definite matrix A via its Cholesky factor: det(A) = det(L)² = (∏ L_ii)²
// so log det(A) = 2·Σ ln(L_ii) (numpy.linalg.slogdet returns sign=+1 and this value
// for SPD input). Computing through L is both stable (no overflow from a direct
// product) and cheap. The result is a scalar; the SPD/shape errors come from the
// shared Cholesky factorization. F32 input factors in f64 and returns an f32 scalar
// (§V10).
func logdetKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	lout, err := choleskyKernel(ctx, in, nil) // validates square + SPD, reads lower triangle
	if err != nil {
		return nil, err
	}
	l := lout[0]
	n := l.Shape()[0]
	var acc float64 // f64 accumulation (§V10)
	//perfscan:ignore PS1001 reference oracle: intentionally simple, correctness baseline not an optimization target
	for i := range n {
		acc += math.Log(l.AtF64(i, i))
	}
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), tensor.Shape{})
	out.SetF64(2 * acc)
	return []*tensor.Tensor{out}, nil
}

func init() {
	//perfscan:ignore PS3062 reference oracle: intentionally simple, correctness baseline not an optimization target
	std.add(backend.OpLogDet, tensor.F32, logdetKernel)
	std.add(backend.OpLogDet, tensor.F64, logdetKernel)
}
