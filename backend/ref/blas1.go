package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// BLAS-1 reference kernels (§T8). dot and nrm2 accumulate in float64 (§V10).
// nrm2 uses the classic scaled algorithm (LAPACK dnrm2) so it does not overflow
// for large magnitudes nor underflow for small ones — the scientifically correct
// reference (§G4, §V9). Inputs share shape and dtype; results are scalars (dot,
// nrm2) or elementwise (axpy).

func sameShapeDtype(a, b *tensor.Tensor) error {
	if !a.Shape().Equal(b.Shape()) {
		return fmt.Errorf("ref: shape mismatch %v vs %v", a.Shape(), b.Shape())
	}
	if a.Dtype() != b.Dtype() {
		return fmt.Errorf("ref: dtype mismatch %v vs %v", a.Dtype(), b.Dtype())
	}
	return nil
}

// dotKernel computes the full inner product sum(a_i*b_i) over all elements.
func dotKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: dot wants 2 inputs, got %d", len(in))
	}
	a, b := in[0], in[1]
	if err := sameShapeDtype(a, b); err != nil {
		return nil, err
	}
	var acc float64
	for pos := range a.Numel() {
		idx := tensor.Unravel(pos, a.Shape())
		acc += a.AtF64(idx...) * b.AtF64(idx...)
	}
	out := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{})
	out.SetF64(acc)
	return []*tensor.Tensor{out}, nil
}

// nrm2Kernel computes the Euclidean norm via the scaled LAPACK dnrm2 algorithm.
func nrm2Kernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: nrm2 wants 1 input, got %d", len(in))
	}
	x := in[0]
	scale, ssq := 0.0, 1.0
	for pos := range x.Numel() {
		v := x.AtF64(tensor.Unravel(pos, x.Shape())...)
		if v == 0 {
			continue
		}
		a := math.Abs(v)
		if scale < a {
			r := scale / a
			ssq = 1 + ssq*r*r
			scale = a
		} else {
			r := a / scale
			ssq += r * r
		}
	}
	out := tensor.NewOn(ctx.Device(), x.Dtype(), tensor.Shape{})
	out.SetF64(scale * math.Sqrt(ssq))
	return []*tensor.Tensor{out}, nil
}

// axpyKernel computes alpha*x + y elementwise (BLAS axpy, functional form).
func axpyKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: axpy wants 2 inputs, got %d", len(in))
	}
	x, y := in[0], in[1]
	if err := sameShapeDtype(x, y); err != nil {
		return nil, err
	}
	pa, _ := attrs.(backend.AXPYAttrs)
	pa = pa.WithDefaults()
	alpha := pa.Alpha
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	for pos := range x.Numel() {
		idx := tensor.Unravel(pos, x.Shape())
		out.SetF64(alpha*x.AtF64(idx...)+y.AtF64(idx...), idx...)
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	reg := func(op backend.Op, k backend.Kernel) {
		std.add(op, tensor.F32, k)
		std.add(op, tensor.F64, k)
	}
	reg(backend.OpDot, dotKernel)
	reg(backend.OpNrm2, nrm2Kernel)
	reg(backend.OpAXPY, axpyKernel)
}
