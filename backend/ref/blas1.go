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
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
func dotKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: dot wants 2 inputs, got %d", len(in))
	}
	a, b := in[0], in[1]
	if err := sameShapeDtype(a, b); err != nil {
		return nil, err
	}
	var acc float64
	n := a.Numel()
	// Devirtualised traversal (§T646 follow-up): grab the raw typed slices once
	// and index flat instead of the per-element Unravel alloc + AtF64 dispatch.
	// Same ascending order, products and sum in float64 on both paths (§V10) —
	// bit-identical.
	switch a.Dtype() {
	case tensor.F64:
		as := a.Contiguous().Storage().F64()[:n]
		bs := b.Contiguous().Storage().F64()[:n]
		//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
		for i, av := range as {
			acc += av * bs[i]
		}
	case tensor.F32:
		as := a.Contiguous().Storage().F32()[:n]
		bs := b.Contiguous().Storage().F32()[:n]
		//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
		for i, av := range as {
			acc += float64(av) * float64(bs[i])
		}
	default:
		// Generic fallback for exotic dtypes (verbatim original loop).
		for pos := range n {
			idx := tensor.Unravel(pos, a.Shape())
			acc += a.AtF64(idx...) * b.AtF64(idx...)
		}
	}
	out := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{})
	out.SetF64(acc)
	return []*tensor.Tensor{out}, nil
}

// nrm2Kernel computes the Euclidean norm via the scaled LAPACK dnrm2 algorithm.
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
func nrm2Kernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: nrm2 wants 1 input, got %d", len(in))
	}
	x := in[0]
	scale, ssq := 0.0, 1.0
	// step folds one value into the running scaled sum-of-squares (LAPACK dnrm2
	// update, unchanged from the original loop body).
	step := func(v float64) {
		if v == 0 {
			return
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
	n := x.Numel()
	// Devirtualised traversal (§T646 follow-up): flat typed scan in the same
	// ascending order, update math untouched — bit-identical.
	switch x.Dtype() {
	case tensor.F64:
		for _, v := range x.Contiguous().Storage().F64()[:n] {
			step(v)
		}
	case tensor.F32:
		for _, v := range x.Contiguous().Storage().F32()[:n] {
			step(float64(v))
		}
	default:
		// Generic fallback for exotic dtypes (verbatim original loop).
		for pos := range n {
			step(x.AtF64(tensor.Unravel(pos, x.Shape())...))
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
	n := x.Numel()
	// Devirtualised traversal (§T646 follow-up): flat typed loop, alpha·x+y in
	// float64 on both paths; F32 rounds only the STORED result — bit-identical.
	switch x.Dtype() {
	case tensor.F64:
		xs := x.Contiguous().Storage().F64()[:n]
		ys := y.Contiguous().Storage().F64()[:n]
		os := out.Storage().F64()
		for i, xv := range xs {
			os[i] = alpha*xv + ys[i]
		}
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		xs := x.Contiguous().Storage().F32()[:n]
		ys := y.Contiguous().Storage().F32()[:n]
		os := out.Storage().F32()
		for i, xv := range xs {
			os[i] = float32(alpha*float64(xv) + float64(ys[i]))
		}
		return []*tensor.Tensor{out}, nil
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	for pos := range n {
		idx := tensor.Unravel(pos, x.Shape())
		out.SetF64(alpha*x.AtF64(idx...)+y.AtF64(idx...), idx...)
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	reg := func(op backend.Op, k backend.Kernel) {
		//perfscan:ignore PS3052 reference oracle: intentionally simple, correctness baseline not an optimization target
		std.add(op, tensor.F32, k)
		std.add(op, tensor.F64, k)
	}
	reg(backend.OpDot, dotKernel)
	reg(backend.OpNrm2, nrm2Kernel)
	reg(backend.OpAXPY, axpyKernel)
}
