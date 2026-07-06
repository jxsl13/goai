package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/simd"
	"github.com/jxsl13/goai/tensor"
)

// Optimized elementwise kernels (§T11). Binary ops use the internal/simd
// primitives on contiguous typed slices; unary ops use tight typed loops. Both
// parallelize above parThreshold. Non-contiguous inputs are materialized to
// contiguous first (still far cheaper than the reference's per-element Unravel).
// Results are bit-identical to backend/ref (§V3, §V11 tol 0) — same f64 math.

// binOp builds a binary kernel from the per-dtype simd primitives.
func binOp(f64 func(dst, a, b []float64), f32 func(dst, a, b []float32)) backend.Kernel {
	return func(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
		if len(in) != 2 {
			return nil, fmt.Errorf("cpu: binary op wants 2 inputs, got %d", len(in))
		}
		a, b := in[0], in[1]
		if !a.Shape().Equal(b.Shape()) {
			return nil, fmt.Errorf("cpu: binary shape mismatch %v vs %v", a.Shape(), b.Shape())
		}
		if a.Dtype() != b.Dtype() {
			return nil, fmt.Errorf("cpu: binary dtype mismatch %v vs %v", a.Dtype(), b.Dtype())
		}
		ac, bc := a.Contiguous(), b.Contiguous()
		out := tensor.NewOn(ctx.Device(), a.Dtype(), a.Shape())
		switch a.Dtype() {
		case tensor.F64:
			da, db, do := ac.Storage().F64(), bc.Storage().F64(), out.Storage().F64()
			parallel(len(do), func(lo, hi int) { f64(do[lo:hi], da[lo:hi], db[lo:hi]) })
		case tensor.F32:
			da, db, do := ac.Storage().F32(), bc.Storage().F32(), out.Storage().F32()
			parallel(len(do), func(lo, hi int) { f32(do[lo:hi], da[lo:hi], db[lo:hi]) })
		default:
			return nil, fmt.Errorf("cpu: unsupported dtype %v", a.Dtype())
		}
		return []*tensor.Tensor{out}, nil
	}
}

// unOp builds a unary kernel applying scalar f (computed in f64) over contiguous
// data, narrowing for F32.
func unOp(f func(float64) float64) backend.Kernel {
	return func(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
		if len(in) != 1 {
			return nil, fmt.Errorf("cpu: unary op wants 1 input, got %d", len(in))
		}
		xc := in[0].Contiguous()
		out := tensor.NewOn(ctx.Device(), in[0].Dtype(), in[0].Shape())
		switch in[0].Dtype() {
		case tensor.F64:
			d, o := xc.Storage().F64(), out.Storage().F64()
			parallel(len(o), func(lo, hi int) {
				for i := lo; i < hi; i++ {
					o[i] = f(d[i])
				}
			})
		case tensor.F32:
			d, o := xc.Storage().F32(), out.Storage().F32()
			parallel(len(o), func(lo, hi int) {
				for i := lo; i < hi; i++ {
					o[i] = float32(f(float64(d[i])))
				}
			})
		default:
			return nil, fmt.Errorf("cpu: unsupported dtype %v", in[0].Dtype())
		}
		return []*tensor.Tensor{out}, nil
	}
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}
func gelu(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	z := math.Exp(x)
	return z / (1 + z)
}

func init() {
	reg := func(op backend.Op, k backend.Kernel) {
		std.add(op, tensor.F32, k)
		std.add(op, tensor.F64, k)
	}
	reg(backend.OpAdd, binOp(simd.AddF64, simd.AddF32))
	reg(backend.OpSub, binOp(simd.SubF64, simd.SubF32))
	reg(backend.OpMul, binOp(simd.MulF64, simd.MulF32))
	reg(backend.OpDiv, binOp(simd.DivF64, simd.DivF32))

	reg(backend.OpNeg, unOp(func(x float64) float64 { return -x }))
	reg(backend.OpExp, unOp(math.Exp))
	reg(backend.OpLog, unOp(math.Log))
	reg(backend.OpTanh, unOp(math.Tanh))
	reg(backend.OpReLU, unOp(relu))
	reg(backend.OpGELU, unOp(gelu))
	reg(backend.OpSigmoid, unOp(sigmoid))
	reg(backend.OpSiLU, unOp(func(x float64) float64 { return x * sigmoid(x) }))
}
