package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Elementwise reference kernels (§T6). Each computes in float64 and stores
// through the tensor's dtype (narrowing for F32, ADR-0001) — the clear "truth"
// path (§V9); typed/SIMD speedups are §T11. Kernels are dtype-agnostic, so one
// implementation serves both F32 and F64 storage.

// unaryKernel applies scalar f elementwise, reading any layout and writing a
// fresh contiguous output on the context device.
func unaryKernel(f func(float64) float64) backend.Kernel {
	return func(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
		if len(in) != 1 {
			return nil, fmt.Errorf("ref: unary op wants 1 input, got %d", len(in))
		}
		x := in[0]
		out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
		n := x.Numel()
		for pos := range n {
			idx := tensor.Unravel(pos, x.Shape())
			out.SetF64(f(x.AtF64(idx...)), idx...)
		}
		return []*tensor.Tensor{out}, nil
	}
}

// binaryKernel applies scalar op elementwise over two same-shape, same-dtype
// inputs (no broadcasting yet — a later task).
func binaryKernel(op func(a, b float64) float64) backend.Kernel {
	return func(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
		if len(in) != 2 {
			return nil, fmt.Errorf("ref: binary op wants 2 inputs, got %d", len(in))
		}
		a, b := in[0], in[1]
		if !a.Shape().Equal(b.Shape()) {
			return nil, fmt.Errorf("ref: binary shape mismatch %v vs %v", a.Shape(), b.Shape())
		}
		if a.Dtype() != b.Dtype() {
			return nil, fmt.Errorf("ref: binary dtype mismatch %v vs %v", a.Dtype(), b.Dtype())
		}
		out := tensor.NewOn(ctx.Device(), a.Dtype(), a.Shape())
		n := a.Numel()
		for pos := range n {
			idx := tensor.Unravel(pos, a.Shape())
			out.SetF64(op(a.AtF64(idx...), b.AtF64(idx...)), idx...)
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

// gelu is the exact erf-based GELU (ADR-0004).
func gelu(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }

// sigmoid is numerically stable: it avoids exp overflow for large |x|.
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	z := math.Exp(x)
	return z / (1 + z)
}

func init() {
	// reg installs the same dtype-agnostic kernel for F32 and F64.
	reg := func(op backend.Op, k backend.Kernel) {
		std.add(op, tensor.F32, k)
		std.add(op, tensor.F64, k)
	}

	reg(backend.OpNeg, unaryKernel(func(x float64) float64 { return -x }))
	reg(backend.OpExp, unaryKernel(math.Exp))
	reg(backend.OpLog, unaryKernel(math.Log))
	reg(backend.OpTanh, unaryKernel(math.Tanh))
	reg(backend.OpReLU, unaryKernel(relu))
	reg(backend.OpGELU, unaryKernel(gelu))
	reg(backend.OpSigmoid, unaryKernel(sigmoid))
	reg(backend.OpSiLU, unaryKernel(func(x float64) float64 { return x * sigmoid(x) }))

	reg(backend.OpAdd, binaryKernel(func(a, b float64) float64 { return a + b }))
	reg(backend.OpSub, binaryKernel(func(a, b float64) float64 { return a - b }))
	reg(backend.OpMul, binaryKernel(func(a, b float64) float64 { return a * b }))
	reg(backend.OpDiv, binaryKernel(func(a, b float64) float64 { return a / b }))
}
