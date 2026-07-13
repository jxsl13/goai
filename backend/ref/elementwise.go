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
		if a.Dtype() != b.Dtype() {
			return nil, fmt.Errorf("ref: binary dtype mismatch %v vs %v", a.Dtype(), b.Dtype())
		}
		if a.Shape().Equal(b.Shape()) { // same-shape fast path (unchanged)
			out := tensor.NewOn(ctx.Device(), a.Dtype(), a.Shape())
			for pos := range a.Numel() {
				idx := tensor.Unravel(pos, a.Shape())
				out.SetF64(op(a.AtF64(idx...), b.AtF64(idx...)), idx...)
			}
			return []*tensor.Tensor{out}, nil
		}
		// broadcasting path (numpy rules)
		outShape, err := backend.BroadcastShape(a.Shape(), b.Shape())
		if err != nil {
			return nil, err
		}
		offA, offB := len(outShape)-a.Ndim(), len(outShape)-b.Ndim()
		ac, bc := make([]int, a.Ndim()), make([]int, b.Ndim())
		out := tensor.NewOn(ctx.Device(), a.Dtype(), outShape)
		for pos := range out.Numel() {
			oc := tensor.Unravel(pos, outShape)
			backend.BroadcastCoords(ac, oc, a.Shape(), offA)
			backend.BroadcastCoords(bc, oc, b.Shape(), offB)
			out.SetF64(op(a.AtF64(ac...), b.AtF64(bc...)), oc...)
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

// geluGrad is the exact GELU derivative g·(Φ(x)+x·φ(x)), Φ=0.5(1+erf(x/√2)),
// φ=(1/√2π)exp(−x²/2) — matches the autograd VJP (§T353).
const refInvSqrt2Pi = 0.3989422804014327 // 1/√(2π)
func geluGrad(x, g float64) float64 {
	phi := 0.5 * (1 + math.Erf(x/math.Sqrt2))
	pdf := refInvSqrt2Pi * math.Exp(-0.5*x*x)
	return g * (phi + x*pdf)
}

// siluBackwardKernel computes dx = g·silu'(x), silu'(x)=σ(x)(1+x(1−σ(x))), elementwise
// (in = [x, g]); the GPU backends dispatch OpSiLUBackward, falling back here (§T362/§I4).
func siluBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: silu-backward wants (x, g), got %d", len(in))
	}
	x, g := in[0], in[1]
	if !g.Shape().Equal(x.Shape()) {
		return nil, fmt.Errorf("ref: silu-backward g %v != x %v", g.Shape(), x.Shape())
	}
	dx := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	for i := range x.Numel() {
		idx := tensor.Unravel(i, x.Shape())
		xv := x.AtF64(idx...)
		s := sigmoid(xv)
		dx.SetF64(g.AtF64(idx...)*s*(1+xv*(1-s)), idx...)
	}
	return []*tensor.Tensor{dx}, nil
}

// geluBackwardKernel computes dx = g·gelu'(x) elementwise (in = [x, g]); the GPU
// backends dispatch OpGELUBackward, falling back here (§T353/§I4).
func geluBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: gelu-backward wants (x, g), got %d", len(in))
	}
	x, g := in[0], in[1]
	if !g.Shape().Equal(x.Shape()) {
		return nil, fmt.Errorf("ref: gelu-backward g %v != x %v", g.Shape(), x.Shape())
	}
	dx := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	for i := range x.Numel() {
		idx := tensor.Unravel(i, x.Shape())
		dx.SetF64(geluGrad(x.AtF64(idx...), g.AtF64(idx...)), idx...)
	}
	return []*tensor.Tensor{dx}, nil
}

// sigmoid is numerically stable: it avoids exp overflow for large |x|.
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	z := math.Exp(x)
	return z / (1 + z)
}

// clipKernel clamps each element to [Lo,Hi] (numpy.clip); Lo>Hi is an error.
func clipKernel(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: clip wants 1 input, got %d", len(in))
	}
	pa, _ := attrs.(backend.ClipAttrs)
	if pa.Lo > pa.Hi {
		return nil, fmt.Errorf("ref: clip Lo %g > Hi %g", pa.Lo, pa.Hi)
	}
	x := in[0]
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	for pos := range x.Numel() {
		idx := tensor.Unravel(pos, x.Shape())
		v := x.AtF64(idx...)
		out.SetF64(math.Max(pa.Lo, math.Min(v, pa.Hi)), idx...)
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	// reg installs the same dtype-agnostic kernel for F32 and F64.
	reg := func(op backend.Op, k backend.Kernel) {
		std.add(op, tensor.F32, k)
		std.add(op, tensor.F64, k)
	}

	reg(backend.OpStopGradient, unaryKernel(func(x float64) float64 { return x })) // detach: identity forward
	reg(backend.OpNeg, unaryKernel(func(x float64) float64 { return -x }))
	reg(backend.OpExp, unaryKernel(math.Exp))
	reg(backend.OpLog, unaryKernel(math.Log))
	reg(backend.OpTanh, unaryKernel(math.Tanh))
	reg(backend.OpReLU, unaryKernel(relu))
	reg(backend.OpGELU, unaryKernel(gelu))
	reg(backend.OpGELUBackward, geluBackwardKernel)
	reg(backend.OpSiLUBackward, siluBackwardKernel)
	reg(backend.OpSigmoid, unaryKernel(sigmoid))
	reg(backend.OpSiLU, unaryKernel(func(x float64) float64 { return x * sigmoid(x) }))
	reg(backend.OpSqrt, unaryKernel(math.Sqrt))
	reg(backend.OpAbs, unaryKernel(math.Abs))
	reg(backend.OpClip, clipKernel)

	reg(backend.OpAdd, binaryKernel(func(a, b float64) float64 { return a + b }))
	reg(backend.OpSub, binaryKernel(func(a, b float64) float64 { return a - b }))
	reg(backend.OpMul, binaryKernel(func(a, b float64) float64 { return a * b }))
	reg(backend.OpDiv, binaryKernel(func(a, b float64) float64 { return a / b }))
	reg(backend.OpMaximum, binaryKernel(math.Max))
	reg(backend.OpMinimum, binaryKernel(math.Min))
	reg(backend.OpWhere, whereKernel)
}

// whereKernel selects elementwise cond?a:b (numpy.where): out[i] = a[i] if the
// condition is nonzero (true) else b[i]. cond, a, b share a shape; a and b share a
// dtype (the output's); cond may be any dtype (read as nonzero=true).
func whereKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 3 {
		return nil, fmt.Errorf("ref: where wants (cond, a, b), got %d inputs", len(in))
	}
	cond, a, b := in[0], in[1], in[2]
	if !cond.Shape().Equal(a.Shape()) || !b.Shape().Equal(a.Shape()) {
		return nil, fmt.Errorf("ref: where shapes must match, got cond%v a%v b%v", cond.Shape(), a.Shape(), b.Shape())
	}
	if a.Dtype() != b.Dtype() {
		return nil, fmt.Errorf("ref: where a/b dtype mismatch %v vs %v", a.Dtype(), b.Dtype())
	}
	out := tensor.NewOn(ctx.Device(), a.Dtype(), a.Shape())
	for pos := range a.Numel() {
		idx := tensor.Unravel(pos, a.Shape())
		if cond.AtF64(idx...) != 0 {
			out.SetF64(a.AtF64(idx...), idx...)
		} else {
			out.SetF64(b.AtF64(idx...), idx...)
		}
	}
	return []*tensor.Tensor{out}, nil
}
