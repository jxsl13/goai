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
		// Devirtualised traversal (§T646): the generic AtF64/SetF64 loop pays a
		// dtype dispatch + per-element Unravel; here we grab the raw typed slice
		// once (elementwise is flat over Numel) and index directly, still calling
		// the scalar f per element. Iteration order and arithmetic are identical to
		// the generic path; the F32 path reads float64, computes in float64 (f), and
		// rounds only the STORED result.
		switch x.Dtype() {
		case tensor.F64:
			xs := x.Contiguous().Storage().F64()
			os := out.Storage().F64()
			for i := range n {
				os[i] = f(xs[i])
			}
			return []*tensor.Tensor{out}, nil
		case tensor.F32:
			xs := x.Contiguous().Storage().F32()
			os := out.Storage().F32()
			for i := range n {
				os[i] = float32(f(float64(xs[i])))
			}
			return []*tensor.Tensor{out}, nil
		}
		// Generic fallback for exotic dtypes (verbatim original loop).
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
		if a.Shape().Equal(b.Shape()) { // same-shape fast path
			out := tensor.NewOn(ctx.Device(), a.Dtype(), a.Shape())
			n := a.Numel()
			// Devirtualised traversal (§T646): a and b share a dtype (guarded
			// above); grab the raw typed slices once and index flat, still calling
			// the scalar op per element. Order/arithmetic identical to the generic
			// loop; F32 reads float64, computes in float64 (op), rounds the store.
			switch a.Dtype() {
			case tensor.F64:
				as := a.Contiguous().Storage().F64()
				bs := b.Contiguous().Storage().F64()
				os := out.Storage().F64()
				for i := range n {
					os[i] = op(as[i], bs[i])
				}
				return []*tensor.Tensor{out}, nil
			case tensor.F32:
				as := a.Contiguous().Storage().F32()
				bs := b.Contiguous().Storage().F32()
				os := out.Storage().F32()
				for i := range n {
					os[i] = float32(op(float64(as[i]), float64(bs[i])))
				}
				return []*tensor.Tensor{out}, nil
			}
			// Generic fallback for exotic dtypes (verbatim original loop).
			for pos := range n {
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
	n := x.Numel()
	// Devirtualised traversal (§T646): the hottest cpu→ref fallback on the CPU
	// training path (~2500 calls/run). The generic AtF64/SetF64 loop pays a dtype
	// dispatch + per-element Unravel; here x and g share a dtype (g.Shape==x.Shape
	// guarded above, and OpSiLUBackward registers per-dtype so g's dtype matches),
	// so grab the raw typed slices once and index flat. Order/arithmetic identical
	// to the generic loop; F32 reads float64, computes silu' in float64, rounds the
	// store.
	switch x.Dtype() {
	case tensor.F64:
		if g.Dtype() == tensor.F64 {
			xs := x.Contiguous().Storage().F64()
			gs := g.Contiguous().Storage().F64()
			ds := dx.Storage().F64()
			//perfscan:ignore PS4003 reference oracle: intentionally simple, correctness baseline not an optimization target
			for i := range n {
				xv := xs[i]
				s := sigmoid(xv)
				ds[i] = gs[i] * s * (1 + xv*(1-s))
			}
			return []*tensor.Tensor{dx}, nil
		}
	case tensor.F32:
		if g.Dtype() == tensor.F32 {
			xs := x.Contiguous().Storage().F32()
			gs := g.Contiguous().Storage().F32()
			ds := dx.Storage().F32()
			//perfscan:ignore PS4003 reference oracle: intentionally simple, correctness baseline not an optimization target
			for i := range n {
				xv := float64(xs[i])
				s := sigmoid(xv)
				ds[i] = float32(float64(gs[i]) * s * (1 + xv*(1-s)))
			}
			return []*tensor.Tensor{dx}, nil
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	//perfscan:ignore PS4003 reference oracle: intentionally simple, correctness baseline not an optimization target
	for i := range n {
		idx := tensor.Unravel(i, x.Shape())
		xv := x.AtF64(idx...)
		s := sigmoid(xv)
		dx.SetF64(g.AtF64(idx...)*s*(1+xv*(1-s)), idx...)
	}
	return []*tensor.Tensor{dx}, nil
}

// softplusBackwardKernel computes dx = g·softplus'(x) = g·σ(x), elementwise (in = [x, g]);
// the CPU backend vectorizes the F64 case (vsoftplusGradF64), the GPU backends and every
// non-F64/F32 dtype fall back here. Softplus is tolerance-gated, not CPU==Ref exact.
func softplusBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: softplus-backward wants (x, g), got %d", len(in))
	}
	x, g := in[0], in[1]
	if !g.Shape().Equal(x.Shape()) {
		return nil, fmt.Errorf("ref: softplus-backward g %v != x %v", g.Shape(), x.Shape())
	}
	dx := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	n := x.Numel()
	switch x.Dtype() {
	case tensor.F64:
		if g.Dtype() == tensor.F64 {
			xs := x.Contiguous().Storage().F64()
			gs := g.Contiguous().Storage().F64()
			ds := dx.Storage().F64()
			//perfscan:ignore PS4003 reference oracle: intentionally simple, correctness baseline not an optimization target
			for i := range n {
				ds[i] = gs[i] * sigmoid(xs[i])
			}
			return []*tensor.Tensor{dx}, nil
		}
	case tensor.F32:
		if g.Dtype() == tensor.F32 {
			xs := x.Contiguous().Storage().F32()
			gs := g.Contiguous().Storage().F32()
			ds := dx.Storage().F32()
			//perfscan:ignore PS4003 reference oracle: intentionally simple, correctness baseline not an optimization target
			for i := range n {
				ds[i] = float32(float64(gs[i]) * sigmoid(float64(xs[i])))
			}
			return []*tensor.Tensor{dx}, nil
		}
	}
	//perfscan:ignore PS4003 reference oracle: intentionally simple, correctness baseline not an optimization target
	for i := range n {
		idx := tensor.Unravel(i, x.Shape())
		dx.SetF64(g.AtF64(idx...)*sigmoid(x.AtF64(idx...)), idx...)
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
	n := x.Numel()
	// Devirtualised traversal (§T646): a hot cpu→ref fallback on the CPU training
	// path. x and g share a dtype (shape guarded above, op registered per-dtype);
	// grab the raw typed slices once and index flat, still calling geluGrad per
	// element. Order/arithmetic identical to the generic loop; F32 reads float64,
	// computes geluGrad in float64, rounds the store.
	switch x.Dtype() {
	case tensor.F64:
		if g.Dtype() == tensor.F64 {
			xs := x.Contiguous().Storage().F64()
			gs := g.Contiguous().Storage().F64()
			ds := dx.Storage().F64()
			//perfscan:ignore PS4003 reference oracle: intentionally simple, correctness baseline not an optimization target
			for i := range n {
				ds[i] = geluGrad(xs[i], gs[i])
			}
			return []*tensor.Tensor{dx}, nil
		}
	case tensor.F32:
		if g.Dtype() == tensor.F32 {
			xs := x.Contiguous().Storage().F32()
			gs := g.Contiguous().Storage().F32()
			ds := dx.Storage().F32()
			//perfscan:ignore PS4003 reference oracle: intentionally simple, correctness baseline not an optimization target
			for i := range n {
				ds[i] = float32(geluGrad(float64(xs[i]), float64(gs[i])))
			}
			return []*tensor.Tensor{dx}, nil
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	//perfscan:ignore PS4003 reference oracle: intentionally simple, correctness baseline not an optimization target
	for i := range n {
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
	n := x.Numel()
	// Devirtualised traversal (§T646): grab the raw typed slice once and index
	// flat instead of the per-element AtF64/SetF64 dispatch + Unravel. Clamp math
	// stays in float64 on both paths; F32 rounds only the STORED result.
	switch x.Dtype() {
	case tensor.F64:
		xs := x.Contiguous().Storage().F64()
		os := out.Storage().F64()
		for i := range n {
			//perfscan:ignore PS3077,PS3082 reference oracle: intentionally simple, correctness baseline not an optimization target
			os[i] = math.Max(pa.Lo, math.Min(xs[i], pa.Hi))
		}
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		xs := x.Contiguous().Storage().F32()
		os := out.Storage().F32()
		for i := range n {
			//perfscan:ignore PS3077,PS3082 reference oracle: intentionally simple, correctness baseline not an optimization target
			os[i] = float32(math.Max(pa.Lo, math.Min(float64(xs[i]), pa.Hi)))
		}
		return []*tensor.Tensor{out}, nil
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	for pos := range n {
		idx := tensor.Unravel(pos, x.Shape())
		v := x.AtF64(idx...)
		//perfscan:ignore PS3077,PS3082 reference oracle: intentionally simple, correctness baseline not an optimization target
		out.SetF64(math.Max(pa.Lo, math.Min(v, pa.Hi)), idx...)
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	// reg installs the same dtype-agnostic kernel for F32 and F64.
	reg := func(op backend.Op, k backend.Kernel) {
		//perfscan:ignore PS3052 reference oracle: intentionally simple, correctness baseline not an optimization target
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
	reg(backend.OpSoftplusBackward, softplusBackwardKernel)
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
	n := a.Numel()
	// Devirtualised traversal (§T646): a and b share the output dtype (guarded);
	// when cond shares that plain float dtype too we grab the raw typed slices once
	// and index flat instead of the per-element AtF64/SetF64 dispatch + Unravel.
	// The select is a verbatim copy (same dtype, no narrowing), so byte-for-byte
	// identical to the generic loop. cond of a different dtype falls through to the
	// generic path (which reads it via AtF64), preserving the any-dtype contract.
	switch a.Dtype() {
	case tensor.F64:
		if cond.Dtype() == tensor.F64 {
			cs := cond.Contiguous().Storage().F64()
			as := a.Contiguous().Storage().F64()
			bs := b.Contiguous().Storage().F64()
			os := out.Storage().F64()
			for i := range n {
				if cs[i] != 0 {
					os[i] = as[i]
				} else {
					os[i] = bs[i]
				}
			}
			return []*tensor.Tensor{out}, nil
		}
	case tensor.F32:
		if cond.Dtype() == tensor.F32 {
			cs := cond.Contiguous().Storage().F32()
			as := a.Contiguous().Storage().F32()
			bs := b.Contiguous().Storage().F32()
			os := out.Storage().F32()
			for i := range n {
				if cs[i] != 0 {
					os[i] = as[i]
				} else {
					os[i] = bs[i]
				}
			}
			return []*tensor.Tensor{out}, nil
		}
	}
	// Generic fallback for exotic dtypes (verbatim original loop).
	for pos := range n {
		idx := tensor.Unravel(pos, a.Shape())
		if cond.AtF64(idx...) != 0 {
			out.SetF64(a.AtF64(idx...), idx...)
		} else {
			out.SetF64(b.AtF64(idx...), idx...)
		}
	}
	return []*tensor.Tensor{out}, nil
}
