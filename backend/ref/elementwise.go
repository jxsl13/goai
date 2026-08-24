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

type unaryOp interface {
	apply(float64) float64
}

type (
	identityOp struct{}
	negOp      struct{}
	expOp      struct{}
	logOp      struct{}
	tanhOp     struct{}
	reluOp     struct{}
	geluOp     struct{}
	sigmoidOp  struct{}
	siluOp     struct{}
	sqrtOp     struct{}
	absOp      struct{}
)

func (identityOp) apply(x float64) float64 { return x }
func (negOp) apply(x float64) float64      { return -x }
func (expOp) apply(x float64) float64      { return math.Exp(x) }
func (logOp) apply(x float64) float64      { return math.Log(x) }
func (tanhOp) apply(x float64) float64     { return math.Tanh(x) }
func (reluOp) apply(x float64) float64     { return relu(x) }
func (geluOp) apply(x float64) float64     { return gelu(x) }
func (sigmoidOp) apply(x float64) float64  { return sigmoid(x) }
func (siluOp) apply(x float64) float64     { return x * sigmoid(x) }
func (sqrtOp) apply(x float64) float64     { return math.Sqrt(x) }
func (absOp) apply(x float64) float64      { return math.Abs(x) }

func elemUnary[T refFloat, O unaryOp](xs, os []T, op O) {
	if len(os) == 0 {
		return
	}
	_ = xs[len(os)-1]
	switch any(op).(type) {
	case identityOp:
		for i := range os {
			os[i] = xs[i]
		}
	case negOp:
		for i := range os {
			os[i] = T(-float64(xs[i]))
		}
	case expOp:
		for i := range os {
			os[i] = T(math.Exp(float64(xs[i])))
		}
	case logOp:
		for i := range os {
			os[i] = T(math.Log(float64(xs[i])))
		}
	case tanhOp:
		for i := range os {
			os[i] = T(math.Tanh(float64(xs[i])))
		}
	case reluOp:
		for i := range os {
			os[i] = T(relu(float64(xs[i])))
		}
	case geluOp:
		for i := range os {
			os[i] = T(gelu(float64(xs[i])))
		}
	case sigmoidOp:
		for i := range os {
			os[i] = T(sigmoid(float64(xs[i])))
		}
	case siluOp:
		for i := range os {
			x := float64(xs[i])
			os[i] = T(x * sigmoid(x))
		}
	case sqrtOp:
		for i := range os {
			os[i] = T(math.Sqrt(float64(xs[i])))
		}
	case absOp:
		for i := range os {
			os[i] = T(math.Abs(float64(xs[i])))
		}
	}
}

// unaryKernel applies a statically selected scalar operation elementwise,
// reading any layout and writing a fresh contiguous output on the context
// device. The zero-size operation type lets the compiler inline trivial
// arithmetic into elemUnary instead of issuing an indirect call per element.
func unaryKernel[O unaryOp](op O) backend.Kernel {
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
			xs := x.Contiguous().Storage().F64()[:n]
			os := out.Storage().F64()[:n]
			elemUnary(xs, os, op)
			return []*tensor.Tensor{out}, nil
		case tensor.F32:
			xs := x.Contiguous().Storage().F32()[:n]
			os := out.Storage().F32()[:n]
			elemUnary(xs, os, op)
			return []*tensor.Tensor{out}, nil
		}
		// Generic fallback for exotic dtypes (verbatim original loop).
		for pos := range n {
			idx := tensor.Unravel(pos, x.Shape())
			out.SetF64(op.apply(x.AtF64(idx...)), idx...)
		}
		return []*tensor.Tensor{out}, nil
	}
}

type binaryOp interface {
	apply(float64, float64) float64
}

type (
	addOp     struct{}
	subOp     struct{}
	mulOp     struct{}
	divOp     struct{}
	maximumOp struct{}
	minimumOp struct{}
)

func (addOp) apply(a, b float64) float64     { return a + b }
func (subOp) apply(a, b float64) float64     { return a - b }
func (mulOp) apply(a, b float64) float64     { return a * b }
func (divOp) apply(a, b float64) float64     { return a / b }
func (maximumOp) apply(a, b float64) float64 { return math.Max(a, b) }
func (minimumOp) apply(a, b float64) float64 { return math.Min(a, b) }

func elemBinary[T refFloat, O binaryOp](as, bs, os []T, op O) {
	if len(os) == 0 {
		return
	}
	_ = as[len(os)-1]
	_ = bs[len(os)-1]
	switch any(op).(type) {
	case addOp:
		for i := range os {
			os[i] = T(float64(as[i]) + float64(bs[i]))
		}
	case subOp:
		for i := range os {
			os[i] = T(float64(as[i]) - float64(bs[i]))
		}
	case mulOp:
		for i := range os {
			os[i] = T(float64(as[i]) * float64(bs[i]))
		}
	case divOp:
		for i := range os {
			os[i] = T(float64(as[i]) / float64(bs[i]))
		}
	case maximumOp:
		for i := range os {
			os[i] = T(math.Max(float64(as[i]), float64(bs[i])))
		}
	case minimumOp:
		for i := range os {
			os[i] = T(math.Min(float64(as[i]), float64(bs[i])))
		}
	}
}

// binaryKernel applies a statically selected scalar operation elementwise.
// The zero-size operation type removes the function-value call from the hot
// same-shape loops while preserving the generic broadcasting fallback.
func binaryKernel[O binaryOp](op O) backend.Kernel {
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
				as := a.Contiguous().Storage().F64()[:n]
				bs := b.Contiguous().Storage().F64()[:n]
				os := out.Storage().F64()[:n]
				elemBinary(as, bs, os, op)
				return []*tensor.Tensor{out}, nil
			case tensor.F32:
				as := a.Contiguous().Storage().F32()[:n]
				bs := b.Contiguous().Storage().F32()[:n]
				os := out.Storage().F32()[:n]
				elemBinary(as, bs, os, op)
				return []*tensor.Tensor{out}, nil
			}
			// Generic fallback for exotic dtypes (verbatim original loop).
			for pos := range n {
				idx := tensor.Unravel(pos, a.Shape())
				out.SetF64(op.apply(a.AtF64(idx...), b.AtF64(idx...)), idx...)
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
			out.SetF64(op.apply(a.AtF64(ac...), b.AtF64(bc...)), oc...)
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

	reg(backend.OpStopGradient, unaryKernel(identityOp{})) // detach: identity forward
	reg(backend.OpNeg, unaryKernel(negOp{}))
	reg(backend.OpExp, unaryKernel(expOp{}))
	reg(backend.OpLog, unaryKernel(logOp{}))
	reg(backend.OpTanh, unaryKernel(tanhOp{}))
	reg(backend.OpReLU, unaryKernel(reluOp{}))
	reg(backend.OpGELU, unaryKernel(geluOp{}))
	reg(backend.OpGELUBackward, geluBackwardKernel)
	reg(backend.OpSiLUBackward, siluBackwardKernel)
	reg(backend.OpSoftplusBackward, softplusBackwardKernel)
	reg(backend.OpSigmoid, unaryKernel(sigmoidOp{}))
	reg(backend.OpSiLU, unaryKernel(siluOp{}))
	reg(backend.OpSqrt, unaryKernel(sqrtOp{}))
	reg(backend.OpAbs, unaryKernel(absOp{}))
	reg(backend.OpClip, clipKernel)

	reg(backend.OpAdd, binaryKernel(addOp{}))
	reg(backend.OpSub, binaryKernel(subOp{}))
	reg(backend.OpMul, binaryKernel(mulOp{}))
	reg(backend.OpDiv, binaryKernel(divOp{}))
	reg(backend.OpMaximum, binaryKernel(maximumOp{}))
	reg(backend.OpMinimum, binaryKernel(minimumOp{}))
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
