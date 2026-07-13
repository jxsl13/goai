// Package ops is the public eager op API (layer L1). Each function dispatches
// through backend.Execute on the default backend, falling back to the Pure-Go
// reference when needed (§I4). Autograd (L2) builds on the same dispatch path
// via a recording context, so these functions need no grad awareness (ADR-0003).
//
// A backend must be registered; importing the root goai package (or
// backend/ref) does so.
package ops

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func eager(op backend.Op, inputs ...*tensor.Tensor) (*tensor.Tensor, error) {
	return eagerA(op, nil, inputs...)
}

func eagerA(op backend.Op, attrs backend.Attrs, inputs ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(backend.NewContext(), op, inputs, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Binary elementwise (same shape, same dtype).

// Add returns a + b elementwise.
func Add(a, b *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpAdd, a, b) }

// Sub returns a − b elementwise.
func Sub(a, b *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpSub, a, b) }

// Mul returns the elementwise (Hadamard) product a ⊙ b.
func Mul(a, b *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpMul, a, b) }

// Div returns a / b elementwise.
func Div(a, b *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpDiv, a, b) }

// Reductions. axes==nil reduces over all axes; keepdims retains reduced axes as
// size 1. F32 inputs accumulate in f64 then narrow (§V10).
func reduce(op backend.Op, x *tensor.Tensor, axes []int, keepdims bool) (*tensor.Tensor, error) {
	return eagerA(op, backend.ReduceAttrs{Axes: axes, KeepDims: keepdims}, x)
}

// Sum reduces x by summation over axes.
func Sum(x *tensor.Tensor, axes []int, keepdims bool) (*tensor.Tensor, error) {
	return reduce(backend.OpSum, x, axes, keepdims)
}

// Mean reduces x by averaging over axes.
func Mean(x *tensor.Tensor, axes []int, keepdims bool) (*tensor.Tensor, error) {
	return reduce(backend.OpMean, x, axes, keepdims)
}

// Max reduces x to the maximum over axes.
func Max(x *tensor.Tensor, axes []int, keepdims bool) (*tensor.Tensor, error) {
	return reduce(backend.OpMax, x, axes, keepdims)
}

// Min reduces x to the minimum over axes.
func Min(x *tensor.Tensor, axes []int, keepdims bool) (*tensor.Tensor, error) {
	return reduce(backend.OpMin, x, axes, keepdims)
}

// Transpose returns the rank-2 matrix transpose of x, out[j,i] = x[i,j] (numpy.T).
// Unlike a transposed view it is a dispatched op, so it is differentiable on the tape
// (its gradient is the transpose of the upstream gradient).
func Transpose(x *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpTranspose, x) }

// Prod reduces x by multiplication over axes (numpy.prod); an empty reduction is 1.
// It is differentiable: ∂prod/∂xᵢ = ∏_{j≠i} xⱼ over the element's group (handled
// stably when the group contains zeros).
func Prod(x *tensor.Tensor, axes []int, keepdims bool) (*tensor.Tensor, error) {
	return reduce(backend.OpProd, x, axes, keepdims)
}

// Var is the population variance of x over the given axes (numpy.var, ddof=0): the
// mean of the squared deviations from the mean, mean((x − mean)²). It composes
// Mean/Sub/Mul with broadcasting (the keepdims mean broadcasts against x), so it is
// differentiable through the same dispatch path. keepdims retains the reduced axes
// as size 1.
func Var(x *tensor.Tensor, axes []int, keepdims bool) (*tensor.Tensor, error) {
	mean, err := Mean(x, axes, true) // keepdims → broadcastable against x
	if err != nil {
		return nil, err
	}
	diff, err := Sub(x, mean) // broadcasts mean back up to x's shape
	if err != nil {
		return nil, err
	}
	sq, err := Mul(diff, diff)
	if err != nil {
		return nil, err
	}
	return Mean(sq, axes, keepdims)
}

// Std is the population standard deviation of x over the given axes (numpy.std,
// ddof=0) — the square root of Var.
func Std(x *tensor.Tensor, axes []int, keepdims bool) (*tensor.Tensor, error) {
	v, err := Var(x, axes, keepdims)
	if err != nil {
		return nil, err
	}
	return Sqrt(v)
}

// VarDDof is the variance of x over the given axes with Bessel-style delta-degrees-of-freedom
// (numpy.var(a, axis, ddof)): the sum of squared deviations from the mean divided by (N − ddof),
// where N is the number of elements reduced per output entry. ddof=0 is the population variance
// (mean of squared deviations, == Var); ddof=1 is the UNBIASED sample variance (the standard
// estimator with Bessel's correction). Since Var already divides by N, VarDDof rescales it by the
// exact factor N/(N − ddof), so it stays differentiable through the same path. ddof must satisfy
// 0 ≤ ddof < N. Definitional source: numpy semantics (§R140; no paper).
func VarDDof(x *tensor.Tensor, axes []int, keepdims bool, ddof int) (*tensor.Tensor, error) {
	if ddof < 0 {
		return nil, fmt.Errorf("ops: VarDDof ddof must be ≥ 0, got %d", ddof)
	}
	v, err := Var(x, axes, keepdims) // population variance = SS/N
	if err != nil {
		return nil, err
	}
	if ddof == 0 {
		return v, nil
	}
	n := x.Numel() / v.Numel() // elements reduced per output entry (numel drops the reduced dims)
	if n-ddof <= 0 {
		return nil, fmt.Errorf("ops: VarDDof needs ddof < N (N=%d reduced elements), got ddof=%d", n, ddof)
	}
	// SS/(N−ddof) = (SS/N)·N/(N−ddof); Full matches v's shape so no broadcast is needed.
	factor := tensor.Full(v.Dtype(), v.Shape(), float64(n)/float64(n-ddof))
	return Mul(v, factor)
}

// StdDDof is the standard deviation of x over the given axes with delta-degrees-of-freedom
// (numpy.std(a, axis, ddof)) — the square root of VarDDof. ddof=1 gives the sample standard
// deviation. Definitional source: numpy semantics (§R140).
func StdDDof(x *tensor.Tensor, axes []int, keepdims bool, ddof int) (*tensor.Tensor, error) {
	v, err := VarDDof(x, axes, keepdims, ddof)
	if err != nil {
		return nil, err
	}
	return Sqrt(v)
}

// Cumsum returns the cumulative sum of x along axis (numpy.cumsum): element i is
// the sum of all elements up to and including i along that axis (a negative axis
// counts from the end). Differentiable — its gradient is the reverse cumulative sum.
func Cumsum(x *tensor.Tensor, axis int) (*tensor.Tensor, error) {
	return eagerA(backend.OpCumsum, backend.CumsumAttrs{Axis: axis}, x)
}

// ArgMax returns indices of the maximum along axis (index is position along that
// axis, ties → lowest index).
func ArgMax(x *tensor.Tensor, axis int) (*tensor.Tensor, error) {
	return eagerA(backend.OpArgMax, backend.ArgMaxAttrs{Axis: axis}, x)
}

// ArgMaxFlat returns the flattened index of the global maximum as a scalar.
func ArgMaxFlat(x *tensor.Tensor) (*tensor.Tensor, error) {
	return eagerA(backend.OpArgMax, nil, x)
}

// MatMul returns the row-major matrix product A[M,K]·B[K,N] = C[M,N]. Transposed
// operands are supported by passing transpose views.
func MatMul(a, b *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpMatMul, a, b) }

// Conv2D cross-correlates x[N,C,H,W] with w[F,C,KH,KW] (+ optional bias[F];
// pass nil for none) using zero padding.
func Conv2D(x, w, bias *tensor.Tensor, stride, pad int) (*tensor.Tensor, error) {
	ins := []*tensor.Tensor{x, w}
	if bias != nil {
		ins = append(ins, bias)
	}
	out, err := backend.Execute(backend.NewContext(), backend.OpConv2D, ins,
		backend.ConvAttrs{Stride: stride, Pad: pad})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// MaxPool2D / AvgPool2D pool k×k windows; stride 0 means stride = kernel.
func MaxPool2D(x *tensor.Tensor, kernel, stride int) (*tensor.Tensor, error) {
	return poolOp(backend.OpMaxPool2D, x, kernel, stride)
}

// AvgPool2D pools k×k windows of x[NCHW] by average; stride 0 means stride = kernel.
func AvgPool2D(x *tensor.Tensor, kernel, stride int) (*tensor.Tensor, error) {
	return poolOp(backend.OpAvgPool2D, x, kernel, stride)
}

func poolOp(op backend.Op, x *tensor.Tensor, kernel, stride int) (*tensor.Tensor, error) {
	attrs := backend.PoolAttrs{Kernel: kernel}
	if stride > 0 {
		attrs.Stride = stride
	}
	return eagerA(op, attrs, x)
}

// Softmax applies the numerically stable softmax over the last axis.
func Softmax(x *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpSoftmax, x) }

// LayerNorm normalizes over the last axis with scale gamma and shift beta
// (torch semantics: biased variance, eps inside the sqrt).
func LayerNorm(x, gamma, beta *tensor.Tensor, eps float64) (*tensor.Tensor, error) {
	return eagerA(backend.OpLayerNorm, backend.NormAttrs{Eps: eps}, x, gamma, beta)
}

// RMSNorm normalizes over the last axis by root-mean-square with scale gamma
// (no mean subtraction, no bias; Zhang & Sennrich 2019).
func RMSNorm(x, gamma *tensor.Tensor, eps float64) (*tensor.Tensor, error) {
	return eagerA(backend.OpRMSNorm, backend.NormAttrs{Eps: eps}, x, gamma)
}

// RoPE applies rotary position embeddings (HF rotate_half) to q[seq,headDim],
// position = row index. base defaults to 10000 when ≤ 0.
func RoPE(q *tensor.Tensor, base float64) (*tensor.Tensor, error) {
	if base <= 0 {
		base = 10000
	}
	return eagerA(backend.OpRoPE, backend.RoPEAttrs{Base: base}, q)
}

// XPos applies the length-extrapolatable xPos positional encoding (Sun et al.
// 2022, §R125) to q[seq,headDim]: the standard RoPE rotation followed by a
// per-pair magnitude scale ζ_i^(±n), n = position. Set downscale=false for
// queries (ζ^(+n)) and true for keys (ζ^(−n)); the attention score q·k then
// carries an exponential ζ^(n−m) decay in the relative distance, which
// extrapolates to longer contexts. base defaults to 10000; γ defaults to 0.4.
func XPos(q *tensor.Tensor, base float64, downscale bool) (*tensor.Tensor, error) {
	if base <= 0 {
		base = 10000
	}
	return eagerA(backend.OpRoPE, backend.RoPEAttrs{Base: base, XPos: true, XPosDownscale: downscale}, q)
}

// Slice extracts the half-open range [start,end) along axis (numpy x[..,start:end,..]):
// the result matches x except along axis, whose extent becomes end−start. A negative
// axis counts from the end. Differentiable — the gradient scatters back into x's shape.
func Slice(x *tensor.Tensor, axis, start, end int) (*tensor.Tensor, error) {
	return eagerA(backend.OpSlice, backend.SliceAttrs{Axis: axis, Start: start, End: end}, x)
}

// Split cuts x into consecutive pieces of the given sizes along axis (like a
// sized numpy.split) — e.g. splitting a fused QKV projection. The sizes must be
// positive and sum to x's extent on that axis. Each piece is a differentiable
// Slice, so gradients from all pieces accumulate back into x. A negative axis
// counts from the end.
func Split(x *tensor.Tensor, axis int, sizes ...int) ([]*tensor.Tensor, error) {
	nd := x.Ndim()
	ax := axis
	if ax < 0 {
		ax += nd
	}
	if ax < 0 || ax >= nd {
		return nil, fmt.Errorf("ops: Split axis %d out of range for rank %d", axis, nd)
	}
	sum := 0
	for _, s := range sizes {
		if s <= 0 {
			return nil, fmt.Errorf("ops: Split sizes must be positive, got %v", sizes)
		}
		sum += s
	}
	if sum != x.Shape()[ax] {
		return nil, fmt.Errorf("ops: Split sizes %v sum to %d != axis %d extent %d", sizes, sum, ax, x.Shape()[ax])
	}
	outs := make([]*tensor.Tensor, len(sizes))
	off := 0
	for i, s := range sizes {
		piece, err := Slice(x, ax, off, off+s)
		if err != nil {
			return nil, err
		}
		outs[i] = piece
		off += s
	}
	return outs, nil
}

// Reshape returns x with a new shape (numpy.reshape), preserving row-major element
// order; the new shape must have the same number of elements. Differentiable — the
// gradient reshapes back.
func Reshape(x *tensor.Tensor, shape ...int) (*tensor.Tensor, error) {
	return eagerA(backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape(shape)}, x)
}

// ExpandDims inserts a size-1 axis at position axis (numpy.expand_dims), raising
// the rank by one. axis may be negative (counting from the end, allowing the new
// last axis). Differentiable.
func ExpandDims(x *tensor.Tensor, axis int) (*tensor.Tensor, error) {
	nd := x.Ndim()
	ax := axis
	if ax < 0 {
		ax += nd + 1
	}
	if ax < 0 || ax > nd {
		return nil, fmt.Errorf("ops: ExpandDims axis %d out of range for rank %d", axis, nd)
	}
	ns := make([]int, 0, nd+1)
	ns = append(ns, x.Shape()[:ax]...)
	ns = append(ns, 1)
	ns = append(ns, x.Shape()[ax:]...)
	return Reshape(x, ns...)
}

// Squeeze removes the size-1 axis at position axis (numpy.squeeze with an axis),
// lowering the rank by one; it errors if that axis is not size 1. axis may be
// negative. Differentiable.
func Squeeze(x *tensor.Tensor, axis int) (*tensor.Tensor, error) {
	nd := x.Ndim()
	ax := axis
	if ax < 0 {
		ax += nd
	}
	if ax < 0 || ax >= nd {
		return nil, fmt.Errorf("ops: Squeeze axis %d out of range for rank %d", axis, nd)
	}
	if x.Shape()[ax] != 1 {
		return nil, fmt.Errorf("ops: Squeeze axis %d has size %d ≠ 1", ax, x.Shape()[ax])
	}
	ns := make([]int, 0, nd-1)
	ns = append(ns, x.Shape()[:ax]...)
	ns = append(ns, x.Shape()[ax+1:]...)
	return Reshape(x, ns...)
}

// Stack joins tensors along a NEW axis (numpy.stack): all inputs must share shape,
// and the result has rank one higher with the stacked axis at position axis (which
// may be negative). Equivalent to expand-dims each input then concatenate.
// Differentiable.
func Stack(axis int, tensors ...*tensor.Tensor) (*tensor.Tensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("ops: Stack needs ≥1 tensor")
	}
	expanded := make([]*tensor.Tensor, len(tensors))
	for i, t := range tensors {
		if !t.Shape().Equal(tensors[0].Shape()) {
			return nil, fmt.Errorf("ops: Stack input %d shape %v != %v", i, t.Shape(), tensors[0].Shape())
		}
		e, err := ExpandDims(t, axis)
		if err != nil {
			return nil, err
		}
		expanded[i] = e
	}
	// after expand-dims the new axis is normalized; concat along it
	ax := axis
	if ax < 0 {
		ax += tensors[0].Ndim() + 1
	}
	return Concat(ax, expanded...)
}

// Concat joins tensors along axis (numpy.concatenate): the inputs must share rank
// and dtype and match in every dimension except the concat axis, whose extents
// add. A negative axis counts from the end. Differentiable — the gradient splits
// back into each input's segment. Needs ≥1 tensor.
func Concat(axis int, tensors ...*tensor.Tensor) (*tensor.Tensor, error) {
	return eagerA(backend.OpConcat, backend.ConcatAttrs{Axis: axis}, tensors...)
}

// AddBias returns x[m,n] + b[n] with b broadcast over rows (§B18 for general
// broadcasting).
func AddBias(x, b *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpAddBias, x, b) }

// BLAS-1. Dot and Nrm2 accumulate in f64 (§V10); Nrm2 is overflow-safe.

// Dot returns the vector dot product of a and b.
func Dot(a, b *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpDot, a, b) }

// Nrm2 returns the Euclidean (L2) norm of x.
func Nrm2(x *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpNrm2, x) }

// Axpy returns alpha*x + y elementwise.
func Axpy(alpha float64, x, y *tensor.Tensor) (*tensor.Tensor, error) {
	return eagerA(backend.OpAXPY, backend.AXPYAttrs{Alpha: alpha}, x, y)
}

// Unary elementwise.

// Neg returns −x elementwise.
func Neg(x *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpNeg, x) }

// StopGradient returns x unchanged (identity forward) but severs the gradient path:
// no gradient flows back through it (numpy/torch detach). Use it to hold a tensor
// constant w.r.t. differentiation — straight-through estimators, EMA/teacher targets,
// RL baselines, or a frozen reference branch.
func StopGradient(x *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpStopGradient, x) }

// Exp returns eˣ elementwise.
func Exp(x *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpExp, x) }

// Log returns the natural logarithm elementwise.
func Log(x *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpLog, x) }

// Tanh returns the hyperbolic tangent elementwise.
func Tanh(x *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpTanh, x) }

// ReLU returns max(0, x) elementwise.
func ReLU(x *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpReLU, x) }

// GELU returns the exact Gaussian Error Linear Unit elementwise.
func GELU(x *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpGELU, x) }

// Sigmoid returns 1/(1+e⁻ˣ) elementwise.
func Sigmoid(x *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpSigmoid, x) }

// SiLU returns x·sigmoid(x) (swish) elementwise.
func SiLU(x *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpSiLU, x) }

// Sqrt returns √x elementwise (numpy.sqrt). Differentiable (grad g/(2√x); undefined at x=0).
func Sqrt(x *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpSqrt, x) }

// Abs returns |x| elementwise (numpy.abs). Differentiable (grad sign(x), 0 at x=0).
func Abs(x *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpAbs, x) }

// Clip clamps each element to [lo,hi] (numpy.clip); lo must be ≤ hi. Differentiable —
// the gradient passes through where x is unclipped and is 0 where it was clamped.
func Clip(x *tensor.Tensor, lo, hi float64) (*tensor.Tensor, error) {
	return eagerA(backend.OpClip, backend.ClipAttrs{Lo: lo, Hi: hi}, x)
}

// Maximum returns the elementwise maximum of a and b (numpy.maximum); same shape and
// dtype. Differentiable — the gradient routes to the larger input (ties to a).
func Maximum(a, b *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpMaximum, a, b) }

// Minimum returns the elementwise minimum of a and b (numpy.minimum); same shape and
// dtype. Differentiable — the gradient routes to the smaller input (ties to a).
func Minimum(a, b *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpMinimum, a, b) }

// BroadcastTo expands x to the target shape by replicating along size-1 (or missing)
// axes (numpy.broadcast_to): the shapes are right-aligned, and each of x's dims must
// be 1 or equal to the corresponding target dim. Differentiable — the gradient sums
// over the broadcast axes back into x's shape. This is the broadcasting primitive
// (§B18); compose it with an elementwise op for broadcasting arithmetic.
func BroadcastTo(x *tensor.Tensor, shape ...int) (*tensor.Tensor, error) {
	return eagerA(backend.OpBroadcast, backend.BroadcastAttrs{Shape: tensor.Shape(shape)}, x)
}

// Where selects elementwise cond?a:b (numpy.where): the result takes a where cond is
// nonzero and b elsewhere. cond, a and b share a shape; a and b share a dtype.
// Differentiable in a and b (the gradient routes by the mask); cond is not differentiated.
func Where(cond, a, b *tensor.Tensor) (*tensor.Tensor, error) {
	return eager(backend.OpWhere, cond, a, b)
}
