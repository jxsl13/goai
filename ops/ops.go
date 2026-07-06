// Package ops is the public eager op API (layer L1). Each function dispatches
// through backend.Execute on the default backend, falling back to the Pure-Go
// reference when needed (§I4). Autograd (L2) builds on the same dispatch path
// via a recording context, so these functions need no grad awareness (ADR-0003).
//
// A backend must be registered; importing the root goai package (or
// backend/ref) does so.
package ops

import (
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
func Add(a, b *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpAdd, a, b) }
func Sub(a, b *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpSub, a, b) }
func Mul(a, b *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpMul, a, b) }
func Div(a, b *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpDiv, a, b) }

// Reductions. axes==nil reduces over all axes; keepdims retains reduced axes as
// size 1. F32 inputs accumulate in f64 then narrow (§V10).
func reduce(op backend.Op, x *tensor.Tensor, axes []int, keepdims bool) (*tensor.Tensor, error) {
	return eagerA(op, backend.Attrs{"axes": axes, "keepdims": keepdims}, x)
}

func Sum(x *tensor.Tensor, axes []int, keepdims bool) (*tensor.Tensor, error) {
	return reduce(backend.OpSum, x, axes, keepdims)
}
func Mean(x *tensor.Tensor, axes []int, keepdims bool) (*tensor.Tensor, error) {
	return reduce(backend.OpMean, x, axes, keepdims)
}
func Max(x *tensor.Tensor, axes []int, keepdims bool) (*tensor.Tensor, error) {
	return reduce(backend.OpMax, x, axes, keepdims)
}
func Min(x *tensor.Tensor, axes []int, keepdims bool) (*tensor.Tensor, error) {
	return reduce(backend.OpMin, x, axes, keepdims)
}

// ArgMax returns indices of the maximum along axis (index is position along that
// axis, ties → lowest index).
func ArgMax(x *tensor.Tensor, axis int) (*tensor.Tensor, error) {
	return eagerA(backend.OpArgMax, backend.Attrs{"axis": axis}, x)
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
		backend.Attrs{"stride": stride, "pad": pad})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// MaxPool2D / AvgPool2D pool k×k windows; stride 0 means stride = kernel.
func MaxPool2D(x *tensor.Tensor, kernel, stride int) (*tensor.Tensor, error) {
	return poolOp(backend.OpMaxPool2D, x, kernel, stride)
}
func AvgPool2D(x *tensor.Tensor, kernel, stride int) (*tensor.Tensor, error) {
	return poolOp(backend.OpAvgPool2D, x, kernel, stride)
}

func poolOp(op backend.Op, x *tensor.Tensor, kernel, stride int) (*tensor.Tensor, error) {
	attrs := backend.Attrs{"kernel": kernel}
	if stride > 0 {
		attrs["stride"] = stride
	}
	return eagerA(op, attrs, x)
}

// Softmax applies the numerically stable softmax over the last axis.
func Softmax(x *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpSoftmax, x) }

// LayerNorm normalizes over the last axis with scale gamma and shift beta
// (torch semantics: biased variance, eps inside the sqrt).
func LayerNorm(x, gamma, beta *tensor.Tensor, eps float64) (*tensor.Tensor, error) {
	return eagerA(backend.OpLayerNorm, backend.Attrs{"eps": eps}, x, gamma, beta)
}

// RMSNorm normalizes over the last axis by root-mean-square with scale gamma
// (no mean subtraction, no bias; Zhang & Sennrich 2019).
func RMSNorm(x, gamma *tensor.Tensor, eps float64) (*tensor.Tensor, error) {
	return eagerA(backend.OpRMSNorm, backend.Attrs{"eps": eps}, x, gamma)
}

// RoPE applies rotary position embeddings (HF rotate_half) to q[seq,headDim],
// position = row index. base defaults to 10000 when ≤ 0.
func RoPE(q *tensor.Tensor, base float64) (*tensor.Tensor, error) {
	if base <= 0 {
		base = 10000
	}
	return eagerA(backend.OpRoPE, backend.Attrs{"base": base}, q)
}

// AddBias returns x[m,n] + b[n] with b broadcast over rows (§B18 for general
// broadcasting).
func AddBias(x, b *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpAddBias, x, b) }

// BLAS-1. Dot and Nrm2 accumulate in f64 (§V10); Nrm2 is overflow-safe.
func Dot(a, b *tensor.Tensor) (*tensor.Tensor, error)  { return eager(backend.OpDot, a, b) }
func Nrm2(x *tensor.Tensor) (*tensor.Tensor, error)    { return eager(backend.OpNrm2, x) }

// Axpy returns alpha*x + y elementwise.
func Axpy(alpha float64, x, y *tensor.Tensor) (*tensor.Tensor, error) {
	return eagerA(backend.OpAXPY, backend.Attrs{"alpha": alpha}, x, y)
}

// Unary elementwise.
func Neg(x *tensor.Tensor) (*tensor.Tensor, error)     { return eager(backend.OpNeg, x) }
func Exp(x *tensor.Tensor) (*tensor.Tensor, error)     { return eager(backend.OpExp, x) }
func Log(x *tensor.Tensor) (*tensor.Tensor, error)     { return eager(backend.OpLog, x) }
func Tanh(x *tensor.Tensor) (*tensor.Tensor, error)    { return eager(backend.OpTanh, x) }
func ReLU(x *tensor.Tensor) (*tensor.Tensor, error)    { return eager(backend.OpReLU, x) }
func GELU(x *tensor.Tensor) (*tensor.Tensor, error)    { return eager(backend.OpGELU, x) }
func Sigmoid(x *tensor.Tensor) (*tensor.Tensor, error) { return eager(backend.OpSigmoid, x) }
func SiLU(x *tensor.Tensor) (*tensor.Tensor, error)    { return eager(backend.OpSiLU, x) }
