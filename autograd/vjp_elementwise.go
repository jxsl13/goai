package autograd

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Elementwise VJPs (§T14). Implemented as direct scalar loops over (x, y, g) —
// exact, dtype-agnostic, and free of op-composition overhead. x = forward input,
// y = forward output (reused where the derivative is cheapest in terms of y).

// unaryVJP lifts a scalar derivative rule into a VJP.
func unaryVJP(f func(x, y, g float64) float64) VJP {
	return func(_ *backend.Context, in, out []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x, y := in[0], out[0]
		gin := tensor.New(x.Dtype(), x.Shape())
		for i := range x.Numel() {
			idx := tensor.Unravel(i, x.Shape())
			gin.SetF64(f(x.AtF64(idx...), y.AtF64(idx...), g.AtF64(idx...)), idx...)
		}
		return []*tensor.Tensor{gin}, nil
	}
}

const invSqrt2Pi = 0.3989422804014327 // 1/√(2π)

func init() {
	// d(a/b): (g/b, -g·a/b²)
	RegisterVJP(backend.OpDiv, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		a, b := in[0], in[1]
		ga := tensor.New(a.Dtype(), a.Shape())
		gb := tensor.New(b.Dtype(), b.Shape())
		for i := range a.Numel() {
			idx := tensor.Unravel(i, a.Shape())
			av, bv, gv := a.AtF64(idx...), b.AtF64(idx...), g.AtF64(idx...)
			ga.SetF64(gv/bv, idx...)
			gb.SetF64(-gv*av/(bv*bv), idx...)
		}
		return []*tensor.Tensor{ga, gb}, nil
	})

	RegisterVJP(backend.OpExp, unaryVJP(func(_, y, g float64) float64 { return g * y }))
	RegisterVJP(backend.OpLog, unaryVJP(func(x, _, g float64) float64 { return g / x }))
	RegisterVJP(backend.OpTanh, unaryVJP(func(_, y, g float64) float64 { return g * (1 - y*y) }))
	RegisterVJP(backend.OpReLU, unaryVJP(func(x, _, g float64) float64 {
		if x > 0 {
			return g
		}
		return 0
	}))
	// gelu'(x) = Φ(x) + x·φ(x), exact erf form (ADR-0004)
	RegisterVJP(backend.OpGELU, unaryVJP(func(x, _, g float64) float64 {
		phi := 0.5 * (1 + math.Erf(x/math.Sqrt2))
		pdf := invSqrt2Pi * math.Exp(-0.5*x*x)
		return g * (phi + x*pdf)
	}))
	RegisterVJP(backend.OpSigmoid, unaryVJP(func(_, y, g float64) float64 { return g * y * (1 - y) }))
	// silu(x)=x·σ(x); silu'(x)=σ(x)·(1+x·(1−σ(x)))
	RegisterVJP(backend.OpSiLU, unaryVJP(func(x, _, g float64) float64 {
		s := 1 / (1 + math.Exp(-x))
		return g * s * (1 + x*(1-s))
	}))

	// softmax (last axis): gin = y ⊙ (g − Σⱼ gⱼyⱼ) per row
	RegisterVJP(backend.OpSoftmax, func(_ *backend.Context, in, out []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		y := out[0]
		d := y.Shape()[y.Ndim()-1]
		rows := y.Numel() / d
		gin := tensor.New(in[0].Dtype(), in[0].Shape())
		for r := range rows {
			var dot float64
			for j := range d {
				idx := tensor.Unravel(r*d+j, y.Shape())
				dot += g.AtF64(idx...) * y.AtF64(idx...)
			}
			for j := range d {
				idx := tensor.Unravel(r*d+j, y.Shape())
				gin.SetF64(y.AtF64(idx...)*(g.AtF64(idx...)-dot), idx...)
			}
		}
		return []*tensor.Tensor{gin}, nil
	})
}
