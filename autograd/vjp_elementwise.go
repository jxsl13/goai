package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Elementwise VJPs (§T14). Implemented as direct scalar loops over (x, y, g) —
// exact, dtype-agnostic, and free of op-composition overhead. x = forward input,
// y = forward output (reused where the derivative is cheapest in terms of y).

// unaryVJP lifts a scalar derivative rule into a VJP. x = forward input, y =
// forward output, g = upstream gradient — all the output's shape (unary). The fast
// path is a typed flat loop over contiguous []T storage: no per-element Unravel
// heap-alloc and no AtF64/SetF64 dtype dispatch, which on the training backward path
// is the dominant cost (§base-perf, C25). The generic per-element loop stays as the
// fallback for mixed-dtype / exotic tensors.
func unaryVJP(f func(x, y, g float64) float64) VJP {
	return func(_ *backend.Context, in, out []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x, y := in[0], out[0]
		gin := tensor.New(x.Dtype(), x.Shape())
		n := x.Numel()
		xc, yc, gc := x.Contiguous(), y.Contiguous(), g.Contiguous()
		switch x.Dtype() {
		case tensor.F64:
			if yc.Dtype() == tensor.F64 && gc.Dtype() == tensor.F64 {
				xs, ys, gs := xc.Storage().F64(), yc.Storage().F64(), gc.Storage().F64()
				ds := gin.Storage().F64()
				for i := 0; i < n; i++ {
					ds[i] = f(xs[i], ys[i], gs[i])
				}
				return []*tensor.Tensor{gin}, nil
			}
		case tensor.F32:
			if yc.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
				xs, ys, gs := xc.Storage().F32(), yc.Storage().F32(), gc.Storage().F32()
				ds := gin.Storage().F32()
				for i := 0; i < n; i++ {
					ds[i] = float32(f(float64(xs[i]), float64(ys[i]), float64(gs[i])))
				}
				return []*tensor.Tensor{gin}, nil
			}
		}
		for i := 0; i < n; i++ { // generic fallback (mixed dtype / exotic layout)
			idx := tensor.Unravel(i, x.Shape())
			gin.SetF64(f(x.AtF64(idx...), y.AtF64(idx...), g.AtF64(idx...)), idx...)
		}
		return []*tensor.Tensor{gin}, nil
	}
}

// selectVJP lifts a two-input elementwise selection (max/min) into a VJP: the
// upstream gradient goes wholly to input a where pick(a,b) is true, else to b.
func selectVJP(pick func(a, b float64) bool) VJP {
	return func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		a, b := in[0], in[1]
		outShape := g.Shape()
		offA, offB := len(outShape)-a.Ndim(), len(outShape)-b.Ndim()
		ac, bc := make([]int, a.Ndim()), make([]int, b.Ndim())
		ga := tensor.New(a.Dtype(), a.Shape())
		gb := tensor.New(b.Dtype(), b.Shape())
		for pos := range g.Numel() {
			oc := tensor.Unravel(pos, outShape)
			backend.BroadcastCoords(ac, oc, a.Shape(), offA)
			backend.BroadcastCoords(bc, oc, b.Shape(), offB)
			gv := g.AtF64(oc...)
			if pick(a.AtF64(ac...), b.AtF64(bc...)) {
				ga.SetF64(ga.AtF64(ac...)+gv, ac...)
			} else {
				gb.SetF64(gb.AtF64(bc...)+gv, bc...)
			}
		}
		return []*tensor.Tensor{ga, gb}, nil
	}
}

func init() {
	// d(a/b): (g/b, -g·a/b²), broadcast-aware — read a,b at their broadcast coords
	// and accumulate into each input's shape (accumulation performs the reduction).
	RegisterVJP(backend.OpDiv, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		a, b := in[0], in[1]
		outShape := g.Shape()
		offA, offB := len(outShape)-a.Ndim(), len(outShape)-b.Ndim()
		ac, bc := make([]int, a.Ndim()), make([]int, b.Ndim())
		ga := tensor.New(a.Dtype(), a.Shape())
		gb := tensor.New(b.Dtype(), b.Shape())
		for pos := range g.Numel() {
			oc := tensor.Unravel(pos, outShape)
			backend.BroadcastCoords(ac, oc, a.Shape(), offA)
			backend.BroadcastCoords(bc, oc, b.Shape(), offB)
			av, bv, gv := a.AtF64(ac...), b.AtF64(bc...), g.AtF64(oc...)
			ga.SetF64(ga.AtF64(ac...)+gv/bv, ac...)
			gb.SetF64(gb.AtF64(bc...)-gv*av/(bv*bv), bc...)
		}
		return []*tensor.Tensor{ga, gb}, nil
	})

	// detach: the forward is the identity, but no gradient flows back through x.
	RegisterVJP(backend.OpStopGradient, unaryVJP(func(_, _, _ float64) float64 { return 0 }))
	RegisterVJP(backend.OpExp, unaryVJP(func(_, y, g float64) float64 { return g * y }))
	RegisterVJP(backend.OpLog, unaryVJP(func(x, _, g float64) float64 { return g / x }))
	RegisterVJP(backend.OpTanh, unaryVJP(func(_, y, g float64) float64 { return g * (1 - y*y) }))
	RegisterVJP(backend.OpReLU, unaryVJP(func(x, _, g float64) float64 {
		if x > 0 {
			return g
		}
		return 0
	}))
	// gelu'(x) = Φ(x) + x·φ(x), exact erf form (ADR-0004). Dispatch OpGELUBackward on the
	// tape's active backend (like the norm/CE VJPs) — the elementwise scalar-loop VJP was
	// ~30ms at the FFN's 256×2048, a dominant training-step cost (§T353).
	RegisterVJP(backend.OpGELU, func(ctx *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := backend.Execute(ctx, backend.OpGELUBackward, []*tensor.Tensor{in[0], g}, nil)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out[0]}, nil
	})
	RegisterVJP(backend.OpSigmoid, unaryVJP(func(_, y, g float64) float64 { return g * y * (1 - y) }))
	// silu(x)=x·σ(x); silu'(x)=σ(x)·(1+x·(1−σ(x))). Dispatch OpSiLUBackward on the tape's
	// active backend (like GELU §T353) — SiLU is SwiGLU's activation, so its elementwise
	// scalar-loop VJP was a training-step cost for Llama-style models (§T362).
	RegisterVJP(backend.OpSiLU, func(ctx *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		out, err := backend.Execute(ctx, backend.OpSiLUBackward, []*tensor.Tensor{in[0], g}, nil)
		if err != nil {
			return nil, err
		}
		return []*tensor.Tensor{out[0]}, nil
	})
	// √x: dx = g/(2√x) = g/(2y) (undefined at x=0)
	RegisterVJP(backend.OpSqrt, unaryVJP(func(_, y, g float64) float64 { return g / (2 * y) }))
	// |x|: dx = g·sign(x) (0 at x=0, matching numpy/torch)
	RegisterVJP(backend.OpAbs, unaryVJP(func(x, _, g float64) float64 {
		switch {
		case x > 0:
			return g
		case x < 0:
			return -g
		default:
			return 0
		}
	}))
	// clip(x,Lo,Hi): the gradient passes through where x is unclipped (Lo≤x≤Hi), 0 where clamped
	RegisterVJP(backend.OpClip, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		pa, _ := attrs.(backend.ClipAttrs)
		gin := tensor.New(x.Dtype(), x.Shape())
		for i := range x.Numel() {
			idx := tensor.Unravel(i, x.Shape())
			v := x.AtF64(idx...)
			if v >= pa.Lo && v <= pa.Hi {
				gin.SetF64(g.AtF64(idx...), idx...)
			}
		}
		return []*tensor.Tensor{gin}, nil
	})

	// max(a,b)/min(a,b): the gradient routes to the selected input; ties go to a (a
	// subgradient choice — the boundary a==b is measure-zero and not probed by gradcheck).
	RegisterVJP(backend.OpMaximum, selectVJP(func(a, b float64) bool { return a >= b }))
	RegisterVJP(backend.OpMinimum, selectVJP(func(a, b float64) bool { return a <= b }))

	// where(cond,a,b): da = g where cond≠0, db = g where cond==0; cond is non-differentiable (nil).
	RegisterVJP(backend.OpWhere, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		cond, a, b := in[0], in[1], in[2]
		da := tensor.New(a.Dtype(), a.Shape())
		db := tensor.New(b.Dtype(), b.Shape())
		for i := range a.Numel() {
			idx := tensor.Unravel(i, a.Shape())
			if cond.AtF64(idx...) != 0 {
				da.SetF64(g.AtF64(idx...), idx...)
			} else {
				db.SetF64(g.AtF64(idx...), idx...)
			}
		}
		return []*tensor.Tensor{nil, da, db}, nil
	})

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
