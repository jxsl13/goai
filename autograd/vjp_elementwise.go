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
		// No-broadcast fast path (see OpDiv): equal shapes ⇒ each coord maps 1:1 at
		// flat i, so ravel + BroadcastCoords + dispatch are pure overhead. ga/gb are
		// pre-zeroed; the f32 store is an exact round-trip of AtF64's widen and pick's
		// f64 comparison of widened f32s matches comparing the f32s — bit-identical.
		if a.Shape().Equal(outShape) && b.Shape().Equal(outShape) {
			if a.Dtype() == tensor.F64 && b.Dtype() == tensor.F64 && g.Dtype() == tensor.F64 {
				as, bs, gs := a.Contiguous().Storage().F64(), b.Contiguous().Storage().F64(), g.Contiguous().Storage().F64()
				gas, gbs := ga.Storage().F64(), gb.Storage().F64()
				for i := range gs {
					if pick(as[i], bs[i]) {
						gas[i] = gs[i]
					} else {
						gbs[i] = gs[i]
					}
				}
				return []*tensor.Tensor{ga, gb}, nil
			}
			if a.Dtype() == tensor.F32 && b.Dtype() == tensor.F32 && g.Dtype() == tensor.F32 {
				as, bs, gs := a.Contiguous().Storage().F32(), b.Contiguous().Storage().F32(), g.Contiguous().Storage().F32()
				gas, gbs := ga.Storage().F32(), gb.Storage().F32()
				for i := range gs {
					if pick(float64(as[i]), float64(bs[i])) {
						gas[i] = gs[i]
					} else {
						gbs[i] = gs[i]
					}
				}
				return []*tensor.Tensor{ga, gb}, nil
			}
		}
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

// reluVJP is the specialized OpReLU backward: gin[i] = x[i]>0 ? g[i] : 0 — a raw-slice
// F32/F64 select with NO per-element closure and NO f64 round-trip (the generic
// unaryVJP path profiled at ~17% of the ReLU training backward). Bit-identical to the
// generic rule: ReLU's derivative is a pure pass-through/zero select, so
// float32(select_f64) == select_f32 exactly, and gin is pre-zeroed so x<=0 stays 0.
func reluVJP(_ *backend.Context, in, out []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
	x := in[0]
	gin := tensor.New(x.Dtype(), x.Shape())
	n := x.Numel()
	xc, gc := x.Contiguous(), g.Contiguous()
	switch x.Dtype() {
	case tensor.F64:
		if gc.Dtype() == tensor.F64 {
			xs, gs := xc.Storage().F64(), gc.Storage().F64()
			ds := gin.Storage().F64()
			for i := 0; i < n; i++ {
				if xs[i] > 0 {
					ds[i] = gs[i]
				}
			}
			return []*tensor.Tensor{gin}, nil
		}
	case tensor.F32:
		if gc.Dtype() == tensor.F32 {
			xs, gs := xc.Storage().F32(), gc.Storage().F32()
			ds := gin.Storage().F32()
			for i := 0; i < n; i++ {
				if xs[i] > 0 {
					ds[i] = gs[i]
				}
			}
			return []*tensor.Tensor{gin}, nil
		}
	}
	// mixed-dtype / exotic-layout fallback: the generic per-element rule.
	return unaryVJP(func(xv, _, gv float64) float64 {
		if xv > 0 {
			return gv
		}
		return 0
	})(nil, in, out, attrs, g)
}

// tanhVJP / expVJP / sigmoidVJP are raw-slice, CLOSURE-FREE specializations of the
// hot activation backwards. The generic unaryVJP applied a per-element closure with
// an f32->f64->f32 round-trip (a P1 anti-pattern; the sibling reluVJP shaved ~1.24×
// off the f64 train step). BIT-IDENTICAL: the f64 branch runs the exact arithmetic
// the closure did, and the f32 branch inlines the identical float32(<f64 math>).
func tanhVJP(_ *backend.Context, in, out []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
	x, y := in[0], out[0]
	gin := tensor.New(x.Dtype(), x.Shape())
	n := x.Numel()
	yc, gc := y.Contiguous(), g.Contiguous()
	if x.Dtype() == tensor.F64 && yc.Dtype() == tensor.F64 && gc.Dtype() == tensor.F64 {
		ys, gs, ds := yc.Storage().F64(), gc.Storage().F64(), gin.Storage().F64()
		for i := 0; i < n; i++ {
			ds[i] = gs[i] * (1 - ys[i]*ys[i])
		}
		return []*tensor.Tensor{gin}, nil
	}
	if x.Dtype() == tensor.F32 && yc.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
		ys, gs, ds := yc.Storage().F32(), gc.Storage().F32(), gin.Storage().F32()
		for i := 0; i < n; i++ {
			yv := float64(ys[i])
			ds[i] = float32(float64(gs[i]) * (1 - yv*yv))
		}
		return []*tensor.Tensor{gin}, nil
	}
	return unaryVJP(func(_, yv, gv float64) float64 { return gv * (1 - yv*yv) })(nil, in, out, attrs, g)
}

func expVJP(_ *backend.Context, in, out []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
	x, y := in[0], out[0]
	gin := tensor.New(x.Dtype(), x.Shape())
	n := x.Numel()
	yc, gc := y.Contiguous(), g.Contiguous()
	if x.Dtype() == tensor.F64 && yc.Dtype() == tensor.F64 && gc.Dtype() == tensor.F64 {
		ys, gs, ds := yc.Storage().F64(), gc.Storage().F64(), gin.Storage().F64()
		for i := 0; i < n; i++ {
			ds[i] = gs[i] * ys[i]
		}
		return []*tensor.Tensor{gin}, nil
	}
	if x.Dtype() == tensor.F32 && yc.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
		ys, gs, ds := yc.Storage().F32(), gc.Storage().F32(), gin.Storage().F32()
		for i := 0; i < n; i++ {
			ds[i] = float32(float64(gs[i]) * float64(ys[i]))
		}
		return []*tensor.Tensor{gin}, nil
	}
	return unaryVJP(func(_, yv, gv float64) float64 { return gv * yv })(nil, in, out, attrs, g)
}

func sigmoidVJP(_ *backend.Context, in, out []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
	x, y := in[0], out[0]
	gin := tensor.New(x.Dtype(), x.Shape())
	n := x.Numel()
	yc, gc := y.Contiguous(), g.Contiguous()
	if x.Dtype() == tensor.F64 && yc.Dtype() == tensor.F64 && gc.Dtype() == tensor.F64 {
		ys, gs, ds := yc.Storage().F64(), gc.Storage().F64(), gin.Storage().F64()
		for i := 0; i < n; i++ {
			ds[i] = gs[i] * ys[i] * (1 - ys[i])
		}
		return []*tensor.Tensor{gin}, nil
	}
	if x.Dtype() == tensor.F32 && yc.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
		ys, gs, ds := yc.Storage().F32(), gc.Storage().F32(), gin.Storage().F32()
		for i := 0; i < n; i++ {
			yv := float64(ys[i])
			ds[i] = float32(float64(gs[i]) * yv * (1 - yv))
		}
		return []*tensor.Tensor{gin}, nil
	}
	return unaryVJP(func(_, yv, gv float64) float64 { return gv * yv * (1 - yv) })(nil, in, out, attrs, g)
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
		// No-broadcast fast path: when a, b, g share one shape each output element maps
		// to exactly one (a,b) element at flat i, so no reduction is needed and the
		// per-element ravel + BroadcastCoords + multi-index dispatch are pure overhead.
		// ga/gb are pre-zeroed, so the single write per coord matches the +=. F32 keeps
		// the arithmetic in f64 (as AtF64 does) and rounds only on store — bit-identical.
		if a.Shape().Equal(outShape) && b.Shape().Equal(outShape) {
			if a.Dtype() == tensor.F64 && b.Dtype() == tensor.F64 && g.Dtype() == tensor.F64 {
				as, bs, gs := a.Contiguous().Storage().F64(), b.Contiguous().Storage().F64(), g.Contiguous().Storage().F64()
				gas, gbs := ga.Storage().F64(), gb.Storage().F64()
				for i := range gs {
					gas[i] = gs[i] / bs[i]
					gbs[i] = -gs[i] * as[i] / (bs[i] * bs[i])
				}
				return []*tensor.Tensor{ga, gb}, nil
			}
			if a.Dtype() == tensor.F32 && b.Dtype() == tensor.F32 && g.Dtype() == tensor.F32 {
				as, bs, gs := a.Contiguous().Storage().F32(), b.Contiguous().Storage().F32(), g.Contiguous().Storage().F32()
				gas, gbs := ga.Storage().F32(), gb.Storage().F32()
				for i := range gs {
					av, bv, gv := float64(as[i]), float64(bs[i]), float64(gs[i])
					gas[i] = float32(gv / bv)
					gbs[i] = float32(-gv * av / (bv * bv))
				}
				return []*tensor.Tensor{ga, gb}, nil
			}
		}
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
	RegisterVJP(backend.OpExp, expVJP)
	RegisterVJP(backend.OpLog, unaryVJP(func(x, _, g float64) float64 { return g / x }))
	RegisterVJP(backend.OpTanh, tanhVJP)
	RegisterVJP(backend.OpReLU, reluVJP)
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
	RegisterVJP(backend.OpSigmoid, sigmoidVJP)
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
		n := x.Numel()
		// Devirtualized fast paths: x and g share the (contiguous) forward shape, so
		// element i is flat i — the ravel and multi-index dispatch are pure overhead.
		// gin is pre-zeroed, so clamped elements (gradient 0) stay 0. Bit-identical.
		xc, gc := x.Contiguous(), g.Contiguous()
		if x.Dtype() == tensor.F64 && gc.Dtype() == tensor.F64 {
			xs, gs, ds := xc.Storage().F64(), gc.Storage().F64(), gin.Storage().F64()
			for i := 0; i < n; i++ {
				if xs[i] >= pa.Lo && xs[i] <= pa.Hi {
					ds[i] = gs[i]
				}
			}
			return []*tensor.Tensor{gin}, nil
		}
		if x.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
			xs, gs, ds := xc.Storage().F32(), gc.Storage().F32(), gin.Storage().F32()
			for i := 0; i < n; i++ {
				if v := float64(xs[i]); v >= pa.Lo && v <= pa.Hi {
					ds[i] = gs[i]
				}
			}
			return []*tensor.Tensor{gin}, nil
		}
		for i := range n { // generic fallback (mixed dtype / exotic layout)
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
		n := a.Numel()
		// Devirtualized fast paths: cond/g/da/db share the (contiguous) forward shape,
		// so element i is flat i. da/db are pre-zeroed, so the unselected branch stays
		// 0. Bit-identical to the generic rule (f32 store is an exact round-trip).
		if a.Dtype() == tensor.F64 && b.Dtype() == tensor.F64 && cond.Dtype() == tensor.F64 && g.Dtype() == tensor.F64 {
			cs, gs := cond.Contiguous().Storage().F64(), g.Contiguous().Storage().F64()
			das, dbs := da.Storage().F64(), db.Storage().F64()
			for i := 0; i < n; i++ {
				if cs[i] != 0 {
					das[i] = gs[i]
				} else {
					dbs[i] = gs[i]
				}
			}
			return []*tensor.Tensor{nil, da, db}, nil
		}
		if a.Dtype() == tensor.F32 && b.Dtype() == tensor.F32 && cond.Dtype() == tensor.F32 && g.Dtype() == tensor.F32 {
			cs, gs := cond.Contiguous().Storage().F32(), g.Contiguous().Storage().F32()
			das, dbs := da.Storage().F32(), db.Storage().F32()
			for i := 0; i < n; i++ {
				if cs[i] != 0 {
					das[i] = gs[i]
				} else {
					dbs[i] = gs[i]
				}
			}
			return []*tensor.Tensor{nil, da, db}, nil
		}
		for i := range n { // generic fallback (mixed dtype / exotic layout)
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
		// Devirtualized fast paths: y and g are row-major over the last axis, so
		// element (r,j) lives at flat r*d+j — the same value tensor.Unravel+AtF64
		// computes, but without the per-element ravel and multi-index dispatch (this
		// VJP is the attention softmax backward, run once per layer per step). gin is
		// freshly contiguous. The F32 branch keeps the dot and the product in f64
		// (matching AtF64's widen) and rounds only on store, so both fast paths are
		// bit-identical to the generic rule below.
		yc, gc := y.Contiguous(), g.Contiguous()
		if in[0].Dtype() == tensor.F64 && yc.Dtype() == tensor.F64 && gc.Dtype() == tensor.F64 {
			ys, gs, ds := yc.Storage().F64(), gc.Storage().F64(), gin.Storage().F64()
			for r := range rows {
				base := r * d
				var dot float64
				for j := range d {
					dot += gs[base+j] * ys[base+j]
				}
				for j := range d {
					ds[base+j] = ys[base+j] * (gs[base+j] - dot)
				}
			}
			return []*tensor.Tensor{gin}, nil
		}
		if in[0].Dtype() == tensor.F32 && yc.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
			ys, gs, ds := yc.Storage().F32(), gc.Storage().F32(), gin.Storage().F32()
			for r := range rows {
				base := r * d
				var dot float64
				for j := range d {
					dot += float64(gs[base+j]) * float64(ys[base+j])
				}
				for j := range d {
					ds[base+j] = float32(float64(ys[base+j]) * (float64(gs[base+j]) - dot))
				}
			}
			return []*tensor.Tensor{gin}, nil
		}
		for r := range rows { // generic fallback (mixed dtype / exotic layout)
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
