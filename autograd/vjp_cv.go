package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// CV VJPs (§T24b, closes §B24). Direct-loop implementations mirroring the
// forward geometry, accumulating in f64 (§V10):
//
//	conv2d:  gx[n,c,iy,ix] += g·w[f,c,ky,kx];  gw += g·x;  gb[f] = Σ g[n,f,·,·]
//	maxpool: g routes to the FIRST window element attaining the max (§B16 rule)
//	avgpool: g/k² spreads uniformly over the window

// poolAccessors returns flat-index readers for x, y, g and an accumulating writer
// for gx on raw contiguous storage (§T463). The writer replicates the accessor
// path's exact numerics — f32 accumulates with a narrow after every add
// (float32(float64(old)+gv)), f64 adds directly; reads widen to f64 like AtF64.
// Non-f32/f64 dtypes fall back to the (slow) tensor accessors.
//
//perfscan:ignore PS6004 explicitly not-perf: unverified-invariant correctness check
func poolAccessors(x, y, g, gx *tensor.Tensor) (getX, getY, getG func(int) float64, addGX func(int, float64)) {
	reader := func(t *tensor.Tensor) func(int) float64 {
		tc := t.Contiguous()
		switch t.Dtype() {
		case tensor.F64:
			s := tc.Storage().F64()
			return func(i int) float64 { return s[i] }
		case tensor.F32:
			s := tc.Storage().F32()
			return func(i int) float64 { return float64(s[i]) }
		default:
			return func(i int) float64 { return tc.AtF64(tensor.Unravel(i, tc.Shape())...) }
		}
	}
	switch gx.Dtype() {
	case tensor.F64:
		s := gx.Storage().F64()
		addGX = func(i int, v float64) { s[i] += v }
	case tensor.F32:
		s := gx.Storage().F32()
		addGX = func(i int, v float64) { s[i] = float32(float64(s[i]) + v) }
	default:
		addGX = func(i int, v float64) {
			idx := tensor.Unravel(i, gx.Shape())
			gx.SetF64(gx.AtF64(idx...)+v, idx...)
		}
	}
	return reader(x), reader(y), reader(g), addGX
}

func init() {
	// conv2d backward is the fused OpConv2DBackward, dispatched on the tape's active
	// backend (GPU when available, §T101) — the kernel returns (dX,dW,dBias); drop
	// dBias when the forward had no bias input.
	RegisterVJP(backend.OpConv2D, func(ctx *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		grads, err := backend.Execute(ctx, backend.OpConv2DBackward, []*tensor.Tensor{in[0], in[1], g}, attrs)
		if err != nil {
			return nil, err
		}
		if len(in) == 3 {
			return grads, nil // dX, dW, dBias
		}
		return grads[:2], nil // dX, dW
	})

	RegisterVJP(backend.OpMaxPool2D, func(_ *backend.Context, in, out []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x, y := in[0], out[0]
		n, c, h, w := x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
		pX, _ := attrs.(backend.PoolAttrs)
		pX = pX.WithDefaults()
		k := pX.Kernel
		s := pX.Stride
		ho, wo := y.Shape()[2], y.Shape()[3]
		gx := tensor.New(x.Dtype(), x.Shape())
		// raw-storage fast path (§T463: the accessor loops were ~half a CNN training
		// step); identical routing — g goes to the FIRST window element attaining the
		// max (§B16), NaN windows route to the first element.
		// Typed arms first. The closure path below costs one INDIRECT CALL per element read and
		// per element written — the AtF64 anti-pattern one level shallower, and invisible to a
		// reader who sees only a helper named "accessors". The routing is character-for-character
		// the same in all three arms: g goes to the FIRST window element attaining the max, and a
		// NaN window falls through to the first element.
		if xs, ys, gs, gxs, ok := poolSlicesF64(x, y, g, gx); ok {
			maxPoolBackF64(xs, ys, gs, gxs, n*c, h, w, ho, wo, k, s)
			return []*tensor.Tensor{gx}, nil
		}
		if xs, ys, gs, gxs, ok := poolSlicesF32(x, y, g, gx); ok {
			maxPoolBackF32(xs, ys, gs, gxs, n*c, h, w, ho, wo, k, s)
			return []*tensor.Tensor{gx}, nil
		}
		getX, getY, getG, addGX := poolAccessors(x, y, g, gx)
		for pl := range n * c {
			xB, yB := pl*h*w, pl*ho*wo
			for oy := range ho {
				for ox := range wo {
					m := getY(yB + oy*wo + ox)
					gv := getG(yB + oy*wo + ox)
					routed := false
					for ky := 0; ky < k && !routed; ky++ {
						row := xB + (oy*s+ky)*w + ox*s
						for kx := 0; kx < k && !routed; kx++ {
							if getX(row+kx) == m {
								addGX(row+kx, gv)
								routed = true
							}
						}
					}
					if !routed { // NaN window: max is NaN, == fails; route to first
						addGX(xB+oy*s*w+ox*s, gv)
					}
				}
			}
		}
		return []*tensor.Tensor{gx}, nil
	})

	RegisterVJP(backend.OpAvgPool2D, func(_ *backend.Context, in, out []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x, y := in[0], out[0]
		n, c, h, w := x.Shape()[0], x.Shape()[1], x.Shape()[2], x.Shape()[3]
		pX, _ := attrs.(backend.PoolAttrs)
		pX = pX.WithDefaults()
		k := pX.Kernel
		s := pX.Stride
		ho, wo := y.Shape()[2], y.Shape()[3]
		inv := 1 / float64(k*k)
		gx := tensor.New(x.Dtype(), x.Shape())
		if _, _, gs, gxs, ok := poolSlicesF64(x, y, g, gx); ok {
			avgPoolBackF64(gs, gxs, n*c, h, w, ho, wo, k, s, inv)
			return []*tensor.Tensor{gx}, nil
		}
		if _, _, gs, gxs, ok := poolSlicesF32(x, y, g, gx); ok {
			avgPoolBackF32(gs, gxs, n*c, h, w, ho, wo, k, s, inv)
			return []*tensor.Tensor{gx}, nil
		}
		_, _, getG, addGX := poolAccessors(x, y, g, gx)
		for pl := range n * c {
			xB, yB := pl*h*w, pl*ho*wo
			for oy := range ho {
				for ox := range wo {
					gv := getG(yB+oy*wo+ox) * inv
					for ky := range k {
						row := xB + (oy*s+ky)*w + ox*s
						for kx := range k {
							addGX(row+kx, gv)
						}
					}
				}
			}
		}
		return []*tensor.Tensor{gx}, nil
	})
}

// poolSlicesF64 and poolSlicesF32 expose the four pooling tensors as raw contiguous slices when
// every one of them has that dtype. All four must match: a mixed set has to keep the widening
// accessor path, since the arms differ in where they round.
func poolSlicesF64(x, y, g, gx *tensor.Tensor) (xs, ys, gs, gxs []float64, ok bool) {
	if x.Dtype() != tensor.F64 || y.Dtype() != tensor.F64 ||
		g.Dtype() != tensor.F64 || gx.Dtype() != tensor.F64 {
		return nil, nil, nil, nil, false
	}
	return x.Contiguous().Storage().F64(), y.Contiguous().Storage().F64(),
		g.Contiguous().Storage().F64(), gx.Storage().F64(), true
}

func poolSlicesF32(x, y, g, gx *tensor.Tensor) (xs, ys, gs, gxs []float32, ok bool) {
	if x.Dtype() != tensor.F32 || y.Dtype() != tensor.F32 ||
		g.Dtype() != tensor.F32 || gx.Dtype() != tensor.F32 {
		return nil, nil, nil, nil, false
	}
	return x.Contiguous().Storage().F32(), y.Contiguous().Storage().F32(),
		g.Contiguous().Storage().F32(), gx.Storage().F32(), true
}

// maxPoolBackF64 routes each output gradient to the first window element attaining the max.
func maxPoolBackF64(xs, ys, gs, gxs []float64, planes, h, w, ho, wo, k, s int) {
	//perfscan:ignore PS3059 stale line (file 131 lines); conv2d bwd now fused OpConv2DBackward
	for pl := range planes {
		xB, yB := pl*h*w, pl*ho*wo
		for oy := range ho {
			for ox := range wo {
				o := yB + oy*wo + ox
				m, gv := ys[o], gs[o]
				routed := false
				for ky := 0; ky < k && !routed; ky++ {
					row := xB + (oy*s+ky)*w + ox*s
					for kx := 0; kx < k && !routed; kx++ {
						if xs[row+kx] == m {
							//perfscan:ignore PS3075 stale line; pool VJP already raw-storage fast path §T463
							gxs[row+kx] += gv
							routed = true
						}
					}
				}
				if !routed { // NaN window: max is NaN, == fails; route to first
					gxs[xB+oy*s*w+ox*s] += gv
				}
			}
		}
	}
}

// maxPoolBackF32 is maxPoolBackF64 with the accessor path's rounding: reads widen to f64 and every
// accumulation narrows again, so the arms agree bit for bit.
func maxPoolBackF32(xs, ys, gs, gxs []float32, planes, h, w, ho, wo, k, s int) {
	//perfscan:ignore PS3059 stale line; conv2d bwd fused to backend kernel
	for pl := range planes {
		xB, yB := pl*h*w, pl*ho*wo
		for oy := range ho {
			for ox := range wo {
				o := yB + oy*wo + ox
				m, gv := float64(ys[o]), float64(gs[o])
				routed := false
				for ky := 0; ky < k && !routed; ky++ {
					row := xB + (oy*s+ky)*w + ox*s
					for kx := 0; kx < k && !routed; kx++ {
						if float64(xs[row+kx]) == m {
							gxs[row+kx] = float32(float64(gxs[row+kx]) + gv)
							routed = true
						}
					}
				}
				if !routed {
					i := xB + oy*s*w + ox*s
					gxs[i] = float32(float64(gxs[i]) + gv)
				}
			}
		}
	}
}

// avgPoolBackF64 spreads each output gradient over its whole window.
func avgPoolBackF64(gs, gxs []float64, planes, h, w, ho, wo, k, s int, inv float64) {
	//perfscan:ignore PS3059 stale line; pool VJP already typed fast path
	for pl := range planes {
		xB, yB := pl*h*w, pl*ho*wo
		for oy := range ho {
			for ox := range wo {
				// The conversion forces the intermediate rounding. Without it the product and
				// the accumulating add below fuse into an FMA — Go may fuse ACROSS statements —
				// and this arm drifts one ulp from the accessor path, where the add lives inside
				// a closure and cannot be contracted with the multiply (§PS3025). The gate caught
				// exactly this, on an element that nine overlapping windows accumulate into.
				gv := float64(gs[yB+oy*wo+ox] * inv)
				for ky := range k {
					row := xB + (oy*s+ky)*w + ox*s
					for kx := range k {
						//perfscan:ignore PS3075 stale line; already-optimized raw-storage fast path
						gxs[row+kx] += gv
					}
				}
			}
		}
	}
}

// avgPoolBackF32 keeps the accessor path's per-add narrowing.
func avgPoolBackF32(gs, gxs []float32, planes, h, w, ho, wo, k, s int, inv float64) {
	//perfscan:ignore PS3059 stale line; conv2d bwd fused/GPU-dispatched
	for pl := range planes {
		xB, yB := pl*h*w, pl*ho*wo
		for oy := range ho {
			for ox := range wo {
				gv := float64(float64(gs[yB+oy*wo+ox]) * inv) // see the F64 arm: blocks FMA fusion
				for ky := range k {
					row := xB + (oy*s+ky)*w + ox*s
					for kx := range k {
						gxs[row+kx] = float32(float64(gxs[row+kx]) + gv)
					}
				}
			}
		}
	}
}
