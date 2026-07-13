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
