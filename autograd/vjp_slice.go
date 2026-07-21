package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Slice VJP: the forward extracts x[..,Start:End,..], so the gradient scatters g
// back into a zero tensor of the input's shape at the sliced range (zeros outside
// it) — dx[coords with axis+Start] = g[coords], 0 elsewhere. When Split composes
// several slices of the same x, the tape accumulates their scatters into the full
// gradient.
func init() {
	RegisterVJP(backend.OpSlice, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		pa, _ := attrs.(backend.SliceAttrs)
		ax, _, err := backend.SlicePlan(x.Shape(), pa.Axis, pa.Start, pa.End)
		if err != nil {
			return nil, err
		}
		dx := tensor.New(x.Dtype(), x.Shape()) // zero-initialized
		xShape := x.Shape()
		// The slice backward scatters g into dx's [Start:End] band along axis ax and
		// leaves the rest zero. For a fixed set of the dims BEFORE ax, dx's band and
		// g's block are each a CONTIGUOUS run (ax·inner elements) — so the whole VJP
		// is one bulk copy per outer index, not a per-element Unravel+SetF64 scatter
		// (which also alloc'd a coord slice per element). Bit-identical: same values,
		// dx pre-zeroed outside the band. Typed F32/F64 fast paths; generic fallback
		// for exotic dtypes / non-contiguous g.
		inner := 1
		for d := ax + 1; d < len(xShape); d++ {
			inner *= xShape[d]
		}
		outer := 1
		for d := 0; d < ax; d++ {
			outer *= xShape[d]
		}
		gAx, xAx := g.Shape()[ax], xShape[ax]
		run := gAx * inner
		gc := g.Contiguous()
		copied := false
		switch x.Dtype() {
		case tensor.F64:
			if gc.Dtype() == tensor.F64 {
				gs, ds := gc.Storage().F64(), dx.Storage().F64()
				for o := 0; o < outer; o++ {
					copy(ds[o*xAx*inner+pa.Start*inner:], gs[o*run:o*run+run])
				}
				copied = true
			}
		case tensor.F32:
			if gc.Dtype() == tensor.F32 {
				gs, ds := gc.Storage().F32(), dx.Storage().F32()
				for o := 0; o < outer; o++ {
					copy(ds[o*xAx*inner+pa.Start*inner:], gs[o*run:o*run+run])
				}
				copied = true
			}
		}
		if !copied { // generic per-element fallback (exotic dtype / layout)
			gShape := g.Shape()
			for e := range g.Numel() {
				c := tensor.Unravel(e, gShape)
				xc := append([]int(nil), c...)
				xc[ax] += pa.Start
				dx.SetF64(g.AtF64(c...), xc...)
			}
		}
		return []*tensor.Tensor{dx}, nil
	})
}
