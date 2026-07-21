package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Concat VJP: the forward joins the inputs along an axis, so the gradient is
// simply g sliced back into each input's segment along that axis — dInₖ[coords] =
// g[coords with axisᵢ += offsetₖ]. One gradient tensor per input.
func init() {
	RegisterVJP(backend.OpConcat, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		pa, _ := attrs.(backend.ConcatAttrs)
		shapes := make([]tensor.Shape, len(in))
		for i, t := range in {
			shapes[i] = t.Shape()
		}
		_, offsets, ax, err := backend.ConcatPlan(shapes, pa.Axis)
		if err != nil {
			return nil, err
		}
		// Each input's gradient is g's band [off:off+tAx] along ax. For a fixed set of
		// the dims BEFORE ax, that band is a CONTIGUOUS run, so each piece is one bulk
		// copy() per outer index (typed F32/F64) — not a per-element Unravel+AtF64+
		// SetF64 gather with an []int coord alloc every element. Bit-identical (each
		// gradient is fully written from g's band); generic fallback for exotic
		// dtypes / non-contiguous g.
		gc := g.Contiguous()
		gAx := g.Shape()[ax]
		grads := make([]*tensor.Tensor, len(in))
		for i, t := range in {
			off := offsets[i]
			s := t.Shape()
			d := tensor.New(t.Dtype(), s)
			inner := 1
			for dd := ax + 1; dd < len(s); dd++ {
				inner *= s[dd]
			}
			outer := 1
			for dd := 0; dd < ax; dd++ {
				outer *= s[dd]
			}
			tAx := s[ax]
			run := tAx * inner
			done := false
			switch t.Dtype() {
			case tensor.F64:
				if gc.Dtype() == tensor.F64 {
					gs, ds := gc.Storage().F64(), d.Storage().F64()
					for o := 0; o < outer; o++ {
						copy(ds[o*run:o*run+run], gs[o*gAx*inner+off*inner:])
					}
					done = true
				}
			case tensor.F32:
				if gc.Dtype() == tensor.F32 {
					gs, ds := gc.Storage().F32(), d.Storage().F32()
					for o := 0; o < outer; o++ {
						copy(ds[o*run:o*run+run], gs[o*gAx*inner+off*inner:])
					}
					done = true
				}
			}
			if !done { // generic per-element fallback (exotic dtype / layout)
				for e := range t.Numel() {
					coords := tensor.Unravel(e, s)
					gco := append([]int(nil), coords...)
					gco[ax] += off
					d.SetF64(g.AtF64(gco...), coords...)
				}
			}
			grads[i] = d
		}
		return grads, nil
	})
}
