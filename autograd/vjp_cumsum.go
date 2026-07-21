package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Cumsum VJP: since out[i] = Σ_{j≤i} x[j], each x[j] feeds every out[i] with i≥j, so
// the gradient is the REVERSE cumulative sum — dx[j] = Σ_{i≥j} g[i] along the axis.
func init() {
	RegisterVJP(backend.OpCumsum, func(_ *backend.Context, in, _ []*tensor.Tensor, attrs backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		pa, _ := attrs.(backend.CumsumAttrs)
		ax, reduced, err := backend.CumsumPlan(x.Shape(), pa.Axis)
		if err != nil {
			return nil, err
		}
		L := x.Shape()[ax]
		dx := tensor.New(x.Dtype(), x.Shape())
		xShape := x.Shape()
		// typed fast path: the ax elements of one line sit at flat offset base + i·inner
		// (inner = product of dims after ax), so the reverse cumsum is a strided scan
		// over []T with no per-element AtF64/SetF64 dispatch. Bit-identical: same
		// descending-i accumulation order.
		inner := 1
		for d := ax + 1; d < len(xShape); d++ {
			inner *= xShape[d]
		}
		outer := 1
		for d := 0; d < ax; d++ {
			outer *= xShape[d]
		}
		gc := g.Contiguous()
		if x.Dtype() == tensor.F64 && gc.Dtype() == tensor.F64 {
			gs, ds := gc.Storage().F64(), dx.Storage().F64()
			for o := 0; o < outer; o++ {
				for j := 0; j < inner; j++ {
					base := o*L*inner + j
					var sum float64
					for i := L - 1; i >= 0; i-- {
						sum += gs[base+i*inner]
						ds[base+i*inner] = sum
					}
				}
			}
			return []*tensor.Tensor{dx}, nil
		}
		if x.Dtype() == tensor.F32 && gc.Dtype() == tensor.F32 {
			gs, ds := gc.Storage().F32(), dx.Storage().F32()
			for o := 0; o < outer; o++ {
				for j := 0; j < inner; j++ {
					base := o*L*inner + j
					var sum float64
					for i := L - 1; i >= 0; i-- {
						sum += float64(gs[base+i*inner])
						ds[base+i*inner] = float32(sum)
					}
				}
			}
			return []*tensor.Tensor{dx}, nil
		}
		coord := make([]int, x.Ndim())
		for l := range reduced.Numel() { // generic per-element fallback
			backend.FillLineCoord(coord, l, reduced, ax)
			var sum float64
			for i := L - 1; i >= 0; i-- {
				coord[ax] = i
				sum += g.AtF64(coord...)
				dx.SetF64(sum, coord...)
			}
		}
		return []*tensor.Tensor{dx}, nil
	})
}
