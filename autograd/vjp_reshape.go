package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Reshape VJP: reshape is a pure re-layout (row-major order preserved), so the
// gradient is g reshaped back to the input's shape — dx[flat i] = g[flat i].
//
// This runs on the training BACKWARD path for every reshape/flatten/head-split,
// so the per-element SetF64(Unravel(i)…) loop was a real cost (an index-slice
// allocation and a strided dispatch per element). When the incoming cotangent g
// is contiguous and offset-0 its backing storage IS its logical row-major order,
// and dx is a freshly allocated (therefore contiguous, offset-0) tensor of the
// same dtype (reshape preserves dtype), so the whole gradient is a single typed
// copy(). The result is bit-identical to the slow path: the F32 round-trip
// float32(float64(v)) is exact, so the copied bits match SetF64(AtF64(·)) element
// for element. A view is deliberately NOT returned — a fresh allocation keeps the
// gradient's storage un-aliased from the reshape output's cotangent, matching the
// old code's ownership exactly. Non-contiguous g and the half-float dtypes
// (F16/BF16, whose SetF64 carries the correct rounding) fall through to the
// general accessor loop.
func init() {
	RegisterVJP(backend.OpReshape, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		dx := tensor.New(x.Dtype(), x.Shape())
		n := x.Numel()
		if g.IsContiguous() && g.Offset() == 0 {
			switch x.Dtype() {
			case tensor.F64:
				copy(dx.Storage().F64(), g.Storage().F64()[:n])
				return []*tensor.Tensor{dx}, nil
			case tensor.F32:
				copy(dx.Storage().F32(), g.Storage().F32()[:n])
				return []*tensor.Tensor{dx}, nil
			}
		}
		xs, gs := x.Shape(), g.Shape()
		for i := range n {
			dx.SetF64(g.AtF64(tensor.Unravel(i, gs)...), tensor.Unravel(i, xs)...)
		}
		return []*tensor.Tensor{dx}, nil
	})
}
