package cpu

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/simd"
	"github.com/jxsl13/goai/tensor"
)

var _ backend.SwiGLUInPlaceFuser = (*Backend)(nil)

// FuseSwiGLUInPlace overwrites the private gate projection with
// SiLU(gate)*up. Quantized decode owns both projection outputs and immediately
// feeds the product to the down projection, so this removes the separate SiLU
// and Mul output tensors without changing any caller-owned input.
func (b *Backend) FuseSwiGLUInPlace(gate, up *tensor.Tensor) bool {
	if gate == nil || up == nil || gate.Dtype() != tensor.F32 || up.Dtype() != tensor.F32 ||
		!gate.Shape().Equal(up.Shape()) || !gate.IsContiguous() || !up.IsContiguous() ||
		gate.Offset() != 0 || up.Offset() != 0 || gate.Storage() == up.Storage() {
		return false
	}
	g, u := gate.Storage().F32(), up.Storage().F32()
	parallel(len(g), func(lo, hi int) {
		gd := g[lo:hi]
		if vexpF32Fast {
			vsiluF32(gd, gd)
		} else {
			for i := range gd {
				x := float64(gd[i])
				gd[i] = float32(x / (1 + math.Exp(-x)))
			}
		}
		simd.MulF32(gd, gd, u[lo:hi])
	})
	return true
}
