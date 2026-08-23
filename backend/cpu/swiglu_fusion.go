package cpu

import (
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/simd"
	"github.com/jxsl13/goai/tensor"
)

var _ backend.SwiGLUInPlaceFuser = (*Backend)(nil)
var _ backend.SwiGLUF32ChunkFuser = (*Backend)(nil)

func fuseSwiGLUF32(gate, up []float32) {
	if vexpF32Fast {
		vsiluF32(gate, gate)
	} else {
		for i := range gate {
			x := float64(gate[i])
			//perfscan:ignore PS4002 portable exact scalar-f64 fallback; SIMD builds take vsiluF32 above
			gate[i] = float32(x / (1 + math.Exp(-x)))
		}
	}
	simd.MulF32(gate, gate, up)
}

// FuseSwiGLUF32Chunk overwrites one raw producer chunk with SiLU(gate)*up.
// The fused Q4_K producer gives each concurrent call disjoint, equal-length
// slices and aligns non-final chunks to the widest supported SIMD lane count.
func (b *Backend) FuseSwiGLUF32Chunk(gate, up []float32) {
	if len(gate) != len(up) {
		panic("cpu: SwiGLU chunk lengths differ")
	}
	fuseSwiGLUF32(gate, up)
}

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
		fuseSwiGLUF32(g[lo:hi], u[lo:hi])
	})
	return true
}
