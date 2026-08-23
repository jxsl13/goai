package backend

import "github.com/jxsl13/goai/tensor"

// SwiGLUInPlaceFuser is an optional eager-inference capability for the
// elementwise middle of a SwiGLU block. Implementations overwrite gate with
// SiLU(gate)*up and leave up unchanged. Returning false means the inputs are
// unsupported and the caller must retain the ordinary OpSiLU plus OpMul path.
//
// The caller owns gate exclusively: it must never pass a public input, a
// recorded activation, or a tensor that survives the fused call.
type SwiGLUInPlaceFuser interface {
	FuseSwiGLUInPlace(gate, up *tensor.Tensor) bool
}

// SwiGLUF32ChunkFuser is the raw-slice form of SwiGLUInPlaceFuser for fused
// producers. Implementations overwrite gate with SiLU(gate)*up, leave up
// unchanged, and support every pair of equal-length slices. Calls may run
// concurrently on disjoint chunks. Implementations must not retain either
// slice after the call returns.
type SwiGLUF32ChunkFuser interface {
	FuseSwiGLUF32Chunk(gate, up []float32)
}
