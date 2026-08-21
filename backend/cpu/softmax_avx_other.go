//go:build !(goexperiment.simd && (amd64 || arm64))

package cpu

import "math"

// Builds without an architecture-specific SIMD implementation keep the scalar
// row-max, scale, and affine loops of the f32 softmax fast path bit-for-bit.

func rowMaxF32(x []float32) float32 {
	m := float32(math.Inf(-1))
	for _, v := range x {
		if v > m {
			m = v
		}
	}
	return m
}

func scaleRowF32(x []float32, inv float32) {
	for j := range x {
		x[j] *= inv
	}
}

func axpbRowF32(x []float32, a, b float32) {
	for j := range x {
		x[j] = x[j]*a + b
	}
}
