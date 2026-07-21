//go:build !(goexperiment.simd && amd64)

package cpu

import "math"

// Non-(amd64 SIMD) builds keep the scalar row-max and scale loops of the f32 softmax fast path
// bit-for-bit (arm64's SIMD build included — vectorizing these is an amd64-only change).

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
