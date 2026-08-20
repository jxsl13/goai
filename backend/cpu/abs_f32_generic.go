//go:build !arm64

package cpu

import "math"

const absF32ParallelThreshold = 0

func absF32(dst, src []float32) {
	src = src[:len(dst)]
	for i, value := range src {
		//perfscan:ignore PS5007 exact portable path; sign clearing does not quiet signaling NaNs
		dst[i] = float32(math.Abs(float64(value)))
	}
}
