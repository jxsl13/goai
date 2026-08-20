//go:build !arm64

package cpu

import "math"

const negF32ParallelThreshold = 0

func negF32(dst, src []float32) {
	src = src[:len(dst)]
	for i, value := range src {
		dst[i] = math.Float32frombits(math.Float32bits(value) ^ uint32(1<<31))
	}
}
