//go:build arm64

package cpu

import "math"

// A single NEON stream is faster than worker-pool fan-out below this measured
// complete-operation crossover on M2 Pro.
const negF32ParallelThreshold = 1 << 20

// negF32BlocksNeon toggles only the sign bit for 16*blocks values.
//
//go:noescape
func negF32BlocksNeon(dst, src *float32, blocks int)

func negF32(dst, src []float32) {
	src = src[:len(dst)]
	nv := len(dst) &^ 15
	if nv > 0 {
		negF32BlocksNeon(&dst[0], &src[0], nv>>4)
	}
	for i := nv; i < len(dst); i++ {
		dst[i] = math.Float32frombits(math.Float32bits(src[i]) ^ uint32(1<<31))
	}
}
