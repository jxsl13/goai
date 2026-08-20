//go:build arm64

package cpu

import "math"

// A single NEON stream is faster than worker-pool fan-out below this measured
// complete-operation crossover on M2 Pro.
const absF32ParallelThreshold = 1 << 18

// absF32BlocksNeon clears every sign bit and quiets signaling NaNs for
// 16*blocks values. It preserves all remaining payload bits.
//
//go:noescape
func absF32BlocksNeon(dst, src *float32, blocks int)

func absF32(dst, src []float32) {
	src = src[:len(dst)]
	nv := len(dst) &^ 15
	if nv > 0 {
		absF32BlocksNeon(&dst[0], &src[0], nv>>4)
	}
	for i := nv; i < len(dst); i++ {
		dst[i] = float32(math.Abs(float64(src[i])))
	}
}
