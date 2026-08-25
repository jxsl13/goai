//go:build arm64

package cpu

import "math"

// A single NEON stream is faster than worker-pool fan-out below this measured
// complete-operation crossover on M2 Pro. Reprice this whenever the leaf
// kernel, worker pool, or Go toolchain changes materially.
const absF32ParallelThreshold = 1 << 21

// absF32BlocksNeon computes the active Go toolchain's scalar Abs result for
// 16*blocks values. Go release build tags select the matching NaN behavior.
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
		//perfscan:ignore PS5007 exact scalar tail; sign clearing does not quiet signaling NaNs
		dst[i] = float32(math.Abs(float64(src[i])))
	}
}
