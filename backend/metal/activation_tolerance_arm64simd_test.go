//go:build darwin && cgo && arm64 && goexperiment.simd

package metal_test

import "math"

// CPU-002: typed f32 SIMD activations use rel=2e-3, abs=1e-4.
func metalActivationCrossClose(got, want float64) bool {
	return math.Abs(got-want) <= 1e-4+2e-3*math.Abs(want)
}
