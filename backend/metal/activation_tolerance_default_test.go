//go:build darwin && cgo && !(arm64 && goexperiment.simd)

package metal_test

import "math"

func metalActivationCrossClose(got, want float64) bool {
	return math.Abs(got-want) <= 1e-5*math.Max(1, math.Abs(want))
}
