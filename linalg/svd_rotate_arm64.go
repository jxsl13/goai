//go:build arm64

package linalg

import "math"

// svdRotateVSecond pins the contraction order used by Go 1.26 on ARM64. Go 1.27
// otherwise chooses the other valid FMA contraction for the same source expression.
func svdRotateVSecond(c, b, sn, a float64) float64 {
	return math.FMA(c, b, sn*a)
}
