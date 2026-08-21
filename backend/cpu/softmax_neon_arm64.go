//go:build goexperiment.simd

package cpu

import "math"

//go:noescape
func rowMaxF32BlocksNeon(lanes, x *float32, blocks int, negInf float32)

//go:noescape
func scaleRowF32BlocksNeon(x *float32, blocks int, scale float32)

//go:noescape
func axpbRowF32BlocksNeon(x *float32, blocks int, a, b float32)

func rowMaxF32(x []float32) float32 {
	m := float32(math.Inf(-1))
	n16 := len(x) &^ 15
	if n16 > 0 {
		var lanes [16]float32
		rowMaxF32BlocksNeon(&lanes[0], &x[0], n16>>4, m)
		for _, v := range lanes {
			if v > m {
				m = v
			}
		}
	}
	for i := n16; i < len(x); i++ {
		if x[i] > m {
			m = x[i]
		}
	}
	if m == 0 {
		// FMAXNM intentionally skips NaNs, but IEEE maximumNumber resolves a
		// mixed ±0 tie to +0. The scalar greater-than reduction instead keeps
		// the first zero that became the maximum, so repair that rare result.
		for _, v := range x {
			if v == 0 {
				return v
			}
		}
	}
	return m
}

func scaleRowF32(x []float32, scale float32) {
	n16 := len(x) &^ 15
	if n16 > 0 {
		scaleRowF32BlocksNeon(&x[0], n16>>4, scale)
	}
	for i := n16; i < len(x); i++ {
		x[i] *= scale
	}
}

func axpbRowF32(x []float32, a, b float32) {
	n16 := len(x) &^ 15
	if n16 > 0 {
		axpbRowF32BlocksNeon(&x[0], n16>>4, a, b)
	}
	for i := n16; i < len(x); i++ {
		x[i] = x[i]*a + b
	}
}
