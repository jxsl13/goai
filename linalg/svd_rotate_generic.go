//go:build !arm64

package linalg

func svdRotateVSecond(c, b, sn, a float64) float64 {
	return sn*a + c*b
}
