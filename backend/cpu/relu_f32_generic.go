//go:build !arm64

package cpu

func reluF32(dst, src []float32) {
	src = src[:len(dst)]
	for i := range dst {
		if src[i] > 0 {
			dst[i] = src[i]
		} else {
			dst[i] = 0
		}
	}
}
