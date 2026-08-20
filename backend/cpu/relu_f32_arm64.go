//go:build arm64

package cpu

// reluF32BlocksNeon applies x > 0 ? x : +0 to 16*blocks values. The ordered
// comparison makes every NaN false; selecting from an integer zero vector also
// turns both input zero signs into +0, exactly matching the scalar Go kernel.
//
//go:noescape
func reluF32BlocksNeon(dst, src *float32, blocks int)

func reluF32(dst, src []float32) {
	src = src[:len(dst)]
	nv := len(dst) &^ 15
	if nv > 0 {
		reluF32BlocksNeon(&dst[0], &src[0], nv>>4)
	}
	for i := nv; i < len(dst); i++ {
		if src[i] > 0 {
			dst[i] = src[i]
		} else {
			dst[i] = 0
		}
	}
}
