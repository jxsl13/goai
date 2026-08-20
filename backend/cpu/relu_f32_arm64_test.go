//go:build arm64

package cpu

import (
	"math"
	"testing"
)

func TestReLUF32Arm64ExactAllLengths(t *testing.T) {
	edges := []uint32{
		0x00000000, 0x80000000,
		0x00000001, 0x80000001,
		0x007fffff, 0x807fffff,
		0x00800000, 0x80800000,
		0x3f800000, 0xbf800000,
		0x7f7fffff, 0xff7fffff,
		0x7f800000, 0xff800000,
		0x7fc00001, 0xffc00001,
		0x7f800001, 0xff800001,
	}
	for n := 0; n <= 129; n++ {
		srcBacking := make([]float32, n+3)
		dstBacking := make([]float32, n+7)
		src := srcBacking[1 : 1+n]
		dst := dstBacking[3 : 3+n]
		state := uint32(0x9e3779b9)
		for i := range src {
			bits := edges[i%len(edges)]
			if i >= len(edges) {
				state ^= state << 13
				state ^= state >> 17
				state ^= state << 5
				bits = state
			}
			src[i] = math.Float32frombits(bits)
			dst[i] = math.Float32frombits(0x7fc0dead)
		}

		reluF32(dst, src)
		for i, value := range src {
			want := uint32(0)
			if value > 0 {
				want = math.Float32bits(value)
			}
			if got := math.Float32bits(dst[i]); got != want {
				t.Fatalf("n=%d i=%d input=%08x: got %08x, want %08x", n, i, math.Float32bits(value), got, want)
			}
		}
	}
}
