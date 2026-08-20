package cpu

import (
	"math"
	"testing"
)

var negF32EdgeBits = []uint32{
	0x00000000, 0x80000000,
	0x00000001, 0x80000001,
	0x007fffff, 0x807fffff,
	0x00800000, 0x80800000,
	0x3f800000, 0xbf800000,
	0x7f7fffff, 0xff7fffff,
	0x7f800000, 0xff800000,
	0x7f800001, 0xff800001,
	0x7fa00001, 0xffa00001,
	0x7fc00000, 0xffc00000,
	0x7fffffff, 0xffffffff,
}

func nextNegF32TestBits(state *uint32, i int) uint32 {
	if i < len(negF32EdgeBits) {
		return negF32EdgeBits[i]
	}
	*state ^= *state << 13
	*state ^= *state >> 17
	*state ^= *state << 5
	return *state
}

func TestNegF32ExactAllLengths(t *testing.T) {
	const guardBits = uint32(0x7fc0dead)
	for n := 0; n <= 257; n++ {
		srcBacking := make([]float32, n+3)
		dstBacking := make([]float32, n+7)
		src := srcBacking[1 : 1+n]
		dst := dstBacking[3 : 3+n]
		state := uint32(0x9e3779b9)
		for i := range dstBacking {
			dstBacking[i] = math.Float32frombits(guardBits)
		}
		for i := range src {
			src[i] = math.Float32frombits(nextNegF32TestBits(&state, i))
		}

		negF32(dst, src)
		if got := math.Float32bits(dstBacking[0]); got != guardBits {
			t.Fatalf("n=%d: leading guard changed to %08x", n, got)
		}
		if got := math.Float32bits(dstBacking[len(dstBacking)-1]); got != guardBits {
			t.Fatalf("n=%d: trailing guard changed to %08x", n, got)
		}
		for i, value := range src {
			want := math.Float32bits(value) ^ uint32(1<<31)
			if got := math.Float32bits(dst[i]); got != want {
				t.Fatalf("n=%d i=%d input=%08x: got %08x, want %08x", n, i, math.Float32bits(value), got, want)
			}
		}
	}
}
