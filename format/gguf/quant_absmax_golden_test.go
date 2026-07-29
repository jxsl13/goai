package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// quantizeQ8_0AbsRef is the frozen pre-change quantizer (math.Abs via f64 round-trip),
// the byte-exactness oracle for the sign-bit-clear abs optimization.
func quantizeQ8_0AbsRef(x []float32) []byte {
	nb := len(x) / blockElems
	out := make([]byte, nb*34)
	for b := range nb {
		blk := x[b*blockElems : (b+1)*blockElems]
		var amax float32
		for _, v := range blk {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		d := amax / 127
		var id float32
		if d != 0 {
			id = 1 / d
		}
		o := b * 34
		binary.LittleEndian.PutUint16(out[o:], f32ToF16(d))
		for j, v := range blk {
			out[o+2+j] = byte(int8(roundHalfAway(v * id)))
		}
	}
	return out
}

func TestQuantizeQ8_0AbsByteIdentical(t *testing.T) {
	for _, n := range []int{32, 256, 4096, 1 << 16} {
		x := benchF32(n)
		got := quantizeQ8_0(x)
		want := quantizeQ8_0AbsRef(x)
		if !bytes.Equal(got, want) {
			t.Fatalf("n=%d: sign-bit-clear abs changed quantized bytes vs math.Abs oracle", n)
		}
	}
}
