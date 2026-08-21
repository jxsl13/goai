//go:build arm64

package gguf

import (
	"math"
	"testing"
)

func TestDotQ6KBlockNeonKnownValue(t *testing.T) {
	x := [qkK]float32{}
	raw := [q6kBlockSize]byte{}
	for i := range x {
		x[i] = 1
	}
	for i := range 128 {
		raw[i] = 0x11 // low nibbles of all four streams are one
	}
	for i := 128; i < 192; i++ {
		raw[i] = 0xaa // high two bits of all four streams are two: q6=33
	}
	for i := 192; i < 208; i++ {
		raw[i] = 1
	}
	got := dotQ6KBlockNeon(&x[0], &raw[0], 1, &qKByteToF32Indexes[0])
	if got != 256 {
		t.Fatalf("dotQ6KBlockNeon = %v (%#08x), want 256", got, math.Float32bits(got))
	}
}
