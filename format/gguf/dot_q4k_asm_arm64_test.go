//go:build arm64

package gguf

import (
	"math"
	"testing"
)

func TestDotQ4KBlockNeonKnownValue(t *testing.T) {
	x := [qkK]float32{}
	qs := [128]byte{}
	coeff := [16]float32{}
	for i := range x {
		x[i] = 1
	}
	for i := range qs {
		qs[i] = 1 // low nibble 1, high nibble 0
	}
	for pair := range 4 {
		coeff[pair*4+0] = 1
		coeff[pair*4+2] = 1
	}
	got := dotQ4KBlockNeon(&x[0], &qs[0], &coeff[0], &qKByteToF32Indexes[0])
	if got != 128 {
		t.Fatalf("dotQ4KBlockNeon = %v (%#08x), want 128", got, math.Float32bits(got))
	}
}
