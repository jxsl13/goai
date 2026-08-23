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

func TestDotQ4KPairRowASMMatchesIndependentASM(t *testing.T) {
	for _, k := range []int{256, 512, 2048, 4096} {
		x := make([]float32, k)
		w0 := make([]float32, k)
		w1 := make([]float32, k)
		for i := range k {
			x[i] = float32(math.Sin(float64(i) * 0.017))
			w0[i] = float32(math.Cos(float64(i) * 0.013))
			w1[i] = float32(math.Sin(float64(i)*0.023) + 0.03125)
		}
		raw0, raw1 := quantizeQ4_K(w0), quantizeQ4_K(w1)
		got0, got1 := dotQ4KPairRowASM(x, raw0, raw1, k)
		want0, want1 := dotQ4_KRowASM(x, raw0, k), dotQ4_KRowASM(x, raw1, k)
		if math.Float64bits(got0) != math.Float64bits(want0) || math.Float64bits(got1) != math.Float64bits(want1) {
			t.Fatalf("k=%d pair=(%016x,%016x), independent=(%016x,%016x)", k, math.Float64bits(got0), math.Float64bits(got1), math.Float64bits(want0), math.Float64bits(want1))
		}
	}
}
