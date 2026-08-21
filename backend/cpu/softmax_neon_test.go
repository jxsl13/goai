//go:build goexperiment.simd

package cpu

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

func scalarRowMaxF32(x []float32) float32 {
	m := float32(math.Inf(-1))
	for _, v := range x {
		if v > m {
			m = v
		}
	}
	return m
}

func scalarScaleRowF32(x []float32, scale float32) {
	for i := range x {
		x[i] *= scale
	}
}

func scalarAxpbRowF32(x []float32, a, b float32) {
	for i := range x {
		x[i] = x[i]*a + b
	}
}

func sameF32Bits(x, y []float32) bool {
	if len(x) != len(y) {
		return false
	}
	for i := range x {
		if math.Float32bits(x[i]) != math.Float32bits(y[i]) {
			return false
		}
	}
	return true
}

func TestSoftmaxRowPassNeonParity(t *testing.T) {
	rng := rand.New(rand.NewSource(81))
	for _, n := range []int{0, 1, 15, 16, 17, 31, 32, 33, 65, 2048} {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		if n >= 16 {
			x[1] = float32(math.NaN())
			x[7] = float32(math.Inf(-1))
			x[15] = float32(math.Inf(1))
		}
		before := slices.Clone(x)
		gotMax := rowMaxF32(x)
		wantMax := scalarRowMaxF32(x)
		if math.Float32bits(gotMax) != math.Float32bits(wantMax) {
			t.Fatalf("row max n=%d: got %#08x want %#08x", n, math.Float32bits(gotMax), math.Float32bits(wantMax))
		}
		if !sameF32Bits(x, before) {
			t.Fatalf("row max n=%d mutated its input", n)
		}

		gotScale, wantScale := slices.Clone(x), slices.Clone(x)
		scaleRowF32(gotScale, 0.125)
		scalarScaleRowF32(wantScale, 0.125)
		if !sameF32Bits(gotScale, wantScale) {
			t.Fatalf("scale n=%d differs from scalar", n)
		}

		gotAxpb, wantAxpb := slices.Clone(x), slices.Clone(x)
		axpbRowF32(gotAxpb, 0.75, -0.125)
		scalarAxpbRowF32(wantAxpb, 0.75, -0.125)
		if !sameF32Bits(gotAxpb, wantAxpb) {
			t.Fatalf("axpb n=%d differs from scalar", n)
		}
	}
}

func TestRowMaxF32NeonSpecialValues(t *testing.T) {
	negZero := math.Float32frombits(1 << 31)
	allNaN := make([]float32, 33)
	for i := range allNaN {
		allNaN[i] = float32(math.NaN())
	}
	cases := [][]float32{
		allNaN,
		{float32(math.NaN()), float32(math.NaN())},
		{negZero, 0, -1, float32(math.NaN())},
		{0, negZero, -1, float32(math.NaN())},
		{float32(math.Inf(-1)), float32(math.NaN()), float32(math.Inf(-1))},
	}
	for _, prefix := range cases {
		x := make([]float32, 33)
		for i := range x {
			x[i] = float32(math.Inf(-1))
		}
		copy(x, prefix)
		got, want := rowMaxF32(x), scalarRowMaxF32(x)
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("input %08x: got %#08x want %#08x", f32Bits(x), math.Float32bits(got), math.Float32bits(want))
		}
	}
}

func f32Bits(x []float32) []uint32 {
	bits := make([]uint32, len(x))
	for i := range x {
		bits[i] = math.Float32bits(x[i])
	}
	return bits
}
