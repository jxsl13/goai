//go:build goexperiment.simd && amd64

package cpu

import (
	"math"
	"math/big"
	"slices"
	"testing"
)

// avxMaxSecondF32 models VMAXPS lane semantics: unordered operands and equal
// operands select the second source operand, including its NaN or zero sign.
// Intel SDM Vol. 2B, MAXPS, Operation (Vol. 2B 4-15):
// https://cdrdv2-public.intel.com/774492/325383-sdm-vol-2abcd.pdf
func avxMaxSecondF32(a, b float32) float32 {
	if math.IsNaN(float64(a)) || math.IsNaN(float64(b)) || a == b {
		return b
	}
	if a > b {
		return a
	}
	return b
}

// avxRowMaxOracle models the production operand order: VMAXPS accumulates
// eight independent lanes, then Go performs the ordered horizontal and tail
// reductions with scalar > comparisons.
func avxRowMaxOracle(x []float32) float32 {
	if len(x) < 8 {
		m := float32(math.Inf(-1))
		for _, v := range x {
			if v > m {
				m = v
			}
		}
		return m
	}
	lanes := [8]float32{}
	for i := range lanes {
		lanes[i] = float32(math.Inf(-1))
	}
	n8 := len(x) &^ 7
	for i := 0; i < n8; i += 8 {
		for lane := range lanes {
			lanes[lane] = avxMaxSecondF32(lanes[lane], x[i+lane])
		}
	}
	m := lanes[0]
	for _, v := range lanes[1:] {
		if v > m {
			m = v
		}
	}
	for _, v := range x[n8:] {
		if v > m {
			m = v
		}
	}
	return m
}

// fusedF32Oracle computes the exact rational product-plus-sum and lets
// big.Rat perform the single correctly rounded conversion to float32. It is
// independent of both archsimd.Float32x8.MulAdd and math.FMA.
func fusedF32Oracle(x, a, b float32) float32 {
	xr := new(big.Rat).SetFloat64(float64(x))
	ar := new(big.Rat).SetFloat64(float64(a))
	br := new(big.Rat).SetFloat64(float64(b))
	exact := new(big.Rat).Mul(xr, ar)
	exact.Add(exact, br)
	f, _ := exact.Float32()
	return f
}

// splitF32Oracle performs two independent correctly rounded rational-to-F32
// conversions, making the scalar tail's multiply and add rounding points
// explicit without relying on host contraction behavior.
func splitF32Oracle(x, a, b float32) float32 {
	xr := new(big.Rat).SetFloat64(float64(x))
	ar := new(big.Rat).SetFloat64(float64(a))
	br := new(big.Rat).SetFloat64(float64(b))
	product := new(big.Rat).Mul(xr, ar)
	p, _ := product.Float32()
	sum := new(big.Rat).Add(new(big.Rat).SetFloat64(float64(p)), br)
	f, _ := sum.Float32()
	return f
}

func sameF32BitsAVX(x, y []float32) bool {
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

func TestSoftmaxRowPassAVXParity(t *testing.T) {
	if !vexpHasAVX {
		t.Fatal("amd64 SIMD test requires native AVX and FMA; scalar fallback must not masquerade as AVX coverage")
	}

	sizes := []int{0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 2048}
	// With this fixed a,b pair every listed x is an adversarial finite normal:
	// the independently rounded fused and split results differ by one ulp.
	a := math.Float32frombits(0x34438863)
	b := math.Float32frombits(0x434fff83)
	xbits := [...]uint32{
		0x4d2b88c6, 0x4d2b88cc, 0x4d2b88d6, 0x4d2b88db,
		0x4d2b88e1, 0x4d2b88eb, 0x4d2b88f0, 0x4d2b88f6,
	}
	for i, bits := range xbits {
		x := math.Float32frombits(bits)
		if math.Float32bits(fusedF32Oracle(x, a, b)) == math.Float32bits(splitF32Oracle(x, a, b)) {
			t.Fatalf("affine fixture lane %d does not distinguish fused and split rounding", i)
		}
	}

	for _, n := range sizes {
		t.Run(fmtInt(n), func(t *testing.T) {
			x := make([]float32, n)
			for i := range x {
				x[i] = math.Float32frombits(xbits[i%len(xbits)])
			}

			before := slices.Clone(x)
			gotMax := rowMaxF32(x)
			wantMax := avxRowMaxOracle(x)
			if math.Float32bits(gotMax) != math.Float32bits(wantMax) {
				t.Fatalf("row max: got %#08x want %#08x", math.Float32bits(gotMax), math.Float32bits(wantMax))
			}
			if !sameF32BitsAVX(x, before) {
				t.Fatal("row max mutated its input")
			}

			gotScale := slices.Clone(x)
			wantScale := slices.Clone(x)
			for i := range wantScale {
				wantScale[i] = float32(float64(wantScale[i]) * 0.125)
			}
			scaleRowF32(gotScale, 0.125)
			if !sameF32BitsAVX(gotScale, wantScale) {
				t.Fatal("scale differs from the exact dyadic product")
			}

			gotAffine := slices.Clone(x)
			wantAffine := make([]float32, n)
			n8 := n &^ 7
			for i, v := range x {
				if i < n8 {
					wantAffine[i] = fusedF32Oracle(v, a, b)
				} else {
					wantAffine[i] = splitF32Oracle(v, a, b)
				}
			}
			axpbRowF32(gotAffine, a, b)
			if !sameF32BitsAVX(gotAffine, wantAffine) {
				for i := range gotAffine {
					if math.Float32bits(gotAffine[i]) != math.Float32bits(wantAffine[i]) {
						t.Fatalf("affine lane %d: got %#08x want %#08x", i, math.Float32bits(gotAffine[i]), math.Float32bits(wantAffine[i]))
					}
				}
			}
		})
	}
}

func TestRowMaxF32AVXSpecialValues(t *testing.T) {
	if !vexpHasAVX {
		t.Fatal("amd64 SIMD test requires native AVX and FMA; scalar fallback must not masquerade as AVX coverage")
	}
	negZero := math.Float32frombits(1 << 31)
	qnan1 := math.Float32frombits(0x7fc00001)
	qnan2 := math.Float32frombits(0x7fc00002)
	cases := []struct {
		name string
		x    []float32
	}{
		{"empty-scalar-fallback", nil},
		{"short-scalar-fallback", []float32{negZero, 0, -1, qnan1, float32(math.Inf(-1)), 3, qnan2}},
		{"source-nan-second-wins", []float32{qnan1, -1, -2, -3, -4, -5, -6, -7}},
		{"later-finite-replaces-lane-nan", []float32{qnan1, -1, -2, -3, -4, -5, -6, -7, 5, -8, -9, -10, -11, -12, -13, -14}},
		{"later-nan-replaces-lane-finite", []float32{5, -1, -2, -3, -4, -5, -6, -7, qnan2, -8, -9, -10, -11, -12, -13, -14}},
		{"equal-zero-second-negative", []float32{0, -1, -2, -3, -4, -5, -6, -7, negZero, -8, -9, -10, -11, -12, -13, -14}},
		{"equal-zero-second-positive", []float32{negZero, -1, -2, -3, -4, -5, -6, -7, 0, -8, -9, -10, -11, -12, -13, -14}},
		{"infinities-and-tail-nan", []float32{float32(math.Inf(-1)), 1, 2, 3, 4, 5, 6, float32(math.Inf(1)), qnan1}},
		{"all-nan-32", slices.Repeat([]float32{qnan1}, 32)},
		{"all-nan-33", slices.Repeat([]float32{qnan2}, 33)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			before := slices.Clone(tc.x)
			got, want := rowMaxF32(tc.x), avxRowMaxOracle(tc.x)
			if math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("got %#08x want %#08x; input=%08x", math.Float32bits(got), math.Float32bits(want), f32BitsAVX(tc.x))
			}
			if !sameF32BitsAVX(tc.x, before) {
				t.Fatal("row max mutated its input")
			}
		})
	}

	// Exercise both vector blocks and the scalar tail for the affine and scale
	// passes with zeros, infinities, and NaNs. Finite-normal exactness and fused
	// versus split rounding are covered independently above.
	special := []float32{0, negZero, float32(math.Inf(1)), float32(math.Inf(-1)), qnan1}
	x := slices.Repeat(special, 5)[:21]
	gotScale := slices.Clone(x)
	scaleRowF32(gotScale, 0.5)
	for i, v := range gotScale {
		input := x[i]
		if math.IsNaN(float64(input)) {
			if !math.IsNaN(float64(v)) {
				t.Fatalf("special scale lane %d: NaN became %v", i, v)
			}
			continue
		}
		// Multiplication by positive 0.5 preserves both infinity and zero signs.
		wantBits := math.Float32bits(input)
		if math.Float32bits(v) != wantBits {
			t.Fatalf("special scale lane %d: got %#08x want %#08x", i, math.Float32bits(v), wantBits)
		}
	}

	gotAffine := slices.Clone(x)
	axpbRowF32(gotAffine, 2, 0)
	for i, v := range gotAffine {
		input := x[i]
		if math.IsNaN(float64(input)) {
			if !math.IsNaN(float64(v)) {
				t.Fatalf("special affine lane %d: NaN became %v", i, v)
			}
			continue
		}
		var wantBits uint32
		if math.IsInf(float64(input), 1) {
			wantBits = math.Float32bits(float32(math.Inf(1)))
		} else if math.IsInf(float64(input), -1) {
			wantBits = math.Float32bits(float32(math.Inf(-1)))
		} else {
			// For either signed-zero input, x*2 + (+0) is +0.
			wantBits = math.Float32bits(0)
		}
		if math.Float32bits(v) != wantBits {
			t.Fatalf("special affine lane %d: got %#08x want %#08x", i, math.Float32bits(v), wantBits)
		}
	}
}

func f32BitsAVX(x []float32) []uint32 {
	bits := make([]uint32, len(x))
	for i := range x {
		bits[i] = math.Float32bits(x[i])
	}
	return bits
}

func fmtInt(n int) string {
	if n == 0 {
		return "n=0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return "n=" + string(buf[i:])
}
