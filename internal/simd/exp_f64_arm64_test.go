//go:build arm64 && goexperiment.simd

package simd

import (
	"math"
	"slices"
	"testing"
)

func TestExpF64SpecialAndFallbackSemantics(t *testing.T) {
	src := []float64{0, math.Inf(-1), -1000, -709, 1, math.Inf(1), math.NaN()}
	dst := make([]float64, len(src))
	ExpScaledF64(dst, src, 1)
	for i, x := range src {
		want := math.Exp(x)
		if !sameF64Within(dst[i], want, 0) {
			t.Fatalf("scaled dst[%d]=%g want %g", i, dst[i], want)
		}
	}

	neg := []float64{math.Inf(-1), -1, -0.0}
	ExpScaledF64(dst[:len(neg)], neg, 1)
	if dst[0] != 0 || math.Signbit(dst[0]) {
		t.Fatalf("exp(-Inf)=%v, want exact +0", dst[0])
	}
	for i, x := range neg[1:] {
		if want := math.Exp(x); !sameF64Within(dst[i+1], want, 1e-13) {
			t.Fatalf("safe special dst[%d]=%g want %g", i+1, dst[i+1], want)
		}
	}
}

func sigmoidScalarForTest(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	z := math.Exp(x)
	return z / (1 + z)
}

func TestSigmoidF64SpecialAliasAndInputSemantics(t *testing.T) {
	src := []float64{math.Inf(-1), -709, -30, -0.0, 0, 30, 709, math.Inf(1), math.NaN()}
	original := slices.Clone(src)
	dst := make([]float64, len(src))
	SigmoidF64(dst, src)
	if !slices.EqualFunc(src, original, func(a, b float64) bool {
		return math.Float64bits(a) == math.Float64bits(b)
	}) {
		t.Fatal("distinct sigmoid source was modified")
	}
	for i, x := range original {
		want := sigmoidScalarForTest(x)
		if !sameF64Within(dst[i], want, 1e-13) {
			t.Fatalf("distinct dst[%d]=%g want %g", i, dst[i], want)
		}
	}

	alias := slices.Clone(original)
	SigmoidF64(alias, alias)
	for i, x := range original {
		want := sigmoidScalarForTest(x)
		if !sameF64Within(alias[i], want, 0) {
			t.Fatalf("alias dst[%d]=%g want exact scalar %g", i, alias[i], want)
		}
	}
}

func TestSoftplusNegLLSumF64SpecialAndInputSemantics(t *testing.T) {
	f := []float64{math.Inf(-1), -709, -20, -0.0, 0, 20, 709, math.Inf(1), math.NaN()}
	y := []float64{0, 1, 0, 1, 0, 1, 0, 1, 0}
	wantF, wantY := slices.Clone(f), slices.Clone(y)
	got := SoftplusNegLLSumF64(f, y)
	var want float64
	for i := range wantF {
		x := (1 - 2*wantY[i]) * wantF[i]
		if x > 0 {
			want += x + math.Log1p(math.Exp(-x))
		} else {
			want += math.Log1p(math.Exp(x))
		}
	}
	if !sameF64Within(got, want, 0) {
		t.Fatalf("sum=%g want %g", got, want)
	}
	for i := range f {
		if math.Float64bits(f[i]) != math.Float64bits(wantF[i]) || math.Float64bits(y[i]) != math.Float64bits(wantY[i]) {
			t.Fatalf("input modified at %d", i)
		}
	}
}
