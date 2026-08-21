package simd

import (
	"math"
	"slices"
	"testing"
)

func TestSigmoidF64Parity(t *testing.T) {
	for _, n := range []int{1, 3, 4, 7, 16, 17, 1000} {
		src := make([]float64, n)
		for i := range src {
			src[i] = -30 + 60*float64(i)/float64(max(n-1, 1)) // spans ±30 (saturation both ends)
		}
		dst := make([]float64, n)
		original := slices.Clone(src)
		SigmoidF64(dst, src)
		if !slices.Equal(src, original) {
			t.Fatalf("n=%d: sigmoid source was modified", n)
		}
		var maxRel float64
		for i, x := range src {
			w := 1 / (1 + math.Exp(-x))
			if rel := math.Abs(dst[i]-w) / math.Max(1e-300, w); rel > maxRel {
				maxRel = rel
			}
		}
		if maxRel > 1e-13 {
			t.Fatalf("n=%d: maxRel=%.2e exceeds 1e-13", n, maxRel)
		}
	}
}

func TestSoftplusNegLLSumF64Parity(t *testing.T) {
	for _, n := range []int{1, 3, 4, 7, 16, 17, 1000} {
		f := make([]float64, n)
		y := make([]float64, n)
		for i := range f {
			f[i] = -20 + 40*float64(i)/float64(max(n-1, 1)) // ±20 (both saturation tails)
			if i%2 == 0 {
				y[i] = 1
			}
		}
		originalF, originalY := slices.Clone(f), slices.Clone(y)
		got := SoftplusNegLLSumF64(f, y)
		if !slices.Equal(f, originalF) || !slices.Equal(y, originalY) {
			t.Fatalf("n=%d: softplus input was modified", n)
		}
		var want float64
		for i := range f {
			x := (1 - 2*y[i]) * f[i]
			if x > 0 {
				want += x + math.Log1p(math.Exp(-x))
			} else {
				want += math.Log1p(math.Exp(x))
			}
		}
		if rel := math.Abs(got-want) / math.Max(1e-300, math.Abs(want)); rel > 1e-13 {
			t.Fatalf("n=%d: got=%.15g want=%.15g rel=%.2e", n, got, want, rel)
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
