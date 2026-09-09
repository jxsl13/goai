//go:build arm64 && goexperiment.simd

package cpu

import (
	"math"
	"slices"
	"testing"
)

func geluF64Oracle(x float64) float64 {
	return 0.5 * x * (1 + math.Erf(x/math.Sqrt2))
}

func geluGradF64Oracle(x, g float64) float64 {
	phi := 0.5 * (1 + math.Erf(x/math.Sqrt2))
	pdf := 0.3989422804014327 * math.Exp(-0.5*x*x)
	return g * (phi + x*pdf)
}

func assertGELUF64Value(t *testing.T, got, want float64) {
	t.Helper()
	if math.IsNaN(want) {
		if !math.IsNaN(got) {
			t.Fatalf("got %g, want NaN", got)
		}
		return
	}
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("got %.17g (%016x), want %.17g (%016x)", got, math.Float64bits(got), want, math.Float64bits(want))
	}
}

func TestVGELUF64NeonBaselineNumerics(t *testing.T) {
	values := []float64{
		0, math.Copysign(0, -1), math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64,
		math.Inf(1), math.Inf(-1), math.NaN(), math.MaxFloat64, -math.MaxFloat64,
		1e-300, -1e-300, 1e150, -1e150,
	}
	for i := -4096; i <= 4096; i++ {
		values = append(values, float64(i)/128)
	}
	for _, boundary := range []float64{math.Sqrt2, -math.Sqrt2, 6 * math.Sqrt2, -6 * math.Sqrt2, 32, -32} {
		values = append(values, math.Nextafter(boundary, math.Inf(-1)), boundary, math.Nextafter(boundary, math.Inf(1)))
	}
	for k := 0; k <= 64; k++ {
		x := math.Sqrt(2 * (float64(k) + 0.5) * math.Ln2)
		values = append(values, x, -x)
	}
	g := make([]float64, len(values))
	for i := range g {
		g[i] = -2 + 4*float64((i*37)%257)/256
	}
	xBefore, gBefore := slices.Clone(values), slices.Clone(g)
	forward, backward := make([]float64, len(values)), make([]float64, len(values))
	vgeluF64(forward, values)
	vgeluGradF64(backward, values, g)
	if !equalF64Bits(values, xBefore) || !equalF64Bits(g, gBefore) {
		t.Fatal("GELU leaf changed an input")
	}
	for i, x := range values {
		assertGELUF64Value(t, forward[i], geluF64Oracle(x))
		assertGELUF64Value(t, backward[i], geluGradF64Oracle(x, g[i]))
	}
}

func equalF64Bits(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float64bits(a[i]) != math.Float64bits(b[i]) {
			return false
		}
	}
	return true
}

func TestVGELUF64NeonLengthsAliasesAndBodyTail(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 7, 17, 2049} {
		x := make([]float64, n)
		g := make([]float64, n)
		for i := range x {
			x[i] = -1 + 2*float64(i+1)/float64(n+1)
			g[i] = -2 + 4*float64((i*19)%101)/100
		}
		wantF, wantB := make([]float64, n), make([]float64, n)
		for i := range x {
			wantF[i] = geluF64Oracle(x[i])
			wantB[i] = geluGradF64Oracle(x[i], g[i])
		}
		gotF, gotB := make([]float64, n), make([]float64, n)
		vgeluF64(gotF, x)
		vgeluGradF64(gotB, x, g)
		xAlias := slices.Clone(x)
		vgeluF64(xAlias, xAlias)
		xBackAlias := slices.Clone(x)
		vgeluGradF64(xBackAlias, xBackAlias, g)
		gAlias := slices.Clone(g)
		vgeluGradF64(gAlias, x, gAlias)
		for i := range x {
			assertGELUF64Value(t, gotF[i], wantF[i])
			assertGELUF64Value(t, xAlias[i], wantF[i])
			assertGELUF64Value(t, gotB[i], wantB[i])
			assertGELUF64Value(t, xBackAlias[i], wantB[i])
			assertGELUF64Value(t, gAlias[i], wantB[i])
		}
	}
}
