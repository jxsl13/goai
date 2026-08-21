//go:build goexperiment.simd && arm64

package cpu

import (
	"math"
	"testing"
)

func TestNormNormalizeF32NeonParityAndInputs(t *testing.T) {
	for _, n := range []int{1, 15, 16, 17, 31, 32, 33, 65} {
		x := make([]float32, n)
		gamma := make([]float32, n)
		beta := make([]float32, n)
		for i := range n {
			x[i] = float32(i%11-5) / 7
			gamma[i] = float32(i%7+1) / 5
			beta[i] = float32(i%5-2) / 9
		}
		if n >= 4 {
			x[0] = float32(math.NaN())
			x[1] = float32(math.Inf(1))
			x[2] = float32(math.Inf(-1))
			x[3] = float32(math.Copysign(0, -1))
		}
		x0, g0, b0 := append([]float32(nil), x...), append([]float32(nil), gamma...), append([]float32(nil), beta...)
		rmsWant, layerWant := make([]float32, n), make([]float32, n)
		const mean, inv = float32(0.125), float32(1.75)
		for i := range n {
			rmsWant[i] = x[i] * inv * gamma[i]
			layerWant[i] = (x[i]-mean)*inv*gamma[i] + beta[i]
		}
		rmsGot, layerGot := make([]float32, n), make([]float32, n)
		rmsNormNormalizeF32(x, gamma, rmsGot, inv)
		layerNormNormalizeF32(x, gamma, beta, layerGot, mean, inv)
		assertNormF32SliceClose(t, "rms", n, rmsGot, rmsWant)
		assertNormF32SliceClose(t, "layer", n, layerGot, layerWant)
		assertNormF32SliceBits(t, "x", n, x, x0)
		assertNormF32SliceBits(t, "gamma", n, gamma, g0)
		assertNormF32SliceBits(t, "beta", n, beta, b0)
	}
}

func assertNormF32SliceClose(t *testing.T, name string, n int, got, want []float32) {
	t.Helper()
	for i := range n {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(w) {
			if !math.IsNaN(g) {
				t.Fatalf("%s/%d[%d]: got %v, want NaN", name, n, i, g)
			}
			continue
		}
		if math.IsInf(w, 0) {
			if !math.IsInf(g, int(math.Copysign(1, w))) {
				t.Fatalf("%s/%d[%d]: got %v, want %v", name, n, i, g, w)
			}
			continue
		}
		tol := 5e-5 * math.Max(1, math.Abs(w))
		if math.Abs(g-w) > tol {
			t.Fatalf("%s/%d[%d]: got %v, want %v (tol %g)", name, n, i, g, w, tol)
		}
	}
}

func assertNormF32SliceBits(t *testing.T, name string, n int, got, want []float32) {
	t.Helper()
	for i := range n {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s/%d[%d] input mutated: bits %08x != %08x", name, n, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}
