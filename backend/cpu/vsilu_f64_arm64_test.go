//go:build arm64 && goexperiment.simd

package cpu

import (
	"math"
	"testing"
)

func TestVsiluF64Arm64Accuracy(t *testing.T) {
	const n = 1<<18 + 1 // odd: vector body plus scalar tail
	src := make([]float64, n)
	for i := range src {
		src[i] = -80 + 160*float64(i)/float64(n-1)
	}
	dst := make([]float64, n)
	vsiluF64(dst, src)

	var maxRel float64
	for i, x := range src {
		want := x / (1 + math.Exp(-x))
		rel := math.Abs(dst[i]-want) / math.Max(1, math.Abs(want))
		if rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("ARM64 NEON F64 SiLU max relative error %.3e over %d values", maxRel, n)
	if maxRel > 1e-13 {
		t.Fatalf("max relative error %.3e exceeds 1e-13", maxRel)
	}
}

func TestVsiluF64Arm64VectorTailBitIdentity(t *testing.T) {
	for x := -80.0; x <= 80.0; x += 0.00031 {
		body := []float64{0, 0}
		vsiluF64(body, []float64{x, x})
		tail := siluF64poly(x)
		if math.Float64bits(body[0]) != math.Float64bits(tail) ||
			math.Float64bits(body[1]) != math.Float64bits(tail) {
			t.Fatalf("x=%g: vector=(%#x,%#x) tail=%#x", x,
				math.Float64bits(body[0]), math.Float64bits(body[1]), math.Float64bits(tail))
		}
	}
}

func TestVsiluF64Arm64Edges(t *testing.T) {
	negZero := math.Copysign(0, -1)
	edges := []float64{0, negZero, 1, -1, 80, -80, math.Inf(1), math.Inf(-1), math.NaN()}
	body := make([]float64, len(edges)) // odd length also exercises the scalar tail
	vsiluF64(body, edges)
	for i, x := range edges {
		want := siluF64poly(x)
		if math.IsNaN(want) {
			if !math.IsNaN(body[i]) {
				t.Errorf("x=%v: got %v, want NaN", x, body[i])
			}
			continue
		}
		if math.Float64bits(body[i]) != math.Float64bits(want) {
			t.Errorf("x=%v: vector=%v (%#x), scalar=%v (%#x)", x, body[i],
				math.Float64bits(body[i]), want, math.Float64bits(want))
		}
	}
	if !math.Signbit(body[1]) || body[1] != 0 {
		t.Errorf("SiLU(-0)=%v, want negative zero", body[1])
	}
	if !math.IsInf(body[6], 1) {
		t.Errorf("SiLU(+Inf)=%v, want +Inf", body[6])
	}
	if !math.IsNaN(body[7]) {
		t.Errorf("SiLU(-Inf)=%v, want NaN like scalar x/(1+exp(-x))", body[7])
	}
}

// siluF64scalarRef is byte-for-byte the loop the non-SIMD build runs for OpSiLU/F64
// (elementwise.go's fallback). It exists so the NEON lane can be A/B'd against the
// path it replaces inside ONE binary, with no build-tag juggling to confound the
// measurement.
func siluF64scalarRef(dst, src []float64) {
	for i, x := range src {
		dst[i] = x / (1 + math.Exp(-x))
	}
}

// BenchmarkVsiluF64Arm64 measures the NEON lane against the scalar fallback it
// replaces, on the SwiGLU-shaped length the FFN gate actually issues.
func BenchmarkVsiluF64Arm64(b *testing.B) {
	const n = 256 * 1408 // [256,1408] f64, the T985 SwiGLU FFN shape
	src := make([]float64, n)
	for i := range src {
		src[i] = -8 + 16*float64(i)/float64(n-1)
	}
	dst := make([]float64, n)
	b.Run("neon", func(b *testing.B) {
		b.SetBytes(int64(n * 8))
		for b.Loop() {
			vsiluF64(dst, src)
		}
	})
	b.Run("scalar", func(b *testing.B) {
		b.SetBytes(int64(n * 8))
		for b.Loop() {
			siluF64scalarRef(dst, src)
		}
	})
}
