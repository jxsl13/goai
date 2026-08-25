//go:build arm64 && goexperiment.simd

package cpu

import (
	"math"
	"testing"
)

func sigmoidF64ScalarReference(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	z := math.Exp(x)
	return z / (1 + z)
}

func TestVsigmoidF64Arm64Accuracy(t *testing.T) {
	const n = 1<<18 + 1
	src := make([]float64, n)
	for i := range src {
		src[i] = -80 + 160*float64(i)/float64(n-1)
	}
	dst := make([]float64, n)
	vsigmoidF64(dst, src)
	var maxRel float64
	for i, x := range src {
		want := sigmoidF64ScalarReference(x)
		rel := math.Abs(dst[i]-want) / math.Max(1, math.Abs(want))
		if rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("ARM64 NEON F64 sigmoid max relative error %.3e over %d values", maxRel, n)
	if maxRel > 1e-13 {
		t.Fatalf("max relative error %.3e exceeds 1e-13", maxRel)
	}
}

func TestVsigmoidF64Arm64VectorTailBitIdentity(t *testing.T) {
	for x := -80.0; x <= 80.0; x += 0.00031 {
		body := []float64{0, 0}
		vsigmoidF64(body, []float64{x, x})
		tail := sigmoidF64poly(x)
		if math.Float64bits(body[0]) != math.Float64bits(tail) ||
			math.Float64bits(body[1]) != math.Float64bits(tail) {
			t.Fatalf("x=%g: vector=(%#x,%#x) tail=%#x", x,
				math.Float64bits(body[0]), math.Float64bits(body[1]), math.Float64bits(tail))
		}
	}
}

func TestVsigmoidF64Arm64Edges(t *testing.T) {
	negZero := math.Copysign(0, -1)
	nan := math.Float64frombits(0x7ff8000000000042)
	edges := []float64{0, negZero, 1, -1, 80, -80, math.Inf(1), math.Inf(-1), nan}
	got := make([]float64, len(edges))
	vsigmoidF64(got, edges)
	for i, x := range edges {
		want := sigmoidF64poly(x)
		if math.Float64bits(got[i]) != math.Float64bits(want) {
			t.Errorf("x=%v: vector=%v (%#x), scalar=%v (%#x)", x, got[i],
				math.Float64bits(got[i]), want, math.Float64bits(want))
		}
	}
	if got[0] != 0.5 || got[1] != 0.5 {
		t.Errorf("sigmoid(±0)=(%v,%v), want (0.5,0.5)", got[0], got[1])
	}
	if got[6] != 1 || got[7] != 0 {
		t.Errorf("sigmoid(+Inf,-Inf)=(%v,%v), want (1,0)", got[6], got[7])
	}
}

func sigmoidF64ScalarControl(dst, src []float64) {
	for i, x := range src {
		dst[i] = sigmoidF64ScalarReference(x)
	}
}

func BenchmarkVsigmoidF64Arm64(b *testing.B) {
	const n = 65536
	src := make([]float64, n)
	for i := range src {
		src[i] = -8 + 16*float64(i)/float64(n-1)
	}
	dst := make([]float64, n)
	b.Run("neon", func(b *testing.B) {
		b.SetBytes(n * 8)
		for b.Loop() {
			vsigmoidF64(dst, src)
		}
	})
	b.Run("scalar", func(b *testing.B) {
		b.SetBytes(n * 8)
		for b.Loop() {
			sigmoidF64ScalarControl(dst, src)
		}
	})
}
