package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// assertMatMul is the dtype/build-aware matmul parity check. F64 (and F32 on
// the default f64-accumulating build) must be BIT-EXACT vs ref (§V3/§V11 tol 0).
// Under the amd64+simd experiment, F32 matmul is f32-native (ADR-0021) and is
// checked within a K-scaled f32 tolerance instead. gemmF32Tolerant is the
// build-tagged switch (gemm_f32policy_{default,simd}_test.go).
func assertMatMul(t *testing.T, got, want *tensor.Tensor, label string) {
	t.Helper()
	if got.Dtype() == tensor.F32 && gemmF32Tolerant {
		assertMatMulF32Close(t, got, want, label)
		return
	}
	assertEqualExact(t, got, want, label)
}

// assertMatMulF32Close bounds the f32-native result against the f64 reference.
// f32 dot-product error grows ~K·u (u = 2^-24 ≈ 6e-8); with FMA it is smaller
// still. rtol 2e-3 leaves ample margin over the observed error for the test
// shapes (K ≤ 128) while staying far tighter than the ~1e-2 an f16 path needs;
// atol handles near-zero entries where relative error is meaningless.
func assertMatMulF32Close(t *testing.T, got, want *tensor.Tensor, label string) {
	t.Helper()
	if !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%s: shape %v vs %v", label, got.Shape(), want.Shape())
	}
	const rtol, atol = 2e-3, 1e-4
	var maxRel float64
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		g, w := got.AtF64(idx...), want.AtF64(idx...)
		d := math.Abs(g - w)
		if d > atol+rtol*math.Abs(w) {
			t.Fatalf("%s [%d]: cpu %v vs ref %v (|Δ|=%g > tol %g)", label, i, g, w, d, atol+rtol*math.Abs(w))
		}
		if r := d / (math.Abs(w) + atol); r > maxRel {
			maxRel = r
		}
	}
	t.Logf("%s: f32-native max rel err %.2e (rtol %g)", label, maxRel, rtol)
}
