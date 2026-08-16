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
	if got.Dtype() == tensor.F64 && gemmF64Tolerant {
		assertMatMulF64Close(t, got, want, label)
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

// assertMatMulF64Close gates the FUSED-multiply-add f64 GEMM (VFMADD microkernel,
// single rounding) against the mul+add reference. The per-element deviation is a
// few ulp accumulated over k (bounded well under the tolerance below); this is the
// same class of departure numpy/OpenBLAS have from a double-rounded reference.
func assertMatMulF64Close(t *testing.T, got, want *tensor.Tensor, label string) {
	t.Helper()
	if !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%s: shape %v vs %v", label, got.Shape(), want.Shape())
	}
	const rtol, atol = 1e-11, 1e-13
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
	t.Logf("%s: f64-FMA max rel err %.2e (rtol %g)", label, maxRel, rtol)
}

// f32NativeTolerant reports whether the F32 results of THIS build come out of an
// f32-native SIMD kernel. Those kernels accumulate in f32 and route through the
// NEON/AVX exp and GEMM lanes, which ADR-0021 (amd64) and ADR-0026 (arm64) put
// under a K-scaled tolerance against the f64 reference — NOT under bit-identity.
// F64 results, and every result on the default f64-accumulating build, stay exact.
//
// This exists because two production-path parity tests (cross-entropy, MHA) compared
// bitwise unconditionally. That was correct while it was written and measured, but
// it asserted something STRONGER than the kernels promise, and it only ever ran where
// the stronger claim happened to hold: the arm64 SIMD lane could not build its test
// binary at all (vgelu_f64_internal_test.go pulled amd64-only symbols), so those two
// tests had never executed there. With the lane building, both fail on f32 by less
// than a part in a million — four orders inside the rtol below — while their F64
// cases still agree bit-for-bit.
//
// Keeping F64 exact is what preserves the mutation-detection scope those tests
// document: a one-ulp change to the QK inner product still turns MHA red through the
// F64 cases. Only the f32-native branch relaxes, and only to the budget its own ADR
// already defines.
func f32NativeTolerant(f32 bool) bool { return f32 && gemmF32Tolerant }

// parityCloseF32 is the f32-native budget, the same rtol/atol assertMatMulF32Close
// applies to the matmul these kernels route through.
func parityCloseF32(g, w float64) bool {
	const rtol, atol = 2e-3, 1e-4
	return math.Abs(g-w) <= atol+rtol*math.Abs(w)
}

// parityRelErr is the relative error reported alongside a tolerant pass, so the
// margin against the budget stays visible in the log instead of being implied.
func parityRelErr(g, w float64) float64 {
	const atol = 1e-4
	return math.Abs(g-w) / (math.Abs(w) + atol)
}
