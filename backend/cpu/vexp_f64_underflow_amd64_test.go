//go:build goexperiment.simd

package cpu

import (
	"math"
	"simd/archsimd"
	"testing"
)

// evalExpF64x4 runs the 4-wide primitive on a single value (broadcast) and returns lane 0.
func evalExpF64x4(x float64) float64 {
	in := [4]float64{x, x, x, x}
	var out [4]float64
	expF64x4(archsimd.LoadFloat64x4Slice(in[:])).StoreSlice(out[:])
	return out[0]
}

// TestExpF64x4Underflow pins the underflow→0 contract on the F64 exp primitive: for arguments below
// the f64 exp-underflow point (~−745.2) eˣ must be EXACTLY 0, not the ~3.3e-308 the −708 clamp used
// to leak. This is what mask-sensitive consumers (sigmoid/softmax attention masking, which feed −∞)
// rely on — exp(−∞) = 0 exactly. Mirrors the F32 twin's underflow pin (vexp_amd64.go). Also asserts
// the scalar twin expF64poly agrees value-for-value so the 4-lane body and scalar tail stay identical
// (the TestQuantDeepSeekV2DecodeMatchesForward body==tail invariant).
func TestExpF64x4Underflow(t *testing.T) {
	// Deep-underflow and −∞ must produce EXACTLY 0 on both the vector and scalar paths.
	for _, x := range []float64{-745.3, -746, -800, -1000, -1e30, math.Inf(-1)} {
		if got := evalExpF64x4(x); got != 0 {
			t.Errorf("expF64x4(%g) = %g, want exactly 0", x, got)
		}
		if got := expF64poly(x); got != 0 {
			t.Errorf("expF64poly(%g) = %g, want exactly 0", x, got)
		}
	}

	// Normal-range values stay accurate (~1 ulp) vs math.Exp and are NOT masked to 0.
	for _, x := range []float64{-700, -80, -10, -1, 0} {
		want := math.Exp(x)
		got := evalExpF64x4(x)
		if want == 0 {
			t.Fatalf("test setup: math.Exp(%g) unexpectedly 0", x)
		}
		if rel := math.Abs(got-want) / want; rel > 1e-13 {
			t.Errorf("expF64x4(%g) = %g, want ~%g (rel %.2e)", x, got, want, rel)
		}
		if got == 0 {
			t.Errorf("expF64x4(%g) = 0, must not underflow a normal value", x)
		}
	}

	// NaN must NOT collapse to 0 via the underflow mask (Less→false on NaN). The vector path's
	// pre-existing x.Max(−708) clamp still eats NaN → ~3.3e-308 (unchanged by this fix; the scalar
	// twin propagates NaN — a long-standing body/tail quirk out of scope here). We only pin that the
	// underflow mask itself leaves NaN alone, i.e. does not force it to exact 0.
	if got := evalExpF64x4(math.NaN()); got == 0 {
		t.Errorf("expF64x4(NaN) = 0, underflow mask must not fire on NaN")
	}

	// body==tail: the vector primitive and its scalar twin agree bit-for-bit across finite inputs,
	// including the underflow boundary and the −745.2..−708 clamp band.
	for _, x := range []float64{-745.2, -745.19, -720, -708, -300, -50, -0.5, 0} {
		v, s := evalExpF64x4(x), expF64poly(x)
		if v != s {
			t.Errorf("body==tail mismatch at %g: expF64x4=%v expF64poly=%v", x, v, s)
		}
	}
}
