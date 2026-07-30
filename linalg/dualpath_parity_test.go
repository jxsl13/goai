package linalg_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// Bit-exact gates for the devirtualized fast paths in this package that had none.
//
// perfscan PS6004 lists eight functions in this package as "unverified dual paths": a contiguous
// f64 fast path plus a generic AtF64 fallback, which is a bit-identity claim between two arms.
// Checking them one at a time rather than trusting the report split them three ways, and only the
// third group needed anything:
//
//   - NormFro and NormInf were ALREADY gated, by TestNormFlatArmsBitIdentical, which computes the
//     accessor-arm reference in its pre-change form at tolerance 0 over four shapes. Tests written
//     for them here were redundant and were removed rather than left as duplicate coverage.
//   - Pinv's fallback is UNREACHABLE (see TestSVDColMajorFallbackBitExact), so no test can gate it.
//   - toColMajor and toFlat had fast-path coverage from the goldens but nothing reached their
//     accessor arms. Those are the two gates below, and each is the only test in the package that
//     catches a one-ulp perturbation of the arm it covers.
//
// THE TWO ARMS COME FROM ONE DATASET, selected by DTYPE. Entries are small integers, exactly
// representable in float32 and float64 alike, so AtF64 on the f32 twin returns bit-identical
// values to the f64 slice read and every intermediate is float64 in both arms. That is what lets
// these assertions be bit-exact instead of tolerance-based — a tolerance would pass exactly the
// index or ordering mistakes a fast path can introduce.

// dualPathPair builds the same matrix as a contiguous f64 tensor (fast arm) and an f32 tensor
// (fallback arm, since flatRowMajor declines a non-f64 dtype), asserting the f32 round-trip is
// exact so a fixture error cannot masquerade as a code difference.
func dualPathPair(t *testing.T, m, n int, gen func(i, j int) float64) (f64t, f32t *tensor.Tensor) {
	t.Helper()
	d := make([]float64, m*n)
	for i := range m {
		for j := range n {
			d[i*n+j] = gen(i, j)
		}
	}
	f64t = tensor.FromFloat64(tensor.Shape{m, n}, d)
	f32t = tensor.New(tensor.F32, tensor.Shape{m, n})
	s := f32t.Storage().F32()
	for i := range d {
		s[i] = float32(d[i])
		if float64(s[i]) != d[i] {
			t.Fatalf("fixture entry %d is not exact in f32 (%v vs %v); pick integer-valued data",
				i, float64(s[i]), d[i])
		}
	}
	return f64t, f32t
}

// TestSVDColMajorFallbackBitExact drives toColMajor's ACCESSOR arm through Pinv, and its name says
// toColMajor rather than Pinv on purpose.
//
// Pinv's own dual path cannot be tested and is not tested here. Its fast path is guarded on
// flatRowMajor(u) and flatRowMajor(v), where u and v are SVD's OWN outputs — freshly built
// contiguous f64 tensors — so the guard always succeeds and the accessor arm is unreachable
// whatever dtype the caller passes. Verified rather than assumed: panicking in that arm leaves the
// entire linalg suite green. PS6004 will keep reporting Pinv as an unverified dual path, correctly
// by its own predicate and unfixably by a test, which is recorded at the site.
//
// What an f32 input DOES reach is toColMajor, which is called with the caller's matrix. Perturbing
// its accessor arm by one ulp reddens this test and nothing else in the package.
func TestSVDColMajorFallbackBitExact(t *testing.T) {
	// Full column rank and integer-valued, so the pseudoinverse is well conditioned and the
	// singular values are not near the rank-cut tolerance where a tie could legitimately differ.
	fast, slow := dualPathPair(t, 12, 5, func(i, j int) float64 {
		if i == j {
			return 6
		}
		return float64((i*3+j*2)%5 - 2)
	})
	pf, err := linalg.Pinv(fast)
	if err != nil {
		t.Fatal(err)
	}
	ps, err := linalg.Pinv(slow)
	if err != nil {
		t.Fatal(err)
	}
	if pf.Numel() != ps.Numel() {
		t.Fatalf("%d values from the fast arm, %d from the fallback", pf.Numel(), ps.Numel())
	}
	sh := pf.Shape()
	for i := range sh[0] {
		for j := range sh[1] {
			a, b := pf.AtF64(i, j), ps.AtF64(i, j)
			if math.Float64bits(a) != math.Float64bits(b) {
				t.Fatalf("Pinv[%d,%d]: fast %v (%016x) != fallback %v (%016x)",
					i, j, a, math.Float64bits(a), b, math.Float64bits(b))
			}
		}
	}
}

// TestQRDualPathBitExact reaches toFlat's ACCESSOR arm. QR's working copy is built by toFlat,
// whose fast path is one copy from the contiguous backing slice and whose fallback is an AtF64
// walk. The goldens already cover the fast path — perturbing it reddens TestQRBitIdenticalToGolden,
// TestQRFlatEquivRef and TestSolveBitStableGoldens — but nothing drove the FALLBACK, because every
// other test passes contiguous f64. Perturbing the accessor arm by one ulp reddens this test alone.
func TestQRDualPathBitExact(t *testing.T) {
	fast, slow := dualPathPair(t, 14, 9, func(i, j int) float64 { return float64((i*11+j*5)%13 - 6) })
	qf, rf, err := linalg.QR(fast)
	if err != nil {
		t.Fatal(err)
	}
	qs, rs, err := linalg.QR(slow)
	if err != nil {
		t.Fatal(err)
	}
	for name, pair := range map[string][2]*tensor.Tensor{"Q": {qf, qs}, "R": {rf, rs}} {
		x, y := pair[0], pair[1]
		if x.Numel() != y.Numel() {
			t.Fatalf("%s: %d values fast, %d fallback", name, x.Numel(), y.Numel())
		}
		sh := x.Shape()
		for i := range sh[0] {
			for j := range sh[1] {
				a, b := x.AtF64(i, j), y.AtF64(i, j)
				if math.Float64bits(a) != math.Float64bits(b) {
					t.Fatalf("%s[%d,%d]: fast %v (%016x) != fallback %v (%016x)",
						name, i, j, a, math.Float64bits(a), b, math.Float64bits(b))
				}
			}
		}
	}
}
