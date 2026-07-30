package linalg_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// TestCholSolveFallbackMatchesFastPath gates the AtF64 fallback in cholesky.go, which no test
// reached after the contiguous fast paths were added.
//
// requireSymmetric, cholFactor and CholSolve each now take the typed backing slice when the input
// is contiguous f64 and keep the per-element accessor otherwise. Every existing test passes
// contiguous f64, so they all exercise the FAST path and the fallback — the code that used to be
// the only path — became unexercised in one commit. That is the same silent-coverage-loss the
// threshold-guarded fast paths in this package produced earlier, arriving by a different route:
// here the arms are selected by DTYPE rather than by size.
//
// The two arms are driven from ONE source matrix, so the comparison cannot drift. Entries are
// small integers, which are exactly representable in float32 and float64 alike, so AtF64 on the
// f32 tensor returns bit-identical values to the f64 slice read. Every intermediate is float64 in
// both paths and the result tensor is f64 in both, so the outputs must agree to the BIT — a
// tolerance here would hide exactly the kind of index mistake the fast path could introduce.
func TestCholSolveFallbackMatchesFastPath(t *testing.T) {
	const n, cols = 24, 5
	// A diagonally dominant symmetric integer matrix: positive definite, and every entry exact
	// in f32.
	af := make([]float64, n*n)
	for i := range n {
		for j := range n {
			v := float64((i*7+j*3)%5 - 2) // small integers, symmetric-ized below
			af[i*n+j] = v
		}
	}
	for i := range n { // symmetrize, then make the diagonal dominate
		for j := i + 1; j < n; j++ {
			af[j*n+i] = af[i*n+j]
		}
		af[i*n+i] = float64(4 * n)
	}
	bf := make([]float64, n*cols)
	for i := range bf {
		bf[i] = float64(i%9 - 4)
	}

	a64 := tensor.FromFloat64(tensor.Shape{n, n}, af)
	b64 := tensor.FromFloat64(tensor.Shape{n, cols}, bf)

	// The f32 twin holds the identical VALUES, but its dtype makes flatRowMajor/flatContig
	// decline, which is what routes it through the AtF64 fallback.
	a32 := tensor.New(tensor.F32, tensor.Shape{n, n})
	for i, s := 0, a32.Storage().F32(); i < len(s); i++ {
		s[i] = float32(af[i])
	}
	b32 := tensor.New(tensor.F32, tensor.Shape{n, cols})
	for i, s := 0, b32.Storage().F32(); i < len(s); i++ {
		s[i] = float32(bf[i])
	}
	// Guard the premise: if any entry were not exactly representable the two arms would differ
	// for a reason that has nothing to do with the code under test.
	for i, s := 0, a32.Storage().F32(); i < len(s); i++ {
		if float64(s[i]) != af[i] {
			t.Fatalf("fixture entry %d is not exact in f32: %v vs %v", i, float64(s[i]), af[i])
		}
	}
	for i, s := 0, b32.Storage().F32(); i < len(s); i++ {
		if float64(s[i]) != bf[i] {
			t.Fatalf("rhs entry %d is not exact in f32: %v vs %v", i, float64(s[i]), bf[i])
		}
	}

	fast, err := linalg.CholSolve(a64, b64)
	if err != nil {
		t.Fatalf("fast path: %v", err)
	}
	slow, err := linalg.CholSolve(a32, b32)
	if err != nil {
		t.Fatalf("fallback path: %v", err)
	}
	if fast.Numel() != slow.Numel() {
		t.Fatalf("%d values from the fast path, %d from the fallback", fast.Numel(), slow.Numel())
	}
	for i := range n {
		for j := range cols {
			f, s := fast.AtF64(i, j), slow.AtF64(i, j)
			if math.Float64bits(f) != math.Float64bits(s) {
				t.Fatalf("x[%d,%d]: fast %v (%016x) != fallback %v (%016x) — the contiguous path "+
					"disagrees with the accessor path", i, j, f, math.Float64bits(f), s, math.Float64bits(s))
			}
		}
	}
}

// TestRequireSymmetricFallbackAgrees pins the same two paths for the symmetry check, whose
// rejection is a behavior in its own right: it must reject the same matrices and name the same
// first offending pair whichever path runs. The asymmetry is introduced at a known cell so the
// reported indices can be checked rather than just the fact of an error.
func TestRequireSymmetricFallbackAgrees(t *testing.T) {
	const n = 16
	af := make([]float64, n*n)
	for i := range n {
		af[i*n+i] = float64(4 * n)
	}
	af[3*n+7] = 1 // breaks symmetry at (3,7) vs (7,3)=0
	b := tensor.FromFloat64(tensor.Shape{n}, make([]float64, n))

	a64 := tensor.FromFloat64(tensor.Shape{n, n}, af)
	a32 := tensor.New(tensor.F32, tensor.Shape{n, n})
	for i, s := 0, a32.Storage().F32(); i < len(s); i++ {
		s[i] = float32(af[i])
	}

	_, errFast := linalg.CholSolve(a64, b)
	_, errSlow := linalg.CholSolve(a32, b)
	if errFast == nil || errSlow == nil {
		t.Fatalf("both paths must reject an asymmetric matrix; fast=%v slow=%v", errFast, errSlow)
	}
	if errFast.Error() != errSlow.Error() {
		t.Fatalf("the two paths report different errors:\n fast: %v\n slow: %v", errFast, errSlow)
	}
}
