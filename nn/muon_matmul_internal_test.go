package nn

import (
	"math"
	"math/rand"
	"testing"
)

// matmulABtRef is matmulABt EXACTLY as it stood before the ikj rewrite: the
// straightforward i/j/p dot product, one serial accumulator per output element.
// It is the bit-identity oracle for the rewrite — the claim being gated is that
// reordering the loops and exploiting symmetry changes the schedule but not a
// single accumulation, so every output bit must match. Keep this function frozen;
// "fixing" it to match a new implementation would make the gate self-fulfilling
// (§self-policing-guard: a guard that reads the table it checks proves nothing).
func matmulABtRef(a, b []float64, m, k int) []float64 {
	c := make([]float64, m*m)
	for i := range m {
		ai := a[i*k : i*k+k]
		ci := c[i*m : i*m+m]
		for j := range m {
			bj := b[j*k : j*k+k]
			var s float64
			for p := range ai {
				s += ai[p] * bj[p]
			}
			ci[j] = s
		}
	}
	return c
}

// randMat returns a deterministic [m,k] matrix with a wide exponent spread, so a
// reassociated accumulation would round differently and be caught.
func randMat(rng *rand.Rand, m, k int) []float64 {
	x := make([]float64, m*k)
	for i := range x {
		x[i] = rng.NormFloat64() * math.Pow(2, float64(rng.Intn(21)-10))
	}
	return x
}

// TestMatmulABtCrossReferenceExact holds the rewritten matmulABt bit-identical to
// the pre-rewrite dot-product form, with NO tolerance (§V22): same values, same
// accumulation order, so the bits must be equal. Covers the distinct-operand case
// and the aliased X·Xᵀ case that newtonSchulz5 actually calls, since only the
// latter takes the symmetry path.
func TestMatmulABtCrossReferenceExact(t *testing.T) {
	rng := rand.New(rand.NewSource(20260728))
	for _, d := range []struct{ m, k int }{
		{1, 1}, {1, 7}, {2, 3}, {3, 2}, {5, 5}, {8, 16}, {17, 9}, {33, 64}, {64, 33},
	} {
		x := randMat(rng, d.m, d.k)
		y := randMat(rng, d.m, d.k)

		// aliased: the X·Xᵀ call at newtonSchulz5 — this is the symmetric path
		got, want := matmulABt(x, x, d.m, d.k), matmulABtRef(x, x, d.m, d.k)
		assertBitsEqual(t, got, want, "aliased", d.m, d.k)

		// distinct operands: the general, non-symmetric path
		got, want = matmulABt(x, y, d.m, d.k), matmulABtRef(x, y, d.m, d.k)
		assertBitsEqual(t, got, want, "distinct", d.m, d.k)
	}
}

// TestMatmulABtAliasedIsSymmetric documents WHY the mirror is legal: X·Xᵀ is
// symmetric to the last bit, because c[i][j] and c[j][i] accumulate the same
// products in the same order and IEEE multiplication is commutative. If this ever
// fails, the mirror in matmulABt is unsound and must be removed, not loosened.
func TestMatmulABtAliasedIsSymmetric(t *testing.T) {
	rng := rand.New(rand.NewSource(4242))
	m, k := 24, 40
	x := randMat(rng, m, k)
	c := matmulABtRef(x, x, m, k)
	for i := range m {
		for j := range m {
			if a, b := c[i*m+j], c[j*m+i]; math.Float64bits(a) != math.Float64bits(b) {
				t.Fatalf("X·Xᵀ not bit-symmetric at (%d,%d): %v vs %v", i, j, a, b)
			}
		}
	}
}

// TestNewtonSchulz5CrossReferenceExact gates the whole orthogonalization, not just
// the kernel: the scratch-buffer reuse must not perturb a single output bit either.
func TestNewtonSchulz5CrossReferenceExact(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for _, d := range []struct{ r, c int }{{8, 16}, {16, 8}, {32, 32}, {13, 29}} {
		x := randMat(rng, d.r, d.c)
		a := append([]float64(nil), x...)
		b := append([]float64(nil), x...)
		// Two independent runs on equal inputs must agree bit-for-bit; with a
		// reused scratch buffer this also catches stale-state leakage between calls.
		got := newtonSchulz5(a, d.r, d.c, 5)
		want := newtonSchulz5(b, d.r, d.c, 5)
		assertBitsEqual(t, got, want, "newtonSchulz5", d.r, d.c)
	}
}

// TestMatmulABtKeepsZeroTerms holds the claim matmulABt's comment makes: it must
// NOT adopt matmulFlat's `if av == 0 { continue }` skip. A skipped 0·(±Inf) term is
// a dropped NaN, so the skip would silently turn a NaN output into a finite one.
// Random fixtures never contain an exact zero, so without this the exactness gate
// passes with the skip in place — it was measured doing exactly that.
func TestMatmulABtKeepsZeroTerms(t *testing.T) {
	const m, k = 3, 4
	a := make([]float64, m*k)
	b := make([]float64, m*k)
	for i := range a {
		a[i], b[i] = 1, 1
	}
	a[0] = 0                     // A[0][0] = 0 …
	b[0] = math.Inf(1)           // … against B[0][0] = +Inf ⇒ 0·Inf = NaN
	b[k] = math.Inf(-1)          // and a -Inf in row 1, reached with a nonzero a
	got := matmulABt(a, b, m, k) // C[0][0] must be NaN, not the finite sum of the rest

	if !math.IsNaN(got[0]) {
		t.Fatalf("C[0][0] = %v, want NaN: a zero term multiplying ±Inf was skipped", got[0])
	}
	if want := matmulABtRef(a, b, m, k); !math.IsNaN(want[0]) {
		t.Fatalf("oracle disagrees: reference C[0][0] = %v, want NaN", want[0])
	}
	// Everything NaN-free must still match the oracle bit-for-bit.
	ref := matmulABtRef(a, b, m, k)
	for i := range ref {
		if math.IsNaN(ref[i]) && math.IsNaN(got[i]) {
			continue
		}
		if math.Float64bits(got[i]) != math.Float64bits(ref[i]) {
			t.Fatalf("element %d: got %v want %v", i, got[i], ref[i])
		}
	}
}

func assertBitsEqual(t *testing.T, got, want []float64, what string, m, k int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s %dx%d: length %d != %d", what, m, k, len(got), len(want))
	}
	for i := range want {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
			t.Fatalf("%s %dx%d: element %d differs: got %v (%#x) want %v (%#x)",
				what, m, k, i, got[i], math.Float64bits(got[i]), want[i], math.Float64bits(want[i]))
		}
	}
}

// matmulFlatSerialRef is matmulFlat EXACTLY as it stood before the row bands: one goroutine
// walking every row in ikj order. It is the bit-identity oracle for the split, and it must stay
// frozen — rewriting it to follow the implementation would make the gate self-fulfilling.
func matmulFlatSerialRef(a, b []float64, m, k, n int) []float64 {
	c := make([]float64, m*n)
	for i := range m {
		ci := c[i*n : i*n+n]
		for p := range k {
			av := a[i*k+p]
			if av == 0 {
				continue
			}
			bp := b[p*n : p*n+n]
			for j := range ci {
				ci[j] += av * bp[j]
			}
		}
	}
	return c
}

// TestMatmulFlatBandsMatchSerialExactly holds the parallel matmulFlat bit-identical to the serial
// form, with no tolerance: a band owns whole rows of C, so every element still accumulates its k
// products in the same ascending-p order. A tolerance here would accept exactly the reassociation
// the split is claiming not to do.
//
// The shapes are chosen to clear parallelRows' m*k*n >= 1<<14 gate — below it the helper runs the
// body serially and the test would compare the serial path against itself — and to include a row
// count that does NOT divide evenly across workers, since an off-by-one in the band arithmetic
// shows up only in the last, short band. The wide exponent spread in randMat is what makes a
// reordered accumulation round differently and be caught at all.
func TestMatmulFlatBandsMatchSerialExactly(t *testing.T) {
	rng := rand.New(rand.NewSource(20260802))
	for _, d := range []struct{ m, k, n int }{
		{64, 64, 64},   // square, evenly divisible
		{37, 48, 53},   // none of the three divides the worker count
		{256, 64, 512}, // the Newton-Schulz shape: many rows, heavy per row
		{3, 512, 512},  // few rows, huge per-row work — the case a row-count gate would miss
		{1, 256, 256},  // single row: the split degenerates
		{129, 17, 8},   // odd row count, narrow output
	} {
		a := randMat(rng, d.m, d.k)
		b := randMat(rng, d.k, d.n)
		// A column of exact zeros exercises the av == 0 skip inside a band.
		for i := range d.m {
			a[i*d.k] = 0
		}
		got := matmulFlat(a, b, d.m, d.k, d.n)
		want := matmulFlatSerialRef(a, b, d.m, d.k, d.n)
		if len(got) != len(want) {
			t.Fatalf("[%d,%d,%d]: %d values, want %d", d.m, d.k, d.n, len(got), len(want))
		}
		for i := range want {
			if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
				t.Fatalf("[%d,%d,%d] element %d (row %d): %v, serial %v — the band split changed"+
					" an accumulation", d.m, d.k, d.n, i, i/d.n, got[i], want[i])
			}
		}
	}
}
