package nn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/tensor"
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

// TestMatmulIntoClearsADirtyDestination pins the precondition that makes buffer reuse safe here:
// both kernels ACCUMULATE into their destination, so a reused buffer must be zeroed first.
//
// The existing cross-reference tests cannot see this. They call matmulFlat and matmulABt, which
// pass a nil destination and get a fresh make — already zero — so the clear is never exercised.
// This calls the Into forms twice on one buffer, with the second result compared against a fresh
// computation: if the clear is dropped, the second answer is the sum of both products.
func TestMatmulIntoClearsADirtyDestination(t *testing.T) {
	rng := rand.New(rand.NewSource(20260803))
	const m, k, n = 12, 9, 7
	a1, b1 := randMat(rng, m, k), randMat(rng, k, n)
	a2, b2 := randMat(rng, m, k), randMat(rng, k, n)

	dst := make([]float64, m*n)
	matmulFlatInto(dst, a1, b1, m, k, n) // dirty the buffer with a different product
	got := matmulFlatInto(dst, a2, b2, m, k, n)
	want := matmulFlat(a2, b2, m, k, n)
	for i := range want {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
			t.Fatalf("matmulFlatInto element %d on a reused buffer: %v, fresh %v — the destination"+
				" was not cleared, so the previous product is still in it", i, got[i], want[i])
		}
	}

	// Same for the AᐧBᵀ kernel, in both its shapes: the symmetric path writes only the lower
	// triangle before mirroring, so a stale upper triangle would survive there too.
	for _, aliased := range []bool{false, true} {
		x := randMat(rng, m, k)
		y := x
		if !aliased {
			y = randMat(rng, m, k)
		}
		d := make([]float64, m*m)
		matmulABtInto(a1, a1, m, k, nil, d) // dirty
		got := matmulABtInto(x, y, m, k, nil, d)
		want := matmulABt(x, y, m, k)
		for i := range want {
			if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
				t.Fatalf("matmulABtInto (aliased=%v) element %d on a reused buffer: %v, fresh %v",
					aliased, i, got[i], want[i])
			}
		}
	}
}

// TestMuonStepIgnoresStaleScratch pins the receiver-held direction buffer. It is reused across
// steps, so every entry must be written before it is read — poisoning it with NaN between steps
// must change nothing. This is the check PS3035's advice calls for, run against the real optimizer
// rather than against a fixture that mimics it.
func TestMuonStepIgnoresStaleScratch(t *testing.T) {
	mk := func() ([]*tensor.Tensor, GradFn) {
		p := tensor.New(tensor.F64, tensor.Shape{24, 16})
		g := tensor.New(tensor.F64, tensor.Shape{24, 16})
		for i := range 24 {
			for j := range 16 {
				p.SetF64(math.Sin(float64(i*16+j))*0.1, i, j)
				g.SetF64(math.Cos(float64(i*7+j*3))*0.05, i, j)
			}
		}
		return []*tensor.Tensor{p}, func(*tensor.Tensor) *tensor.Tensor { return g }
	}
	const steps = 4
	pc, gc := mk()
	clean := NewMuon(pc, 0.02)
	pp, gp := mk()
	poisoned := NewMuon(pp, 0.02)
	for range steps {
		if err := clean.Step(gc); err != nil {
			t.Fatal(err)
		}
		for i := range poisoned.dir {
			for j := range poisoned.dir[i] {
				poisoned.dir[i][j] = math.NaN()
			}
		}
		if err := poisoned.Step(gp); err != nil {
			t.Fatal(err)
		}
	}
	for i := range 24 {
		for j := range 16 {
			a, b := pc[0].AtF64(i, j), pp[0].AtF64(i, j)
			if math.Float64bits(a) != math.Float64bits(b) {
				t.Fatalf("param[%d,%d] after %d steps: clean %v, NaN-poisoned scratch %v — the"+
					" direction buffer is read before it is written", i, j, steps, a, b)
			}
		}
	}
}

// newtonSchulz5Ref is an INDEPENDENT implementation of the quintic Newton-Schulz iteration:
// textbook triple-loop products, a fresh buffer for every intermediate, no reuse of any kind.
// It exists to be an oracle, so it is deliberately written the slow, obvious way and must stay
// that way — optimizing it would delete the only thing in this file that does not share code with
// the implementation it checks.
func newtonSchulz5Ref(x []float64, rows, cols, steps int) []float64 {
	const a, b, c = 3.4445, -4.7750, 2.0315
	mul := func(p, q []float64, m, k, n int) []float64 { // plain i,j,k dot products
		o := make([]float64, m*n)
		for i := range m {
			for j := range n {
				var s float64
				for t := range k {
					s += p[i*k+t] * q[t*n+j]
				}
				o[i*n+j] = s
			}
		}
		return o
	}
	tr := func(p []float64, m, n int) []float64 {
		o := make([]float64, m*n)
		for i := range m {
			for j := range n {
				o[j*m+i] = p[i*n+j]
			}
		}
		return o
	}
	X := append([]float64(nil), x...)
	r, cc := rows, cols
	transposed := rows > cols
	if transposed {
		X = tr(X, rows, cols)
		r, cc = cols, rows
	}
	var ss float64
	for _, v := range X {
		ss += v * v
	}
	inv := 1 / (math.Sqrt(ss) + 1e-7)
	for i := range X {
		X[i] *= inv
	}
	for range steps {
		A := mul(X, tr(X, r, cc), r, cc, r)
		A2 := mul(A, A, r, r, r)
		bm := make([]float64, r*r)
		for i := range bm {
			bm[i] = b*A[i] + c*A2[i]
		}
		bx := mul(bm, X, r, r, cc)
		next := make([]float64, len(X))
		for i := range X {
			next[i] = a*X[i] + bx[i]
		}
		X = next
	}
	if transposed {
		X = tr(X, r, cc)
	}
	return X
}

// TestNewtonSchulz5MatchesIndependentReference gives the iteration an ORACLE.
//
// TestNewtonSchulz5CrossReferenceExact runs the implementation against itself on equal inputs. That
// catches stale state carried BETWEEN calls, and nothing else: any mistake inside the iteration —
// wiring two intermediates to one buffer, clearing the wrong one — changes both arms identically
// and it stays green. A mutation proved that, aliasing the A and A² buffers left it passing.
//
// The comparison is to a tolerance rather than to bits because the reference multiplies with plain
// dot products while the implementation accumulates in ikj order over a triangle; the values agree,
// the summation order does not. 1e-12 relative is far tighter than any real defect would survive.
func TestNewtonSchulz5MatchesIndependentReference(t *testing.T) {
	rng := rand.New(rand.NewSource(4711))
	for _, d := range []struct{ r, c int }{{8, 16}, {16, 8}, {32, 32}, {13, 29}, {1, 5}} {
		x := randMat(rng, d.r, d.c)
		got := newtonSchulz5(append([]float64(nil), x...), d.r, d.c, 5)
		want := newtonSchulz5Ref(append([]float64(nil), x...), d.r, d.c, 5)
		if len(got) != len(want) {
			t.Fatalf("[%d,%d]: %d values, want %d", d.r, d.c, len(got), len(want))
		}
		for i := range want {
			if math.Abs(got[i]-want[i]) > 1e-12*(1+math.Abs(want[i])) {
				t.Fatalf("[%d,%d] element %d: %v, independent reference %v", d.r, d.c, i,
					got[i], want[i])
			}
		}
	}
}
