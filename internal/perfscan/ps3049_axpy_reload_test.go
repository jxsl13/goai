package main

import "testing"

func axpyFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "axpy-reloads-its-destination" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3049_GemvOneRowPerPass is the measured shape: a matrix-vector product that walks one
// source row per pass over the destination, so every destination element is loaded and stored once
// per row of the matrix.
func TestDetectPS3049_GemvOneRowPerPass(t *testing.T) {
	src := `package p

func gemv(A, B, C []float64, k, n, lo, hi int) {
	c := C[lo:hi:hi]
	for p := range k {
		ap := A[p]
		bp := B[p*n+lo : p*n+hi : p*n+hi]
		for j, bv := range bp {
			c[j] += ap * bv
		}
	}
}`
	fs := axpyFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The transform is worthless if it is applied as a summed group, and the test shapes are
	// what catch the two ways an unrolled body goes wrong, so both have to survive.
	if !containsAll(fs[0].msg, "ONE AT A TIME, NOT AS A SUM", "BIT-IDENTICAL",
		"TEST THE REMAINDER AND A NON-ZERO WINDOW") {
		t.Fatalf("message omits the ordering rule, the exactness claim or the test guidance:\n%s",
			fs[0].msg)
	}
}

// TestDetectPS3049_SilentWhenUnrolled pins the APPLIED form. The STEP is what distinguishes it,
// and the fixture is written so that nothing else can: it still accumulates with += into the same
// destination from rows the outer loop chooses, exactly as the reported form does.
//
// The loop is written WITH an init clause on purpose. The init-less spelling the real kernel uses
// — p := 0 outside, then for ; p+3 < k; p += 4 — is already invisible to the check, because the
// shared loop helper only treats a for statement with a defining init as an induction loop. That
// makes it silent for a reason that has nothing to do with the step, so it would not test this.
func TestDetectPS3049_SilentWhenUnrolled(t *testing.T) {
	src := `package p

func gemv(A, B, C []float64, k, n, lo, hi int) {
	c := C[lo:hi:hi]
	for p := 0; p+3 < k; p += 4 {
		a0, a1 := A[p], A[p+1]
		b0 := B[p*n+lo : p*n+hi : p*n+hi]
		b1 := B[(p+1)*n+lo : (p+1)*n+hi : (p+1)*n+hi]
		for j, v0 := range b0 {
			c[j] += a0 * v0
			c[j] += a1 * b1[j]
		}
	}
}`
	if fs := axpyFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the outer loop already takes several rows per pass:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3049_SilentWhenTheOuterLoopOwnsTheDestination pins the load-bearing condition. If
// the outer loop chooses the destination row, nothing is re-read: each row is touched once, and
// the answer there is to split the outer loop, not to unroll it.
func TestDetectPS3049_SilentWhenTheOuterLoopOwnsTheDestination(t *testing.T) {
	src := `package p

func rows(A, B, C []float64, m, n int) {
	for i := range m {
		ci := C[i*n : i*n+n]
		ai := A[i*n : i*n+n]
		bi := B[i*n : i*n+n]
		for j, bv := range bi {
			ci[j] += ai[j] * bv
		}
	}
}`
	if fs := axpyFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — each destination row is touched once:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3049_SilentWithoutAFullRowPass pins the narrowing that made the check usable. An
// inner loop that is not a pass over the outer step's own row has no row-sized reuse to recover,
// and admitting those took the tree-wide count from 26 to 95, nearly all of them a couple of
// elements wide.
func TestDetectPS3049_SilentWithoutAFullRowPass(t *testing.T) {
	src := `package p

func scatter(A, C []float64, k, width int) {
	for p := range k {
		ap := A[p]
		for j := range width {
			C[j] += ap
		}
	}
}`
	if fs := axpyFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the inner loop is not a pass over the outer row:\n%s",
			len(fs), fs[0].msg)
	}
}
