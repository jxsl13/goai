package main

import (
	"strings"
	"testing"
)

func twoDeepFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "two-deep-index-not-ranged" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3016_IntBoundedRowIndex is the measured shape: Cholesky's forward substitution loops
// `for k := range i` and indexes l[i][k]. Converting it to range the row gave geomean -2.08%.
func TestDetectPS3016_IntBoundedRowIndex(t *testing.T) {
	src := `package p

func solve(n int, l [][]float64, y []float64) {
	for i := range n {
		var s float64
		for k := range i {
			s -= l[i][k] * y[k]
		}
		y[i] = s
	}
}`
	fs := twoDeepFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Which half pays is the actionable part, and it is counterintuitive.
	if !strings.Contains(fs[0].msg, "RANGE IS THE PART THAT PAYS") {
		t.Fatalf("message omits that the hoist alone is a wash:\n%s", fs[0].msg)
	}
}

// TestDetectPS3016_SilentOnAppliedRange pins the applied form: a loop already ranging the row.
func TestDetectPS3016_SilentOnAppliedRange(t *testing.T) {
	src := `package p

func solve(n int, l [][]float64, y []float64) {
	for i := range n {
		var s float64
		li := l[i]
		for k, lik := range li[:i] {
			s -= lik * y[k]
		}
		y[i] = s
	}
}`
	if fs := twoDeepFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — this is the fix:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3016_ReportsColumnWalkWithRowSliceAdvice pins the arm added after this shape paid
// twice under a check that used to decline it.
//
// When the outer subscript moves — l[k][i] — there is no row to range ALONG, which is what the
// earlier version keyed on. But the SLICE OF ROWS is itself rangeable, and pairing a companion to
// it leaves only the fixed subscript's check. Cholesky back substitution went 3 checks to 1 and
// autograd's logdet solve 4 to 1, both measured wins, and both were found by hand because this
// check was silent on them.
func TestDetectPS3016_ReportsColumnWalkWithRowSliceAdvice(t *testing.T) {
	src := `package p

func back(n int, l [][]float64, y []float64) {
	for i := n - 1; i >= 0; i-- {
		s := y[i]
		for k := i + 1; k < n; k++ {
			s -= l[k][i] * y[k]
		}
		y[i] = s
	}
}`
	fs := twoDeepFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — the slice of rows is rangeable", len(fs))
	}
	if !strings.Contains(fs[0].msg, "SLICE OF ROWS") {
		t.Fatalf("column walk got the row-ranging advice, which does not apply:\n%s", fs[0].msg)
	}
}

// TestDetectPS3016_OuterLoopDeclinesTheNestedLoopVar is the INVARIANCE floor, and it caught a real
// false positive.
//
// Requiring only that the row index differ from THIS loop's variable let the OUTER loop of a nested
// pair match l[k][i]: from its point of view i moves and k is the row, when k is the inner loop's
// own variable and no fixed row exists at all. The INNER loop reports it — correctly, as a column
// walk — so the assertion is exactly one finding, not none: two would mean the outer loop matched
// as well.
func TestDetectPS3016_OuterLoopDeclinesTheNestedLoopVar(t *testing.T) {
	src := `package p

func back(n int, l [][]float64, y []float64) {
	for i := 0; i < n; i++ {
		for k := 0; k < n; k++ {
			y[i] -= l[k][i]
		}
	}
}`
	if fs := twoDeepFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want exactly 1 — the inner loop's column walk, and NOT a second "+
			"from the outer loop reading k as an invariant row", len(fs))
	}
}

// TestDetectPS3016_ReportsEachDistinctRow pins that two rows in one inner loop are two findings:
// ranging one still leaves the other indexed, so a reader has to see both.
func TestDetectPS3016_ReportsEachDistinctRow(t *testing.T) {
	src := `package p

func gram(m, n int, b, rinv [][]float64) float64 {
	var s float64
	for i := range m {
		for j := range n {
			for k := j; k < n; k++ {
				s += b[i][k] * rinv[j][k]
			}
		}
	}
	return s
}`
	if fs := twoDeepFindings(t, src); len(fs) != 2 {
		t.Fatalf("%d findings, want 2 — b[i] and rinv[j] are separate rows", len(fs))
	}
}

// The three floors below were added after mutation testing showed the ones above do not exercise
// the clauses they appear to. Each fixture here is silent because of exactly one clause.

// TestDetectPS3016_SilentWhenAlreadyRangingTheRowExpression floors the applied-form suppression.
// The earlier applied-form fixture has no two-deep index left at all, so it stays silent however
// that clause is mutated; this one keeps the index and ranges the row expression.
func TestDetectPS3016_SilentWhenAlreadyRangingTheRowExpression(t *testing.T) {
	src := `package p

func f(i int, l [][]float64, y []float64) float64 {
	var s float64
	for k := range l[i] {
		s += l[i][k] * y[k]
	}
	return s
}`
	if fs := twoDeepFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the loop already ranges the row:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3016_SilentWhenNeitherSubscriptMoves floors the requirement that the INNER subscript
// be this loop's variable. A two-deep index whose subscripts are both invariant is loop-invariant
// entirely — a different finding, and not one about bounds checks.
func TestDetectPS3016_SilentWhenNeitherSubscriptMoves(t *testing.T) {
	src := `package p

func f(n, i, j int, l [][]float64) float64 {
	var s float64
	for k := range n {
		s += l[i][j]
	}
	return s
}`
	if fs := twoDeepFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — neither subscript moves with the loop:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3016_SilentOnDiagonalWalk floors the row-differs-from-the-loop-variable requirement.
// l[k][k] walks a diagonal: both subscripts move together, so every step is a different row and
// there is nothing to hoist.
func TestDetectPS3016_SilentOnDiagonalWalk(t *testing.T) {
	src := `package p

func trace(n int, l [][]float64) float64 {
	var s float64
	for k := range n {
		s += l[k][k]
	}
	return s
}`
	if fs := twoDeepFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a diagonal walk has no fixed row:\n%s", len(fs), fs[0].msg)
	}
}
