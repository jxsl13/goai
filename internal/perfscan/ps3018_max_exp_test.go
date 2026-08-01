package main

import "testing"

func maxExpFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "max-normalized-exp" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3018_MaxNormalizedExp is the measured shape, from the RWKV WKV scan: a two-term
// max-normalization where both exponentials are evaluated even though the max guarantees one of
// them is exp(0). Applying it went -12.18% there and -14.87% on the linear-time WKV backward.
func TestDetectPS3018_MaxNormalizedExp(t *testing.T) {
	src := `package p

func scan(pp, ww, aa, bb, vv float64) float64 {
	q := math.Max(pp, ww)
	return (math.Exp(pp-q)*aa + math.Exp(ww-q)*vv) / (math.Exp(pp-q)*bb + math.Exp(ww-q))
}`
	// Four, not two: each CALL SITE is reported, and this fixture writes each exponent twice. That
	// is deliberate. The same duplication was real in the WKV backward, where math.Exp(l-z) and
	// math.Exp(dgt-z) each appeared twice — math.Exp is not inlined, so the repeat is a second
	// evaluation, not a common subexpression the compiler folds. Reporting per site is what made
	// that visible; collapsing to one finding per exponent would have hidden half the cost.
	fs := maxExpFindings(t, src)
	if len(fs) != 4 {
		t.Fatalf("%d findings, want 4 — one per call site, including repeats of the same exponent", len(fs))
	}
	// The NaN clause is the part an implementer gets wrong, so it must survive into the advice:
	// branching on `pp >= ww` substitutes a 1 the original never produced.
	if !containsAll(fs[0].msg, "NaN", "1/N") {
		t.Fatalf("message omits the NaN or the 1/N caveat:\n%s", fs[0].msg)
	}
}

// TestDetectPS3018_SilentOnGuarded pins the applied form. Once each call sits behind a test of the
// max against its own argument, the fix is in place; reporting it files the fix as the defect
// (A-CHECK-NEEDS-ITS-OWN-BEFORE-AND-AFTER-001 — this floor is why the check was rewritten, the
// first version flagged its own motivating fix).
func TestDetectPS3018_SilentOnGuarded(t *testing.T) {
	src := `package p

func scan(pp, ww, aa, vv float64) float64 {
	q := math.Max(pp, ww)
	e1, e2 := 1.0, 1.0
	if q != pp {
		e1 = math.Exp(pp - q)
	}
	if q != ww {
		e2 = math.Exp(ww - q)
	}
	return e1*aa + e2*vv
}`
	if fs := maxExpFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a guarded call is the applied form:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3018_SilentWithoutMax keeps the check to exponents normalized by a MAX. Subtracting
// an arbitrary reference is just a shift, and nothing says the difference is ever zero.
func TestDetectPS3018_SilentWithoutMax(t *testing.T) {
	src := `package p

func shift(x, y, ref float64) float64 {
	z := ref * 0.5
	return math.Exp(x-z) + math.Exp(y-z)
}`
	if fs := maxExpFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — no max, so no term is guaranteed to be exp(0)", len(fs))
	}
}

// TestDetectPS3018_SilentOnUnrelatedArgument is the floor that keeps this off ordinary softmax. In
// a softmax the shift IS a max, but over a slice whose elements the check cannot tie to the max's
// arguments — and a shift by an unrelated quantity saves nothing.
func TestDetectPS3018_SilentOnUnrelatedArgument(t *testing.T) {
	src := `package p

func sm(a, b, c float64) float64 {
	m := math.Max(a, b)
	return math.Exp(c - m)
}`
	if fs := maxExpFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — c is not an argument of the max, so the term is never exp(0)", len(fs))
	}
}

// TestDetectPS3018_SilentOnNonSubtraction pins the SHAPE. math.Exp(x*m) or math.Exp(x+m) carry no
// cancellation, and matching every exp whose argument merely mentions a max name would flag them.
func TestDetectPS3018_SilentOnNonSubtraction(t *testing.T) {
	src := `package p

func scale(a, b, x float64) float64 {
	m := math.Max(a, b)
	return math.Exp(a*m) + math.Exp(a+m)
}`
	if fs := maxExpFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — only a subtraction of the max can cancel", len(fs))
	}
}

// TestDetectPS3018_SilentOnReversedOperands pins the ORDER. math.Exp(m - x) is the reflected
// quantity: it is exp(0) in the same case, but it is also unbounded ABOVE for the other argument,
// so it is not the stable normalization this check is about and rewriting it is not the same edit.
func TestDetectPS3018_SilentOnReversedOperands(t *testing.T) {
	src := `package p

func rev(a, b float64) float64 {
	m := math.Max(a, b)
	return math.Exp(m-a) + math.Exp(m-b)
}`
	if fs := maxExpFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — exp(m-x) is not the max-normalized form", len(fs))
	}
}

// TestDetectPS3018_SilentOnDifferentSubtrahend is the floor SilentWithoutMax cannot supply. There a
// max is absent entirely, so a detector that accepted ANY max in scope would still stay quiet and
// the floor would pass for the wrong reason. Here a max EXISTS and one of its arguments is
// exponentiated — but shifted by a DIFFERENT quantity, so nothing cancels and the site is clean.
func TestDetectPS3018_SilentOnDifferentSubtrahend(t *testing.T) {
	src := `package p

func other(a, b float64) float64 {
	m := math.Max(a, b)
	z := a * 0.5
	return math.Exp(a-z) + m
}`
	if fs := maxExpFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the shift is z, not the max, so no term is exp(0)", len(fs))
	}
}
