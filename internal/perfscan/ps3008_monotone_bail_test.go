package main

import (
	"strings"
	"testing"
)

func monotoneBailFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "monotone-bail-per-element" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3008_SquaredTerm is the measured shape: classic's ballTree.within, whose per-element
// bail-out branch cost more than the arithmetic it guarded.
func TestDetectPS3008_SquaredTerm(t *testing.T) {
	src := `package p

func within(a, b []float64, eps2 float64) bool {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
		if s > eps2 {
			return false
		}
	}
	return true
}`
	fs := monotoneBailFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The NaN trap is the one thing a reader acting on this finding can get silently wrong, so the
	// message has to carry it.
	if !strings.Contains(fs[0].msg, "NaN") {
		t.Fatalf("message omits the NaN tail caveat:\n%s", fs[0].msg)
	}
}

// TestDetectPS3008_AbsTerm pins the L1 spelling. math.Abs is non-negative for the same reason a
// square is, and the repo's own remaining instance of this pattern is an Abs one.
func TestDetectPS3008_AbsTerm(t *testing.T) {
	src := `package p

import "math"

func within(a, b []float64, eps float64) bool {
	var s float64
	for i := range a {
		s += math.Abs(a[i] - b[i])
		if s > eps {
			return false
		}
	}
	return true
}`
	if fs := monotoneBailFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
}

// TestDetectPS3008_SilentOnSignedTerm is the CORRECTNESS floor, not a precision one.
//
// If the accumulated term can be negative the accumulator is not monotone: it can cross the
// threshold and come back under it, so a run that the per-element test would have bailed on might
// legitimately end up within. Moving that test then changes the ANSWER, not just the speed. This is
// the one silence in this check that is about being wrong rather than being noisy.
func TestDetectPS3008_SilentOnSignedTerm(t *testing.T) {
	src := `package p

func drift(a, b []float64, lim float64) bool {
	var s float64
	for i := range a {
		s += a[i] - b[i]
		if s > lim {
			return false
		}
	}
	return true
}`
	if fs := monotoneBailFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a signed term is not monotone and moving the test would "+
			"change the result:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3008_SilentOnProductOfDifferentOperands guards the square test specifically. x*y is
// not a square and can be negative; only identical operands prove non-negativity.
func TestDetectPS3008_SilentOnProductOfDifferentOperands(t *testing.T) {
	src := `package p

func dot(a, b []float64, lim float64) bool {
	var s float64
	for i := range a {
		s += a[i] * b[i]
		if s > lim {
			return false
		}
	}
	return true
}`
	if fs := monotoneBailFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a*b is not a square and may be negative", len(fs))
	}
}

// TestDetectPS3008_SilentOnceBlocked pins the applied form. A loop already advancing by 4 has had
// this transform done to it; reporting it again would flag the fix as the defect.
func TestDetectPS3008_SilentOnceBlocked(t *testing.T) {
	src := `package p

func within(a, b []float64, eps2 float64) bool {
	var s float64
	i := 0
	for ; i+4 <= len(a); i += 4 {
		d0 := a[i] - b[i]
		s += d0 * d0
		d1 := a[i+1] - b[i+1]
		s += d1 * d1
		if s > eps2 {
			return false
		}
	}
	return !(s > eps2)
}`
	if fs := monotoneBailFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the loop already strides by 4, which is the applied form", len(fs))
	}
}

// TestDetectPS3008_SilentWhenTestDoesNotBail keeps the check to the shape whose branch can actually
// be moved. A threshold test that merely records something still has to run every iteration.
func TestDetectPS3008_SilentWhenTestDoesNotBail(t *testing.T) {
	src := `package p

func count(a, b []float64, lim float64) int {
	var s float64
	n := 0
	for i := range a {
		d := a[i] - b[i]
		s += d * d
		if s > lim {
			n++
		}
	}
	return n
}`
	if fs := monotoneBailFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the test does not exit the loop, so it cannot be deferred", len(fs))
	}
}
