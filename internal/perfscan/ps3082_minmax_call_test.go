package main

import (
	"go/parser"
	"go/token"
	"testing"
)

// minMaxCallFindingsIn parses a fixture and keeps only PS3082's findings.
func minMaxCallFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var out []finding
	for _, fd := range scanFile(fset, f, testSets(t)) {
		if fd.category == "minmax-call-in-a-loop" {
			out = append(out, fd)
		}
	}
	return out
}

// TestDetectPS3082_CallInLoop is the measured shape: a clamp and a two-value choice, both
// written as math calls, both inside the element loop of a reduction.
func TestDetectPS3082_CallInLoop(t *testing.T) {
	src := `package p

import "math"

func loss(xs, ys, as []float64, eps float64, n int) float64 {
	var total float64
	for i := 0; i < n; i++ {
		r := math.Exp(xs[i] - ys[i])
		a := as[i]
		total += math.Min(r*a, math.Max(1-eps, math.Min(1+eps, r))*a)
	}
	return total
}`
	fs := minMaxCallFindingsIn(t, src)
	if len(fs) != 3 {
		t.Fatalf("%d findings, want 3 (two Min, one Max)", len(fs))
	}
	// The three facts a reader needs and cannot get from the code: that this is a call and not
	// an instruction, that the builtin is NOT a drop-in, and that the gate has to plant the
	// value because a reduction hides the difference.
	if !containsAll(fs[0].msg, "is a CALL, not an instruction", "NOT THE SAME FUNCTION",
		"internal/fmath", "ONE PLANTED VALUE PER CALL") {
		t.Fatalf("message omits the mechanism, the trap or the gate:\n%s", fs[0].msg)
	}
	// It names the instruction for the function it found, not a fixed one.
	var sawMin, sawMax bool
	for _, f := range fs {
		if containsAll(f.msg, "math.Min", "FMIND") {
			sawMin = true
		}
		if containsAll(f.msg, "math.Max", "FMAXD") {
			sawMax = true
		}
	}
	if !sawMin || !sawMax {
		t.Fatalf("message does not name the matching builtin instruction (min=%v max=%v)", sawMin, sawMax)
	}
}

// TestDetectPS3082_SilentWhenGuarded pins the suppression that makes the check idempotent: a
// function that already takes the builtin and falls back to math on a NaN result has both
// spellings of the same pair, and the math one is the recovery this check is asking for.
func TestDetectPS3082_SilentWhenGuarded(t *testing.T) {
	src := `package p

import "math"

func clampAll(xs []float64, lo, hi float64, n int) {
	for i := 0; i < n; i++ {
		m := min(hi, xs[i])
		if m != m {
			m = math.Min(hi, xs[i])
		}
		xs[i] = m
	}
}`
	if fs := minMaxCallFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the builtin already runs and this call is the NaN"+
			" recovery:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3082_SilentOutsideALoop pins that the cost is PER ELEMENT. One call at function
// scope is a call, and no rewrite of it is worth the divergence risk.
func TestDetectPS3082_SilentOutsideALoop(t *testing.T) {
	src := `package p

import "math"

func bound(v, lo, hi float64) float64 {
	return math.Max(lo, math.Min(hi, v))
}`
	if fs := minMaxCallFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — not per element:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3082_SilentOnAnotherPackagesMin pins that the selector has to be math. A Min on
// some other package, or a local one, has neither the call cost nor the NaN contract.
func TestDetectPS3082_SilentOnAnotherPackagesMin(t *testing.T) {
	src := `package p

import "othermath"

func walk(xs []float64, hi float64, n int) {
	for i := 0; i < n; i++ {
		xs[i] = othermath.Min(hi, xs[i])
	}
}`
	if fs := minMaxCallFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — not math.Min:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3082_SilentOnADifferentPair pins that the guard suppression is matched on the
// ARGUMENTS, not merely on a builtin appearing somewhere in the function. A loop that guards
// one pair and leaves another as a bare call still has the bare call.
func TestDetectPS3082_SilentOnADifferentPair(t *testing.T) {
	src := `package p

import "math"

func two(xs, ys []float64, lo, hi float64, n int) {
	for i := 0; i < n; i++ {
		m := min(hi, xs[i])
		if m != m {
			m = math.Min(hi, xs[i])
		}
		xs[i] = m
		ys[i] = math.Min(lo, ys[i])
	}
}`
	fs := minMaxCallFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — the unguarded pair is still a call", len(fs))
	}
}
