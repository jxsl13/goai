package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func sharedThresholdFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	thresholdUses = map[string]map[string]string{}
	collectThresholdComparisons([]*ast.File{f})
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "one-threshold-two-regimes" {
			out = append(out, fnd)
		}
	}
	return out
}

// TestDetectPS3070_TwoRegimes is the measured shape: one cutoff gating a per-node sort in one
// function and a whole-matrix presort in another.
func TestDetectPS3070_TwoRegimes(t *testing.T) {
	src := `package p

const cutoff = 512

func perNode(order []int) bool {
	return len(order) < cutoff
}

func presort(rows int) bool {
	return rows >= cutoff
}`
	fs := sharedThresholdFindingsIn(t, src)
	if len(fs) == 0 {
		t.Fatal("0 findings, want at least 1")
	}
	// Two things the message has to carry. The order of operations, which is easy to get
	// wrong — sweeping a shared constant measures the sum of two answers — and the triage,
	// because both candidates this check found in its own repository were false positives:
	// one a numeric guard rather than a knob, one a size gate whose crossover no benchmark
	// reaches.
	if !containsAll(fs[0].msg, "SPLIT FIRST, THEN SWEEP", "missing second constant",
		"TRIAGE BEFORE SPLITTING", "IS THE CROSSOVER BENCHMARKED") {
		t.Fatalf("message omits the order of operations or the diagnosis:\n%s", fs[0].msg)
	}
}

// TestDetectPS3070_SilentOnOneQuantity pins the condition. The same quantity tested in two
// places is ONE regime, however many call sites it has.
//
// The quantity count and the function count catch this case TOGETHER: the use map is keyed by
// the compared expression, so two functions testing the same quantity collapse to one entry and
// the function set collapses with it. Removing either condition alone leaves this green — that
// is cooperation rather than a gap, since either is sufficient.
func TestDetectPS3070_SilentOnOneQuantity(t *testing.T) {
	src := `package p

const cutoff = 512

func a(rows int) bool {
	return rows < cutoff
}

func b(rows int) bool {
	return rows >= cutoff
}`
	if fs := sharedThresholdFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — one quantity is one regime:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3070_SilentWithinOneFunction pins that two regimes must be in two FUNCTIONS. A
// single function testing two quantities against one constant is choosing one policy, and
// splitting the constant there is a judgement call rather than a defect.
func TestDetectPS3070_SilentWithinOneFunction(t *testing.T) {
	src := `package p

const cutoff = 512

func both(rows, cols int) bool {
	return rows < cutoff && cols >= cutoff
}`
	if fs := sharedThresholdFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — one function, one policy:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3070_SilentOnANonConstant pins that the shape is a named CONSTANT. A variable
// compared in two places is not a tuning knob shared between regimes.
func TestDetectPS3070_SilentOnANonConstant(t *testing.T) {
	src := `package p

func a(rows, cutoff int) bool {
	return rows < cutoff
}

func b(cols, cutoff int) bool {
	return cols >= cutoff
}`
	if fs := sharedThresholdFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — not a constant:\n%s", len(fs), fs[0].msg)
	}
}
