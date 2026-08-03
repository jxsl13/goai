package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// serialOverParallelFindingsIn primes both registries from the fixture itself; plain scanSrc
// would leave them holding whatever an earlier test file put there.
func serialOverParallelFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	fanoutReg = map[string]map[string]bool{}
	fansOutReg = map[string]map[string]bool{}
	collectFanoutHelpers([]*ast.File{f})
	collectFanningFuncs([]*ast.File{f})
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "serial-loop-over-parallel-work" {
			out = append(out, fnd)
		}
	}
	return out
}

// TestDetectPS3060_SerialLoopOverFanningWork is the measured shape: an optimizer stepping each
// parameter in turn, where the per-parameter work already fans out and the parameter loop does
// not.
func TestDetectPS3060_SerialLoopOverFanningWork(t *testing.T) {
	src := `package p

func parallelRows(n, work int, body func(lo, hi int)) { body(0, n) }

func orthogonalize(x []float64, r, c int) {
	parallelRows(r, c, func(lo, hi int) { _ = x[lo:hi] })
}

func step(params [][]float64, r, c int) {
	for pi := range params {
		orthogonalize(params[pi], r, c)
	}
}`
	fs := serialOverParallelFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The three things the measurement produced: where the win actually comes from, the
	// callback constraint, and the test-sizing trap that this very change fell into first.
	if !containsAll(fs[0].msg, "THE WIN IS NOT ONLY THE WORK, IT IS THE STALLS",
		"CALL ANY CALLER-SUPPLIED CALLBACK SERIALLY FIRST",
		"SIZE THE TEST ABOVE THE FAN-OUT HELPER'S WORK GATE") {
		t.Fatalf("message omits where the win is, the callback rule or the sizing trap:\n%s",
			fs[0].msg)
	}
}

// TestDetectPS3060_SilentOnARefinementLoop pins the narrowing that stopped this check reporting
// the inside of the function it was written from. A loop with no variable — "for range steps" —
// repeats a refinement rather than iterating over items, and the Newton-Schulz iteration is
// exactly that: step i+1 reads what step i wrote.
func TestDetectPS3060_SilentOnARefinementLoop(t *testing.T) {
	src := `package p

func parallelRows(n, work int, body func(lo, hi int)) { body(0, n) }

func matmulInto(dst, a, b []float64, r, c int) []float64 {
	parallelRows(r, c, func(lo, hi int) { _ = dst[lo:hi] })
	return dst
}

func newtonSchulz(x, buf []float64, r, c, steps int) {
	for range steps {
		x = matmulInto(buf, x, x, r, c)
	}
}`
	if fs := serialOverParallelFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the loop refines, it does not iterate:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3060_SilentWhenTheCallerAlreadyFansOut pins that the check is about a MISSING
// level, not a second one. A function that already dispatches has made the choice.
func TestDetectPS3060_SilentWhenTheCallerAlreadyFansOut(t *testing.T) {
	src := `package p

func parallelRows(n, work int, body func(lo, hi int)) { body(0, n) }

func orthogonalize(x []float64, r, c int) {
	parallelRows(r, c, func(lo, hi int) { _ = x[lo:hi] })
}

func step(params [][]float64, r, c int) {
	parallelRows(len(params), r*c, func(plo, phi int) {
		for pi := plo; pi < phi; pi++ {
			orthogonalize(params[pi], r, c)
		}
	})
}`
	if fs := serialOverParallelFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the outer level is already there:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3060_SilentWhenTheCalleeIsSerial pins the condition the finding rests on. A loop
// over a function that never fans out is an ordinary serial loop, and whether banding it pays
// is a different question this check does not answer.
//
// The package HAS a fanning function that the loop does not call. Without it the fixture was
// vacuous: the detector returns early when a package contains no fanning function at all, so
// the fixture passed for that reason and a mutation letting any callee through left it green.
func TestDetectPS3060_SilentWhenTheCalleeIsSerial(t *testing.T) {
	src := `package p

func parallelRows(n, work int, body func(lo, hi int)) { body(0, n) }

func orthogonalize(x []float64, r, c int) {
	parallelRows(r, c, func(lo, hi int) { _ = x[lo:hi] })
}

func scale(x []float64, k float64) {
	for i := range x {
		x[i] *= k
	}
}

func step(params [][]float64, k float64) {
	for pi := range params {
		scale(params[pi], k)
	}
}`
	if fs := serialOverParallelFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the callee is serial:\n%s", len(fs), fs[0].msg)
	}
}
