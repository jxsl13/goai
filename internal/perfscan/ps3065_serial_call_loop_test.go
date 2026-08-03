package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func serialCallLoopFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	fanoutReg = map[string]map[string]bool{}
	loopyFuncReg = map[string]map[string]bool{}
	collectFanoutHelpers([]*ast.File{f})
	collectLoopyFuncs([]*ast.File{f})
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "serial-loop-over-an-expensive-call" {
			out = append(out, fnd)
		}
	}
	return out
}

const fanoutAndKernel = `
func parallelBands(n, work int, body func(lo, hi int)) { body(0, n) }

func kernel(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
`

// TestDetectPS3065_LoopOverAnExpensiveCall is the measured shape: a kernel column, filled one
// independent evaluation at a time, with the per-item cost hidden inside the callee.
func TestDetectPS3065_LoopOverAnExpensiveCall(t *testing.T) {
	src := "package p\n" + fanoutAndKernel + `
func column(x [][]float64, xi []float64, n int) []float64 {
	col := make([]float64, n)
	for t := 0; t < n; t++ {
		col[t] = kernel(xi, x[t])
	}
	return col
}`
	fs := serialCallLoopFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Two things the measurement produced: why the nest checks cannot see this, and that the
	// payoff is Amdahl-bounded rather than core-count-bounded.
	if !containsAll(fs[0].msg, "EVERY NEST-BASED CHECK MISSES THIS",
		"EXPECT AMDAHL, NOT THE CORE COUNT") {
		t.Fatalf("message omits why it is separate or the honest expectation:\n%s", fs[0].msg)
	}
}

// TestDetectPS3065_SilentOnACheapCall pins the condition that keeps this usable. A callee with
// no loop of its own is O(1) per item, and a loop over one is not worth a fan-out.
func TestDetectPS3065_SilentOnACheapCall(t *testing.T) {
	src := "package p\n" + fanoutAndKernel + `
func scale(v float64) float64 { return v * 2 }

func fill(src, dst []float64, n int) {
	for t := 0; t < n; t++ {
		dst[t] = scale(src[t])
	}
}`
	if fs := serialCallLoopFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the callee is O(1):\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3065_SilentOnANest pins the boundary with the nest checks. A loop with an inner
// loop is their shape, and reporting it here would only duplicate them.
//
// The write is a BARE call on purpose. An earlier version added a term to it, which made the
// right-hand-side test reject the fixture before the depth test was reached and left a mutation
// of the depth test green.
func TestDetectPS3065_SilentOnANest(t *testing.T) {
	src := "package p\n" + fanoutAndKernel + `
func fill(x [][]float64, dst, acc []float64, n, d int) {
	for t := 0; t < n; t++ {
		for j := 0; j < d; j++ {
			acc[t] += x[t][j]
		}
		dst[t] = kernel(x[t], x[t])
	}
}`
	if fs := serialCallLoopFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the nest checks own this:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3065_SilentWhenTheFunctionAlreadyFansOut pins that this is about a MISSING level.
func TestDetectPS3065_SilentWhenTheFunctionAlreadyFansOut(t *testing.T) {
	src := "package p\n" + fanoutAndKernel + `
func column(x [][]float64, xi []float64, n int) []float64 {
	col := make([]float64, n)
	parallelBands(n, len(xi), func(lo, hi int) {
		for t := lo; t < hi; t++ {
			col[t] = kernel(xi, x[t])
		}
	})
	return col
}`
	if fs := serialCallLoopFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — already fanned out:\n%s", len(fs), fs[0].msg)
	}
}
