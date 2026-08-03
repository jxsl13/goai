package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func bestOfScanFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	fanoutReg = map[string]map[string]bool{}
	collectFanoutHelpers([]*ast.File{f})
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "serial-best-of-scan" {
			out = append(out, fnd)
		}
	}
	return out
}

const scanHelper = `
func parallelRows(n, work int, body func(lo, hi int)) { body(0, n) }
`

// TestDetectPS3068_BestOfScan is the measured shape: a feature scan keeping the lowest cost and
// recording the winner.
func TestDetectPS3068_BestOfScan(t *testing.T) {
	src := "package p\n" + scanHelper + `
func best(feats []int, cost func(int) float64) (feat int, c float64, ok bool) {
	bestCost := 1e300
	for _, f := range feats {
		cv := cost(f)
		if cv < bestCost {
			bestCost = cv
			feat = f
			ok = true
		}
	}
	return feat, bestCost, ok
}`
	fs := bestOfScanFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The fold rule and the testing lesson are what the measurement produced; the argument
	// alone gives neither.
	if !containsAll(fs[0].msg, "FOLD THE CHUNKS IN ASCENDING ORDER",
		"A DIGEST WILL NOT GATE THE TIE RULE", "in BOTH placements") {
		t.Fatalf("message omits the fold rule, the digest warning or the two placements:\n%s",
			fs[0].msg)
	}
}

// TestDetectPS3068_SilentOnARunningTotal pins the condition that keeps this usable. A loop that
// updates ONE outer name is an accumulator, not a record of a winner, and it has no tie rule to
// reproduce.
func TestDetectPS3068_SilentOnARunningTotal(t *testing.T) {
	src := "package p\n" + scanHelper + `
func maxOf(xs []float64) float64 {
	m := xs[0]
	for _, v := range xs {
		if v > m {
			m = v
		}
	}
	return m
}`
	if fs := bestOfScanFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — one name is an accumulator:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3068_SilentOnANonStrictComparison pins that the shape is a FIRST-wins rule. A
// loop taking a new best on <= keeps the LAST of equal items, which a chunked fold reproduces
// differently and which this check's advice would silently change.
func TestDetectPS3068_SilentOnANonStrictComparison(t *testing.T) {
	src := "package p\n" + scanHelper + `
func best(feats []int, cost func(int) float64) (feat int, c float64) {
	bestCost := 1e300
	for _, f := range feats {
		cv := cost(f)
		if cv <= bestCost {
			bestCost = cv
			feat = f
		}
	}
	return feat, bestCost
}`
	if fs := bestOfScanFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — <= keeps the last, not the first:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3068_SilentWhenTheFunctionAlreadyFansOut pins that this is about a MISSING level.
func TestDetectPS3068_SilentWhenTheFunctionAlreadyFansOut(t *testing.T) {
	src := "package p\n" + scanHelper + `
func best(feats []int, cost func(int) float64, out []float64) (feat int, c float64, ok bool) {
	parallelRows(len(feats), 64, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			out[i] = cost(feats[i])
		}
	})
	bestCost := 1e300
	for i, cv := range out {
		if cv < bestCost {
			bestCost = cv
			feat = i
			ok = true
		}
	}
	return feat, bestCost, ok
}`
	if fs := bestOfScanFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the function already fans out:\n%s", len(fs), fs[0].msg)
	}
}
