package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func interchangeFindingsIn(t *testing.T, src string) []finding {
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
		if fnd.category == "sequential-outer-with-an-independent-inner-loop" {
			out = append(out, fnd)
		}
	}
	return out
}

const bandHelper = `
func parallelRows(n, work int, body func(lo, hi int)) { body(0, n) }
`

// TestDetectPS3067_ConstantExtentSweep is the measured shape: a sequential column sweep with an
// independent row loop inside it, each row doing O(in) compensation work.
func TestDetectPS3067_ConstantExtentSweep(t *testing.T) {
	src := "package p\n" + bandHelper + `
func sweep(wm [][]float64, hinv [][]float64, out, in int) {
	for i := 0; i < in; i++ {
		for r := 0; r < out; r++ {
			e := wm[r][i] / hinv[i][i]
			for j := i + 1; j < in; j++ {
				wm[r][j] -= e * hinv[i][j]
			}
		}
	}
}`
	fs := interchangeFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The three things the two measurements produced: why this is not PS3040's advice, that
	// reused scratch must go per worker, and that a caller's callback becomes concurrent.
	if !containsAll(fs[0].msg, "INTERCHANGE FIRST", "must become PER WORKER",
		"CALLER-SUPPLIED CALLBACK") {
		t.Fatalf("message omits the fix, the scratch rule or the callback hazard:\n%s", fs[0].msg)
	}
}

// TestDetectPS3067_SilentOnAShrinkingExtent pins the boundary with PS3040. A factorization's
// inner loop shrinks as the outer index advances, so the rows really must be split per pivot
// and the interchange is not available.
func TestDetectPS3067_SilentOnAShrinkingExtent(t *testing.T) {
	src := "package p\n" + bandHelper + `
func factor(m []float64, n int) {
	for k := 0; k < n; k++ {
		for i := k + 1; i < n; i++ {
			for j := k; j < n; j++ {
				m[i*n+j] -= m[k*n+j]
			}
		}
	}
}`
	if fs := interchangeFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the inner extent shrinks, so PS3040 owns it:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3067_SilentWhenAWriteIsNotOwned pins the independence the finding rests on. A
// write whose indices name NEITHER loop variable is shared across the inner iterations, so
// banding them is a race rather than an optimization.
//
// An earlier version wrote acc[r*in+i], which names the OUTER variable — rejected by a
// different clause, which left a mutation of the ownership test green.
func TestDetectPS3067_SilentWhenAWriteIsNotOwned(t *testing.T) {
	src := "package p\n" + bandHelper + `
func sweep(tot []float64, wm [][]float64, out, in int) {
	for i := 0; i < in; i++ {
		for r := 0; r < out; r++ {
			for j := 0; j < in; j++ {
				tot[j] += wm[r][j]
			}
		}
	}
}`
	if fs := interchangeFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the write is shared across rows:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3067_SilentOnAShallowInnerLoop pins the narrowing that took the tree-wide count
// from 210 to 33. An inner loop with no loop of its own is elementwise: the interchange buys
// nothing and the work gate would keep it serial anyway.
//
// The write is OWNED by the inner variable on purpose. An earlier version wrote wm[r][i],
// whose index chain names the outer variable, so the fixture was rejected by the ownership
// test and a mutation of the depth test left it green.
func TestDetectPS3067_SilentOnAShallowInnerLoop(t *testing.T) {
	src := "package p\n" + bandHelper + `
func sweep(acc, s []float64, out, in int) {
	for i := 0; i < in; i++ {
		for r := 0; r < out; r++ {
			acc[r] += s[i]
		}
	}
}`
	if fs := interchangeFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the inner loop carries no work of its own:\n%s",
			len(fs), fs[0].msg)
	}
}
