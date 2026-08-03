package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func unsizedFanoutFindingsIn(t *testing.T, src string) []finding {
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
		if fnd.category == "fanout-not-sized-to-the-work" {
			out = append(out, fnd)
		}
	}
	return out
}

// TestDetectPS3061_UnsizedFanout is the measured shape: GOMAXPROCS workers, an on/off work
// gate, and a chunk division that partitions a range whose worker count was already fixed.
func TestDetectPS3061_UnsizedFanout(t *testing.T) {
	src := `package p

import (
	"runtime"
	"sync"
)

func parallelRows(n, workPerItem int, body func(lo, hi int)) {
	nw := runtime.GOMAXPROCS(0)
	if nw <= 1 || n*workPerItem < 1<<14 {
		body(0, n)
		return
	}
	if nw > n {
		nw = n
	}
	chunk := (n + nw - 1) / nw
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) { defer wg.Done(); body(lo, hi) }(lo, hi)
	}
	wg.Wait()
}`
	fs := unsizedFanoutFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Three things the measurements produced that the argument alone would not: that the on/off
	// gate cannot express the fix, that the grain has to be chosen against BOTH the clock and
	// the CPU because they disagree, and — from a later round — that the transform is NEUTRAL
	// outside the decode regime, so the check's own candidate list must not be worked through
	// blindly.
	if !containsAll(fs[0].msg, "THE ON/OFF GATE CANNOT",
		"PICK THE GRAIN BY MEASURING BOTH THE CLOCK AND THE CPU",
		"DO NOT APPLY THIS BLINDLY TO EVERY HELPER") {
		t.Fatalf("message omits why the gate is not enough or how to pick the grain:\n%s", fs[0].msg)
	}
}

// TestDetectPS3061_SilentWhenSized pins the fix. The worker count is reduced by a division of
// the work, so the helper already scales itself.
func TestDetectPS3061_SilentWhenSized(t *testing.T) {
	src := `package p

import (
	"runtime"
	"sync"
)

const grain = 1 << 15

func parallelRows(n, workPerItem int, body func(lo, hi int)) {
	nw := runtime.GOMAXPROCS(0)
	if w := n * workPerItem / grain; w < nw {
		nw = w
	}
	if nw <= 1 || n*workPerItem < 1<<14 {
		body(0, n)
		return
	}
	chunk := (n + nw - 1) / nw
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) { defer wg.Done(); body(lo, hi) }(lo, hi)
	}
	wg.Wait()
}`
	if fs := unsizedFanoutFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the worker count already scales with the work:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3061_SilentWithoutGOMAXPROCS pins that the shape is a helper that asks the
// machine how wide it is. One that takes a fixed width has already been told.
func TestDetectPS3061_SilentWithoutGOMAXPROCS(t *testing.T) {
	src := `package p

import "sync"

func parallelRows(n, nw int, body func(lo, hi int)) {
	chunk := (n + nw - 1) / nw
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) { defer wg.Done(); body(lo, hi) }(lo, hi)
	}
	wg.Wait()
}`
	if fs := unsizedFanoutFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the width is a parameter:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3061_SilentOnAPlainFunction pins that only registered fan-out helpers qualify —
// a function whose last parameter is not a callback over a work range is not one.
func TestDetectPS3061_SilentOnAPlainFunction(t *testing.T) {
	src := `package p

import "runtime"

func width(n int) int {
	nw := runtime.GOMAXPROCS(0)
	if nw > n {
		nw = n
	}
	return nw
}`
	if fs := unsizedFanoutFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — not a fan-out helper:\n%s", len(fs), fs[0].msg)
	}
}
