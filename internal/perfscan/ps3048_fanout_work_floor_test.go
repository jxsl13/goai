package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func workFloorFindingsIn(t *testing.T, src string) []finding {
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
		if fnd.category == "fanout-without-a-work-floor" {
			out = append(out, fnd)
		}
	}
	return out
}

// TestDetectPS3048_NoFloorOnWorkPerWorker is the measured shape: a pool that decides WHETHER to
// fan out from the total work and then splits to the full width of the machine regardless of how
// little each band ends up with.
func TestDetectPS3048_NoFloorOnWorkPerWorker(t *testing.T) {
	src := `package p

import "runtime"

const parThreshold = 1 << 15

func parallelWork(n, workPerItem int, body func(lo, hi int)) {
	workers := runtime.GOMAXPROCS(0)
	total := n * workPerItem
	if workers <= 1 || total < parThreshold {
		body(0, n)
		return
	}
	chunk := (n + workers - 1) / workers
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		body(lo, hi)
	}
}`
	fs := workFloorFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Three things the fix needs: the formula, the warning that the curve is not monotone so the
	// floor is a measurement, and the safety condition on changing the band count.
	if !containsAll(fs[0].msg, "DERIVE THE COUNT FROM THE WORK", "THE CURVE IS NOT MONOTONE",
		"CHANGING THE BAND COUNT MUST NOT CHANGE A VALUE") {
		t.Fatalf("message omits the fix, the sweep warning or the safety condition:\n%s", fs[0].msg)
	}
}

// TestDetectPS3048_SilentWithAFloor pins the APPLIED form: the worker count derived from the work
// and capped at the core count.
func TestDetectPS3048_SilentWithAFloor(t *testing.T) {
	src := `package p

import "runtime"

const parThreshold = 1 << 15

func parallelWork(n, workPerItem int, body func(lo, hi int)) {
	workers := runtime.GOMAXPROCS(0)
	total := n * workPerItem
	if workers <= 1 || total < parThreshold {
		body(0, n)
		return
	}
	if w := total / parThreshold; w < workers {
		workers = max(w, 1)
	}
	if workers <= 1 {
		body(0, n)
		return
	}
	chunk := (n + workers - 1) / workers
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		body(lo, hi)
	}
}`
	if fs := workFloorFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the worker count is derived from the work:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3048_ChunkDivisionIsNotAFloor is the discriminator, and it is the one that decides
// whether this check is worth anything. EVERY band splitter divides to get its chunk size —
// chunk := (n + workers - 1) / workers — and reading that as "the work is being divided" made the
// check silent on the pool it was written for. A division BY the worker count sizes a chunk; a
// division that BOUNDS the worker count is the floor.
func TestDetectPS3048_ChunkDivisionIsNotAFloor(t *testing.T) {
	src := `package p

import "runtime"

func parallelRows(n, workPerItem int, body func(lo, hi int)) {
	nw := runtime.GOMAXPROCS(0)
	if nw <= 1 || n*workPerItem < 1<<14 {
		body(0, n)
		return
	}
	chunk := (n + nw - 1) / nw
	for lo := 0; lo < n; lo += chunk {
		body(lo, min(lo+chunk, n))
	}
}`
	if fs := workFloorFindingsIn(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — a chunk division is not a work floor", len(fs))
	}
}

// TestDetectPS3048_SilentWithoutGOMAXPROCS pins that the finding is about taking the width from
// the machine. A helper handed its band size by the caller has already had that decision made
// somewhere this check cannot see. The fixture divides nothing, so it discriminates the
// GOMAXPROCS condition rather than the division test that the other silent cases rest on.
func TestDetectPS3048_SilentWithoutGOMAXPROCS(t *testing.T) {
	src := `package p

func parallelFixed(n, chunk int, body func(lo, hi int)) {
	for lo := 0; lo < n; lo += chunk {
		body(lo, min(lo+chunk, n))
	}
}`
	if fs := workFloorFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the width comes from the caller:\n%s", len(fs), fs[0].msg)
	}
}
