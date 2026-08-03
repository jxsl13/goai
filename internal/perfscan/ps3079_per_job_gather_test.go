package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func perJobGatherFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	fanoutReg = map[string]map[string]bool{}
	fansOutReg = map[string]map[string]bool{}
	allocGatherReg = map[string]map[string]bool{}
	collectFanoutHelpers([]*ast.File{f})
	collectAllocGathers([]*ast.File{f})
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "per-job-whole-input-allocation" {
			out = append(out, fnd)
		}
	}
	return out
}

// gatherFixture is the measured shape: a fan-out over trees, each job materializing its own copy
// of the whole training set.
const gatherFixture = `package p

import "sync"

func parallelBuild(n int, work func(t int) error) error {
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { defer wg.Done(); _ = work(0) }()
	wg.Wait()
	return nil
}

func gatherInt(x [][]float64, y []int, idx []int) ([][]float64, []int) {
	bx := make([][]float64, len(idx))
	by := make([]int, len(idx))
	for i, j := range idx {
		bx[i] = x[j]
		by[i] = y[j]
	}
	return bx, by
}

func fit(x [][]float64, y []int, samples [][]int, nt int) error {
	return parallelBuild(nt, func(t int) error {
		bx, by := gatherInt(x, y, samples[t])
		_, _ = bx, by
		return nil
	})
}`

func TestDetectPS3079_PerJobGather(t *testing.T) {
	fs := perJobGatherFindingsIn(t, gatherFixture)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The transform, the honest expectation, and the question that decides whether it is legal.
	if !containsAll(fs[0].msg, "RECYCLE THE BUFFERS", "EXPECT BYTES, NOT TIME",
		"THE SAFETY QUESTION IS RETENTION") {
		t.Fatalf("message omits the transform, the expectation or the safety question:\n%s",
			fs[0].msg)
	}
}

// TestDetectPS3079_SilentWhenAlreadyPooled pins the suppression that stops the check reporting
// its own fix.
func TestDetectPS3079_SilentWhenAlreadyPooled(t *testing.T) {
	src := replaceOnce(t, gatherFixture, `		bx, by := gatherInt(x, y, samples[t])`,
		`		gb := bufPool.Get()
		bx, by := gatherInt(x, y, samples[t])
		bufPool.Put(gb)`)
	src = replaceOnce(t, src, "func fit(", "var bufPool sync.Pool\n\nfunc fit(")
	if fs := perJobGatherFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — this job already recycles:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3079_SilentOnASingleResult pins the two-result shape. One returned slice is the
// function's RESULT; two or more is an input being materialized alongside it.
func TestDetectPS3079_SilentOnASingleResult(t *testing.T) {
	src := replaceOnce(t, gatherFixture, `func gatherInt(x [][]float64, y []int, idx []int) ([][]float64, []int) {
	bx := make([][]float64, len(idx))
	by := make([]int, len(idx))
	for i, j := range idx {
		bx[i] = x[j]
		by[i] = y[j]
	}
	return bx, by
}`, `func gatherInt(x [][]float64, y []int, idx []int) [][]float64 {
	bx := make([][]float64, len(idx))
	by := make([]int, len(idx))
	for i, j := range idx {
		bx[i] = x[j]
		by[i] = y[j]
	}
	_ = by
	return bx
}`)
	src = replaceOnce(t, src, "		bx, by := gatherInt(x, y, samples[t])\n		_, _ = bx, by",
		"		bx := gatherInt(x, y, samples[t])\n		_ = bx")
	if fs := perJobGatherFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — one result is a result:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3079_SilentOutsideAFanOut pins that the closure must be handed to a FAN-OUT HELPER
// OF THIS PACKAGE. The identical body wrapped in a stdlib call runs once and pays nothing a pool
// would win back — and a locally declared sequential runner is NOT usable here, because the
// helper collector registers anything taking a job callback, which is what it is for.
func TestDetectPS3079_SilentOutsideAFanOut(t *testing.T) {
	src := replaceOnce(t, gatherFixture, `	return parallelBuild(nt, func(t int) error {
		bx, by := gatherInt(x, y, samples[t])
		_, _ = bx, by
		return nil
	})`, `	f := sync.OnceFunc(func() {
		bx, by := gatherInt(x, y, samples[0])
		_, _ = bx, by
	})
	f()
	_ = nt
	return nil`)
	if fs := perJobGatherFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — that runner is sequential:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3079_SilentOnASingleAllocation pins the count. A function returning one allocated
// slice beside an error allocates its own result, not a copy of somebody's input.
func TestDetectPS3079_SilentOnASingleAllocation(t *testing.T) {
	src := replaceOnce(t, gatherFixture, `	bx := make([][]float64, len(idx))
	by := make([]int, len(idx))
	for i, j := range idx {
		bx[i] = x[j]
		by[i] = y[j]
	}
	return bx, by`, `	bx := make([][]float64, len(idx))
	for i, j := range idx {
		bx[i] = x[j]
	}
	return bx, y`)
	if fs := perJobGatherFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — one allocation is a result:\n%s", len(fs), fs[0].msg)
	}
}
