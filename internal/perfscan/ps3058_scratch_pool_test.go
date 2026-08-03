package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// scratchFindingsIn primes the package-level registry from the fixture itself. Reaching for
// plain scanSrc would leave the registry holding whatever an earlier test put there, which is
// a fixture that passes for a reason unrelated to its own source.
func scratchFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	scratchTypeReg = map[string]map[string]bool{}
	scratchInitializers = map[string]bool{}
	collectScratchTypes([]*ast.File{f})
	var out []finding
	for _, f := range scanFile(fset, f, testSets(t)) {
		if f.category == "per-iteration-scratch-allocation" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3058_UnpooledScratchInitializer is the measured shape: a method that allocates a
// whole working set into its receiver's fields, with nothing recycling it.
func TestDetectPS3058_UnpooledScratchInitializer(t *testing.T) {
	src := `package p

type builder struct {
	sortBuf []int
	keys    []uint64
	vals    []float64
	tmp     []int
}

func (b *builder) init(n int) {
	b.sortBuf = make([]int, n)
	b.keys = make([]uint64, n)
	b.vals = make([]float64, n)
	b.tmp = make([]int, n)
}`
	fs := scratchFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The ranking instruction, the safety argument and the honest expectation all came out of
	// the measurement, and the last one matters most: this transform moved no wall clock.
	if !containsAll(fs[0].msg, "RANK IT BY WHERE THE TYPE IS CONSTRUCTED",
		"THE SAFETY ARGUMENT IS USUALLY ALREADY MADE", "EXPECT MEMORY, NOT NECESSARILY TIME") {
		t.Fatalf("message omits the ranking rule, the safety argument or the honest\n"+
			"expectation:\n%s", fs[0].msg)
	}
}

// TestDetectPS3058_SilentBelowThreeBuffers pins the count the finding rests on. One or two
// buffers is an ordinary constructor, not a working set worth recycling.
func TestDetectPS3058_SilentBelowThreeBuffers(t *testing.T) {
	src := `package p

type small struct {
	a []int
	b []int
}

func (s *small) init(n int) {
	s.a = make([]int, n)
	s.b = make([]int, n)
}`
	if fs := scratchFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — two buffers is a constructor:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3058_SilentWhenTheInitializerPools pins the suppression, and it is PER METHOD on
// purpose. A package-level test was tried and is useless: one sync.Pool anywhere in a large
// package silenced every initializer in it, taking the tree-wide count from 8 to 0.
func TestDetectPS3058_SilentWhenTheInitializerPools(t *testing.T) {
	src := `package p

import "sync"

type scratch struct {
	sortBuf []int
	keys    []uint64
	vals    []float64
}

var pool sync.Pool

type builder struct {
	sortBuf []int
	keys    []uint64
	vals    []float64
	sc      *scratch
}

func (b *builder) init(n int) {
	sc, _ := pool.Get().(*scratch)
	if sc == nil {
		sc = &scratch{}
	}
	b.sc = sc
	b.sortBuf = make([]int, n)
	b.keys = make([]uint64, n)
	b.vals = make([]float64, n)
}`
	if fs := scratchFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — this initializer already recycles:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3058_SilentOnAPlainFunction pins that the shape is a METHOD writing its own
// receiver. A free function allocating three slices it returns owns nothing to recycle.
//
// This one is guaranteed by construction rather than by a condition worth mutating: the
// collector only ever enters methods into the registry, so a plain function cannot reach the
// report. It is here to say so, not to gate a branch.
func TestDetectPS3058_SilentOnAPlainFunction(t *testing.T) {
	src := `package p

func alloc(n int) ([]int, []uint64, []float64) {
	return make([]int, n), make([]uint64, n), make([]float64, n)
}`
	if fs := scratchFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — no receiver owns these:\n%s", len(fs), fs[0].msg)
	}
}
