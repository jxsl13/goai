package main

import (
	"go/parser"
	"go/token"
	"testing"
)

func intMapFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var out []finding
	for _, fd := range scanFile(fset, f, testSets(t)) {
		if fd.category == "integer-keyed-map-in-a-loop" {
			out = append(out, fd)
		}
	}
	return out
}

// TestDetectPS3083_MapBuiltPerPass is the measured shape: a membership set rebuilt for every
// query and probed once per key of the inner loop.
func TestDetectPS3083_MapBuiltPerPass(t *testing.T) {
	src := `package p

func attend(scores []float64, blocks []int, seq, blockSize, topK int) {
	for i := range seq {
		cur := i / blockSize
		selected := map[int]bool{cur: true}
		for gi := 0; gi < len(blocks) && len(selected) < topK; gi++ {
			selected[blocks[gi]] = true
		}
		for j := 0; j <= i; j++ {
			if !selected[j/blockSize] {
				scores[j] = 0
			}
		}
	}
}`
	fs := intMapFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The container swap, the free reset, the len trap and the gate the change needs.
	if !containsAll(fs[0].msg, "STAMPING IT WITH A GENERATION COUNTER", "WATCH THE LENGTH",
		"DISTINCT", "THE GATE IS A DIGEST") {
		t.Fatalf("message omits the stamp, the len trap or the gate:\n%s", fs[0].msg)
	}
}

// TestDetectPS3083_MakeForm pins that the make spelling is the same finding as a literal.
func TestDetectPS3083_MakeForm(t *testing.T) {
	src := `package p

func f(xs []int, n, m int) int {
	total := 0
	for i := range n {
		seen := make(map[int]struct{}, 8)
		seen[i] = struct{}{}
		for j := range m {
			if _, ok := seen[xs[j]]; ok {
				total++
			}
		}
	}
	return total
}`
	if fs := intMapFindingsIn(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — make is the same map", len(fs))
	}
}

// TestDetectPS3083_SilentOnAStringKey pins the key-type condition. A string key has no dense
// index to fall back on, so there is no slice to swap in and the advice would be wrong.
func TestDetectPS3083_SilentOnAStringKey(t *testing.T) {
	src := `package p

func f(names []string, n, m int) int {
	total := 0
	for i := range n {
		seen := map[string]bool{}
		for j := range m {
			if seen[names[j]] {
				total++
			}
		}
	}
	return total
}`
	if fs := intMapFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a string key has no dense index:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3083_SilentWhenBuiltOnce pins that the allocation is PER PASS. A map built once
// and probed in a loop is a lookup table, and the fix here would buy nothing.
func TestDetectPS3083_SilentWhenBuiltOnce(t *testing.T) {
	src := `package p

func f(xs []int, n, m int) int {
	seen := map[int]bool{}
	total := 0
	for i := range n {
		for j := range m {
			if seen[xs[j]] {
				total++
			}
		}
	}
	return total
}`
	if fs := intMapFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — built once, not per pass:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3083_SilentWithoutANestedProbe pins that the cost this reports is PER ELEMENT of
// an inner loop. A map built and read a constant number of times per pass is one allocation,
// which is a different and much smaller finding.
func TestDetectPS3083_SilentWithoutANestedProbe(t *testing.T) {
	src := `package p

func f(n int) int {
	total := 0
	for i := range n {
		seen := map[int]bool{i: true}
		if seen[i] {
			total++
		}
	}
	return total
}`
	if fs := intMapFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — no inner loop pays per element:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3083_SilentWhenTheInnerLoopProbesSomethingElse pins that the probe has to be OF
// THIS MAP. An inner loop indexing an ordinary slice pays no hash, and a check that reported
// any index expression at all would fire on almost every loop that happens to build a map.
//
// This one was green under mutation until it was written: dropping the name test from the
// probe left every other fixture passing.
func TestDetectPS3083_SilentWhenTheInnerLoopProbesSomethingElse(t *testing.T) {
	src := `package p

func f(xs []int, n, m int) int {
	total := 0
	for i := range n {
		seen := map[int]bool{i: true}
		if seen[i] {
			total++
		}
		for j := range m {
			total += xs[j]
		}
	}
	return total
}`
	if fs := intMapFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the inner loop indexes a slice, not the map:\n%s",
			len(fs), fs[0].msg)
	}
}
