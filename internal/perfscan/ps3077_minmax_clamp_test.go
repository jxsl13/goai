package main

import (
	"go/parser"
	"go/token"
	"testing"
)

func clampFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "minmax-clamp-in-a-loop" {
			out = append(out, fnd)
		}
	}
	return out
}

// clampFixture is the measured shape: a value held between two bounds by two function calls,
// once per element.
const clampFixture = `package p

import "math"

func round(q, w []float64, s, z, maxLevel float64) {
	for i, v := range w {
		q[i] = math.Min(maxLevel, math.Max(0, math.Round(v/s+z)))
	}
}`

func TestDetectPS3077_ClampInLoop(t *testing.T) {
	fs := clampFindingsIn(t, clampFixture)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The trap in the rewrite and the two-part gate are the whole value of this finding.
	if !containsAll(fs[0].msg, "WRITE THE CHAIN THAT IS ACTUALLY EQUIVALENT",
		"NEGATIVE ZERO", "GATE IT TWICE") {
		t.Fatalf("message omits the equivalent form, the negative-zero trap or the gate:\n%s",
			fs[0].msg)
	}
}

// TestDetectPS3077_SilentOnALoneMin pins the NESTED form. A single math.Min is often a genuine
// two-value choice rather than a clamp, and rewriting it buys one call, not two.
func TestDetectPS3077_SilentOnALoneMin(t *testing.T) {
	src := replaceOnce(t, clampFixture, "math.Min(maxLevel, math.Max(0, math.Round(v/s+z)))",
		"math.Min(maxLevel, math.Round(v/s+z))")
	if fs := clampFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — one bound is not a clamp:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3077_SilentOnTheSameFunctionNested pins that the two calls must be DIFFERENT ones.
// math.Min around math.Min is a three-way minimum, not a value held between bounds.
func TestDetectPS3077_SilentOnTheSameFunctionNested(t *testing.T) {
	src := replaceOnce(t, clampFixture, "math.Min(maxLevel, math.Max(0, math.Round(v/s+z)))",
		"math.Min(maxLevel, math.Min(0, math.Round(v/s+z)))")
	if fs := clampFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — that is a three-way minimum:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3077_SilentOutsideALoop pins where the cost is. One clamp per call is a few
// nanoseconds; the finding is about paying it once per element.
func TestDetectPS3077_SilentOutsideALoop(t *testing.T) {
	src := replaceOnce(t, clampFixture, `	for i, v := range w {
		q[i] = math.Min(maxLevel, math.Max(0, math.Round(v/s+z)))
	}`, `	q[0] = math.Min(maxLevel, math.Max(0, math.Round(w[0]/s+z)))`)
	if fs := clampFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a single clamp is not a hot path:\n%s", len(fs), fs[0].msg)
	}
}
