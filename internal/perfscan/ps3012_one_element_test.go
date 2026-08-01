package main

import (
	"strings"
	"testing"
)

func oneElementFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "slice-built-for-one-element" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3012_IndexedCall is the measured shape: nlp rows2D materializes every row of a
// tensor and the decode caller keeps row 0, once per layer per token.
func TestDetectPS3012_IndexedCall(t *testing.T) {
	src := `package p

func step(x *T) float64 {
	row := rows2D(x)[0]
	return row[0]
}`
	fs := oneElementFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Where the scratch lives is the part that turns this from a speedup into a data race, so
	// the message has to carry it.
	if !strings.Contains(fs[0].msg, "per-stream") {
		t.Fatalf("message omits where the scratch may live:\n%s", fs[0].msg)
	}
}

// TestDetectPS3012_SilentOnMethodChain is the precision floor that decided the check's shape.
//
// t.Shape()[0] and s.Storage().F64()[0] return a view or a field and allocate nothing. Including
// method chains matched them everywhere and buried the real class; the repo has 20 findings with
// this restriction and far more without, essentially all of them noise.
func TestDetectPS3012_SilentOnMethodChain(t *testing.T) {
	src := `package p

func f(x *T) int {
	a := x.Shape()[0]
	b := x.Storage().F64()[0]
	_ = b
	return a
}`
	if fs := oneElementFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a method chain returns a view, not a fresh collection:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3012_SilentOnBuiltins keeps len/cap/append and friends out. Indexing their result is
// either impossible or free, and flagging them would be pure noise.
func TestDetectPS3012_SilentOnBuiltins(t *testing.T) {
	src := `package p

func f(a, b []int) int {
	return append(a, b...)[0] + min(a, b)[0]
}`
	if fs := oneElementFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — builtins are not collection builders worth avoiding", len(fs))
	}
}

// TestDetectPS3012_SilentOnVariableIndex keeps the check to the case where ONE element is provably
// all the caller wants. A variable index may walk the whole collection across iterations, which is
// exactly what building it was for.
func TestDetectPS3012_SilentOnVariableIndex(t *testing.T) {
	src := `package p

func f(x *T, i int) float64 {
	return rows2D(x)[i][0]
}`
	if fs := oneElementFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a variable index does not prove only one element is wanted", len(fs))
	}
}

// TestDetectPS3012_ReportsNonZeroConstantIndex pins that the check is about building a collection
// for ONE element, not about the literal zero. Taking row 3 wastes just as much as taking row 0.
func TestDetectPS3012_ReportsNonZeroConstantIndex(t *testing.T) {
	src := `package p

func f(x *T) []float64 {
	return rows2D(x)[3]
}`
	if fs := oneElementFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — any constant index keeps exactly one element", len(fs))
	}
}
