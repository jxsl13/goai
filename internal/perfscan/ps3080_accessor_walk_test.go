package main

import (
	"go/parser"
	"go/token"
	"testing"
)

func accessorWalkFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "one-dimensional-accessor-walk" {
			out = append(out, fnd)
		}
	}
	return out
}

// walkFixture is the measured shape: a flat tensor read and written element by element through
// the dispatching accessors.
const walkFixture = `package p

type T struct{}

func (T) AtF64(idx ...int) float64   { return 0 }
func (T) SetF64(v float64, idx ...int) {}

func grad(a, b, c, out T, n int) {
	for i := 0; i < n; i++ {
		r := a.AtF64(i) - b.AtF64(i)
		out.SetF64(r*c.AtF64(i), i)
	}
}`

func TestDetectPS3080_AccessorWalk(t *testing.T) {
	fs := accessorWalkFindingsIn(t, walkFixture)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The relationship to PS1005, the fallback requirement and the arm-comparison trap.
	if !containsAll(fs[0].msg, "PS1005 REPORTS THE MULTI-DIMENSIONAL VERSION",
		"KEEP THE ACCESSOR ARM", "CANNOT BE COMPARED AS EQUAL BITS") {
		t.Fatalf("message omits the PS1005 relationship, the fallback or the arm trap:\n%s",
			fs[0].msg)
	}
}

// TestDetectPS3080_SilentOnAMultiDimWalk pins the division of labor. Two index arguments is the
// shape PS1005 already reports, and reporting it twice would double every triage.
func TestDetectPS3080_SilentOnAMultiDimWalk(t *testing.T) {
	src := replaceOnce(t, walkFixture, `		r := a.AtF64(i) - b.AtF64(i)
		out.SetF64(r*c.AtF64(i), i)`, `		r := a.AtF64(i, 0) - b.AtF64(i, 0)
		out.SetF64(r*c.AtF64(i, 0), i, 0)`)
	if fs := accessorWalkFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — PS1005 owns that shape:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3080_SilentWithATypedFastPath pins the suppression that stops the check reporting
// its own fix: the accessor loop is kept as a deliberate fallback beside the typed path.
func TestDetectPS3080_SilentWithATypedFastPath(t *testing.T) {
	src := replaceOnce(t, walkFixture, "	for i := 0; i < n; i++ {", `	if fast {
		s := a.Storage().F64()
		_ = s
	}
	for i := 0; i < n; i++ {`)
	src = replaceOnce(t, src, "func grad(a, b, c, out T, n int) {", `type S struct{}

func (S) F64() []float64 { return nil }
func (T) Storage() S     { return S{} }

var fast bool

func grad(a, b, c, out T, n int) {`)
	if fs := accessorWalkFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a typed path already exists here:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3080_SilentBelowThreeCalls pins the density. One accessor in a loop is an
// incidental read; the finding is a loop built out of them.
func TestDetectPS3080_SilentBelowThreeCalls(t *testing.T) {
	src := replaceOnce(t, walkFixture, `		r := a.AtF64(i) - b.AtF64(i)
		out.SetF64(r*c.AtF64(i), i)`, `		r := a.AtF64(i)
		out.SetF64(r, i)`)
	if fs := accessorWalkFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — two accessors is incidental:\n%s", len(fs), fs[0].msg)
	}
}
