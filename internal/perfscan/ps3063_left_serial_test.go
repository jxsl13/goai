package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func leftSerialFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	fanoutReg = map[string]map[string]bool{}
	collectFanoutHelpers([]*ast.File{f})
	ns := testSets(t)
	if !ns.accessors["SetF64"] {
		t.Fatal("perfscan.json must list the element accessors, or the setter-call half of this" +
			" check is silent for a reason unrelated to the fixture")
	}
	var out []finding
	for _, fnd := range scanFile(fset, f, ns) {
		if fnd.category == "one-loop-left-serial-in-a-fanning-function" {
			out = append(out, fnd)
		}
	}
	return out
}

// TestDetectPS3063_OneLoopLeftSerial is the measured shape: a function that fans out one n³
// product and leaves the next one serial, writing its result through a setter rather than an
// index expression — which is why the checks that came before could not see it.
func TestDetectPS3063_OneLoopLeftSerial(t *testing.T) {
	src := `package p

func parallelIdx(n, work int, body func(i int)) {
	for i := 0; i < n; i++ {
		body(i)
	}
}

type T struct{}

func (t *T) SetF64(v float64, idx ...int) {}

func vjp(v [][]float64, tmp [][]float64, out *T, n int) {
	parallelIdx(n, n*n*n, func(a int) {
		for j := 0; j < n; j++ {
			tmp[a][j] = v[a][j]
		}
	})
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			var g float64
			for a := 0; a < n; a++ {
				g += v[i][a] * tmp[a][j]
			}
			out.SetF64(g, i, j)
			out.SetF64(g, j, i)
		}
	}
}`
	fs := leftSerialFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The message has to say why this is separate from PS3034 and PS3059, and carry both
	// measurements — the check exists because their shared suppression hid this shape twice.
	if !containsAll(fs[0].msg, "PS3034 and PS3059 both suppress this case",
		"TWICE MEASURED, BOTH LARGE", "CHECK THE WRITES OWN DISJOINT OUTPUT") {
		t.Fatalf("message omits why it is separate, the evidence or the safety check:\n%s",
			fs[0].msg)
	}
}

// TestDetectPS3063_SilentWhenNothingFansOut pins the boundary with PS3034 and PS3059. A
// function that never fans out at all is their shape, not this one.
func TestDetectPS3063_SilentWhenNothingFansOut(t *testing.T) {
	src := `package p

func parallelIdx(n, work int, body func(i int)) {
	for i := 0; i < n; i++ {
		body(i)
	}
}

type T struct{}

func (t *T) SetF64(v float64, idx ...int) {}

func vjp(v [][]float64, tmp [][]float64, out *T, n int) {
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			var g float64
			for a := 0; a < n; a++ {
				g += v[i][a] * tmp[a][j]
			}
			out.SetF64(g, i, j)
		}
	}
}`
	if fs := leftSerialFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — PS3034 and PS3059 own this:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3063_SilentWhenAWriteEscapesTheIteration pins the disjointness the finding rests
// on. A write whose index does not come from the outer variable is shared across iterations,
// and banding it is a race rather than an optimization.
//
// THREE LOOPS DEEP ON PURPOSE. At two, the check's own nesting requirement silenced this
// fixture before the ownership test was reached, and a mutation removing that test left it
// green — the same trap the PS3056 fixtures fell into.
func TestDetectPS3063_SilentWhenAWriteEscapesTheIteration(t *testing.T) {
	src := `package p

func parallelIdx(n, work int, body func(i int)) {
	for i := 0; i < n; i++ {
		body(i)
	}
}

type T struct{}

func (t *T) SetF64(v float64, idx ...int) {}

func vjp(v [][]float64, tmp [][]float64, tot []float64, out *T, n int) {
	parallelIdx(n, n*n*n, func(a int) {
		for j := 0; j < n; j++ {
			tmp[a][j] = v[a][j]
		}
	})
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			for a := 0; a < n; a++ {
				out.SetF64(v[i][a], i, j)
				tot[j] += v[i][a]
			}
		}
	}
}`
	if fs := leftSerialFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — tot[j] is shared across iterations:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3063_SilentInsideTheCallback pins that a nest already running inside a fan-out
// callback is parallel one level out and is not this finding. Deep enough to clear the nesting
// requirement, for the same reason as the fixture above, and its writes are OWNED by the nest's
// own outer variable — with an unowned write it would have been silent on that instead, which
// is what a mutation removing the callback test exposed.
func TestDetectPS3063_SilentInsideTheCallback(t *testing.T) {
	src := `package p

func parallelIdx(n, work int, body func(i int)) {
	for i := 0; i < n; i++ {
		body(i)
	}
}

type T struct{}

func (t *T) SetF64(v float64, idx ...int) {}

func vjp(v [][]float64, dst []float64, out *T, n int) {
	parallelIdx(n, n*n*n, func(i int) {
		for j := 0; j < n; j++ {
			for a := 0; a < n; a++ {
				for b := 0; b < n; b++ {
					dst[j*n+b] = v[i][a]
				}
			}
		}
	})
}`
	if fs := leftSerialFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — already inside the callback:\n%s", len(fs), fs[0].msg)
	}
}
