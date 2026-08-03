package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// invariantOperandFindingsIn primes the reduction-helper registry from the fixture itself, so a
// fixture is silent or not for a reason contained in its own source.
func invariantOperandFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	reductionHelperReg = map[string]map[string][]bool{}
	collectReductionHelpers([]*ast.File{f})
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "invariant-operand-reloaded-per-iteration" {
			out = append(out, fnd)
		}
	}
	return out
}

// invariantFixture is the measured shape: a row update calling a multi-accumulator dot with the
// pivot row as an operand that does not vary with the row.
const invariantFixture = `package p

func dot4(x, y []float64) float64 {
	var s0, s1 float64
	k := 0
	for ; k+2 <= len(x); k += 2 {
		s0 += x[k] * y[k]
		s1 += x[k+1] * y[k+1]
	}
	return s0 + s1
}

func factor(lf []float64, n, j int, ljj float64) {
	lj := lf[j*n : j*n+j]
	for i := j + 1; i < n; i++ {
		li := lf[i*n : i*n+j]
		li[j] = (li[j] - dot4(li, lj)) / ljj
	}
}`

func TestDetectPS3073_InvariantOperandReload(t *testing.T) {
	fs := invariantOperandFindingsIn(t, invariantFixture)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The transform, the reason a plain unroll is not it, and the arm the digest nearly missed.
	if !containsAll(fs[0].msg, "JAM THE LOOP", "THE SHARING IS THE SINGLE PASS",
		"THE JAM MUST REACH EVERY ARM") {
		t.Fatalf("message omits the transform, the single-pass distinction or the second arm:\n%s",
			fs[0].msg)
	}
}

// This fixture is ALSO what pins the rule that a name bound inside the body varies with the loop.
// Its per-row operand reaches the call as the bare identifier "li", which does not spell "i": drop
// that rule and the check finds no varying operand here and reports nothing at all.

// TestDetectPS3073_SilentWhenTheCalleeIsNotAReduction pins what makes the reload expensive. A
// helper that indexes its arguments without accumulating streams nothing worth sharing.
func TestDetectPS3073_SilentWhenTheCalleeIsNotAReduction(t *testing.T) {
	src := replaceOnce(t, invariantFixture, `	var s0, s1 float64
	k := 0
	for ; k+2 <= len(x); k += 2 {
		s0 += x[k] * y[k]
		s1 += x[k+1] * y[k+1]
	}
	return s0 + s1`, `	return x[0] * y[0]`)
	if fs := invariantOperandFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — nothing is being streamed:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3073_SilentWhenEveryOperandVaries pins the half that makes the finding: with both
// operands moving there is no shared memory to hoist into one pass.
func TestDetectPS3073_SilentWhenEveryOperandVaries(t *testing.T) {
	src := replaceOnce(t, invariantFixture, "dot4(li, lj)", "dot4(li, lf[i*n:i*n+j])")
	if fs := invariantOperandFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — no operand is shared across iterations:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3073_SilentWhenNothingVaries pins the other half. A call whose operands are BOTH
// invariant is not reloading anything per iteration — it is loop-invariant outright, which is a
// different and larger finding than jamming.
func TestDetectPS3073_SilentWhenNothingVaries(t *testing.T) {
	src := replaceOnce(t, invariantFixture, `		li := lf[i*n : i*n+j]
		li[j] = (li[j] - dot4(li, lj)) / ljj`, `		lf[i*n+j] = (lf[i*n+j] - dot4(lj, lj)) / ljj`)
	if fs := invariantOperandFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the whole call is loop-invariant:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3073_SilentOnAScalarOperand pins that an operand must be a SLICE. A scalar argument
// that happens not to vary with the loop streams no memory and shares nothing — reading every
// argument alike reported the trailing LENGTH of a dot as the shared operand, which is what this
// fixture would catch again.
func TestDetectPS3073_SilentOnAScalarOperand(t *testing.T) {
	src := replaceOnce(t, invariantFixture, "func dot4(x, y []float64) float64 {",
		"func dot4(x []float64, y float64) float64 {")
	src = replaceOnce(t, src, "s0 += x[k] * y[k]", "s0 += x[k] * y")
	src = replaceOnce(t, src, "s1 += x[k+1] * y[k+1]", "s1 += x[k+1] * y")
	src = replaceOnce(t, src, "dot4(li, lj)", "dot4(li, lj[0])")
	if fs := invariantOperandFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — one slice operand cannot be shared:\n%s",
			len(fs), fs[0].msg)
	}
}
