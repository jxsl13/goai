package main

import (
	"go/parser"
	"go/token"
	"testing"
)

func perUnitFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "operand-streamed-once-per-output-unit" {
			out = append(out, fnd)
		}
	}
	return out
}

// unitFixture is the measured shape: a closure run once per output column that derives a
// per-column operand and then walks the WHOLE shared matrix, so the matrix is streamed once per
// column. The counter is declared outside its own for statement, which is what a jam looks like.
const unitFixture = `package p

func matmul(x []float32, out []float32, m, k, n int, weight []byte) {
	scratch := make([]float32, k)
	process := func(sc []float32, ni int) {
		dequant(sc, weight, ni)
		mi := 0
		for ; mi+8 <= m; mi += 8 {
			r0 := x[(mi+0)*k : (mi+0)*k+k]
			var a0 float64
			for ki := 0; ki < k; ki++ {
				a0 += float64(r0[ki]) * float64(sc[ki])
			}
			out[mi*n+ni] = float32(a0)
		}
	}
	for ni := 0; ni < n; ni++ {
		process(scratch, ni)
	}
}

func dequant(dst []float32, src []byte, ni int) {}`

func TestDetectPS3081_PerUnitStream(t *testing.T) {
	fs := perUnitFindingsIn(t, unitFixture)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The transform, the swept factor, and the harness trap that understated the sweep.
	if !containsAll(fs[0].msg, "BLOCK THE UNIT", "THREE, AND SWEPT",
		"NAMED PARAMETERS, NOT A SLICE OF SLICES") {
		t.Fatalf("message omits the transform, the swept factor or the harness trap:\n%s",
			fs[0].msg)
	}
}

// TestDetectPS3081_SilentWhenTheUnitIsBlocked pins the suppression that stops the check
// reporting its own fix: the per-unit path stays as the tail of the blocked loop.
func TestDetectPS3081_SilentWhenTheUnitIsBlocked(t *testing.T) {
	src := replaceOnce(t, unitFixture, `	for ni := 0; ni < n; ni++ {
		process(scratch, ni)
	}`, `	nb := 0
	for ; nb+3 <= n; nb += 3 {
		process(scratch, nb)
	}
	for ni := nb; ni < n; ni++ {
		process(scratch, ni)
	}`)
	if fs := perUnitFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the unit is already blocked:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3081_SilentOnPerUnitScratch pins what makes an operand SHARED. A buffer the
// closure itself allocates belongs to the unit, and cutting it by an inner index says nothing —
// which is how a mixture-model fitter read as a finding.
func TestDetectPS3081_SilentOnPerUnitScratch(t *testing.T) {
	src := replaceOnce(t, unitFixture, "		mi := 0", `		own := make([]float32, m*k)
		mi := 0`)
	src = replaceOnce(t, src, "			r0 := x[(mi+0)*k : (mi+0)*k+k]",
		"			r0 := own[(mi+0)*k : (mi+0)*k+k]")
	if fs := perUnitFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — that buffer belongs to the unit:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3081_SilentWithoutAUnitWrite pins that the closure must PRODUCE the unit's output.
// One that writes nothing indexed by its index is an ordinary callback, not an output unit.
func TestDetectPS3081_SilentWithoutAUnitWrite(t *testing.T) {
	src := replaceOnce(t, unitFixture, "			out[mi*n+ni] = float32(a0)", "			out[mi] = float32(a0)")
	if fs := perUnitFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — nothing is written per unit:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3081_SilentOnAConstantCut pins that the shared operand must be cut BY THE LOOP,
// which is what makes it streamed per unit. A fixed window read once is loop-invariant — a hoist
// if anything, and nothing a unit block would help.
func TestDetectPS3081_SilentOnAConstantCut(t *testing.T) {
	src := replaceOnce(t, unitFixture, "			r0 := x[(mi+0)*k : (mi+0)*k+k]", "			r0 := x[0:k]")
	if fs := perUnitFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — that cut does not move with the loop:\n%s",
			len(fs), fs[0].msg)
	}
}
