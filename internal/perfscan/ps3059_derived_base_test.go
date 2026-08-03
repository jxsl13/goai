package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// derivedBaseFindingsIn primes fanoutReg from the fixture itself; reaching for plain scanSrc
// would leave the registry holding whatever an earlier test file put there.
func derivedBaseFindingsIn(t *testing.T, src string) []finding {
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
		if fnd.category == "serial-nest-writing-through-a-derived-base" {
			out = append(out, fnd)
		}
	}
	return out
}

// TestDetectPS3059_WriteThroughADerivedBase is the measured shape: the KAN fused spline, whose
// row offset is hoisted into a local so no write names the batch variable directly.
func TestDetectPS3059_WriteThroughADerivedBase(t *testing.T) {
	src := `package p

func parallelRows(n, work int, body func(lo, hi int)) { body(0, n) }

func fused(ys, bs, cs []float64, B, in, out, nbasis int) {
	for b := 0; b < B; b++ {
		obase := b * out
		for i := 0; i < in; i++ {
			for j := 0; j < out; j++ {
				var acc float64
				for c := 0; c < nbasis; c++ {
					acc += bs[(b*in+i)*nbasis+c] * cs[(i*out+j)*nbasis+c]
				}
				ys[obase+j] += acc
			}
		}
	}
}`
	fs := derivedBaseFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The message has to carry why this check exists beside PS3034, and which gate earns its
	// keep on which kind of destination — both came out of the two finds.
	if !containsAll(fs[0].msg, "PS3034'S BLIND SPOT", "WIDENING PS3034 WAS TRIED AND REVERTED",
		"GATE WITH BOTH A BIT-EXACT DIGEST AND -race") {
		t.Fatalf("message omits why this is separate from PS3034 or how to gate it:\n%s", fs[0].msg)
	}
}

// TestDetectPS3059_SilentWhenEveryWriteNamesTheOuterVar pins the boundary with PS3034. A nest
// whose writes name the outermost variable directly is that check's shape, and reporting it
// here would only duplicate it.
func TestDetectPS3059_SilentWhenEveryWriteNamesTheOuterVar(t *testing.T) {
	src := `package p

func parallelRows(n, work int, body func(lo, hi int)) { body(0, n) }

func fill(dst, src []float64, B, in, out int) {
	for b := 0; b < B; b++ {
		for i := 0; i < in; i++ {
			for j := 0; j < out; j++ {
				dst[b*out+j] += src[b*in+i]
			}
		}
	}
}`
	if fs := derivedBaseFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — PS3034 owns this shape:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3059_SilentWhenAWriteEscapesTheOuterIteration pins the disjointness the finding
// rests on. If any write lands somewhere the outer variable does not own, the iterations are
// not independent and banding them is a race, not an optimization.
func TestDetectPS3059_SilentWhenAWriteEscapesTheOuterIteration(t *testing.T) {
	src := `package p

func parallelRows(n, work int, body func(lo, hi int)) { body(0, n) }

func fill(dst, tot, src []float64, B, in, out int) {
	for b := 0; b < B; b++ {
		obase := b * out
		for i := 0; i < in; i++ {
			for j := 0; j < out; j++ {
				dst[obase+j] += src[b*in+i]
				tot[j] += src[b*in+i]
			}
		}
	}
}`
	if fs := derivedBaseFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — tot[j] is shared across outer iterations:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3059_SilentOnABandBody pins that a function already handed its band is not this
// shape. Applying this check produces exactly such a function: the nest is split into a helper
// so the gated arm and the below-gate arm share one copy of it.
func TestDetectPS3059_SilentOnABandBody(t *testing.T) {
	src := `package p

func parallelRows(n, work int, body func(lo, hi int)) { body(0, n) }

func fusedBand(ys, src []float64, blo, bhi, in, out int) {
	for b := blo; b < bhi; b++ {
		obase := b * out
		for i := 0; i < in; i++ {
			for j := 0; j < out; j++ {
				ys[obase+j] += src[b*in+i]
			}
		}
	}
}`
	if fs := derivedBaseFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the band is a parameter, so the fan-out is the"+
			" caller's:\n%s", len(fs), fs[0].msg)
	}
}
