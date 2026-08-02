package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// scatterFindingsIn rebuilds the package-level fan-out registry the way the real scan does, then
// keeps only this check's findings.
func scatterFindingsIn(t *testing.T, src string) []finding {
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
		if fnd.category == "colliding-scatter-with-partitionable-destination" {
			out = append(out, fnd)
		}
	}
	return out
}

const scatterPrelude = `package p

func parallelRows(n, work int, body func(lo, hi int)) { body(0, n) }
`

// TestDetectPS3045_StrengthReducedHistogram is the measured shape: every sample updates one bin of
// every feature, the bin chosen by the data and the feature's window by a running base advanced by
// a constant stride.
//
// The strength-reduced base is the part that matters. A check that only looked for the loop
// variable inside the index would miss every hand-optimized histogram in the tree, which is
// precisely the set worth reporting.
func TestDetectPS3045_StrengthReducedHistogram(t *testing.T) {
	src := scatterPrelude + `
type bin struct {
	sum float64
	cnt int
}

func build(h []bin, idx []int, y []float64, binned []uint16, d, nb int) {
	for _, i := range idx {
		yi := y[i]
		br := binned[i*d : i*d+d : i*d+d]
		base := 0
		for f := 0; f < d; f++ {
			c := base + int(br[f])
			h[c].sum += yi
			h[c].cnt++
			base += nb
		}
	}
}`
	fs := scatterFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The three things a reader needs: which dimension to split, why it stays exact, and the
	// floor that stops the split from costing more than it saves.
	if !containsAll(fs[0].msg, "SPLIT THE INNER DIMENSION, NOT THE ITEMS", "BIT-IDENTICAL", "FLOOR THE WINDOWS PER WORKER") {
		t.Fatalf("message omits the direction, the exactness claim or the floor:\n%s", fs[0].msg)
	}
}

// TestDetectPS3045_SilentWhenBanded pins the APPLIED form. This is not hypothetical: the band
// function is reached through a raw goroutine rather than a registered fan-out helper, so nothing
// about it looks parallel — the signal is that its dimension loop takes its bounds from the
// caller.
func TestDetectPS3045_SilentWhenBanded(t *testing.T) {
	src := scatterPrelude + `
type bin struct {
	sum float64
	cnt int
}

func buildBand(h []bin, idx []int, y []float64, binned []uint16, d, nb, f0, f1 int) {
	for _, i := range idx {
		yi := y[i]
		br := binned[i*d : i*d+d : i*d+d]
		base := f0 * nb
		for f := f0; f < f1; f++ {
			c := base + int(br[f])
			h[c].sum += yi
			h[c].cnt++
			base += nb
		}
	}
}`
	if fs := scatterFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the caller already bands the dimension:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3045_SilentOnDenseUpdate pins the discriminator. When the destination index is made
// only of loop variables there is no collision at all: the item loop can fan out directly, and
// nothing about the inner dimension needs to be involved.
func TestDetectPS3045_SilentOnDenseUpdate(t *testing.T) {
	src := scatterPrelude + `
func accumulate(out []float64, x []float64, n, d int) {
	for i := 0; i < n; i++ {
		base := 0
		for f := 0; f < d; f++ {
			out[base+f] += x[i*d+f]
			base += d
		}
	}
}`
	if fs := scatterFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — no data-dependent offset, so nothing collides:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3045_SilentWhenAlreadyFannedOut pins that a loop already handing its work to the
// package's fan-out helper is not reported again.
func TestDetectPS3045_SilentWhenAlreadyFannedOut(t *testing.T) {
	src := scatterPrelude + `
type bin struct {
	sum float64
	cnt int
}

func build(h []bin, idx []int, y []float64, binned []uint16, d, nb int) {
	parallelRows(d, len(idx), func(lo, hi int) {
		for _, i := range idx {
			yi := y[i]
			br := binned[i*d : i*d+d : i*d+d]
			base := lo * nb
			for f := lo; f < hi; f++ {
				c := base + int(br[f])
				h[c].sum += yi
				h[c].cnt++
				base += nb
			}
		}
	})
}`
	if fs := scatterFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — already split over the dimension:\n%s", len(fs), fs[0].msg)
	}
}
