package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// mapReduceFindingsIn rebuilds the package-level fan-out registry from this source the way the
// real scan does, then keeps only this check's findings.
func mapReduceFindingsIn(t *testing.T, src string) []finding {
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
		if fnd.category == "serial-reduction-blocks-parallel-map" {
			out = append(out, fnd)
		}
	}
	return out
}

const mapReducePrelude = `package p

func parallelRows(n, work int, body func(lo, hi int)) { body(0, n) }

func nearest(x []float64, cent [][]float64) int { return len(x) + len(cent) }
`

// TestDetectPS3044_MapThenFold is the measured shape: a k-means assignment step whose expensive
// half is a pure per-item call and whose cheap half collides on shared accumulators.
func TestDetectPS3044_MapThenFold(t *testing.T) {
	src := mapReducePrelude + `
func step(data [][]float64, cent [][]float64, sums [][]float64, cnt []int, dim int) {
	for _, x := range data {
		b := nearest(x, cent)
		cnt[b]++
		for t := range dim {
			sums[b][t] += x[t]
		}
	}
}`
	fs := mapReduceFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Three things have to survive: that the split is exact, how to rank it, and when it is not
	// worth doing at all.
	if !containsAll(fs[0].msg, "BIT-IDENTICAL", "RANK IT AGAINST WALL CLOCK", "If the map is CHEAP") {
		t.Fatalf("message omits the exactness claim, the ranking rule or the null case:\n%s", fs[0].msg)
	}
}

// TestDetectPS3044_SilentWhenAlreadyFannedOut pins the APPLIED form, and it is not a
// hypothetical shape. After the measured site was fixed, its ENCLOSING loop started reporting:
// moving the inner loop into a fan-out closure took it out of the nesting count while the
// accumulating write inside that closure stayed visible in the body, and the per-round call that
// produces the centroids reads as the expensive per-item map.
func TestDetectPS3044_SilentWhenAlreadyFannedOut(t *testing.T) {
	src := mapReducePrelude + `
func fit(data [][]float64, cent [][]float64, codes []int, rounds, dim int) {
	for m := range rounds {
		cb := kmeans(data, dim)
		parallelRows(len(data), dim, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				b := nearest(data[i], cb)
				codes[i*rounds+m] = b
				for t := range dim {
					data[i][t] -= cb[b][t]
				}
			}
		})
	}
}

func kmeans(data [][]float64, dim int) [][]float64 { return data }`
	if fs := mapReduceFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the loop already hands its work to a fan-out helper:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3044_SilentOnPerItemWrite pins the discriminator. A write indexed BY THE ITEM is not
// a reduction: no two items collide, and the loop can fan out with no restructuring at all.
func TestDetectPS3044_SilentOnPerItemWrite(t *testing.T) {
	src := mapReducePrelude + `
func step(data [][]float64, cent [][]float64, acc []float64, dim int) {
	for i := range data {
		b := nearest(data[i], cent)
		acc[i] += float64(b)
	}
}`
	if fs := mapReduceFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a per-item write is not a reduction:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3044_SilentWithoutAPerItemCall pins that there has to be work worth moving. A fold
// with no expensive computation in front of it gains nothing from the split — the extra array and
// the second pass would be the entire cost.
func TestDetectPS3044_SilentWithoutAPerItemCall(t *testing.T) {
	src := mapReducePrelude + `
func step(data []int, cnt []int) {
	for _, b := range data {
		cnt[b]++
	}
}`
	if fs := mapReduceFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — nothing expensive sits in front of the fold:\n%s", len(fs), fs[0].msg)
	}
}
