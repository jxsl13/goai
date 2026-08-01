package main

import (
	"strings"
	"testing"
)

func indirectGatherFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "indirect-column-gather" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3009_GathersColumnThroughPermutation is the measured shape: classic's GBM split scan
// hoisting a feature column through the node's sorted index permutation.
func TestDetectPS3009_GathersColumnThroughPermutation(t *testing.T) {
	src := `package p

func hoist(x [][]float64, idx []int, vals []float64, f, n int) {
	for k := 0; k < n; k++ {
		vals[k] = x[idx[k]][f]
	}
}`
	fs := indirectGatherFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The remedy costs memory, and a reader who acts without seeing that has been misled.
	if !strings.Contains(fs[0].msg, "MEMORY") {
		t.Fatalf("message omits the memory tradeoff:\n%s", fs[0].msg)
	}
}

// TestDetectPS3009_SilentOnDirectRowIndex keeps PS3009 off PS1010's territory. With a direct row
// index there IS a nest to interchange, which is a different remedy and already reported.
func TestDetectPS3009_SilentOnDirectRowIndex(t *testing.T) {
	src := `package p

func walk(x [][]float64, vals []float64, f, n int) {
	for k := 0; k < n; k++ {
		vals[k] = x[k][f]
	}
}`
	if fs := indirectGatherFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a direct row index is PS1010's interchangeable nest, not a "+
			"gather:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3009_SilentWhenColumnVaries pins the column-invariance requirement. If the second
// index advances with the loop the access walks a ROW, which is contiguous and exactly what the
// feature-major copy would destroy.
func TestDetectPS3009_SilentWhenColumnVaries(t *testing.T) {
	src := `package p

func rowwalk(x [][]float64, idx []int, vals []float64, r, n int) {
	for k := 0; k < n; k++ {
		vals[k] = x[idx[r]][k]
	}
}`
	if fs := indirectGatherFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the varying index is the COLUMN, so this walks a row "+
			"contiguously", len(fs))
	}
}

// TestDetectPS3009_SilentOutsideALoop pins that a single gather is not the defect: the cost of this
// shape is paying a cache line per element across a whole column, which needs a loop.
func TestDetectPS3009_SilentOutsideALoop(t *testing.T) {
	src := `package p

func one(x [][]float64, idx []int, k, f int) float64 {
	return x[idx[k]][f]
}`
	if fs := indirectGatherFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — one gather amortizes nothing", len(fs))
	}
}

// TestDetectPS3009_RangeLoopCounts guards the other loop spelling: the detector must not recognize
// only three-clause for statements.
func TestDetectPS3009_RangeLoopCounts(t *testing.T) {
	src := `package p

func hoist(x [][]float64, idx []int, vals []float64, f int) {
	for k := range idx {
		vals[k] = x[idx[k]][f]
	}
}`
	if fs := indirectGatherFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — a range loop is the same defect", len(fs))
	}
}
