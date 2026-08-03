package main

import "testing"

func jaggedFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "jagged-matrix-allocated-row-by-row" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3064_RowByRow is the measured shape: an outer make([][]T, r) followed by one
// make() per row.
func TestDetectPS3064_RowByRow(t *testing.T) {
	src := `package p

func build(r, c int) [][]float64 {
	d := make([][]float64, r)
	for i := range r {
		d[i] = make([]float64, c)
	}
	return d
}`
	fs := jaggedFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Three things the measurement produced: that the fix costs no call-site changes, that
	// the clock is the smaller and less reliable half, and the append hazard the row window
	// introduces.
	if !containsAll(fs[0].msg, "NO CALL SITE CHANGES",
		"THE CLOCK IS THE SMALLER HALF AND IS SHAPE-DEPENDENT", "CAP THE ROW WINDOW") {
		t.Fatalf("message omits the no-churn point, the honest expectation or the append"+
			" hazard:\n%s", fs[0].msg)
	}
}

// TestDetectPS3064_SilentWhenBackedByOneBlock pins the applied form: the rows are windows on a
// single allocation, so the loop assigns a slice expression rather than calling make.
func TestDetectPS3064_SilentWhenBackedByOneBlock(t *testing.T) {
	src := `package p

func build(r, c int) [][]float64 {
	base := make([]float64, r*c)
	d := make([][]float64, r)
	for i := range r {
		d[i] = base[i*c : (i+1)*c : (i+1)*c]
	}
	return d
}`
	if fs := jaggedFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the rows share one block:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3064_SilentOnAOneDimensionalSlice pins that the shape is a MATRIX. The element
// type here is not a slice, so there are no rows to pack into a block — and the loop still
// calls make, so the ELEMENT-TYPE test is what decides it. An earlier version filled the slice
// with new() instead, which made it silent for the wrong reason and left the mutation green.
func TestDetectPS3064_SilentOnAOneDimensionalSlice(t *testing.T) {
	src := `package p

func build(r, c int) []chan int {
	d := make([]chan int, r)
	for i := range r {
		d[i] = make(chan int, c)
	}
	return d
}`
	if fs := jaggedFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — not a [][]T:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3064_SilentWhenTheRowIndexIsNotTheLoopVariable pins that the loop must be filling
// the matrix ROW BY ROW. A single row replaced at a fixed index is not this shape, and one
// block would not serve it. The matrix IS allocated here, so the row-index test is what decides
// it — an earlier version took the matrix as a parameter, which made it silent for the wrong
// reason and left the mutation green.
func TestDetectPS3064_SilentWhenTheRowIndexIsNotTheLoopVariable(t *testing.T) {
	src := `package p

func build(r, c, slot int) [][]float64 {
	d := make([][]float64, r)
	for i := range r {
		if i == 0 {
			d[slot] = make([]float64, c)
		}
	}
	return d
}`
	if fs := jaggedFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the row index is fixed:\n%s", len(fs), fs[0].msg)
	}
}
