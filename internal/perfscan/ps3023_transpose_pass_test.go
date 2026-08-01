package main

import "testing"

func transposePassTestFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "transpose-pass-over-built-matrix" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3023_TransposePass is the measured shape: the autograd logdet VJP, which solved its
// triangular inverse row-major and then transposed it because the contraction needs columns.
func TestDetectPS3023_TransposePass(t *testing.T) {
	src := `package p

func inv(n int) [][]float64 {
	linv := make([][]float64, n)
	linvT := make([][]float64, n)
	for i := 0; i < n; i++ {
		col := make([]float64, n)
		for k := i; k < n; k++ {
			col[k] = linv[k][i]
		}
		linvT[i] = col
	}
	return linvT
}`
	fs := transposePassTestFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The relationship to PS1010 is the reason this check exists at all, and the false-sharing
	// cost is invisible to a profile. Both must reach the reader.
	if !containsAll(fs[0].msg, "PS1010", "contending for a cache line") {
		t.Fatalf("message omits the PS1010 relationship or the false-sharing cost:\n%s", fs[0].msg)
	}
}

// TestDetectPS3023_SilentOnForeignSource pins the LOCALLY-BUILT restriction, which is what makes
// the advice actionable: the remedy is to change the PRODUCER, and a parameter's producer is not in
// reach. The fixture keeps a genuine transpose pass so it discriminates the source, not the shape.
func TestDetectPS3023_SilentOnForeignSource(t *testing.T) {
	src := `package p

func tr(src [][]float64, n int) [][]float64 {
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		col := make([]float64, n)
		for k := 0; k < n; k++ {
			col[k] = src[k][i]
		}
		out[i] = col
	}
	return out
}`
	if fs := transposePassTestFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a parameter's producer cannot be changed here:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3023_SilentOnRowMajorCopy pins the TRANSPOSED read. A straight copy keeps the index
// order, so there is no layout to flip and nothing for the producer to do differently.
func TestDetectPS3023_SilentOnRowMajorCopy(t *testing.T) {
	src := `package p

func cp(n int) [][]float64 {
	src := make([][]float64, n)
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		row := make([]float64, n)
		for k := 0; k < n; k++ {
			row[k] = src[i][k]
		}
		out[i] = row
	}
	return out
}`
	if fs := transposePassTestFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the indexes are not swapped, so this is not a transpose", len(fs))
	}
}

// TestDetectPS3023_SilentOnInterchangeableWalk pins the BOUNDARY WITH PS1010. Here the inner loop
// accumulates into a destination free of the inner variable, so interchange IS the remedy and
// PS1010 owns it. Reporting it here would give the wrong advice — delete a pass that is not a pass.
func TestDetectPS3023_SilentOnInterchangeableWalk(t *testing.T) {
	src := `package p

func sums(n int) []float64 {
	m := make([][]float64, n)
	acc := make([]float64, n)
	for i := 0; i < n; i++ {
		for k := 0; k < n; k++ {
			acc[i] += m[k][i]
		}
	}
	return acc
}`
	if fs := transposePassTestFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — an interchangeable walk is PS1010's, not this check's:\n%s",
			len(fs), fs[0].msg)
	}
}
