package main

import "testing"

// PS4006 must catch the HOISTED-ROW form (row := m[i]; row[j]) and the INDIRECT-FILL form
// (r := make(...); m[i] = r), not only the literal m[i][j] with a direct m[i]=make(...).
func TestDetectRowSliceMatrix_HoistedAndIndirect(t *testing.T) {
	hoisted := `package p
func f(n, cols int) float64 {
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, cols)
	}
	var s float64
	for i := 0; i < n; i++ {
		row := m[i]
		for j := 0; j < cols; j++ {
			s += row[j]
		}
	}
	return s
}`
	if got := countCat(scanSrc(t, hoisted))["row-slice-matrix"]; got != 1 {
		t.Fatalf("hoisted-row form: want 1 row-slice-matrix, got %d", got)
	}
	indirect := `package p
func f(n, cols int) float64 {
	m := make([][]float64, n)
	for i := range m {
		r := make([]float64, cols)
		m[i] = r
	}
	var s float64
	for i := 0; i < n; i++ {
		for j := 0; j < cols; j++ {
			s += m[i][j]
		}
	}
	return s
}`
	if got := countCat(scanSrc(t, indirect))["row-slice-matrix"]; got != 1 {
		t.Fatalf("indirect-fill form: want 1 row-slice-matrix, got %d", got)
	}
}

// Must stay silent when the [][]T is not densely indexed: a borrowed row passed WHOLE
// (never indexed as a vector), or a matrix indexed only outside a nested loop.
func TestDetectRowSliceMatrix_Silent(t *testing.T) {
	borrowed := `package p
func g(row []float64) {}
func f(n, cols int) {
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, cols)
	}
	for i := 0; i < n; i++ {
		row := m[i]
		g(row)
	}
}`
	if got := countCat(scanSrc(t, borrowed))["row-slice-matrix"]; got != 0 {
		t.Fatalf("borrowed-whole-row must be silent, got %d", got)
	}
	shallow := `package p
func f(n, cols int) float64 {
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, cols)
	}
	row := m[0]
	return row[2]
}`
	if got := countCat(scanSrc(t, shallow))["row-slice-matrix"]; got != 0 {
		t.Fatalf("shallow single index must be silent, got %d", got)
	}
}
