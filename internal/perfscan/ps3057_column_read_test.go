package main

import "testing"

func columnReadFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "column-read-through-a-jagged-matrix" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3057_GatheredColumn is the measured shape: a loop gathering one feature's values
// for a node's samples, where the row comes from a permutation array and the column is fixed.
func TestDetectPS3057_GatheredColumn(t *testing.T) {
	src := `package p

func scan(x [][]float64, cf []int, vals []float64, start, n, f int) {
	for k := 0; k < n; k++ {
		vals[k] = x[cf[start+k]][f]
	}
}`
	fs := columnReadFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Three things the measurement produced and the argument alone would not: what the fix is,
	// that only a bit-exact gate can check it, and that it is paid for in memory.
	if !containsAll(fs[0].msg, "Mirror the matrix FEATURE-MAJOR once", "Only a bit-exact digest",
		"COSTS n*d elements of memory") {
		t.Fatalf("message omits the fix, the gate or the memory trade:\n%s", fs[0].msg)
	}
}

// TestDetectPS3057_SilentOnARowRead pins the orientation the finding rests on. When the COLUMN
// varies with the loop the read is already contiguous — it walks one row — and there is nothing
// to mirror.
func TestDetectPS3057_SilentOnARowRead(t *testing.T) {
	src := `package p

func rowdot(x [][]float64, idx []int, w []float64, i, d int) float64 {
	var s float64
	for k := 0; k < d; k++ {
		s += x[idx[i]][k] * w[k]
	}
	return s
}`
	if fs := columnReadFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the column varies, so the read walks a row:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3057_SilentOnAStridedRow pins the narrowing that took the tree-wide count from
// 118 to 3. Reading x[i][f] for ascending i is also a column, but the rows are visited in order
// at a fixed stride and a prefetcher can follow that; only a data-dependent row makes every
// read an independent miss.
func TestDetectPS3057_SilentOnAStridedRow(t *testing.T) {
	src := `package p

func strided(x [][]float64, vals []float64, n, f int) {
	for i := 0; i < n; i++ {
		vals[i] = x[i][f]
	}
}`
	if fs := columnReadFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the rows are visited in order:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3057_SilentOnAColumnWrite pins that the transform is mirror-once-read-many. A
// column WRITE has the same locality and no fix: the value would have to land in both copies.
func TestDetectPS3057_SilentOnAColumnWrite(t *testing.T) {
	src := `package p

func scatter(x [][]float64, cf []int, vals []float64, start, n, f int) {
	for k := 0; k < n; k++ {
		x[cf[start+k]][f] = vals[k]
	}
}`
	if fs := columnReadFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a write cannot be served from a mirror:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3057_SilentWhenTheColumnVariesToo pins the condition M2 exposed: the row-read
// fixture above rejects on its ROW index, so nothing there gates the column test. Here both
// indices move with the loop — a permuted diagonal — and a feature-major mirror does not help,
// because each read lands in a different column and stays just as scattered.
func TestDetectPS3057_SilentWhenTheColumnVariesToo(t *testing.T) {
	src := `package p

func diag(x [][]float64, idx []int, n int) float64 {
	var s float64
	for k := 0; k < n; k++ {
		s += x[idx[k]][k]
	}
	return s
}`
	if fs := columnReadFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the column moves too, so a mirror stays scattered:\n%s",
			len(fs), fs[0].msg)
	}
}
