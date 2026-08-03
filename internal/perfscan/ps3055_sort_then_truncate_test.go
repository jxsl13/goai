package main

import "testing"

func sortTruncFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "sort-then-truncate" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3055_SortThenCut is the measured shape: a candidate list sorted in full and then
// reduced to a small prefix, with the cut guarded by a length test.
func TestDetectPS3055_SortThenCut(t *testing.T) {
	src := `package p

import "slices"

type cand struct {
	score float64
	tok   int
}

func pick(cands []cand, k int) []cand {
	slices.SortFunc(cands, func(a, b cand) int {
		if a.score > b.score {
			return -1
		}
		return 1
	})
	if len(cands) > k {
		cands = cands[:k]
	}
	return cands
}`
	fs := sortTruncFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Three conditions the measurement produced: exactness depends on the comparator, the ORDER
	// has to be gated and not just the set, and the heap must not read the array it overwrites.
	if !containsAll(fs[0].msg, "STRICT TOTAL ORDER", "GATE THE ORDER, NOT JUST THE SET",
		"BUILD THE HEAP FROM A COPY") {
		t.Fatalf("message omits the exactness condition, the ordering gate or the aliasing"+
			" warning:\n%s", fs[0].msg)
	}
}

// TestDetectPS3055_SilentWithoutATruncation pins the condition the finding rests on. A sort whose
// whole result is used is not wasted work — every element's position was asked for.
func TestDetectPS3055_SilentWithoutATruncation(t *testing.T) {
	src := `package p

import "slices"

type cand struct {
	score float64
	tok   int
}

func order(cands []cand) []cand {
	slices.SortFunc(cands, func(a, b cand) int {
		if a.score > b.score {
			return -1
		}
		return 1
	})
	return cands
}`
	if fs := sortTruncFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the whole sorted result is used:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3055_SilentWhenCutToItsOwnLength pins that a reslice to len(x) is not a truncation.
// It appears when code normalizes a capacity and discards nothing.
func TestDetectPS3055_SilentWhenCutToItsOwnLength(t *testing.T) {
	src := `package p

import "slices"

type cand struct {
	score float64
	tok   int
}

func norm(cands []cand) []cand {
	slices.SortFunc(cands, func(a, b cand) int {
		if a.score > b.score {
			return -1
		}
		return 1
	})
	cands = cands[:len(cands)]
	return cands
}`
	if fs := sortTruncFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — nothing is discarded:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3055_ReportsEveryBlock pins that a function which sorts and cuts in a nested loop AND
// again at the end yields both. Reporting only the first hid the per-step selection inside the
// loop, which is the one that costs, behind the cheap one at the end.
func TestDetectPS3055_ReportsEveryBlock(t *testing.T) {
	src := `package p

import "slices"

type cand struct {
	score float64
	tok   int
}

func search(steps int, cands []cand, k int, done []cand) []cand {
	for range steps {
		slices.SortFunc(cands, func(a, b cand) int {
			if a.score > b.score {
				return -1
			}
			return 1
		})
		if len(cands) > k {
			cands = cands[:k]
		}
	}
	slices.SortFunc(done, func(a, b cand) int {
		if a.score > b.score {
			return -1
		}
		return 1
	})
	if len(done) > k {
		done = done[:k]
	}
	return done
}`
	if fs := sortTruncFindingsIn(t, src); len(fs) != 2 {
		t.Fatalf("%d findings, want 2 — the loop's selection and the final one", len(fs))
	}
}
