package main

import (
	"strings"
	"testing"
)

// batchSingleEltFindings returns the PS1003 findings for src.
func batchSingleEltFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "batch-single-elt" {
			out = append(out, f)
		}
	}
	return out
}

// TestPS1003ReportsEveryCallInOneLoop pins the per-call-site positioning.
//
// PS1003 used to position its finding at the enclosing LOOP. dedup collapses findings that share a
// (position, category), so a loop holding two different batch-1 calls produced ONE finding naming
// only the first — the second was not deduplicated, it was invisible. rl.rlRollout was exactly that
// shape, an actor forward whose result the loop reads (not hoistable) beside a critic forward whose
// result it does not (hoistable, and worth 1.59x when it was moved), and PS6015's doc comment
// recorded the gap without closing it.
//
// This is the regression guard: two distinct callees in one loop must yield two findings, each
// naming its own callee.
func TestPS1003ReportsEveryCallInOneLoop(t *testing.T) {
	src := `package p

func forward(net int, states [][]float64) float64 { return 0 }
func evaluate(net int, states [][]float64) float64 { return 0 }

func rollout(actor, critic int, obs []float64, n int) {
	for range n {
		a := forward(actor, [][]float64{obs})
		c := evaluate(critic, [][]float64{obs})
		_, _ = a, c
	}
}`
	fs := batchSingleEltFindings(t, src)
	if len(fs) != 2 {
		t.Fatalf("%d findings, want 2 — one per call site", len(fs))
	}
	var named []string
	for _, f := range fs {
		named = append(named, f.msg)
	}
	joined := strings.Join(named, "\n")
	for _, want := range []string{`"forward"`, `"evaluate"`} {
		if !strings.Contains(joined, want) {
			t.Fatalf("no finding names %s:\n%s", want, joined)
		}
	}
	// Distinct positions are what keep dedup from collapsing them, so assert that directly rather
	// than relying on the count alone.
	if fs[0].pos == fs[1].pos {
		t.Fatalf("both findings share position %s — dedup will collapse them again", fs[0].pos)
	}
	// And the position must be the CALL, not the `for` statement one line above it.
	if fs[0].pos.Line == fs[1].pos.Line {
		t.Fatalf("both findings on line %d; expected the two distinct call lines", fs[0].pos.Line)
	}
}

// TestPS1003StillOneFindingPerCall guards the other direction: a single call carrying TWO
// single-element batch arguments is ONE candidate, not two.
//
// What enforces that is worth stating precisely, because it changed. The detector breaks after the
// first wrapped argument, but since the finding moved to the call position, scanFile's dedup — which
// collapses findings sharing a (position, category) — would collapse the duplicates anyway. Verified:
// removing the break leaves this test and the whole suite green, so it is now belt-and-braces rather
// than the guard. This test is therefore NOT a floor for the break, and claiming otherwise would make
// it the kind of guard that only proves the code equals itself.
//
// It does still guard something real: positioning findings per ARGUMENT instead of per call would
// give them distinct positions, survive dedup, and redden this.
func TestPS1003StillOneFindingPerCall(t *testing.T) {
	src := `package p

func combine(a, b [][]float64) float64 { return 0 }

func run(x, y []float64, n int) {
	for range n {
		_ = combine([][]float64{x}, [][]float64{y})
	}
}`
	if fs := batchSingleEltFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — one call is one candidate however many wrapped args it takes", len(fs))
	}
}
