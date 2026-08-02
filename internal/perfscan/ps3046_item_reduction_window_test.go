package main

import "testing"

func windowFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "item-reduction-into-partitioned-windows" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3046_WindowedGramAccumulation is the measured shape: every sample contributes to
// every (class pair, feature) window of one shared Gram buffer, so the sample loop is a reduction
// and cannot be split, while the pair and feature loops carve the buffer into disjoint pieces.
//
// The offset is STRENGTH-REDUCED — aoff is computed from a rather than written as a*mAug — which
// is how the site was actually written. A syntactic test for the item variable inside the slice
// bounds calls that window item-independent and reports the wrong loop.
func TestDetectPS3046_WindowedGramAccumulation(t *testing.T) {
	src := `package p

func hessian(grams []float64, xa [][]float64, wpair []float64, n, mAug, numPairs, mm int) {
	for i := range n {
		row := xa[i]
		for a := range mAug {
			ra := row[a]
			r := row[a:mAug]
			aoff := a*mAug + a
			aend := a*mAug + mAug
			for q := 0; q < numPairs; q++ {
				wa := wpair[q] * ra
				g := grams[q*mm+aoff : q*mm+aend]
				for j := range g {
					g[j] += wa * r[j]
				}
			}
		}
	}
}`
	fs := windowFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The message has to carry the exactness claim, the balance rule for a triangular inner
	// range, and the gate warning — all three came out of the measurement, not the design.
	if !containsAll(fs[0].msg, "BIT-IDENTICAL", "CUMULATIVE WORK", "CHECK THE GATE AGAINST THE REAL SHAPE") {
		t.Fatalf("message omits the exactness claim, the balance rule or the gate warning:\n%s", fs[0].msg)
	}
}

// TestDetectPS3046_SilentWhenWindowMovesWithTheItem pins the load-bearing condition. A window cut
// at an offset the ITEM chooses is that item's own output; nothing collides, and the item loop can
// fan out directly with no restructuring.
func TestDetectPS3046_SilentWhenWindowMovesWithTheItem(t *testing.T) {
	src := `package p

func rows(out []float64, xa [][]float64, n, mAug, numPairs, mm int) {
	for i := range n {
		row := xa[i]
		ioff := i * mm
		for a := range mAug {
			r := row[a:mAug]
			for q := 0; q < numPairs; q++ {
				g := out[ioff+q*mAug : ioff+q*mAug+mAug-a]
				for j := range g {
					g[j] += r[j]
				}
			}
		}
	}
}`
	if fs := windowFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the window is the item's own output:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3046_SilentWhenBanded pins the APPLIED form: a band function whose window-choosing
// loop starts at a caller-supplied offset. Nothing else about it looks parallel, because the band
// is reached through a raw goroutine rather than a registered fan-out helper.
func TestDetectPS3046_SilentWhenBanded(t *testing.T) {
	src := `package p

func hessianBand(grams []float64, xa [][]float64, wpair []float64, n, mAug, numPairs, mm, a0, a1 int) {
	for i := range n {
		row := xa[i]
		for a := a0; a < a1; a++ {
			ra := row[a]
			r := row[a:mAug]
			aoff := a*mAug + a
			aend := a*mAug + mAug
			for q := 0; q < numPairs; q++ {
				wa := wpair[q] * ra
				g := grams[q*mm+aoff : q*mm+aend]
				for j := range g {
					g[j] += wa * r[j]
				}
			}
		}
	}
}`
	if fs := windowFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the caller already bands the window dimension:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3046_SilentOnShallowNest pins that there has to be enough work under the item loop
// to be worth splitting. One inner loop is a single windowed update per item; the transform pays
// for a fork per call and would not earn it back.
func TestDetectPS3046_SilentOnShallowNest(t *testing.T) {
	src := `package p

func accumulate(acc []float64, xa [][]float64, n, mAug int) {
	for i := range n {
		row := xa[i]
		g := acc[0:mAug]
		for j := range g {
			g[j] += row[j]
		}
	}
}`
	if fs := windowFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — one inner loop is not worth a fork:\n%s", len(fs), fs[0].msg)
	}
}
