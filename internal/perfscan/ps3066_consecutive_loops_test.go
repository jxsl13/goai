package main

import "testing"

func consecutiveLoopFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "consecutive-loops-over-one-buffer" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3066_FourPassesOverOneBuffer is the measured shape: a step that scales a state,
// reduces it, updates it and reduces it again, in four sibling loops.
func TestDetectPS3066_FourPassesOverOneBuffer(t *testing.T) {
	src := `package p

func dot(a, b []float64) float64 { return 0 }

func step(S, kt, qt, ar, sk, o []float64, dv, dk int) {
	for r := 0; r < dv; r++ {
		for c := 0; c < dk; c++ {
			S[r*dk+c] *= ar[c]
		}
	}
	for r := 0; r < dv; r++ {
		sk[r] = dot(S[r*dk:r*dk+dk], kt)
	}
	for r := 0; r < dv; r++ {
		for c := 0; c < dk; c++ {
			S[r*dk+c] += sk[r] * kt[c]
		}
	}
	for r := 0; r < dv; r++ {
		o[r] = dot(S[r*dk:r*dk+dk], qt)
	}
}`
	fs := consecutiveLoopFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Two things the measurement produced: that the payoff tracks the BUFFER size rather than
	// the loop count, and the dependency the merge is only valid without.
	if !containsAll(fs[0].msg, "THE WIN IS THE BUFFER SIZE, NOT THE LOOP COUNT",
		"Check for a cross-index dependency first") {
		t.Fatalf("message omits the size dependence or the safety condition:\n%s", fs[0].msg)
	}
}

// TestDetectPS3066_SilentOnTwoLoops pins the count. Two passes are common and often deliberate
// — a fill and a use — and reporting them would bury the four-pass case this exists for.
func TestDetectPS3066_SilentOnTwoLoops(t *testing.T) {
	src := `package p

func dot(a, b []float64) float64 { return 0 }

func step(S, kt, ar, sk []float64, dv, dk int) {
	for r := 0; r < dv; r++ {
		for c := 0; c < dk; c++ {
			S[r*dk+c] *= ar[c]
		}
	}
	for r := 0; r < dv; r++ {
		sk[r] = dot(S[r*dk:r*dk+dk], kt)
	}
}`
	if fs := consecutiveLoopFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — two passes is not this shape:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3066_SilentWhenNoBufferIsShared pins the condition the finding rests on. Three
// loops over the same bound that touch DIFFERENT buffers do not evict each other's data, and
// merging them buys nothing.
//
// The shared-buffer requirement is enforced in TWO places — the intersection resets the run,
// and the report guard checks what survived — so a mutation has to remove both before this
// fixture reddens. Removing either alone leaves it green, which is cooperation rather than a
// gap: each is sufficient on its own.
func TestDetectPS3066_SilentWhenNoBufferIsShared(t *testing.T) {
	src := `package p

func step(a, b, c []float64, n int) {
	for i := 0; i < n; i++ {
		a[i] = 1
	}
	for i := 0; i < n; i++ {
		b[i] = 2
	}
	for i := 0; i < n; i++ {
		c[i] = 3
	}
}`
	if fs := consecutiveLoopFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — no buffer is shared:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3066_SilentWhenTheBoundsDiffer pins that the loops must cover the SAME range.
// Loops over different bounds cannot be merged element-for-element at all.
func TestDetectPS3066_SilentWhenTheBoundsDiffer(t *testing.T) {
	src := `package p

func step(S []float64, n, m, k int) {
	for i := 0; i < n; i++ {
		S[i] = 1
	}
	for i := 0; i < m; i++ {
		S[i] = 2
	}
	for i := 0; i < k; i++ {
		S[i] = 3
	}
}`
	if fs := consecutiveLoopFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the bounds differ:\n%s", len(fs), fs[0].msg)
	}
}
