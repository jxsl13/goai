package main

import "testing"

func topkFanoutFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "full-fanout-under-topk-gate" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3026_FullFanoutUnderTopKGate is the measured shape: a mixture-of-experts forward that
// chooses two experts per token and then evaluates all of them, relying on the unselected ones
// carrying a zero weight. Skipping them went -23.7% on the decode step.
func TestDetectPS3026_FullFanoutUnderTopKGate(t *testing.T) {
	src := `package p

func moe(ctx *C, x *T, e, topK int) *T {
	for t := range seq {
		for _, i := range topKIndices(scores, t, e, topK) {
			ws[t*e+i] = scores.AtF64(t, i)
		}
	}
	var y *T
	for i := range e {
		out := m.Experts[i].Forward(ctx, x)
		y = add(y, mul(out, weight.Slice(1, i, i+1)))
	}
	return y
}`
	fs := topkFanoutFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The bit-identity argument is what makes the edit safe to make, and the benchmark protocol is
	// what stops the next person mismeasuring it the way it was first mismeasured. Both must reach
	// the reader.
	if !containsAll(fs[0].msg, "exactly zero", "BOTH orders") {
		t.Fatalf("message omits the exactness argument or the interleaving protocol:\n%s", fs[0].msg)
	}
}

// TestDetectPS3026_SilentWhenSkipping pins the applied form — the selection recorded and the
// unselected branches skipped, which is exactly the fix.
func TestDetectPS3026_SilentWhenSkipping(t *testing.T) {
	src := `package p

func moe(ctx *C, x *T, e, topK int) *T {
	used := make([]bool, e)
	for t := range seq {
		for _, i := range topKIndices(scores, t, e, topK) {
			ws[t*e+i] = scores.AtF64(t, i)
			used[i] = true
		}
	}
	var y *T
	for i := range e {
		if !used[i] {
			continue
		}
		out := m.Experts[i].Forward(ctx, x)
		y = add(y, mul(out, weight.Slice(1, i, i+1)))
	}
	return y
}`
	if fs := topkFanoutFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a skipping loop is the applied form:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3026_SilentWithoutGate pins the GATE. A fan-out over every branch is not a defect on
// its own — a dense ensemble genuinely wants all of them. What makes the work removable is that
// something already chose a subset. The fixture is the positive with the selection deleted.
func TestDetectPS3026_SilentWithoutGate(t *testing.T) {
	src := `package p

func ensemble(ctx *C, x *T, e int) *T {
	var y *T
	for i := range e {
		out := m.Experts[i].Forward(ctx, x)
		y = add(y, out)
	}
	return y
}`
	if fs := topkFanoutFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — nothing selected a subset, so nothing is skippable", len(fs))
	}
}

// TestDetectPS3026_SilentOnLoopBeforeGate pins the ORDER. A loop that runs before the selection
// exists cannot skip on it, so reporting it would be advice that cannot be followed. The fixture
// keeps a gate and a full fan-out, and only their positions differ.
func TestDetectPS3026_SilentOnLoopBeforeGate(t *testing.T) {
	src := `package p

func warm(ctx *C, x *T, e, topK int) *T {
	var y *T
	for i := range e {
		out := m.Experts[i].Forward(ctx, x)
		y = add(y, out)
	}
	for t := range seq {
		for _, i := range topKIndices(scores, t, e, topK) {
			ws[t*e+i] = scores.AtF64(t, i)
		}
	}
	return y
}`
	if fs := topkFanoutFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the fan-out precedes the selection:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3026_SilentOnUnindexedLoop pins that the loop must reach its work THROUGH the index.
// The loop here HAS a variable and passes it along, so a fixture with a bare `for range` would not
// discriminate this clause — the empty-name test suppresses that one first. Passing an index to a
// function is not the same as selecting a branch with it: nothing says the callee maps it to one,
// and skipping iterations could change what the function computes.
func TestDetectPS3026_SilentOnUnindexedLoop(t *testing.T) {
	src := `package p

func accum(ctx *C, x *T, e, topK int) *T {
	for t := range seq {
		for _, i := range topKIndices(scores, t, e, topK) {
			ws[t*e+i] = scores.AtF64(t, i)
		}
	}
	var y *T
	for i := range e {
		y = add(y, step(ctx, x, i))
	}
	return y
}`
	if fs := topkFanoutFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the loop indexes nothing, so it is not a branch fan-out:\n%s",
			len(fs), fs[0].msg)
	}
}
