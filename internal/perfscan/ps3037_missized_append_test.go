package main

import "testing"

func misSizedAppendFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "mis-sized-append-buffer" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3037_MisSizedAppendBuffer is the measured shape: a beam search whose candidate
// buffer is hinted at 8 per live beam while the expansion appends one candidate per beam per
// vocabulary entry. That one append line was 2.45 GB of a 2.90 GB benchmark.
func TestDetectPS3037_MisSizedAppendBuffer(t *testing.T) {
	src := `package p

func BeamSearch(live []node, next func([]int) []float64) []Beam {
	var done []Beam
	for len(live) > 0 {
		cand := make([]cnd, 0, len(live)*8)
		for p, h := range live {
			ls := logSoftmaxRow(next(h.toks))
			for tok, l := range ls {
				cand = append(cand, cnd{p, tok, h.score + l})
			}
		}
		slices.SortFunc(cand, byScore)
		live = advance(cand)
	}
	return done
}`
	fs := misSizedAppendFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The mechanism, the metric and the reset obligation all have to survive into the advice.
	if !containsAll(fs[0].msg, "doubles its way up", "B/op FIRST", "truncated before it is refilled") {
		t.Fatalf("message omits the mechanism, the metric or the reset:\n%s", fs[0].msg)
	}
}

// TestDetectPS3037_SilentOnAppliedForm pins the applied form: one buffer above the loop, truncated
// each pass.
func TestDetectPS3037_SilentOnAppliedForm(t *testing.T) {
	src := `package p

func BeamSearch(live []node, next func([]int) []float64) []Beam {
	var done []Beam
	var cand []cnd
	for len(live) > 0 {
		cand = cand[:0]
		for p, h := range live {
			ls := logSoftmaxRow(next(h.toks))
			for tok, l := range ls {
				cand = append(cand, cnd{p, tok, h.score + l})
			}
		}
		slices.SortFunc(cand, byScore)
		live = advance(cand)
	}
	return done
}`
	if fs := misSizedAppendFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the buffer is hoisted and reset:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3037_SilentWhenAppendedAtOneLevel pins the NESTING, which is the whole argument. A
// buffer appended to from the same loop that made it grows to the trip count of that loop, and a
// hint proportional to it is fine. The fixture keeps the make, the hint and the append, and
// removes only the inner loop.
func TestDetectPS3037_SilentWhenAppendedAtOneLevel(t *testing.T) {
	src := `package p

func collect(rows []row) []int {
	var all []int
	for _, r := range rows {
		buf := make([]int, 0, len(r.items))
		for _, it := range r.items {
			buf = append(buf, it.id)
		}
		all = use(all, buf)
	}
	return all
}`
	if fs := misSizedAppendFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the appends are at the hint's own level:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3037_SilentWithoutACapacityHint pins that a capacity must have been STATED. A plain
// two-argument make says nothing about an expected size, and the finding's whole claim is that the
// stated number is wrong by a factor of the inner trip count.
func TestDetectPS3037_SilentWithoutACapacityHint(t *testing.T) {
	src := `package p

func expand(live []node) []cnd {
	var out []cnd
	for len(live) > 0 {
		cand := make([]cnd, 0)
		for _, h := range live {
			for _, l := range h.ls {
				cand = append(cand, cnd{l})
			}
		}
		out = advance(cand)
	}
	return out
}`
	if fs := misSizedAppendFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — no capacity was stated:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3037_SilentWhenTheBufferIsKept pins the retention condition. A buffer stored where it
// outlives the pass cannot be hoisted and reset — the next pass would overwrite what the caller
// kept. Note this is deliberately NARROWER than the escape test PS3033 and PS3035 use: passing the
// buffer to a sort keeps nothing, and treating every call argument as retention made this check
// blind to its own motivating site.
func TestDetectPS3037_SilentWhenTheBufferIsKept(t *testing.T) {
	src := `package p

func expand(live []node, sink *[][]cnd) {
	for len(live) > 0 {
		cand := make([]cnd, 0, len(live)*8)
		for _, h := range live {
			for _, l := range h.ls {
				cand = append(cand, cnd{l})
			}
		}
		*sink = append(*sink, cand)
		live = advance(cand)
	}
}`
	if fs := misSizedAppendFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the buffer is kept past the pass:\n%s", len(fs), fs[0].msg)
	}
}
