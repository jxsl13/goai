package main

import (
	"strings"
	"testing"
)

func serialReductionFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "serial-reduction-chain" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3010_SingleAccumulator is the measured shape: an f64 dot product whose every add
// waits on the previous one. Splitting it four ways went 537.8 -> 177.3 ns at d=512 on this host.
func TestDetectPS3010_SingleAccumulator(t *testing.T) {
	src := `package p

func dot(a, b []float64) float64 {
	var s float64
	for j, v := range a {
		s += b[j] * v
	}
	return s
}`
	fs := serialReductionFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Reassociation is the entire risk of acting on this finding, so the message has to carry it.
	if !strings.Contains(fs[0].msg, "BIT-IDENTICAL") {
		t.Fatalf("message omits the reassociation warning:\n%s", fs[0].msg)
	}
}

// TestDetectPS3010_FusedAccumulators pins the case the first cut of this check MISSED, which is
// also the one that matters most.
//
// Requiring exactly one accumulator sounds conservative and is not: a fused dot-plus-norm loop is
// the canonical hot reduction. With the single-accumulator rule this check flagged the COLD norm
// loop in nlp MaxContextCosine and stayed silent on the hot pair loop directly below it — the
// second-hottest own-package line in the whole nlp profile. Two chains do give the core two
// independent streams, and the split still measured 2.90x at dim=768.
func TestDetectPS3010_FusedAccumulators(t *testing.T) {
	src := `package p

func cos(cand, ctx []float64) (float64, float64) {
	var dot, nb float64
	for i := range cand {
		dot += cand[i] * ctx[i]
		nb += ctx[i] * ctx[i]
	}
	return dot, nb
}`
	if fs := serialReductionFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — two fused chains are still latency-bound", len(fs))
	}
}

// TestDetectPS3010_SilentOnAppliedForm pins the applied shape, and it is the floor that caught a
// real bug in this check. Four partials are four distinct names each written once, which satisfies
// every other test here — only the stride distinguishes the fix from the defect.
func TestDetectPS3010_SilentOnAppliedForm(t *testing.T) {
	src := `package p

func dot4(a, b []float64) float64 {
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	return s0 + s1 + s2 + s3
}`
	if fs := serialReductionFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the loop already strides by 4, which is the applied form:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3010_SilentWhenAccumulatorIsRead is the CORRECTNESS floor, not a precision one.
//
// A loop that tests its accumulator is PS3008's early-bail shape. Four partials cannot be compared
// against a threshold without being summed first, so the split changes WHEN the loop exits and
// therefore what it returns. This silence is about being wrong, not about being noisy.
func TestDetectPS3010_SilentWhenAccumulatorIsRead(t *testing.T) {
	src := `package p

func within(a, b []float64, thr float64) bool {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
		if s > thr {
			return false
		}
	}
	return true
}`
	if fs := serialReductionFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the accumulator is tested, so splitting it changes the exit", len(fs))
	}
}

// TestDetectPS3010_SilentOnLoopInvariantTerm keeps this check off ground PS4006 already owns. A sum
// that does not depend on the index has a far better fix than reassociation: hoist it out entirely.
func TestDetectPS3010_SilentOnLoopInvariantTerm(t *testing.T) {
	src := `package p

func f(n int, x, y float64) float64 {
	var s float64
	for range n {
		s += x * y
	}
	return s
}`
	if fs := serialReductionFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the term is loop-invariant and belongs to PS4006", len(fs))
	}
}

// TestDetectPS3010_SilentOnManyAccumulators pins the upper bound. A loop already carrying more than
// four independent chains has the instruction-level parallelism this check exists to recommend, and
// more partials would only compete for registers.
func TestDetectPS3010_SilentOnManyAccumulators(t *testing.T) {
	src := `package p

func f(a []float64) (float64, float64, float64, float64, float64) {
	var v, w, x, y, z float64
	for i := range a {
		v += a[i]
		w += a[i] * 2
		x += a[i] * 3
		y += a[i] * 4
		z += a[i] * 5
	}
	return v, w, x, y, z
}`
	if fs := serialReductionFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — five chains already saturate the available parallelism", len(fs))
	}
}

// TestDetectPS3010_SilentOnBranchingBody keeps the check to a PURE reduction. A conditional in the
// body means the adds are not unconditional, so partials would not receive the same terms.
func TestDetectPS3010_SilentOnBranchingBody(t *testing.T) {
	src := `package p

func f(a []float64, lim float64) float64 {
	var s float64
	for i := range a {
		if a[i] > lim {
			continue
		}
		s += a[i] * a[i]
	}
	return s
}`
	if fs := serialReductionFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a branch in the body makes the adds conditional", len(fs))
	}
}

// TestDetectPS3010_SilentWhenAccumulatorIsReadWithoutBranching is the UNMASKED write-only floor.
//
// TestDetectPS3010_SilentWhenAccumulatorIsRead uses the early-bail shape, whose `if` also trips the
// pure-body check — so it passes even with the write-only test removed, and cannot floor it. Here
// the running total is read by a plain assignment, leaving the body branch-free: the only thing
// standing between this loop and a finding is that the accumulator is observed mid-loop, which
// four partials would change.
func TestDetectPS3010_SilentWhenAccumulatorIsReadWithoutBranching(t *testing.T) {
	src := `package p

func f(a []float64) (float64, float64) {
	var s, m float64
	for i := range a {
		s += a[i]
		m = s * 2
	}
	return s, m
}`
	if fs := serialReductionFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the accumulator is read inside the loop", len(fs))
	}
}
