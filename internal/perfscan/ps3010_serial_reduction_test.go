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

// TestPS3010_IntegerAccumulatorDropsTheContractWarning pins the branch that matters most for
// ACTING on this check.
//
// Integer addition is exactly associative, so splitting an integer accumulator is bit-identical BY
// CONSTRUCTION — it passes even a Float64bits golden-checksum test unchanged. Emitting the generic
// reassociation warning there would send a reader off to clear a contract that cannot be violated.
func TestPS3010_IntegerAccumulatorDropsTheContractWarning(t *testing.T) {
	for _, decl := range []string{"var s int", "s := 0"} {
		src := `package p

func f(a []int) int {
	` + decl + `
	for i := range a {
		s += a[i]
	}
	return s
}`
		fs := serialReductionFindings(t, src)
		if len(fs) != 1 {
			t.Fatalf("decl %q: %d findings, want 1", decl, len(fs))
		}
		if !strings.Contains(fs[0].msg, "INTEGER ACCUMULATOR") {
			t.Fatalf("decl %q: integer accumulator did not get the exact-associativity message:\n%s",
				decl, fs[0].msg)
		}
		if strings.Contains(fs[0].msg, "DO NOT apply it where the exact value is pinned") {
			t.Fatalf("decl %q: integer accumulator still carries the float contract warning:\n%s",
				decl, fs[0].msg)
		}
	}
}

// TestPS3010_FloatAccumulatorKeepsTheContractWarning is the other side of the same branch, and it
// also pins the f32 number — f32 gains MORE than f64 (3.65x vs 2.90x at dim=768), which is the
// opposite of what a reader assuming "smaller type, smaller win" would guess.
func TestPS3010_FloatAccumulatorKeepsTheContractWarning(t *testing.T) {
	for _, decl := range []string{"var s float64", "s := 0.0"} {
		src := `package p

func f(a []float64) float64 {
	` + decl + `
	for i := range a {
		s += a[i]
	}
	return s
}`
		fs := serialReductionFindings(t, src)
		if len(fs) != 1 {
			t.Fatalf("decl %q: %d findings, want 1", decl, len(fs))
		}
		if !strings.Contains(fs[0].msg, "DO NOT apply it where the exact") {
			t.Fatalf("decl %q: float accumulator lost the reassociation warning:\n%s", decl, fs[0].msg)
		}
		if strings.Contains(fs[0].msg, "INTEGER ACCUMULATOR") {
			t.Fatalf("decl %q: float accumulator claimed exact associativity:\n%s", decl, fs[0].msg)
		}
	}
}

// TestPS3010_UnknownAccumulatorTypeFallsBackToTheStrictAdvice keeps the inference FAIL-SAFE. With
// no type checker here the kind is often unknowable — a generic parameter, a named type, a value
// from a call. The fallback has to be the strict float warning, because telling a reader that a
// split is unconditionally safe when it may be a float is the one error with a wrong answer at the
// end of it.
func TestPS3010_UnknownAccumulatorTypeFallsBackToTheStrictAdvice(t *testing.T) {
	src := `package p

func f[T Num](a []T) T {
	var s T
	for i := range a {
		s += a[i]
	}
	return s
}`
	fs := serialReductionFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	if strings.Contains(fs[0].msg, "INTEGER ACCUMULATOR") {
		t.Fatalf("an unknown accumulator type must not be claimed exactly associative:\n%s", fs[0].msg)
	}
	if !strings.Contains(fs[0].msg, "NOT BIT-IDENTICAL") {
		t.Fatalf("an unknown accumulator type must keep the strict warning:\n%s", fs[0].msg)
	}
}

// TestPS3010_UndeclaredAccumulatorFallsBackToTheStrictAdvice is the UNMASKED fail-safe floor.
//
// TestPS3010_UnknownAccumulatorTypeFallsBackToTheStrictAdvice cannot floor the default on its own:
// its `var s T` still runs the classifier, which returns "" and overwrites whatever the default
// was. Here the accumulator is a PARAMETER, so no declaration is found in the body at all and the
// initial value is what survives. It must be the strict one — claiming exact associativity for a
// float is the error with a wrong answer at the end of it.
//
// Reading parameter types would be a genuine recall improvement and is deliberately not done here;
// the fallback is safe, so the gap costs advice quality rather than correctness.
func TestPS3010_UndeclaredAccumulatorFallsBackToTheStrictAdvice(t *testing.T) {
	src := `package p

func f(a []float64, s float64) float64 {
	for i := range a {
		s += a[i]
	}
	return s
}`
	fs := serialReductionFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	if strings.Contains(fs[0].msg, "INTEGER ACCUMULATOR") {
		t.Fatalf("an undeclared accumulator must not be claimed exactly associative:\n%s", fs[0].msg)
	}
}
