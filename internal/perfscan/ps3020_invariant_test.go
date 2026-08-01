package main

import "testing"

func invariantFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "invariant-behind-bounds-check" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3020_InvariantBehindBoundsCheck is the measured shape: the rl Polyak soft update,
// where (1-tau) was rematerialized on every iteration because the two bounds checks split the body
// into blocks SSA will not hoist across.
func TestDetectPS3020_InvariantBehindBoundsCheck(t *testing.T) {
	src := `package p

func soft(to, so []float64, lo, hi int, tau float64) {
	for j := lo; j < hi; j++ {
		to[j] = tau*so[j] + (1-tau)*to[j]
	}
}`
	fs := invariantFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The panic-edge mechanism is the whole insight, and the FMA caveat is what stops a reader
	// claiming bit-identity without re-reading the disassembly. Both must survive into the advice.
	if !containsAll(fs[0].msg, "PANIC EDGE", "FMADDD") {
		t.Fatalf("message omits the mechanism or the FMA caveat:\n%s", fs[0].msg)
	}
}

// TestDetectPS3020_SilentWhenHoisted pins the applied form — the invariant lifted to a local above
// the loop, which is exactly the fix.
func TestDetectPS3020_SilentWhenHoisted(t *testing.T) {
	src := `package p

func soft(to, so []float64, lo, hi int, tau float64) {
	omt := 1 - tau
	for j := lo; j < hi; j++ {
		to[j] = tau*so[j] + omt*to[j]
	}
}`
	if fs := invariantFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a hoisted invariant is the applied form:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3020_SilentOnAddressingArithmetic is the floor for the paired-with-an-indexed-read
// clause. The fixture deliberately CONTAINS an invariant product in a loop that indexes with the
// loop variable — everything the check looks for — but that product only computes an address, and
// addressing arithmetic folds into an addressing mode rather than costing an instruction. Without
// this construct present, dropping the pairing clause would go undetected.
func TestDetectPS3020_SilentOnAddressingArithmetic(t *testing.T) {
	src := `package p

func addr(x, y []float64, n, a, b int) {
	for i := 0; i < n; i++ {
		y[i] = x[a*b+i]
	}
}`
	if fs := invariantFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — addressing arithmetic is not a recomputed value:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3020_SilentOnVaryingOperand pins INVARIANCE. The expression has the same shape and is
// paired with an indexed read, but one operand is written inside the loop, so it is not invariant
// and hoisting it would change the result.
func TestDetectPS3020_SilentOnVaryingOperand(t *testing.T) {
	src := `package p

func varying(to, so []float64, lo, hi int, tau float64) {
	acc := 0.0
	for j := lo; j < hi; j++ {
		acc = acc + 1
		to[j] = tau*so[j] + (1-acc)*to[j]
	}
}`
	if fs := invariantFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — acc is written in the loop:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3020_SilentOnImpureInvariant pins that only literals and plain identifiers count. A
// call may have side effects or return a different value each time; without a type checker there is
// no way to prove it pure, and hoisting it would change behavior rather than just cost.
func TestDetectPS3020_SilentOnImpureInvariant(t *testing.T) {
	src := `package p

func impure(to, so []float64, lo, hi int, tau float64) {
	for j := lo; j < hi; j++ {
		to[j] = tau*so[j] + (1-rate())*to[j]
	}
}`
	if fs := invariantFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a call is not provably invariant:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3020_SilentOnRangeLoop pins the COUNTED loop. A range over the destination already
// bounds the index, so the panic edge this check is about is not there, and the fixture keeps the
// recomputed invariant present so it discriminates the loop form rather than the invariant.
//
// Honest about what this floor proves: the exclusion is STRUCTURAL, not a predicate. A range loop
// is an *ast.RangeStmt, a different node from the *ast.ForStmt this check matches, so it is
// filtered by the type assertion before any clause runs. Blanking the counter test does NOT redden
// this floor — the mutation was run and it stayed green — because a counter-less loop yields an
// empty name that then matches no index. The floor documents the intended boundary; the type
// assertion is what enforces it.
func TestDetectPS3020_SilentOnRangeLoop(t *testing.T) {
	src := `package p

func ranged(to, so []float64, tau float64) {
	for j := range to {
		to[j] = tau*so[j] + (1-tau)*to[j]
	}
}`
	if fs := invariantFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a range loop carries no counter and no un-discharged check:\n%s",
			len(fs), fs[0].msg)
	}
}
