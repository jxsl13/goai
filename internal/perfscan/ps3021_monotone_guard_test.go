package main

import "testing"

func monotoneGuardFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "monotone-guard-in-loop" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3021_MonotoneGuardInLoop is the measured shape: the autograd conv1d backward, whose
// per-tap `j >= 0` was false for only the first K-1 of L positions and whose loop header profiled
// larger than either line the guard protected.
func TestDetectPS3021_MonotoneGuardInLoop(t *testing.T) {
	src := `package p

func conv(dws, dxs, xs, ws []float64, t, K, D, c int, gv float64) {
	for k := 0; k < K; k++ {
		j := t - (K - 1) + k
		if j >= 0 {
			dws[c*K+k] += gv * xs[j*D+c]
			dxs[j*D+c] += gv * ws[c*K+k]
		}
	}
}`
	fs := monotoneGuardFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The direction warning is the part that turns a good idea into a silent bug, and the honest
	// null result stops a reader expecting a win everywhere. Both must survive into the advice.
	if !containsAll(fs[0].msg, "DIRECTION", "p=0.210") {
		t.Fatalf("message omits the direction warning or the null result:\n%s", fs[0].msg)
	}
}

// TestDetectPS3021_SilentWhenBounded pins the applied form: the crossing point computed once and
// folded into the loop's start, which is exactly the fix.
func TestDetectPS3021_SilentWhenBounded(t *testing.T) {
	src := `package p

func conv(dws, dxs, xs, ws []float64, t, K, D, c int, gv float64) {
	base := t - (K - 1)
	k0 := 0
	if base < 0 {
		k0 = -base
	}
	for k := k0; k < K; k++ {
		j := base + k
		dws[c*K+k] += gv * xs[j*D+c]
		dxs[j*D+c] += gv * ws[c*K+k]
	}
}`
	if fs := monotoneGuardFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a bounded loop is the applied form:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3021_SilentOnEqualityGuard pins MONOTONICITY. The fixture keeps everything else
// identical — a counted loop, a value moving with the loop variable, an invariant right-hand side —
// and changes only the operator. An equality selects a single iteration rather than a run, so there
// is no crossing point and no bound to hoist it into; without this construct present, dropping the
// operator test would go undetected.
func TestDetectPS3021_SilentOnEqualityGuard(t *testing.T) {
	src := `package p

func pick(dst, src []float64, t, K int) {
	for k := 0; k < K; k++ {
		j := t + k
		if j == 0 {
			dst[k] = src[k]
		}
	}
}`
	if fs := monotoneGuardFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — equality selects one iteration, not a run", len(fs))
	}
}

// TestDetectPS3021_SilentWhenBothSidesMove pins that the comparison must have a FIXED side. If the
// bound advances with the loop too, the guard can re-enter and there is no single crossing index.
func TestDetectPS3021_SilentWhenBothSidesMove(t *testing.T) {
	src := `package p

func both(dst, src []float64, K int) {
	for k := 0; k < K; k++ {
		j := k - 1
		m := k + 2
		if j >= m {
			dst[k] = src[k]
		}
	}
}`
	if fs := monotoneGuardFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — both sides advance, so there is no fixed crossing point", len(fs))
	}
}

// TestDetectPS3021_SilentOnIfElse pins SKIPPING. An if/else runs work on both paths, so nothing is
// being skipped and the guard cannot become a bound — it is a selection, and rewriting it as one
// would drop the else branch entirely.
func TestDetectPS3021_SilentOnIfElse(t *testing.T) {
	src := `package p

func sel(dst, src []float64, t, K int) {
	for k := 0; k < K; k++ {
		j := t + k
		if j >= 0 {
			dst[k] = src[k]
		} else {
			dst[k] = -src[k]
		}
	}
}`
	if fs := monotoneGuardFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — an if/else selects, it does not skip", len(fs))
	}
}

// TestDetectPS3021_SilentOnNonAdditiveGuard pins the ADDITIVE form. A product of the loop variable
// with something else has no single crossing index the loop bounds can express — with a negative
// factor it reverses, so the guarded region is not a prefix or a suffix at all.
func TestDetectPS3021_SilentOnNonAdditiveGuard(t *testing.T) {
	src := `package p

func prod(dst, src []float64, K, w int) {
	for k := 0; k < K; k++ {
		j := k * w
		if j >= 8 {
			dst[k] = src[k]
		}
	}
}`
	if fs := monotoneGuardFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a product has no single crossing index", len(fs))
	}
}

// TestDetectPS3021_SilentWhenBoundMovesNonAdditively is the floor SilentWhenBothSidesMove cannot
// supply. There the both-sides test suppresses first, so blanking the invariant check changes
// nothing and the floor passes for the wrong reason. Here the bound is a PRODUCT of the loop
// variable: movesWithLoop rejects it for being non-additive, so the comparison looks one-sided —
// but the bound plainly does move, and hoisting the guard into the loop's start would be wrong.
// Only the invariant check catches this, and only this fixture proves it.
func TestDetectPS3021_SilentWhenBoundMovesNonAdditively(t *testing.T) {
	src := `package p

func moving(dst, src []float64, K int) {
	for k := 0; k < K; k++ {
		j := k - 1
		if j >= k*2 {
			dst[k] = src[k]
		}
	}
}`
	if fs := monotoneGuardFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the bound moves with the loop, so it is not a fixed crossing point:\n%s",
			len(fs), fs[0].msg)
	}
}
