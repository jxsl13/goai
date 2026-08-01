package main

import "testing"

func literalArgCount(t *testing.T, src string) int {
	t.Helper()
	n := 0
	for _, f := range scanSrc(t, src) {
		if f.category == "loop-invariant-literal-arg" {
			n++
		}
	}
	return n
}

// TestPS6016SkipsStorageGuardedFallback pins the suppression added after a panic proved these arms
// unreachable in practice.
//
// The shape is a devirtualized branch that reaches typed storage plus an else that keeps the old
// dispatch for whatever the branch declines. A literal rebuilt in that else is not rebuilt on the
// path the benchmark takes — nlp quant_deepseekv2's attnReconstructed is the real instance, where a
// panic in the else never fired under either QuantDeepSeekV2Reconstructed benchmark.
//
// Both halves are asserted in ONE fixture on purpose: the same function carries a literal in the
// else arm AND one after the if/else on the common path, so the test fails if the suppression is
// dropped (2 findings) or if it swallows the common-path site too (0 findings).
func TestPS6016SkipsStorageGuardedFallback(t *testing.T) {
	src := `package p

func f(ctx *C, t *T, rows []R, n int) {
	for i := range rows {
		if s := t.Storage().F64(); s != nil {
			consume(s, i)
		} else {
			one, _ := exec(ctx, OpSlice, SliceAttrs{Axis: 1, Start: 0, End: n}, t)
			_ = one
		}
		out, _ := exec(ctx, OpConcat, ConcatAttrs{Axis: 1, Start: 0, End: n}, t)
		_ = out
	}
}`
	if got := literalArgCount(t, src); got != 1 {
		t.Fatalf("%d findings, want 1: the else-arm literal must be suppressed and the common-path "+
			"one after the if/else must survive", got)
	}
}

// TestPS6016KeepsUnguardedElse is the other side of the clause: an else arm is only inert when its
// sibling actually reaches typed storage. A plain if/else with no such branch is ordinary control
// flow, and a literal rebuilt in either arm still runs every iteration.
func TestPS6016KeepsUnguardedElse(t *testing.T) {
	src := `package p

func f(ctx *C, t *T, rows []R, n int, flag bool) {
	for i := range rows {
		if flag {
			consume(nil, i)
		} else {
			one, _ := exec(ctx, OpSlice, SliceAttrs{Axis: 1, Start: 0, End: n}, t)
			_ = one
		}
	}
}`
	if got := literalArgCount(t, src); got != 1 {
		t.Fatalf("%d findings, want 1: an else with no typed-storage sibling is not a fallback arm", got)
	}
}

// TestPS6016SkipsAllConstantLiteral pins the constness suppression, which was added only after a
// hoist measured ZERO.
//
// A literal whose every field is a compile-time constant costs nothing to box — the compiler emits a
// pointer to a static read-only copy. Hoisting two such sites in nlp quant_deepseekv2 changed
// allocs/op not at all, every sample equal, even though `go build -gcflags=-m` had reported them as
// escaping to heap. An isolated benchmark separated the cases: all-constant is 0 allocs at 1.98ns,
// identical to a pre-boxed package var, while one non-constant field costs 1 alloc, 24 B and 11.4ns.
//
// Both cases live in one fixture so the test fails if the suppression is dropped (2 findings) or if
// it swallows the variable-field literal too (0).
func TestPS6016SkipsAllConstantLiteral(t *testing.T) {
	src := `package p

func f(ctx *C, t *T, rows []R, n int) {
	for i := range rows {
		a, _ := exec(ctx, OpConcat, ConcatAttrs{Axis: 1}, t)
		b, _ := exec(ctx, OpSlice, SliceAttrs{Axis: 1, Start: 0, End: n}, t)
		_, _ = a, b
	}
}`
	if got := literalArgCount(t, src); got != 1 {
		t.Fatalf("%d findings, want 1: the all-constant literal boxes for free and must be skipped, "+
			"the one reading n must survive", got)
	}
}
