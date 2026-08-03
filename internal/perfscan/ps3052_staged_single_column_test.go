package main

import "testing"

func stagedFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "staged-matrix-reduced-against-one-column" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3052_StagedThenReduced is the measured shape: an im2col fill immediately followed by
// the GEMM that consumes it, with the output width passed straight through and never branched on.
func TestDetectPS3052_StagedThenReduced(t *testing.T) {
	src := `package p

func fill(cols, xs []float64, lo, hi, off, k int)          {}
func gemm(A, B, C []float64, lo, hi, k, n int)             {}
func scatter(prod, os []float64, lo, hi, off, f, hw int)   {}

func conv(cols, xs, wt, prod, os []float64, lo, hi, k, f, hw int) {
	for base := lo; base < hi; base++ {
		fill(cols, xs, base, base+1, base, k)
		gemm(cols, wt, prod, 0, 1, k, f)
		scatter(prod, os, base, base+1, base, f, hw)
	}
}`
	fs := stagedFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Three things the measurement produced and the argument would not: the exactness hazard the
	// staging's zeros create, the requirement to test both sides of the gate, and where the gain
	// actually is.
	if !containsAll(fs[0].msg, "ADD A FUSED PATH FOR WIDTH ONE", "MIND THE ZEROS THE STAGING HOLDS",
		"THE GAIN IS IN THE SHORT KERNELS") {
		t.Fatalf("message omits the fix, the exactness hazard or the where-it-pays note:\n%s", fs[0].msg)
	}
}

// TestDetectPS3052_SilentWithAWidthBranch pins the APPLIED form: the function branches on the
// width, so the fused path is either there or has been considered and rejected.
func TestDetectPS3052_SilentWithAWidthBranch(t *testing.T) {
	src := `package p

func fill(cols, xs []float64, lo, hi, off, k int)        {}
func gemm(A, B, C []float64, lo, hi, k, n int)           {}
func fused(prod, xs, wt []float64, lo, hi, off, k int)   {}
func scatter(prod, os []float64, lo, hi, off, f, hw int) {}

func conv(cols, xs, wt, prod, os []float64, lo, hi, k, f, hw int) {
	one := f == 1
	for base := lo; base < hi; base++ {
		if one {
			fused(prod, xs, wt, base, base+1, base, k)
		} else {
			fill(cols, xs, base, base+1, base, k)
			gemm(cols, wt, prod, 0, 1, k, f)
		}
		scatter(prod, os, base, base+1, base, f, hw)
	}
}`
	if fs := stagedFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the function already branches on the width:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3052_SilentWhenTheCallsDoNotShareABuffer pins the connection the finding rests on.
// Two adjacent calls that happen to be neighbours are not a staging pipeline; there is nothing to
// fuse if the second does not consume what the first produced.
func TestDetectPS3052_SilentWhenTheCallsDoNotShareABuffer(t *testing.T) {
	src := `package p

func fill(cols, xs []float64, lo, hi, off, k int) {}
func gemm(A, B, C []float64, lo, hi, k, n int)    {}

func conv(cols, other, xs, wt, prod []float64, lo, hi, k, f int) {
	for base := lo; base < hi; base++ {
		fill(cols, xs, base, base+1, base, k)
		gemm(other, wt, prod, 0, 1, k, f)
	}
}`
	if fs := stagedFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the consumer reads a different buffer:\n%s", len(fs), fs[0].msg)
	}
}
