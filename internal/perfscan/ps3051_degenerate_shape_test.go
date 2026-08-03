package main

import "testing"

func degenerateFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "blocked-kernel-without-a-degenerate-shape-guard" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3051_BlockedBandNoGuard is the measured shape: a band GEMM that blocks four output
// rows and iterates the column dimension innermost, with no branch for one column.
//
// The blocking loop has NO init clause — i is declared above it — and the innermost range is over
// a NAME rather than the slice expression that produced it. Both are how the kernel is actually
// written, and each of them on its own made an earlier version of this check silent on it.
func TestDetectPS3051_BlockedBandNoGuard(t *testing.T) {
	src := `package p

func band(A, B, C []float64, loRow, hiRow, k, n int) {
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		c0 := C[(i+0)*n : (i+1)*n]
		c1 := C[(i+1)*n : (i+2)*n]
		for p := range k {
			bp := B[p*n : (p+1)*n]
			a0 := A[(i+0)*k+p]
			a1 := A[(i+1)*k+p]
			for j, bv := range bp {
				c0[j] += a0 * bv
				c1[j] += a1 * bv
			}
		}
	}
}`
	fs := degenerateFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The message has to carry where the win comes from, because the obvious reading is wrong:
	// the simpler loop is not the point, the registers are, and the simpler loop was measured
	// slightly worse than the block it would replace.
	if !containsAll(fs[0].msg, "ADD THE DEGENERATE PATH", "as SCALARS",
		"MEASURE THE OBVIOUS FORM TOO, AND EXPECT IT TO LOSE") {
		t.Fatalf("message omits the fix, the register point or the rejected alternative:\n%s", fs[0].msg)
	}
}

// TestDetectPS3051_SilentWithAShapeBranch pins the APPLIED form. The guard test is deliberately
// coarse — ANY int parameter compared against a literal counts — because the fix branches on a
// DIFFERENT dimension than the one reported: it special-cases n == 1 and then blocks over rows
// with the reduction innermost. Keying on the reported dimension made the check report its own
// fix.
func TestDetectPS3051_SilentWithAShapeBranch(t *testing.T) {
	src := `package p

func band(A, B, C []float64, loRow, hiRow, k, n int) {
	if n == 1 {
		i := loRow
		for ; i+3 < hiRow; i += 4 {
			v0, v1 := C[i+0], C[i+1]
			for p, bv := range B[:k] {
				v0 += A[(i+0)*k+p] * bv
				v1 += A[(i+1)*k+p] * bv
			}
			C[i+0], C[i+1] = v0, v1
		}
		return
	}
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		c0 := C[(i+0)*n : (i+1)*n]
		for p := range k {
			bp := B[p*n : (p+1)*n]
			a0 := A[(i+0)*k+p]
			for j, bv := range bp {
				c0[j] += a0 * bv
			}
		}
	}
}`
	if fs := degenerateFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the function already branches on a shape:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3051_SilentWithoutBlocking pins that the finding is about a BLOCKED loop. A loop
// advancing one row at a time has no per-block machinery to waste and nothing to hold in
// registers beyond what it already does.
func TestDetectPS3051_SilentWithoutBlocking(t *testing.T) {
	src := `package p

func rows(A, B, C []float64, loRow, hiRow, k, n int) {
	for i := loRow; i < hiRow; i++ {
		ci := C[i*n : (i+1)*n]
		for p := range k {
			bp := B[p*n : (p+1)*n]
			aip := A[i*k+p]
			for j, bv := range bp {
				ci[j] += aip * bv
			}
		}
	}
}`
	if fs := degenerateFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the loop is not blocked:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3051_SilentWhenTheInnerLengthIsNotAParameter pins the other half. If the innermost
// pass is over a fixed-size window, there is no shape the caller can degenerate.
func TestDetectPS3051_SilentWhenTheInnerLengthIsNotAParameter(t *testing.T) {
	src := `package p

func blocks(A, B, C []float64, loRow, hiRow, k int) {
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		c0 := C[(i+0)*4 : (i+1)*4]
		for p := range k {
			bp := B[p*4 : p*4+4]
			a0 := A[(i+0)*k+p]
			for j, bv := range bp {
				c0[j] += a0 * bv
			}
		}
	}
}`
	if fs := degenerateFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the inner width is fixed:\n%s", len(fs), fs[0].msg)
	}
}
