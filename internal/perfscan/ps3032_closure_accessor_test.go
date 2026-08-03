package main

import "testing"

func closureAccessorFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "closure-accessor-in-loop" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3032_ClosureAccessorInLoop is the measured shape: a pooling backward reading and
// writing every element through closures a helper handed back. Typed arms went about -50% on four
// benchmark cells.
func TestDetectPS3032_ClosureAccessorInLoop(t *testing.T) {
	src := `package p

func back(x, y, g, gx *T, n int) {
	getX, getG, addGX := poolAccessors(x, y, g, gx)
	for i := range n {
		if getX(i) > 0 {
			addGX(i, getG(i))
		}
	}
}`
	fs := closureAccessorFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — one per factory, not one per call", len(fs))
	}
	// Both conversion traps were hit while making the measured change; a reader who converts
	// without them ships an ulp-level divergence or a test that cannot see one.
	if !containsAll(fs[0].msg, "BLOCKS FMA CONTRACTION", "SEVERAL accumulations") {
		t.Fatalf("message omits a conversion trap:\n%s", fs[0].msg)
	}
}

// TestDetectPS3032_SilentWithGuardedFastPath pins the applied form, and it is the unusual one: the
// fix does NOT remove the closure loop, it adds typed arms in front of it and leaves the loop as
// the fallback. So a function that already exits through a guarded fast path must go quiet, or the
// check files its own remedy as the defect.
func TestDetectPS3032_SilentWithGuardedFastPath(t *testing.T) {
	src := `package p

func back(x, y, g, gx *T, n int) {
	if xs, gs, ok := slicesF64(x, g); ok {
		backF64(xs, gs, n)
		return
	}
	getX, getG, addGX := poolAccessors(x, y, g, gx)
	for i := range n {
		if getX(i) > 0 {
			addGX(i, getG(i))
		}
	}
}`
	if fs := closureAccessorFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the closure loop is the fallback behind a fast path:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3032_SilentOnCallOutsideLoop pins the LOOP. A closure called once is a closure call,
// not a per-element cost, and devirtualizing it would buy nothing.
func TestDetectPS3032_SilentOnCallOutsideLoop(t *testing.T) {
	src := `package p

func back(x, y, g, gx *T, n int) float64 {
	getX, _, _ := poolAccessors(x, y, g, gx)
	total := getX(0)
	for i := range n {
		total += float64(i)
	}
	return total
}`
	if fs := closureAccessorFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the closure is called once, outside the loop", len(fs))
	}
}

// TestDetectPS3032_SilentOnDirectCall pins that the callee must be a FUNCTION VALUE bound from a
// factory. A plain function called in a loop is an ordinary call the compiler can inline, and the
// fixture keeps the loop and the call shape so it discriminates the binding alone.
func TestDetectPS3032_SilentOnDirectCall(t *testing.T) {
	src := `package p

func back(x []float64, n int) float64 {
	var total float64
	for i := range n {
		total += scaleOf(x, i)
	}
	return total
}`
	if fs := closureAccessorFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a package function is not a closure value", len(fs))
	}
}
