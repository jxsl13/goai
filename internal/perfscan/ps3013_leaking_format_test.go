package main

import (
	"strings"
	"testing"
)

func leakingFormatFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "leaking-format-param" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3013_NamedSliceParam is the measured shape: tensor.NewOn's invalid-shape panic
// formatted its Shape with %v, which made every caller heap-allocate its shape literal.
func TestDetectPS3013_NamedSliceParam(t *testing.T) {
	src := `package p

func NewOn(dev Device, dtype Dtype, shape Shape) *Tensor {
	if !shape.IsValid() {
		panic(fmt.Sprintf("tensor: invalid shape %v", shape))
	}
	return nil
}`
	fs := leakingFormatFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Without the -gcflags=-m step a reader cannot tell a fix that worked from one that did not,
	// because a second leak in the same function keeps the old verdict.
	if !strings.Contains(fs[0].msg, "gcflags=-m") {
		t.Fatalf("message omits the verification step:\n%s", fs[0].msg)
	}
}

// TestDetectPS3013_ReportsEachLeakingParam pins that both operands of a two-shape error are
// reported. backend.BroadcastShape formats a AND b, and fixing only one leaves the other leaking.
func TestDetectPS3013_ReportsEachLeakingParam(t *testing.T) {
	src := `package p

func BroadcastShape(a, b Shape) error {
	return fmt.Errorf("shapes %v and %v not compatible", a, b)
}`
	if fs := leakingFormatFindings(t, src); len(fs) != 2 {
		t.Fatalf("%d findings, want 2 — each leaking parameter has to be named", len(fs))
	}
}

// TestDetectPS3013_SilentOnValueParams is the precision floor. Leaking an int or a float costs
// nothing at the call site: there is no pointer for the caller to heap-allocate.
func TestDetectPS3013_SilentOnValueParams(t *testing.T) {
	src := `package p

func f(n int, x float64, ok bool) error {
	return fmt.Errorf("bad n=%d x=%v ok=%v", n, x, ok)
}`
	if fs := leakingFormatFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — value parameters carry no pointer to leak:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3013_SilentOnFixedArray keeps the slice test honest. A fixed-size array is a VALUE,
// copied into the interface, so the caller's array is not forced onto the heap by this call.
func TestDetectPS3013_SilentOnFixedArray(t *testing.T) {
	src := `package p

func f(a [4]int) error {
	return fmt.Errorf("bad %v", a)
}`
	if fs := leakingFormatFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a fixed array is a value, not a pointer", len(fs))
	}
}

// TestDetectPS3013_SilentOnLocals keeps the check to PARAMETERS. A local built inside the function
// already lives wherever it lives; formatting it cannot change any caller's allocation.
//
// The fixture carries a leakable PARAMETER that is not formatted, so the function is not skipped
// wholesale for having no candidates — without that, this test passes even with the
// parameter-only rule removed and floors nothing.
func TestDetectPS3013_SilentOnLocals(t *testing.T) {
	src := `package p

func f(s []int) error {
	tmp := make([]int, 2)
	_ = s
	return fmt.Errorf("bad %v", tmp)
}`
	if fs := leakingFormatFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — only a parameter's leak propagates to callers", len(fs))
	}
}

// TestDetectPS3013_SilentOnNonFmtCall keeps the check to fmt specifically.
//
// The fixture uses log.Printf, a package SELECTOR call, on purpose: a bare helper(s) call is not a
// SelectorExpr at all and would be rejected a step earlier, so it could not floor the package test.
// Restricting to fmt is a deliberate under-reach — log and others leak the same way — because fmt
// is where the measured instances were and widening it without measuring would be a guess.
func TestDetectPS3013_SilentOnNonFmtCall(t *testing.T) {
	src := `package p

func f(s []int) {
	log.Printf("bad %v", s)
}`
	if fs := leakingFormatFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a plain call is not an interface boundary", len(fs))
	}
}

// TestDetectPS3013_ReportsPlainSliceParam pins the syntactic half, which needs no configuration —
// a []T parameter is a slice whatever the element type.
func TestDetectPS3013_ReportsPlainSliceParam(t *testing.T) {
	src := `package p

func f(xs []float64) error {
	return fmt.Errorf("bad %v", xs)
}`
	if fs := leakingFormatFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
}
