package main

import "testing"

func variadicCallFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "fixed-arity-variadic-call" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3024_FixedArityVariadicCall is the measured shape: nlp's MHA calling a variadic
// dispatch wrapper with a fixed argument list, once per backend dispatch.
func TestDetectPS3024_FixedArityVariadicCall(t *testing.T) {
	src := `package p

func run(ctx *C, a, b *T) (*T, error) {
	x, err := exec(ctx, OpMatMul, nil, a, b)
	if err != nil {
		return nil, err
	}
	return x, nil
}`
	fs := variadicCallFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The metric caveat is what stops a reader chasing a time win that is not there, and the
	// recorder rule is what stops a pooled rewrite corrupting a tape.
	if !containsAll(fs[0].msg, "NOT ns/op", "MUST DEFER") {
		t.Fatalf("message omits the metric caveat or the recorder rule:\n%s", fs[0].msg)
	}
}

// TestDetectPS3024_SilentOnSpread pins the one case that cannot be fixed: a genuine spread has to
// build a pack, so no fixed-arity sibling can serve it.
func TestDetectPS3024_SilentOnSpread(t *testing.T) {
	src := `package p

func run(ctx *C, parts []*T) (*T, error) {
	return exec(ctx, OpConcat, nil, parts...)
}`
	if fs := variadicCallFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a spread cannot avoid the pack:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3024_SilentInsidePooledHelper pins the APPLIED form, and it is the floor that keeps
// this check from filing the fix as the defect. A pooled helper hands its fixed arguments to the
// variadic form precisely when a recorder is attached, because the tape may retain the slice — that
// call is correct and must not be reported.
func TestDetectPS3024_SilentInsidePooledHelper(t *testing.T) {
	src := `package p

func exec2(ctx *C, op Op, attrs A, a, b *T) (*T, error) {
	if ctx.Recorder != nil {
		return exec(ctx, op, attrs, a, b)
	}
	sp := ins2Pool.Get().(*[]*T)
	s := *sp
	s[0], s[1] = a, b
	out, err := Execute(ctx, op, s, attrs)
	ins2Pool.Put(sp)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}`
	if fs := variadicCallFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a pooled helper's recorder fallback is correct:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3024_SilentOnUnconfiguredName pins that the wrapper set is CONFIGURED. Without types
// or a package view the scanner cannot tell a dispatch wrapper from any other call, so the names
// are supplied; a call to something not on the list must stay quiet.
func TestDetectPS3024_SilentOnUnconfiguredName(t *testing.T) {
	src := `package p

func run(ctx *C, a, b *T) (*T, error) {
	return dispatch(ctx, OpMatMul, nil, a, b)
}`
	if fs := variadicCallFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — dispatch is not a configured wrapper", len(fs))
	}
}
