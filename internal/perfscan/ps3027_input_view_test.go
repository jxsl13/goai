package main

import "testing"

func inputViewFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "input-view-on-output-tensor" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3027_InputViewOnOutputTensor is the measured shape: a backward kernel that took its
// gradient buffers from the input-view helper. On the dtype where that view copies rather than
// aliases, every gradient landed in a buffer nobody read and the op returned zeros.
func TestDetectPS3027_InputViewOnOutputTensor(t *testing.T) {
	src := `package p

func bwd(ctx *C, q *T) []*T {
	dQ := tensor.NewOn(ctx.Device(), q.Dtype(), q.Shape())
	qs, qok := f64Data(q)
	dqs, dqok := f64Data(dQ)
	if qok && dqok {
		for i := range qs {
			dqs[i] += qs[i]
		}
	}
	return []*T{dQ}
}`
	fs := inputViewFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — the output, not the input", len(fs))
	}
	// The silence of the failure is the whole point: right shape, no error, all zeros. And the
	// other-dtype test is the only thing that keeps it from coming back.
	if !containsAll(fs[0].msg, "all zeros", "OTHER dtype") {
		t.Fatalf("message omits the failure mode or the test advice:\n%s", fs[0].msg)
	}
}

// TestDetectPS3027_SilentOnOutputView pins the applied form — the output counterpart, which returns
// a buffer plus a flush.
func TestDetectPS3027_SilentOnOutputView(t *testing.T) {
	src := `package p

func bwd(ctx *C, q *T) []*T {
	dQ := tensor.NewOn(ctx.Device(), q.Dtype(), q.Shape())
	qs, qok := f64Data(q)
	dqs, dqflush, dqok := outF64(dQ)
	if qok && dqok {
		for i := range qs {
			dqs[i] += qs[i]
		}
		dqflush()
	}
	return []*T{dQ}
}`
	if fs := inputViewFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the output view is the applied form:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3027_SilentOnInputs pins that the argument must be a tensor this function ALLOCATED.
// Reading an input through the input view is what the helper is for, and the fixture keeps an
// allocation in the function so it discriminates the argument rather than the presence of one.
func TestDetectPS3027_SilentOnInputs(t *testing.T) {
	src := `package p

func fwd(ctx *C, q, k *T) []*T {
	out := tensor.NewOn(ctx.Device(), q.Dtype(), q.Shape())
	qs, _ := f64Data(q)
	ks, _ := f64Data(k)
	os, flush, _ := outF64(out)
	for i := range qs {
		os[i] = qs[i] * ks[i]
	}
	flush()
	return []*T{out}
}`
	if fs := inputViewFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — both arguments are inputs:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3027_SilentOnUnallocatedName pins the ALLOCATION. Honest about the mutation result:
// blanking the membership test alone does NOT redden this floor, because the function early-outs
// when nothing was allocated at all. Blanking BOTH reddens it, which is what proves the floor is
// live. The early-out is not a second predicate — it is a short-circuit for the same fact. A tensor that arrives as a
// parameter may well be an output the caller owns, but this scanner has no package view and cannot
// tell; reporting it would be a guess. The fixture passes a parameter with an output-shaped name,
// which is exactly the case a name-based heuristic would get wrong.
func TestDetectPS3027_SilentOnUnallocatedName(t *testing.T) {
	src := `package p

func accum(dQ *T, q *T) {
	qs, _ := f64Data(q)
	dqs, _ := f64Data(dQ)
	for i := range qs {
		dqs[i] += qs[i]
	}
}`
	if fs := inputViewFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — dQ is a parameter, not allocated here:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3027_SilentOnNonAllocatorSource pins that the name must come from an ALLOCATOR. A
// tensor handed back by some other call may be a reused buffer the caller owns, or a view, or an
// input in disguise; without a package view the scanner cannot tell, and reporting it would be a
// guess. The fixture is the positive with one word changed — the constructor replaced by an
// ordinary call — so it discriminates the allocator list alone.
func TestDetectPS3027_SilentOnNonAllocatorSource(t *testing.T) {
	src := `package p

func bwd(ctx *C, q *T) []*T {
	dQ := reuseGradBuffer(ctx, q)
	qs, qok := f64Data(q)
	dqs, dqok := f64Data(dQ)
	if qok && dqok {
		for i := range qs {
			dqs[i] += qs[i]
		}
	}
	return []*T{dQ}
}`
	if fs := inputViewFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — dQ did not come from an allocator:\n%s", len(fs), fs[0].msg)
	}
}
