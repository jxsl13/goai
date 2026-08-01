package main

import "testing"

func fmaClaimFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "unrounded-product-under-exactness-claim" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3025_UnroundedProductUnderClaim is the measured shape: KAN's fused spline, whose doc
// asserted the path was bit-identical to the einsum dispatch with "no FMA" while the code let the
// compiler contract both of its products.
func TestDetectPS3025_UnroundedProductUnderClaim(t *testing.T) {
	src := `package p

// fused is bit-identical to the einsum dispatch path: same operand order, no FMA.
func fused(b, c, y []float64, n int) {
	for i := 0; i < n; i++ {
		var acc float64
		for j := 0; j < n; j++ {
			acc += b[j] * c[j]
		}
		y[i] += acc * b[i]
	}
}`
	fs := fmaClaimFindings(t, src)
	if len(fs) != 2 {
		t.Fatalf("%d findings, want 2 — one per bare product", len(fs))
	}
	// The variable-does-not-work caveat is the part that cost a wrong attempt, and the
	// both-sides warning stops someone chasing an unreachable pin. Both must reach the reader.
	if !containsAll(fs[0].msg, "VARIABLE DOES NOT WORK", "BOTH SIDES MUST BE ROUNDED") {
		t.Fatalf("message omits the variable caveat or the both-sides warning:\n%s", fs[0].msg)
	}
}

// TestDetectPS3025_SilentWhenRounded pins the applied form: an explicit conversion is what the Go
// spec guarantees forces the intermediate rounding.
//
// Honest about what this floor proves: the exclusion is STRUCTURAL, not a predicate. float64(x*y)
// is an *ast.CallExpr, and the type assertion in isUnroundedProduct only admits *ast.BinaryExpr,
// so a wrapped product is filtered before any operator test runs. The floor documents the intended
// boundary; the type assertion is what enforces it.
func TestDetectPS3025_SilentWhenRounded(t *testing.T) {
	src := `package p

// fused is bit-identical to the dispatch path.
func fused(b, c, y []float64, n int) {
	for i := 0; i < n; i++ {
		var acc float64
		for j := 0; j < n; j++ {
			acc += float64(b[j] * c[j])
		}
		y[i] = float64(acc * b[i])
	}
}`
	if fs := fmaClaimFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a wrapped product is the applied form:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3025_SilentWithoutClaim pins the EXACTNESS CLAIM. A bare multiply-add is the normal
// way to write a dot product; it is only a defect where something asserts exactness against a
// second implementation. The fixture deliberately NAMES a peer path, so the peer clause cannot
// suppress it and the claim clause is the only thing keeping it quiet — a fixture without the peer
// mention passes for the wrong reason, which is how this floor was first written.
func TestDetectPS3025_SilentWithoutClaim(t *testing.T) {
	src := `package p

// dot mirrors the einsum dispatch path for the contiguous shape and is measurably faster on it.
func dot(a, b []float64, n int) float64 {
	var s float64
	for i := 0; i < n; i++ {
		s += a[i] * b[i]
	}
	return s
}`
	if fs := fmaClaimFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — no exactness claim, so nothing is contradicted", len(fs))
	}
}

// TestDetectPS3025_SilentOnParallelClaim is the floor that keeps this off the common case. A
// "bit-identical" note about splitting the SAME loop across workers is safe under contraction —
// both halves are the same instructions — so the claim must name a second IMPLEMENTATION. The
// fixture keeps the claim and the bare product; only the peer reference is absent.
func TestDetectPS3025_SilentOnParallelClaim(t *testing.T) {
	src := `package p

// sum is bit-identical to the serial loop: chunking changes which worker runs an
// iteration, never the order terms are added.
func sum(a, b []float64, n int) float64 {
	var s float64
	for i := 0; i < n; i++ {
		s += a[i] * b[i]
	}
	return s
}`
	if fs := fmaClaimFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a parallel-split claim is unaffected by contraction:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3025_SilentOnPlainAssignment pins that the product must feed an ADD. The fixture is
// a CHAIN of multiplies, not a single one: with a lone `y = a*b` the operand test suppresses it
// anyway, so the floor would pass for the wrong reason. Here the outer expression has a product on
// its left, exactly the shape the add case matches — only the operator differs, and a product of
// products has no accumulator for the compiler to fuse into, so it rounds the same on both
// architectures.
func TestDetectPS3025_SilentOnPlainAssignment(t *testing.T) {
	src := `package p

// scale is bit-identical to the dispatch path.
func scale(a, b, c, y []float64, n int) {
	for i := 0; i < n; i++ {
		y[i] = a[i]*b[i] * c[i]
	}
}`
	if fs := fmaClaimFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a product with no add cannot be contracted", len(fs))
	}
}

// TestDetectPS3025_SilentOnNonProduct pins that the fused operand must be a PRODUCT. The fixture
// keeps the claim, the peer reference and the accumulation, and changes only the operator: a
// difference feeding an add is two roundings on every architecture, because there is no multiply
// for the compiler to fold into the addition. Without this construct present, blanking the
// multiply test is a mutation no other fixture detects — every one of them accumulates a product.
func TestDetectPS3025_SilentOnNonProduct(t *testing.T) {
	src := `package p

// resid is bit-identical to the reference path.
func resid(a, b []float64, n int) float64 {
	var s float64
	for i := 0; i < n; i++ {
		s += a[i] - b[i]
	}
	return s
}`
	if fs := fmaClaimFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a difference carries no multiply to contract", len(fs))
	}
}
