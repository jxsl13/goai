package main

import (
	"go/parser"
	"go/token"
	"testing"
)

func narrowUnrollFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "unroll-factor-fixed-at-two" {
			out = append(out, fnd)
		}
	}
	return out
}

// narrowUnrollFixture is the measured shape: a gemm band already holding four C rows in locals
// and taking two B rows per pass.
const narrowUnrollFixture = `package p

func band(A, B, C []float64, i, k, n int) {
	c0 := C[(i+0)*n : (i+1)*n]
	c1 := C[(i+1)*n : (i+2)*n]
	c2 := C[(i+2)*n : (i+3)*n]
	p := 0
	for ; p+1 < k; p += 2 {
		bp0 := B[p*n : (p+1)*n]
		bp1 := B[(p+1)*n : (p+2)*n]
		a00, a01 := A[i*k+p], A[i*k+p+1]
		for j, b0 := range bp0 {
			b1 := bp1[j]
			v0 := c0[j]
			v0 += a00 * b0
			v0 += a01 * b1
			c0[j] = v0
			v1 := c1[j]
			v1 += a00 * b0
			v1 += a01 * b1
			c1[j] = v1
			v2 := c2[j]
			v2 += a00 * b0
			v2 += a01 * b1
			c2[j] = v2
		}
	}
}`

func TestDetectPS3076_NarrowUnroll(t *testing.T) {
	fs := narrowUnrollFindingsIn(t, narrowUnrollFixture)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The instruction, the reason the sweep is cheap, and the result that contradicts the usual
	// argument all came out of the measurement.
	if !containsAll(fs[0].msg, "SWEEP, DO NOT ARGUE", "BIT-IDENTICAL AT ANY FACTOR",
		"eight back at the baseline") {
		t.Fatalf("message omits the instruction, the bit-identity argument or the sweep"+
			" result:\n%s", fs[0].msg)
	}
}

// TestDetectPS3076_SilentOnAWiderStride pins that this is about the FACTOR. A loop already taking
// four or six steps has had the choice made deliberately.
func TestDetectPS3076_SilentOnAWiderStride(t *testing.T) {
	src := replaceOnce(t, narrowUnrollFixture, "	for ; p+1 < k; p += 2 {", "	for ; p+3 < k; p += 4 {")
	if fs := narrowUnrollFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — that factor is not two:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3076_SilentBelowThreeAccumulators pins the register-blocking requirement. A stride
// of two on an ordinary loop says nothing about tuning; the finding is about a kernel that
// already holds its accumulators in locals, because that is where the factor is a choice.
func TestDetectPS3076_SilentBelowThreeAccumulators(t *testing.T) {
	src := replaceOnce(t, narrowUnrollFixture, `			v2 := c2[j]
			v2 += a00 * b0
			v2 += a01 * b1
			c2[j] = v2
`, "")
	if fs := narrowUnrollFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — two accumulators is not register blocking:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3076_SilentWithoutHeldAccumulators pins that the store must come FROM A LOCAL.
// Three stores whose right side is an expression read the destination and write it back in one
// step: nothing is being held across the pass, so a wider factor has nothing to amortize.
func TestDetectPS3076_SilentWithoutHeldAccumulators(t *testing.T) {
	src := replaceOnce(t, narrowUnrollFixture, `			v0 := c0[j]
			v0 += a00 * b0
			v0 += a01 * b1
			c0[j] = v0
			v1 := c1[j]
			v1 += a00 * b0
			v1 += a01 * b1
			c1[j] = v1
			v2 := c2[j]
			v2 += a00 * b0
			v2 += a01 * b1
			c2[j] = v2`, `			c0[j] = c0[j] + a00*b0 + a01*b1
			c1[j] = c1[j] + a00*b0 + a01*b1
			c2[j] = c2[j] + a00*b0 + a01*b1`)
	if fs := narrowUnrollFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — nothing is held in a local:\n%s", len(fs), fs[0].msg)
	}
}
