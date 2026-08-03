package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func permutationFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	fanoutReg = map[string]map[string]bool{}
	collectFanoutHelpers([]*ast.File{f})
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "serial-permutation" {
			out = append(out, fnd)
		}
	}
	return out
}

const permPrelude = `package p

func parallelChunks(n, work int, body func(lo, hi int)) { body(0, n) }
`

// TestDetectPS3056_BlockedTranspose is the measured shape: a cache-blocked transpose, entirely
// copies, in a package that has a fan-out helper and does not call it.
//
// The source subscript carries index ARITHMETIC — src[row+j] — and an earlier version of this
// check treated any arithmetic in the right-hand side as disqualifying, which made it silent on
// exactly this. Addressing is not computation on the value.
func TestDetectPS3056_BlockedTranspose(t *testing.T) {
	src := permPrelude + `
func transpose(src, dst []float64, a, b int) {
	const tile = 64
	for ii := 0; ii < a; ii += tile {
		iMax := min(ii+tile, a)
		for jj := 0; jj < b; jj += tile {
			jMax := min(jj+tile, b)
			for i := ii; i < iMax; i++ {
				row := i * b
				for j := jj; j < jMax; j++ {
					dst[j*a+i] = src[row+j]
				}
			}
		}
	}
}`
	fs := permutationFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	if !containsAll(fs[0].msg, "BIT-IDENTICAL at any band count", "CHECK THE BAND OWNS DISJOINT OUTPUT",
		"GATE IT WITH TWO TESTS, NOT ONE") {
		t.Fatalf("message omits the exactness claim or the disjointness warning:\n%s", fs[0].msg)
	}
}

// TestDetectPS3056_SilentOnAccumulation pins the discriminator. It is written THREE loops deep on
// purpose: the check wants a nest, and at two levels these fixtures were silent for that reason
// rather than for the one each of them claims to test — a mutation removing the accumulation
// rejection left them green.
//
// TestDetectPS3056_SilentOnAccumulation pins the discriminator. A nest that adds into its
// destination has a summation order to preserve, so it is one of the other checks' shapes and not
// this one — the whole point of naming permutations separately is that they have no order at all.
func TestDetectPS3056_SilentOnAccumulation(t *testing.T) {
	src := permPrelude + `
func acc(src, dst []float64, a, b, c int) {
	for i := 0; i < a; i++ {
		for j := 0; j < b; j++ {
			for k := 0; k < c; k++ {
				dst[(j*a+i)*c+k] += src[(i*b+j)*c+k]
			}
		}
	}
}`
	if fs := permutationFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — this accumulates:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3056_SilentOnComputedValues pins that the write must be a READ. A nest computing
// values into its destination is an elementwise kernel, which several other checks already cover
// and which needs its arithmetic examined rather than just its indices.
func TestDetectPS3056_SilentOnComputedValues(t *testing.T) {
	src := permPrelude + `
func scale(src, dst []float64, a, b, c int, f float64) {
	for i := 0; i < a; i++ {
		for j := 0; j < b; j++ {
			for k := 0; k < c; k++ {
				dst[(j*a+i)*c+k] = src[(i*b+j)*c+k] * f
			}
		}
	}
}`
	if fs := permutationFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the value is computed, not moved:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3056_SilentWhenAlreadyFannedOut pins the APPLIED form.
func TestDetectPS3056_SilentWhenAlreadyFannedOut(t *testing.T) {
	src := permPrelude + `
func transpose(src, dst []float64, a, b int) {
	parallelChunks(a, a*b, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			for j := 0; j < b; j++ {
				for k := 0; k < b; k++ {
					dst[(j*a+i)*b+k] = src[(i*b+j)*b+k]
				}
			}
		}
	})
}`
	if fs := permutationFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — already banded:\n%s", len(fs), fs[0].msg)
	}
}
