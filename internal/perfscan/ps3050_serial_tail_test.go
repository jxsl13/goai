package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func serialTailFindingsIn(t *testing.T, src string) []finding {
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
		if fnd.category == "serial-tail-after-fanout" {
			out = append(out, fnd)
		}
	}
	return out
}

const tailPrelude = `package p

func parallelWork(n, work int, body func(lo, hi int)) { body(0, n) }

func band(A, B []float32, acc []float64, lo, hi, k, n int) {}
`

// TestDetectPS3050_NarrowingTail is the measured shape: a matmul that accumulates into a wider
// scratch inside the bands and then narrows the whole result to the output afterwards.
//
// The bands touch the SOURCE of that pass, not its destination — they fill the accumulator and
// have never heard of C. A check that only asked whether the bands touch the buffer being written
// is silent here, which is how the first version behaved on the site it was written for.
func TestDetectPS3050_NarrowingTail(t *testing.T) {
	src := tailPrelude + `
func gemm(A, B, C []float32, m, k, n int) {
	acc := make([]float64, m*n)
	parallelWork(m, k*n, func(lo, hi int) {
		band(A, B, acc, lo, hi, k, n)
	})
	for i := range C {
		C[i] = float32(acc[i])
	}
}`
	fs := serialTailFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Why it is exact, and how to gate it — the sentinel is the part that catches the mistake
	// this transform actually makes, which is a band converting the wrong range.
	if !containsAll(fs[0].msg, "FOLD IT INTO THE BAND", "BIT-IDENTICAL", "GATE IT ON A SENTINEL") {
		t.Fatalf("message omits the fix, the exactness claim or the gating advice:\n%s", fs[0].msg)
	}
}

// TestDetectPS3050_SilentWhenFolded pins the APPLIED form: the pass done inside the band over that
// band's own rows.
func TestDetectPS3050_SilentWhenFolded(t *testing.T) {
	src := tailPrelude + `
func gemm(A, B, C []float32, m, k, n int) {
	acc := make([]float64, m*n)
	parallelWork(m, k*n, func(lo, hi int) {
		band(A, B, acc, lo, hi, k, n)
		for i := lo * n; i < hi*n; i++ {
			C[i] = float32(acc[i])
		}
	})
}`
	if fs := serialTailFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the pass is already inside the band:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3050_SilentOnAnUnrelatedPass pins the connection the finding rests on. A serial pass
// over a buffer the bands never touch is not this shape: folding it into them would be a race, not
// an optimization.
func TestDetectPS3050_SilentOnAnUnrelatedPass(t *testing.T) {
	src := tailPrelude + `
func gemm(A, B, C, other, src []float32, m, k, n int) {
	acc := make([]float64, m*n)
	parallelWork(m, k*n, func(lo, hi int) {
		band(A, B, acc, lo, hi, k, n)
	})
	for i := range other {
		other[i] = src[i] * 2
	}
}`
	if fs := serialTailFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the bands never touch that buffer:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3050_SilentOnAReduction pins that the pass has to be ELEMENTWISE. A tail that folds
// the buffer into a scalar cannot simply move into the bands — that is a reduction, and splitting
// it changes the summation order.
//
// Two independent conditions exclude it — the write target is not an indexed element, and the
// scalar it writes is not a buffer the bands touch — so no single mutation isolates this fixture.
// It is kept because the shape is the one a reader is most likely to fold by mistake.
func TestDetectPS3050_SilentOnAReduction(t *testing.T) {
	src := tailPrelude + `
func gemm(A, B []float32, acc []float64, m, k, n int) float64 {
	parallelWork(m, k*n, func(lo, hi int) {
		band(A, B, acc, lo, hi, k, n)
	})
	var total float64
	for i := range acc {
		total += acc[i]
	}
	return total
}`
	if fs := serialTailFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a reduction is not an elementwise pass:\n%s", len(fs), fs[0].msg)
	}
}
