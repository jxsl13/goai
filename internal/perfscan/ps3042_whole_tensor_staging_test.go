package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// stagingFindingsIn rebuilds the package-level fan-out registry from this source the way the
// real scan does before scanning any file, then keeps only this check's findings. scanSrcFanout
// exists but filters to PS3034's category.
func stagingFindingsIn(t *testing.T, src string) []finding {
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
		if fnd.category == "whole-tensor-staging-buffer" {
			out = append(out, fnd)
		}
	}
	return out
}

// stagingPrelude declares the fan-out helper the check keys on, plus the pooled getter whose
// POINTER-then-dereference binding is how the real allocation is written.
const stagingPrelude = `package p

func parallelWork(n, work int, body func(lo, hi int)) { body(0, n) }

func getF64(n int) *[]float64 { s := make([]float64, n); return &s }
func putF64(p *[]float64)     {}
`

// TestDetectPS3042_PooledWholeTensorBuffer is the measured shape: conv2d's im2col column
// matrix, sized rows x k and handed between two stages that both run inside one fan-out band.
//
// The dereference is the part that matters. The getter returns a POINTER and the callback uses
// the dereferenced slice, so a check that only looks for the name it allocated into is silent on
// this — which is exactly how the first version of this check behaved on the site it was written
// for.
func TestDetectPS3042_PooledWholeTensorBuffer(t *testing.T) {
	src := stagingPrelude + `
func conv(xs, wt, os []float64, rows, k, f int) {
	colsP := getF64(rows * k)
	defer putF64(colsP)
	cols := *colsP
	parallelWork(rows, k*f, func(lo, hi int) {
		fill(cols, xs, lo, hi, k)
		gemm(cols, wt, os, lo, hi, k, f)
	})
}

func fill(cols, xs []float64, lo, hi, k int)          {}
func gemm(a, b, c []float64, lo, hi, k, f int)        {}`
	fs := stagingFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The accumulation warning has to survive into the message. It is the trap that turned this
	// rewrite red the first time: the whole-tensor buffer wrote each row's slot exactly once, so
	// an accumulating consumer looked like a store and the pool's zeroing was doing invisible work.
	if !containsAll(fs[0].msg, "CHECK WHETHER THE CONSUMER ACCUMULATES", "CAP THE CHUNK AT ONE BAND") {
		t.Fatalf("message omits the accumulation trap or the sizing cap:\n%s", fs[0].msg)
	}
}

// TestDetectPS3042_SilentWhenSizedPerBand pins the APPLIED form: the buffer sized by a slot
// width rather than by the fan-out's item count.
func TestDetectPS3042_SilentWhenSizedPerBand(t *testing.T) {
	src := stagingPrelude + `
func conv(xs, wt, os []float64, rows, k, f int) {
	nw, cw := 12, 32
	colsP := getF64(nw * cw * k)
	defer putF64(colsP)
	cols := *colsP
	parallelWork(rows, k*f, func(lo, hi int) {
		for base := lo; base < hi; base += cw {
			fill(cols, xs, base, min(base+cw, hi), k)
			gemm(cols, wt, os, 0, min(cw, hi-base), k, f)
		}
	})
}

func fill(cols, xs []float64, lo, hi, k int)   {}
func gemm(a, b, c []float64, lo, hi, k, f int) {}`
	if fs := stagingFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a slot-sized buffer is the applied form:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3042_SilentOnPerItemBuffer pins the discriminator that keeps the check from firing
// on every parallel loop in the tree. A buffer of exactly one slot per item is the natural size
// for a per-item result and has no staging to shrink; only count TIMES a width is a staging area
// holding rows of intermediate work.
func TestDetectPS3042_SilentOnPerItemBuffer(t *testing.T) {
	src := stagingPrelude + `
func score(xs []float64, rows int) {
	accP := getF64(rows)
	defer putF64(accP)
	acc := *accP
	parallelWork(rows, 8, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			acc[i] = xs[i] * 2
		}
	})
}`
	if fs := stagingFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — one slot per item is not staging:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3042_SilentOnReturnedBuffer pins that an OUTPUT is not staging. Its size is the
// caller's contract, not a choice this function is free to shrink.
func TestDetectPS3042_SilentOnReturnedBuffer(t *testing.T) {
	src := stagingPrelude + `
func build(xs []float64, rows, k int) []float64 {
	out := make([]float64, rows*k)
	parallelWork(rows, k, func(lo, hi int) {
		for i := lo * k; i < hi*k; i++ {
			out[i] = xs[i%len(xs)]
		}
	})
	return out
}`
	if fs := stagingFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a returned buffer is the output:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3042_SilentWhenNotUsedInTheCallback pins that the buffer has to be what the bands
// actually hand each other. One allocated for a serial phase and merely sized by the same count
// is a different question with a different answer.
func TestDetectPS3042_SilentWhenNotUsedInTheCallback(t *testing.T) {
	src := stagingPrelude + `
func prep(xs, os []float64, rows, k int) {
	tmpP := getF64(rows * k)
	defer putF64(tmpP)
	tmp := *tmpP
	for i := range tmp {
		tmp[i] = xs[i%len(xs)]
	}
	parallelWork(rows, k, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			os[i] = xs[i]
		}
	})
}`
	if fs := stagingFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the buffer never reaches a band:\n%s", len(fs), fs[0].msg)
	}
}
