package main

import "testing"

func loopScratchFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "loop-hoistable-scratch" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3035_LoopHoistableScratch is the measured shape: a Cholesky solve allocating its
// forward-substitution buffer once per right-hand side, sized by the matrix dimension the loop
// does not vary. At 128 columns, hoisting it took 133 allocations to 43.
func TestDetectPS3035_LoopHoistableScratch(t *testing.T) {
	src := `package p

func CholSolve(lf []float64, b []float64, n, cols int) []float64 {
	out := make([]float64, n*cols)
	for c := range cols {
		y := make([]float64, n)
		for i := range n {
			s := b[i*cols+c]
			for k := range i {
				s -= lf[i*n+k] * y[k]
			}
			y[i] = s / lf[i*n+i]
		}
		for i := range n {
			out[i*cols+c] = y[i]
		}
	}
	return out
}`
	fs := loopScratchFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The correctness precondition is the part a reader must not miss: a fresh make is zeroed and
	// a reused buffer is not. And the metric, since the time win here was nil at both shapes.
	if !containsAll(fs[0].msg, "PROVE THE OVERWRITE", "allocs/op AND B/op", "PS2001") {
		t.Fatalf("message omits the precondition, the metric or why PS2001 misses it:\n%s", fs[0].msg)
	}
}

// TestDetectPS3035_SilentOnAppliedForm pins the applied form.
func TestDetectPS3035_SilentOnAppliedForm(t *testing.T) {
	src := `package p

func CholSolve(lf []float64, b []float64, n, cols int) []float64 {
	out := make([]float64, n*cols)
	y := make([]float64, n)
	for c := range cols {
		for i := range n {
			s := b[i*cols+c]
			for k := range i {
				s -= lf[i*n+k] * y[k]
			}
			y[i] = s / lf[i*n+i]
		}
		for i := range n {
			out[i*cols+c] = y[i]
		}
	}
	return out
}`
	if fs := loopScratchFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the buffer is already hoisted:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3035_SilentOnLoopVaryingSize pins that the size must be invariant. A buffer whose
// length grows with the loop cannot be hoisted without deciding a maximum first, which is a
// different edit with a different cost and a different risk.
func TestDetectPS3035_SilentOnLoopVaryingSize(t *testing.T) {
	src := `package p

func prefixes(x []float64, n int) float64 {
	var t float64
	for i := range n {
		buf := make([]float64, i+1)
		for j := range buf {
			buf[j] = x[j]
		}
		t += buf[0]
	}
	return t
}`
	if fs := loopScratchFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the size varies with the loop:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3035_SilentWhenTheBufferIsKept pins the escape condition, which is what makes the
// difference between a wasted allocation and a required one. A row stored into a matrix that
// outlives the loop must be its own buffer; reusing one would give every row the same array.
func TestDetectPS3035_SilentWhenTheBufferIsKept(t *testing.T) {
	src := `package p

func rows(x []float64, m, n int) [][]float64 {
	out := make([][]float64, m)
	for i := range m {
		r := make([]float64, n)
		for j := range r {
			r[j] = x[i*n+j]
		}
		out[i] = r
	}
	return out
}`
	if fs := loopScratchFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — each row is kept:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3035_SilentWhenHandedToACall pins the conservative half of the escape test: a
// buffer passed to something else may be retained by it, and this scanner cannot see where.
func TestDetectPS3035_SilentWhenHandedToACall(t *testing.T) {
	src := `package p

func fill(c *cache, ks []int, n int) {
	for _, k := range ks {
		buf := make([]float64, n)
		for j := range buf {
			buf[j] = float64(j)
		}
		c.Store(k, buf)
	}
}`
	if fs := loopScratchFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the callee may keep the buffer:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3035_ReportsThroughCopyAndLen keeps that exemption honest. copy and len cannot
// retain anything, and a scratch buffer filled by copy is the single most common shape of this
// finding; excluding it would silence most of the class.
func TestDetectPS3035_ReportsThroughCopyAndLen(t *testing.T) {
	src := `package p

func sums(x []float64, m, n int) float64 {
	var t float64
	for i := range m {
		buf := make([]float64, n)
		copy(buf, x[i*n:i*n+len(buf)])
		for j := range buf {
			t += buf[j]
		}
	}
	return t
}`
	if fs := loopScratchFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — copy and len cannot retain the buffer", len(fs))
	}
}
