package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// scanSrcFanout is scanSrc with the package-level fan-out registry rebuilt from this source, the
// way the real scan builds it from every file of a package before scanning any of them.
func scanSrcFanout(t *testing.T, src string) []finding {
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
		if fnd.category == "serial-nest-with-idle-fanout" {
			out = append(out, fnd)
		}
	}
	return out
}

// fanoutHelperSrc is the seam the package already has — the reason a finding is one call away.
const fanoutHelperSrc = `
func parallelRows(n, workPerItem int, body func(lo, hi int)) {
	body(0, n)
}
`

// TestDetectPS3034_SerialNestWithIdleFanout is the measured shape: a flat matmul filling a
// destination it allocated, one row band per outer iteration, running on one core while the
// package's own fan-out helper sits unused. Splitting it went -60.5%.
func TestDetectPS3034_SerialNestWithIdleFanout(t *testing.T) {
	src := `package p
` + fanoutHelperSrc + `
func matmulFlat(a, b []float64, m, k, n int) []float64 {
	c := make([]float64, m*n)
	for i := range m {
		ci := c[i*n : i*n+n]
		for p := range k {
			av := a[i*k+p]
			bp := b[p*n : p*n+n]
			for j := range ci {
				ci[j] += av * bp[j]
			}
		}
	}
	return c
}`
	fs := scanSrcFanout(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Three things decide whether this is applied correctly, and all three were learned by
	// measuring: gate on the product rather than the row count, expect allocations to rise, and
	// do not bother moving a band onto the caller.
	if !containsAll(fs[0].msg, "GATE ON THE PRODUCT", "allocations to RISE", "BIT-IDENTICAL") {
		t.Fatalf("message omits the gate, the alloc trade or the exactness claim:\n%s", fs[0].msg)
	}
}

// TestDetectPS3034_SilentWhenAlreadyFannedOut pins the applied form.
func TestDetectPS3034_SilentWhenAlreadyFannedOut(t *testing.T) {
	src := `package p
` + fanoutHelperSrc + `
func matmulFlat(a, b []float64, m, k, n int) []float64 {
	c := make([]float64, m*n)
	parallelRows(m, k*n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			ci := c[i*n : i*n+n]
			for p := range k {
				av := a[i*k+p]
				bp := b[p*n : p*n+n]
				for j := range ci {
					ci[j] += av * bp[j]
				}
			}
		}
	})
	return c
}`
	if fs := scanSrcFanout(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the loop already fans out:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3034_SilentOnCrossIterationAccumulator pins the independence condition, which is
// the one that decides whether the split is CORRECT rather than merely faster. A destination
// index that does not name the outer variable is written by every iteration, so bands would race
// on it. The fixture keeps the nest, the allocation and the helper, and changes only the index.
func TestDetectPS3034_SilentOnCrossIterationAccumulator(t *testing.T) {
	src := `package p
` + fanoutHelperSrc + `
func colSums(a []float64, m, k, n int) []float64 {
	c := make([]float64, n)
	for i := range m {
		for p := range k {
			for j := range n {
				c[j] += a[i*k+p] * float64(j)
			}
		}
	}
	return c
}`
	if fs := scanSrcFanout(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — every iteration writes the same slots:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3034_SilentWithoutAFanoutHelper pins that the seam must already exist. Advising a
// package to add its first worker helper is a design change, not a loop rewrite, and it carries
// questions this scanner cannot answer — where the threshold goes, who owns the pool. The fixture
// is the positive with the helper removed.
func TestDetectPS3034_SilentWithoutAFanoutHelper(t *testing.T) {
	src := `package p

func matmulFlat(a, b []float64, m, k, n int) []float64 {
	c := make([]float64, m*n)
	for i := range m {
		ci := c[i*n : i*n+n]
		for p := range k {
			av := a[i*k+p]
			bp := b[p*n : p*n+n]
			for j := range ci {
				ci[j] += av * bp[j]
			}
		}
	}
	return c
}`
	if fs := scanSrcFanout(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the package has no fan-out seam:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3034_SilentOnTwoDeepNest pins the depth. A two-deep fill is usually bandwidth-bound
// and the fork/join can cost more than the loop; three deep is where the arithmetic per element
// pays for the split. The fixture is a plain transpose — independent, allocated here, and
// deliberately not reported.
func TestDetectPS3034_SilentOnTwoDeepNest(t *testing.T) {
	src := `package p
` + fanoutHelperSrc + `
func transposeFlat(x []float64, r, c int) []float64 {
	out := make([]float64, r*c)
	for i := range r {
		for j := range c {
			out[j*r+i] = x[i*c+j]
		}
	}
	return out
}`
	if fs := scanSrcFanout(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — two deep is not worth a fork/join:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3034_ReportsThreeIndexLoopForm pins that the C-style loop counts too. Restricting
// the outer loop to a range statement made the check blind to `for i := 0; i < m; i++`, which is
// how a large share of this tree's numeric kernels are written — and blind checks read as clean
// code. The body is the positive fixture's, in the other loop form.
func TestDetectPS3034_ReportsThreeIndexLoopForm(t *testing.T) {
	src := `package p
` + fanoutHelperSrc + `
func matmulFlat(a, b []float64, m, k, n int) []float64 {
	c := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for p := 0; p < k; p++ {
			av := a[i*k+p]
			for j := 0; j < n; j++ {
				c[i*n+j] += av * b[p*n+j]
			}
		}
	}
	return c
}`
	if fs := scanSrcFanout(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — a three-index loop is the same nest", len(fs))
	}
}

// TestDetectPS3034_SilentOnEliminationIntoLaterColumns is the false positive the tree sweep
// produced, kept as a fixture because it is the worst answer this check can give: a sequential
// elimination reported as parallelizable.
//
// The mask write is indexed by the outer variable, so a test that looks only at the destination
// the function allocated sees an independent nest. The same body then eliminates into columns
// AHEAD of c, on a matrix it did not allocate, which the next iteration of c reads. Splitting the
// outer loop would race on exactly that.
func TestDetectPS3034_SilentOnEliminationIntoLaterColumns(t *testing.T) {
	src := `package p
` + fanoutHelperSrc + `
func prune(wm [][]float64, hinv [][]float64, rows, in int) [][]bool {
	mask := make([][]bool, rows)
	for c := 0; c < in; c++ {
		d := hinv[c][c]
		for r := range rows {
			e := wm[r][c] / d
			mask[r][c] = true
			for j := c + 1; j < in; j++ {
				wm[r][j] -= e * hinv[c][j]
			}
		}
	}
	return mask
}`
	if fs := scanSrcFanout(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the body eliminates into columns a later iteration reads:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3034_ReportsThroughAPerIterationBuffer keeps that widening from silencing everything
// it should not: a buffer the iteration MAKES for itself is invisible to every other iteration, so
// writing through it says nothing about whether the loop can be split. Without this, the natural
// fix to the case above — treat every write as evidence of dependence — would silence any nest
// that uses a scratch row, which is most of them.
func TestDetectPS3034_ReportsThroughAPerIterationBuffer(t *testing.T) {
	src := `package p
` + fanoutHelperSrc + `
func rowNorms(a []float64, m, k, n int) []float64 {
	out := make([]float64, m*n)
	for i := range m {
		tmp := make([]float64, n)
		for p := range k {
			for j := range n {
				tmp[j] += a[i*k+p] * float64(j)
			}
		}
		for j := range n {
			out[i*n+j] = tmp[j]
		}
	}
	return out
}`
	if fs := scanSrcFanout(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — the scratch is made per iteration", len(fs))
	}
}
