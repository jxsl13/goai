package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func innerIndependentFindings(t *testing.T, src string) []finding {
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
		if fnd.category == "inner-independent-under-sequential-outer" {
			out = append(out, fnd)
		}
	}
	return out
}

const parColsHelperSrc = `
func parallelCols(n, workPerItem int, body func(lo, hi int)) {
	body(0, n)
}
`

// TestDetectPS3040_InnerIndependentUnderSequentialOuter is the measured shape: an LU
// factorization's rank-1 update. The pivot loop cannot be split — every step reads the rows the
// previous one wrote — while the row loop under it can, and that one line was 92% of the
// benchmark. Splitting it went -40.8% at 512 wide.
func TestDetectPS3040_InnerIndependentUnderSequentialOuter(t *testing.T) {
	src := `package p
` + parColsHelperSrc + `
func Factor(m []float64, n int) {
	for k := 0; k < n; k++ {
		mk := m[k*n : k*n+n]
		pivot := mk[k]
		for i := k + 1; i < n; i++ {
			mi := m[i*n : i*n+n]
			mult := mi[k] / pivot
			mi[k] = mult
			for j := k + 1; j < n; j++ {
				mi[j] -= mult * mk[j]
			}
		}
	}
}`
	fs := innerIndependentFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Both conversion requirements have to survive, because both were learned the hard way: the
	// work estimate and the duplicated serial body.
	// The last phrase is the calibration a second application produced: banding a serial
	// section that was 40% of a wall clock returned 6.6%, because every step of the outer loop
	// pays its own fork.
	if !containsAll(fs[0].msg, "rows times columns", "PLAIN DUPLICATED LOOP", "one level in",
		"EXPECT LESS THAN THE SERIAL SHARE PROMISES") {
		t.Fatalf("message omits the gate estimate, the closure cost or the advice:\n%s", fs[0].msg)
	}
}

// TestDetectPS3040_SilentOnAppliedForm pins the applied form: the middle loop fans out, with the
// below-gate path kept as a plain duplicated loop.
func TestDetectPS3040_SilentOnAppliedForm(t *testing.T) {
	src := `package p
` + parColsHelperSrc + `
func Factor(m []float64, n int) {
	for k := 0; k < n; k++ {
		mk := m[k*n : k*n+n]
		pivot := mk[k]
		rows := n - k - 1
		if rows*rows < 1<<14 {
			for i := k + 1; i < n; i++ {
				mi := m[i*n : i*n+n]
				mult := mi[k] / pivot
				mi[k] = mult
				for j := k + 1; j < n; j++ {
					mi[j] -= mult * mk[j]
				}
			}
			continue
		}
		parallelCols(rows, rows, func(lo, hi int) {
			for i := k + 1 + lo; i < k+1+hi; i++ {
				mi := m[i*n : i*n+n]
				mult := mi[k] / pivot
				mi[k] = mult
				for j := k + 1; j < n; j++ {
					mi[j] -= mult * mk[j]
				}
			}
		})
	}
}`
	if fs := innerIndependentFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the middle loop already fans out:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3040_SilentWhenTheOuterIsIndependent pins the boundary with PS3034. If nothing in
// the body reads an index carrying the outer variable, the outer loop is not sequential at all and
// the question to ask is whether IT can be split — which is the other check's job. Reporting both
// on one nest would be two contradictory pieces of advice.
func TestDetectPS3040_SilentWhenTheOuterIsIndependent(t *testing.T) {
	src := `package p
` + parColsHelperSrc + `
func gram(a [][]float64, out [][]float64, n int) {
	for i := range n {
		for j := range n {
			var s float64
			for k := range n {
				s += a[j][k] * a[j][k]
			}
			out[i][j] = s
		}
	}
}`
	if fs := innerIndependentFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the outer loop carries no dependence:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3040_SilentOnACrossIterationWrite pins the ownership test: a write that does not
// belong to the middle iteration is shared between them, and splitting the middle loop would race.
func TestDetectPS3040_SilentOnACrossIterationWrite(t *testing.T) {
	src := `package p
` + parColsHelperSrc + `
func colSums(m []float64, acc []float64, n int) {
	for k := 0; k < n; k++ {
		for i := k + 1; i < n; i++ {
			mult := m[i*n+k] // reads the outer variable: the dependence is real
			for j := k + 1; j < n; j++ {
				acc[j] += mult * m[k*n+j]
			}
		}
	}
}`
	if fs := innerIndependentFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the accumulator is shared across middle iterations:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3040_SilentWithoutAFanoutHelper pins that the package must already have the seam,
// for the same reason PS3034 does: advising a package to add its first worker helper is a design
// change this scanner cannot stand behind.
func TestDetectPS3040_SilentWithoutAFanoutHelper(t *testing.T) {
	src := `package p

func Factor(m []float64, n int) {
	for k := 0; k < n; k++ {
		mk := m[k*n : k*n+n]
		pivot := mk[k]
		for i := k + 1; i < n; i++ {
			mi := m[i*n : i*n+n]
			mult := mi[k] / pivot
			mi[k] = mult
			for j := k + 1; j < n; j++ {
				mi[j] -= mult * mk[j]
			}
		}
	}
}`
	if fs := innerIndependentFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the package has no fan-out seam:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3040_SilentWhenTheMiddleLoopFansOutInside pins the last silence. A middle loop that
// itself dispatches — its own body is the fan-out — is already converted, just at a finer grain
// than this check would advise, and telling it to split again is noise.
//
// It is silenced by the DEPTH requirement rather than by a rule of its own: the middle loop body
// holds a call where its inner loop would be, so the nest is two deep, not three. A separate
// already-fans-out guard was written, found to redden nothing this does not, and removed.
func TestDetectPS3040_SilentWhenTheMiddleLoopFansOutInside(t *testing.T) {
	src := `package p
` + parColsHelperSrc + `
func eliminate(m []float64, n int) {
	for k := 0; k < n; k++ {
		mk := m[k*n : k*n+n]
		for i := k + 1; i < n; i++ {
			mi := m[i*n : i*n+n]
			mult := mi[k]
			parallelCols(n-k-1, 1, func(lo, hi int) {
				for j := k + 1 + lo; j < k+1+hi; j++ {
					mi[j] -= mult * mk[j]
				}
			})
		}
	}
}`
	if fs := innerIndependentFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the middle loop's body is already the fan-out:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3040_SilentOnItsOwnBelowGateArm pins the false positive this check produced on
// the very site it had just been used to fix. Applying it leaves a below-gate arm that is a
// PLAIN DUPLICATED LOOP — this check's own advice, because routing small inputs through the
// callback costs a few percent — and that duplicate is exactly the shape it looks for. It is a
// SIBLING of the gated dispatch, not inside it, so neither the depth test nor the in-callback
// test sees it.
func TestDetectPS3040_SilentOnItsOwnBelowGateArm(t *testing.T) {
	src := `package p

func parallelRows(n, work int, body func(lo, hi int)) { body(0, n) }

func solve(aug []float64, n, stride int) {
	for c := 0; c < n; c++ {
		crow := aug[c*stride : c*stride+stride]
		elim := func(rlo, rhi int) {
			for r := rlo; r < rhi; r++ {
				if r == c {
					continue
				}
				f := aug[r*stride+c]
				rrow := aug[r*stride : r*stride+stride]
				for j := c; j < stride; j++ {
					rrow[j] -= f * crow[j]
				}
			}
		}
		if n*(stride-c) >= 1<<14 {
			parallelRows(n, stride-c, elim)
			continue
		}
		for r := 0; r < n; r++ {
			if r == c {
				continue
			}
			f := aug[r*stride+c]
			rrow := aug[r*stride : r*stride+stride]
			for j := c; j < stride; j++ {
				rrow[j] -= f * crow[j]
			}
		}
	}
}`
	if fs := innerIndependentFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the split is already here and this is its below-gate"+
			" twin:\n%s", len(fs), fs[0].msg)
	}
}
