package main

import (
	"go/parser"
	"go/token"
	"testing"
)

func sharedAccumulatorFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "inner-loop-accumulates-into-a-shared-buffer" {
			out = append(out, fnd)
		}
	}
	return out
}

// sharedAccFixture is the measured shape: a weighted sum over keys, where every key loads and
// stores the whole output accumulator to add its one term.
const sharedAccFixture = `package p

func weightedSum(obuf, row []float64, vs []float64, sk, dk, sum int) {
	for j := 0; j < sk; j++ {
		w := row[j] / float64(sum)
		vrow := vs[j*dk : j*dk+dk]
		for d, vv := range vrow {
			obuf[d] += w * vv
		}
	}
}`

func TestDetectPS3075_SharedAccumulator(t *testing.T) {
	fs := sharedAccumulatorFindingsIn(t, sharedAccFixture)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The transform, the ordering condition it rests on, and the gate that can actually see it.
	if !containsAll(fs[0].msg, "JAM THE ITEM LOOP", "same ascending item order",
		"check the gate measures bit identity") {
		t.Fatalf("message omits the transform, the ordering rule or the gate warning:\n%s",
			fs[0].msg)
	}
}

// TestDetectPS3075_SilentOnAPerItemAccumulator pins what makes the round trip wasteful. A buffer
// cut from the item is written once and never revisited.
func TestDetectPS3075_SilentOnAPerItemAccumulator(t *testing.T) {
	src := replaceOnce(t, sharedAccFixture, `		vrow := vs[j*dk : j*dk+dk]`,
		`		vrow := vs[j*dk : j*dk+dk]
		obuf := vs[j*dk : j*dk+dk]`)
	if fs := sharedAccumulatorFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — that buffer belongs to the item:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3075_SilentWhenTheTermDoesNotVary pins the other half. An addition that is the same
// for every item is loop-invariant and wants hoisting, not jamming.
func TestDetectPS3075_SilentWhenTheTermDoesNotVary(t *testing.T) {
	src := replaceOnce(t, sharedAccFixture, `		for d, vv := range vrow {
			obuf[d] += w * vv
		}`, `		for d, vv := range vs[:dk] {
			obuf[d] += vv
		}
		_ = w
		_ = vrow`)
	if fs := sharedAccumulatorFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — every item adds the same thing:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3075_SilentOnAScalarAccumulator pins the INDEXED form. A scalar reduction is one
// register already; there is no buffer being walked once per item.
func TestDetectPS3075_SilentOnAScalarAccumulator(t *testing.T) {
	src := replaceOnce(t, sharedAccFixture, `			obuf[d] += w * vv`,
		`			obuf[0] += w * vv
			_ = d`)
	if fs := sharedAccumulatorFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — this is not indexed by the inner variable:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3075_SilentWhenAlreadyJammed pins the suppression that stops the check reporting its
// own fix: applying it leaves a by-one tail whose body is the reported shape exactly.
func TestDetectPS3075_SilentWhenAlreadyJammed(t *testing.T) {
	src := replaceOnce(t, sharedAccFixture, `	for j := 0; j < sk; j++ {`,
		`	for jj := 0; jj+3 < sk; jj += 4 {
		for d := 0; d < dk; d++ {
			t := obuf[d]
			t += row[jj] * vs[jj*dk+d]
			t += row[jj+1] * vs[(jj+1)*dk+d]
			obuf[d] = t
		}
	}
	for j := 0; j < sk; j++ {`)
	if fs := sharedAccumulatorFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — this function already folds four items at once:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3075_BasePlusInnerVarIndex pins the index form that a bare-identifier test misses.
// A kernel addressing its output as os[ob+d] rather than slicing a row first is the same shared
// accumulator — and this is not hypothetical: it is the arm that sat beside a reported one in
// the same kernel, unreported, until the index test learned to see a sum.
func TestDetectPS3075_BasePlusInnerVarIndex(t *testing.T) {
	src := replaceOnce(t, sharedAccFixture, `			obuf[d] += w * vv`,
		`			obuf[sk+d] += w * vv`)
	fs := sharedAccumulatorFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — a base plus the inner variable is still that element",
			len(fs))
	}
}

// TestDetectPS3075_SilentOnAStridedIndex pins the other side of that widening. An index that
// MULTIPLIES the inner variable addresses a different element per item, so there is no single
// accumulator element to hold in a register.
func TestDetectPS3075_SilentOnAStridedIndex(t *testing.T) {
	src := replaceOnce(t, sharedAccFixture, `			obuf[d] += w * vv`,
		`			obuf[d*sk] += w * vv`)
	if fs := sharedAccumulatorFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a strided index is not one element:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3075_ThreeClauseInnerLoop pins the form the check could not see until T1183: an
// inner loop written as `for s := 0; s < n; s++` rather than as a range. The AWQ
// reconstruction-error kernel is exactly this shape, its accumulate line was 84% of its
// benchmark's profile, and matching only *ast.RangeStmt walked straight past it — while the
// check's own description already claimed the shape.
func TestDetectPS3075_ThreeClauseInnerLoop(t *testing.T) {
	src := `package p

func reconErr(diff []float64, xf []float64, acc []float64, in, samples int) {
	for i := 0; i < in; i++ {
		di, base := diff[i], i*samples
		for s := 0; s < samples; s++ {
			acc[s] += di * xf[base+s]
		}
	}
}`
	fs := sharedAccumulatorFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — a three-clause inner loop is the same accumulator", len(fs))
	}
	// The association trap and the gate that catches it are the round's whole lesson.
	if !containsAll(fs[0].msg, "HOLD THE ACCUMULATOR IN AN EXPLICIT",
		"GATE IT BY WIDTH INVARIANCE", "bit-identical at EVERY width") {
		t.Fatalf("message omits the association trap or the width-invariance gate:\n%s", fs[0].msg)
	}
}

// TestDetectPS3075_SubtractiveAccumulation pins that -= is the same shape as +=. The QR
// backward's rank-1 update is written with -= and was 50.7% of its benchmark; matching only
// += walked past it.
func TestDetectPS3075_SubtractiveAccumulation(t *testing.T) {
	src := `package p

func rank1(qb, qd, mm [][]float64, m, n int) {
	for k := range m {
		qbk, qdk := qb[k], qd[k]
		for i := range n {
			qbki := qbk[i]
			for j := range n {
				mm[i][j] -= qbki * qdk[j]
			}
		}
	}
}`
	if fs := sharedAccumulatorFindingsIn(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — a subtractive accumulation is the same accumulator", len(fs))
	}
}

// TestDetectPS3075_RowAliasedAccumulator pins the alias resolution. `mmi := mm[i]` is rebound
// on every pass of the item loop while the memory behind it is shared, so a per-item test on
// the row variable alone hides the buffer.
func TestDetectPS3075_RowAliasedAccumulator(t *testing.T) {
	src := `package p

func rank1(qb, qd, mm [][]float64, m, n int) {
	for k := range m {
		qbk, qdk := qb[k], qd[k]
		for i := range n {
			qbki, mmi := qbk[i], mm[i]
			for j := range n {
				mmi[j] -= qbki * qdk[j]
			}
		}
	}
}`
	if fs := sharedAccumulatorFindingsIn(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — the row variable aliases a shared buffer", len(fs))
	}
}

// TestDetectPS3075_SilentOnARowAliasOfTheItem pins the other side of that resolution: when the
// row is taken AT the item index, the buffer really is per item and there is nothing to jam.
func TestDetectPS3075_SilentOnARowAliasOfTheItem(t *testing.T) {
	src := `package p

func perItem(src [][]float64, dst [][]float64, m, n int) {
	for k := range m {
		row := dst[k]
		s := src[k]
		for j := range n {
			row[j] += s[j] * 2
		}
	}
}`
	if fs := sharedAccumulatorFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the row is the item's own output:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3075_SilentOnATwoDimensionalPerItemAccumulator pins the index-crossing rule
// inside the root resolution. Walking mm[i][j] down to mm is only sound while every index
// crossed is independent of the item; dst[k][j] with k the item is the item's OWN row, and
// treating it as shared would report a jam that has nothing to hold.
func TestDetectPS3075_SilentOnATwoDimensionalPerItemAccumulator(t *testing.T) {
	src := `package p

func perItemRow(src, dst [][]float64, m, n int) {
	for k := range m {
		s := src[k]
		for j := range n {
			dst[k][j] += s[j] * 2
		}
	}
}`
	if fs := sharedAccumulatorFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — dst[k] is the item's own row:\n%s", len(fs), fs[0].msg)
	}
}
