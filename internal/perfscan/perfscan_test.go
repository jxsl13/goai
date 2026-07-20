package main

import (
	"go/parser"
	"go/token"
	"testing"
)

// scanSrc parses one in-memory source file and returns the findings.
func scanSrc(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return scanFile(fset, f)
}

// countCat tallies findings by category.
func countCat(fs []finding) map[string]int {
	m := map[string]int{}
	for _, f := range fs {
		m[f.category]++
	}
	return m
}

// Detector A fires on a per-element Unravel+SetF64 walk with no fast path…
func TestDetectA_UnravelSetNoFastPath(t *testing.T) {
	src := `package p
func Step(p, g *T) {
	n := p.Numel()
	for i := 0; i < n; i++ {
		idx := Unravel(i, p.Shape())
		g.SetF64(g.AtF64(idx...)*2, idx...)
	}
}`
	got := countCat(scanSrc(t, src))
	if got["per-element-dispatch"] != 1 {
		t.Fatalf("want 1 per-element-dispatch, got %d (%v)", got["per-element-dispatch"], got)
	}
}

// …and stays silent once the enclosing function has a flatF64/flatF32 fast path,
// even though the generic fallback loop is still present (the real shape of every
// optimizer/dropout fix in this repo).
func TestDetectA_SilentWithFastPath(t *testing.T) {
	src := `package p
func Step(p, g *T) {
	if pf := flatF64(p); pf != nil {
		for i := range pf {
			pf[i] *= 2
		}
		return
	}
	n := p.Numel()
	for i := 0; i < n; i++ {
		idx := Unravel(i, p.Shape())
		p.SetF64(p.AtF64(idx...)*2, idx...)
	}
}`
	if got := countCat(scanSrc(t, src)); got["per-element-dispatch"] != 0 {
		t.Fatalf("fast path present, want 0 per-element-dispatch, got %d", got["per-element-dispatch"])
	}
}

// The older typed-fast-path idiom — a Storage().F64()/.F32() bulk slice grab with a
// per-element Unravel loop kept only as the fallback (fillSigmoidFocalConstants) —
// must silence detector A just like flatF64 does.
func TestDetectA_SilentWithStorageTypedPath(t *testing.T) {
	src := `package p
func fill(x, out *T) {
	n := x.Numel()
	if x.Dtype() == F64 {
		sd := x.Storage().F64()
		od := out.Storage().F64()
		for i := range n {
			od[i] = sd[i] * 2
		}
		return
	}
	for i := range n {
		c := Unravel(i, x.Shape())
		out.SetF64(x.AtF64(c...)*2, c...)
	}
}`
	if got := countCat(scanSrc(t, src)); got["per-element-dispatch"] != 0 {
		t.Fatalf("Storage().F64() fast path present, want 0 per-element-dispatch, got %d", got["per-element-dispatch"])
	}
}

// A per-ROW loop (SetF64 indexed by the loop var, no Numel bound, no Unravel) is
// NOT a full-tensor walk and must not be flagged — the key false-positive guard.
func TestDetectA_SilentOnPerRow(t *testing.T) {
	src := `package p
func fill(x *T, rows int) {
	for r := 0; r < rows; r++ {
		x.SetF64(1.0, r, 0)
	}
}`
	if got := countCat(scanSrc(t, src)); got["per-element-dispatch"] != 0 {
		t.Fatalf("per-row loop, want 0 per-element-dispatch, got %d", got["per-element-dispatch"])
	}
}

// Detector B fires on an allocation inside a per-element loop (the AMP roundHalf
// shape: a fresh tensor built per element).
func TestDetectB_AllocInLoop(t *testing.T) {
	src := `package p
func sync(w, m *T) {
	for i := range w.Numel() {
		idx := Unravel(i, w.Shape())
		h := FromFloat64(Shape{1}, []float64{m.AtF64(idx...)}).Cast(F16)
		w.SetF64(h.AtF64(0), idx...)
	}
}`
	got := countCat(scanSrc(t, src))
	if got["alloc-in-loop"] == 0 {
		t.Fatalf("want ≥1 alloc-in-loop, got 0 (%v)", got)
	}
}

// Detector C fires on a batch API fed a one-element slice literal inside a loop
// (the forest-predict shape).
func TestDetectC_BatchSingleElt(t *testing.T) {
	src := `package p
func predict(trees []Tree, rows [][]float64) {
	for _, tree := range trees {
		for _, row := range rows {
			_ = tree.Predict([][]float64{row})
		}
	}
}`
	got := countCat(scanSrc(t, src))
	if got["batch-single-elt"] == 0 {
		t.Fatalf("want ≥1 batch-single-elt, got 0 (%v)", got)
	}
}

// A single-element slice literal OUTSIDE any loop is not a hot-path smell.
func TestDetectC_SilentOutsideLoop(t *testing.T) {
	src := `package p
func once(tree Tree, row []float64) {
	_ = tree.Predict([][]float64{row})
}`
	if got := countCat(scanSrc(t, src)); got["batch-single-elt"] != 0 {
		t.Fatalf("outside a loop, want 0 batch-single-elt, got %d", got["batch-single-elt"])
	}
}

// The canonical single-INPUT op call backend.Execute(ctx, op, []*tensor.Tensor{x},
// …) is a flat 1-element pointer slice, NOT a wrapped row — detector C must ignore
// it (the sole false-positive class on the first module-wide run).
func TestDetectC_SilentOnUnaryOpInput(t *testing.T) {
	src := `package p
func slices(x *T, rows []int) {
	for _, b := range rows {
		_, _ = Execute(ctx, OpSlice, []*T{x}, attrs)
	}
}`
	if got := countCat(scanSrc(t, src)); got["batch-single-elt"] != 0 {
		t.Fatalf("unary op input list, want 0 batch-single-elt, got %d", got["batch-single-elt"])
	}
}

// A once-per-parameter allocation sitting ABOVE an inner per-element Unravel loop
// (the TIESMerge/soup shape) must NOT be flagged as alloc-in-loop: the outer loop
// is per-parameter, not per-element, even though a deeper loop walks elements.
func TestDetectB_SilentOnPerParamAllocAboveInnerLoop(t *testing.T) {
	src := `package p
func merge(base []*T) {
	for i := range base {
		b := base[i]
		n := b.Numel()
		res := New(b.Dtype(), b.Shape()) // per-parameter, fine
		for p := range n {
			res.SetF64(b.AtF64(Unravel(p, b.Shape())...), Unravel(p, b.Shape())...)
		}
		_ = res
	}
}`
	got := countCat(scanSrc(t, src))
	if got["alloc-in-loop"] != 0 {
		t.Fatalf("per-parameter alloc above inner loop, want 0 alloc-in-loop, got %d", got["alloc-in-loop"])
	}
	// …but the inner per-element SetF64/AtF64 walk IS still a candidate.
	if got["per-element-dispatch"] != 1 {
		t.Fatalf("inner per-element walk, want 1 per-element-dispatch, got %d", got["per-element-dispatch"])
	}
}

// One loop holding both an AtF64 and a SetF64 is a single candidate, not two.
func TestDedupPerLoop(t *testing.T) {
	src := `package p
func f(x *T) {
	for i := range x.Numel() {
		idx := Unravel(i, x.Shape())
		x.SetF64(x.AtF64(idx...)+1, idx...)
	}
}`
	n := 0
	for _, f := range scanSrc(t, src) {
		if f.category == "per-element-dispatch" {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("want 1 deduped per-element-dispatch, got %d", n)
	}
}

// The tool must not choke on comment/string occurrences of the trigger tokens —
// AST parsing ignores them where an awk scan would false-positive.
func TestNoMatchInCommentsOrStrings(t *testing.T) {
	src := `package p
// this mentions SetF64 and Unravel in a comment
func f() {
	s := "AtF64 Unravel SetF64"
	_ = s
}`
	if got := scanSrc(t, src); len(got) != 0 {
		t.Fatalf("comments/strings must not match, got %v", got)
	}
}
