package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// scanSrc parses one in-memory source file and returns the findings.
// testSets loads the shipped GoAI vocabulary (perfscan.json, next to this test)
// and compiles it, so the domain-check fixtures run with AtF64/flatF64/Numel/… and
// the tests double as a validation that the shipped config parses and activates the
// checks. The engine itself is repo-agnostic; this is the config a project supplies.
func testSets(t *testing.T) nameSets {
	t.Helper()
	c, err := loadConfig("perfscan.json")
	if err != nil {
		t.Fatalf("load perfscan.json: %v", err)
	}
	return c.compile()
}

func scanSrc(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return scanFile(fset, f, testSets(t))
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

// Detector D: a reflection-based fmt scan/format call inside a loop is a candidate
// (the SPM.Decode fmt.Sscanf-per-token class, T931).
func TestDetectD_FmtScanInLoop(t *testing.T) {
	src := `package p
import "fmt"
func Decode(ids []int, pieces []string) {
	for _, id := range ids {
		var v int
		fmt.Sscanf(pieces[id], "<0x%02X>", &v)
	}
}`
	if got := countCat(scanSrc(t, src)); got["reflection-in-loop"] != 1 {
		t.Fatalf("want 1 reflection-in-loop, got %d (%v)", got["reflection-in-loop"], got)
	}
}

// …but the SAME call outside any loop is fine (one-shot parse, not per-element).
func TestDetectD_SilentOutsideLoop(t *testing.T) {
	src := `package p
import "fmt"
func Parse(s string) int {
	var v int
	fmt.Sscanf(s, "<0x%02X>", &v)
	return v
}`
	if got := countCat(scanSrc(t, src)); got["reflection-in-loop"] != 0 {
		t.Fatalf("call outside a loop, want 0 reflection-in-loop, got %d", got["reflection-in-loop"])
	}
}

// …and a same-named method on another type (not the fmt package) must NOT fire —
// the pkg check guards against that false positive.
func TestDetectD_SilentOnNonFmtSscanf(t *testing.T) {
	src := `package p
type parser struct{}
func (parser) Sscanf(s, f string, a ...any) (int, error) { return 0, nil }
func run(ps []parser, ss []string) {
	for i := range ss {
		ps[i].Sscanf(ss[i], "%d")
	}
}`
	if got := countCat(scanSrc(t, src)); got["reflection-in-loop"] != 0 {
		t.Fatalf("non-fmt Sscanf, want 0 reflection-in-loop, got %d", got["reflection-in-loop"])
	}
}

// Detector E: a strings.Builder written in a loop with no Grow (the T929 pre-size class).
func TestDetectE_UnsizedBuilderInLoop(t *testing.T) {
	src := `package p
import "strings"
func Decode(ids []int, pieces []string) string {
	var b strings.Builder
	for _, id := range ids {
		b.WriteString(pieces[id])
	}
	return b.String()
}`
	if got := countCat(scanSrc(t, src)); got["unsized-builder"] != 1 {
		t.Fatalf("want 1 unsized-builder, got %d (%v)", got["unsized-builder"], got)
	}
}

// …silent once the builder is pre-sized with Grow.
func TestDetectE_SilentWithGrow(t *testing.T) {
	src := `package p
import "strings"
func Decode(ids []int, pieces []string) string {
	var b strings.Builder
	b.Grow(len(ids) * 4)
	for _, id := range ids {
		b.WriteString(pieces[id])
	}
	return b.String()
}`
	if got := countCat(scanSrc(t, src)); got["unsized-builder"] != 0 {
		t.Fatalf("Grow present, want 0 unsized-builder, got %d", got["unsized-builder"])
	}
}

// Detector F: an allocating strings transform in a loop (the builder is Grown so only F fires).
func TestDetectF_StringsAllocInLoop(t *testing.T) {
	src := `package p
import "strings"
func Decode(ids []int, pieces []string, meta string) {
	var b strings.Builder
	b.Grow(8)
	for _, id := range ids {
		b.WriteString(strings.ReplaceAll(pieces[id], meta, " "))
	}
}`
	if got := countCat(scanSrc(t, src)); got["strings-alloc-in-loop"] != 1 {
		t.Fatalf("want 1 strings-alloc-in-loop, got %d (%v)", got["strings-alloc-in-loop"], got)
	}
}

// Detector G: a per-element little-endian bit decode in a loop with no rawCopyLE fast path.
func TestDetectG_LEDecodeInLoop(t *testing.T) {
	src := `package p
import "encoding/binary"
func read(raw []byte, dst []uint32) {
	for i := range dst {
		dst[i] = binary.LittleEndian.Uint32(raw[i*4:])
	}
}`
	if got := countCat(scanSrc(t, src)); got["le-decode-in-loop"] != 1 {
		t.Fatalf("want 1 le-decode-in-loop, got %d (%v)", got["le-decode-in-loop"], got)
	}
}

// …silent when the function has a rawCopyLE fast path (the loop is the big-endian fallback).
func TestDetectG_SilentWithRawCopyLE(t *testing.T) {
	src := `package p
import "encoding/binary"
func read(raw []byte, dst []uint32) bool {
	if rawCopyLE(dst, raw, 4) {
		return true
	}
	for i := range dst {
		dst[i] = binary.LittleEndian.Uint32(raw[i*4:])
	}
	return false
}`
	if got := countCat(scanSrc(t, src)); got["le-decode-in-loop"] != 0 {
		t.Fatalf("rawCopyLE present, want 0 le-decode-in-loop, got %d", got["le-decode-in-loop"])
	}
}

// Detector H: a regexp compile inside a loop recompiles the same pattern every iteration.
func TestDetectH_RegexpCompileInLoop(t *testing.T) {
	src := `package p
import "regexp"
func run(pats []string) {
	for _, p := range pats {
		_ = regexp.MustCompile(p)
	}
}`
	if got := countCat(scanSrc(t, src)); got["regexp-compile-in-loop"] != 1 {
		t.Fatalf("want 1 regexp-compile-in-loop, got %d (%v)", got["regexp-compile-in-loop"], got)
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

// moduleRoot walks up from the test's cwd to the directory holding go.mod.
func moduleRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("go.mod not found walking up from cwd")
		}
		dir = parent
	}
}

// TestScanWholeModule is the always-run meta-test (§T893): it exercises the tool
// over EVERY first-party .go file — a surface the package's import closure can
// never reach — so a detector that panics or a parser regression on some real
// construct fails CI on any push, not just one that touches internal/perfscan.
// It is deliberately ADVISORY: it asserts the scan COMPLETES cleanly and finds a
// non-empty candidate set (proving the detectors still fire on real code), NOT a
// fixed count — candidate counts move as the tree is optimized (§C3), and pinning
// them would turn an advisory tool into a brittle gate.
func TestScanWholeModule(t *testing.T) {
	root := moduleRoot(t)
	files, err := goFiles([]string{root + "/..."}, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		t.Fatal("no .go files discovered under module root")
	}
	fset := token.NewFileSet()
	total := 0
	for _, path := range files {
		f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if err != nil {
			continue // a first-party parse error is not perfscan's concern (main skips too)
		}
		total += len(scanFile(fset, f, testSets(t))) // must not panic on any real file
	}
	if total == 0 {
		t.Error("expected perfscan to surface candidates on the real tree; detectors may have silently stopped matching")
	}
	t.Logf("perfscan scanned %d files, %d candidates", len(files), total)
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

// Detector K fires on a scalar transcendental in a loop beside a vectorized
// v*F32/F64 sibling (the F64 SiLU signature)…
func TestDetectK_ScalarTranscendentalWithVectorSibling(t *testing.T) {
	src := `package p
import "math"
func vsiluF32(a, b []float32) {}
func silu(o, d []float64, f32 bool) {
	if f32 { vsiluF32(nil, nil); return }
	for i := range o { o[i] = d[i] / (1 + math.Exp(-d[i])) }
}`
	if got := countCat(scanSrc(t, src)); got["scalar-transcendental-vectorizable"] != 1 {
		t.Fatalf("want 1 scalar-transcendental-vectorizable, got %d (%v)", got["scalar-transcendental-vectorizable"], got)
	}
}

// …and stays silent when there is no vectorized sibling (nothing proves the SIMD
// path exists) or the transcendental is outside a loop.
func TestDetectK_SilentWithoutSiblingOrLoop(t *testing.T) {
	noSibling := `package p
import "math"
func silu(o, d []float64) {
	for i := range o { o[i] = d[i] / (1 + math.Exp(-d[i])) }
}`
	if got := countCat(scanSrc(t, noSibling)); got["scalar-transcendental-vectorizable"] != 0 {
		t.Fatalf("no-sibling: want 0, got %d", got["scalar-transcendental-vectorizable"])
	}
	notInLoop := `package p
import "math"
func vexpF32(a []float32) {}
func f(x float64) float64 { vexpF32(nil); return math.Exp(x) }`
	if got := countCat(scanSrc(t, notInLoop)); got["scalar-transcendental-vectorizable"] != 0 {
		t.Fatalf("not-in-loop: want 0, got %d", got["scalar-transcendental-vectorizable"])
	}
}

// Detector L fires on a loop that calls a package-local elementwise helper WRAPPING
// a transcendental (the softplus/mixer case class K misses because the math.X is one
// call deep). This is the mamba2.go mixer softplus that stayed scalar after the
// OpSoftplus kernel landed.
func TestDetectL_TranscendentalWrapperInLoop(t *testing.T) {
	src := `package p
import "math"
func softplus(x float64) float64 {
	if x > 0 { return x + math.Log1p(math.Exp(-x)) }
	return math.Log1p(math.Exp(x))
}
func mixer(dt []float64, out []float64) {
	for t := range dt { out[t] = softplus(dt[t]) }
}`
	if got := countCat(scanSrc(t, src)); got["transcendental-wrapper-in-loop"] != 1 {
		t.Fatalf("want 1 transcendental-wrapper-in-loop, got %d (%v)", got["transcendental-wrapper-in-loop"], got)
	}
}

// …and stays silent when the wrapper is called OUTSIDE a loop, when the helper is
// not scalar float→float (takes a slice → already batched), and when the local func
// contains no transcendental at all.
func TestDetectL_Silent(t *testing.T) {
	outsideLoop := `package p
import "math"
func softplus(x float64) float64 { return math.Log1p(math.Exp(x)) }
func f(x float64) float64 { return softplus(x) }`
	if got := countCat(scanSrc(t, outsideLoop)); got["transcendental-wrapper-in-loop"] != 0 {
		t.Fatalf("outside-loop: want 0, got %d", got["transcendental-wrapper-in-loop"])
	}
	batched := `package p
import "math"
func softplusVec(x []float64) { for i := range x { x[i] = math.Log1p(math.Exp(x[i])) } }
func f(x []float64) { for range x { softplusVec(x) } }`
	if got := countCat(scanSrc(t, batched)); got["transcendental-wrapper-in-loop"] != 0 {
		t.Fatalf("batched-helper (slice arg): want 0, got %d", got["transcendental-wrapper-in-loop"])
	}
	noTranscendental := `package p
func scale(x float64) float64 { return x * 0.5 }
func f(x []float64) { for i := range x { x[i] = scale(x[i]) } }`
	if got := countCat(scanSrc(t, noTranscendental)); got["transcendental-wrapper-in-loop"] != 0 {
		t.Fatalf("no-transcendental: want 0, got %d", got["transcendental-wrapper-in-loop"])
	}
}

// Detector I fires on a per-element visitor fed a closure.
func TestDetectI_PerElementClosure(t *testing.T) {
	src := `package p
func avg(dst *T) {
	fillGen(dst, func(i int) float64 { return float64(i) * 0.5 })
}`
	if got := countCat(scanSrc(t, src)); got["per-element-closure"] != 1 {
		t.Fatalf("want 1 per-element-closure, got %d (%v)", got["per-element-closure"], got)
	}
}

// Detector J fires on a package-qualified sort with a comparator, and does NOT
// false-match a non-sort homonym like ops.Slice.
func TestDetectJ_ClosureSortQualified(t *testing.T) {
	fires := `package p
import "sort"
func rank(idx []int, s []float64) {
	sort.SliceStable(idx, func(a, b int) bool { return s[idx[a]] < s[idx[b]] })
}`
	if got := countCat(scanSrc(t, fires)); got["closure-comparator-sort"] != 1 {
		t.Fatalf("want 1 closure-comparator-sort, got %d (%v)", got["closure-comparator-sort"], got)
	}
	homonym := `package p
func crop(x *T, a, b int) *T { return ops.Slice(x, a, b) }`
	if got := countCat(scanSrc(t, homonym)); got["closure-comparator-sort"] != 0 {
		t.Fatalf("ops.Slice homonym must not fire, got %d", got["closure-comparator-sort"])
	}
}

// An inline //perfscan:ignore naming a class silences ONLY that class at the site,
// leaving other detectors live — the staticcheck-style class-granular ignore.
func TestIgnore_ClassGranular(t *testing.T) {
	// two independent findings on the same loop: alloc-in-loop (B) + a per-element
	// closure visitor (I) nearby. Ignoring only B must keep I… but they are on
	// different lines, so test each class on its own site.
	base := `package p
import "math"
func vsiluF32(a, b []float32) {}
func silu(o, d []float64, f32 bool) {
	if f32 { vsiluF32(nil, nil); return }
	for i := range o { o[i] = d[i] / (1 + math.Exp(-d[i])) }
}`
	if got := countCat(scanSrc(t, base))["scalar-transcendental-vectorizable"]; got != 1 {
		t.Fatalf("baseline: want 1, got %d", got)
	}
	// ignore by LETTER on the line above the loop.
	byLetter := `package p
import "math"
func vsiluF32(a, b []float32) {}
func silu(o, d []float64, f32 bool) {
	if f32 { vsiluF32(nil, nil); return }
	//perfscan:ignore K exact-locked reference op
	for i := range o { o[i] = d[i] / (1 + math.Exp(-d[i])) }
}`
	if got := countCat(scanSrc(t, byLetter))["scalar-transcendental-vectorizable"]; got != 0 {
		t.Fatalf("ignore-by-letter: want 0, got %d", got)
	}
	// ignore by CATEGORY string.
	byCat := `package p
import "math"
func vsiluF32(a, b []float32) {}
func silu(o, d []float64, f32 bool) {
	if f32 { vsiluF32(nil, nil); return }
	//perfscan:ignore scalar-transcendental-vectorizable reason
	for i := range o { o[i] = d[i] / (1 + math.Exp(-d[i])) }
}`
	if got := countCat(scanSrc(t, byCat))["scalar-transcendental-vectorizable"]; got != 0 {
		t.Fatalf("ignore-by-category: want 0, got %d", got)
	}
}

// Ignoring an UNRELATED class does not silence a live one (the user's scenario:
// a previously-ignored class must not blind the tool to a new, different pattern).
func TestIgnore_UnrelatedClassDoesNotSilence(t *testing.T) {
	src := `package p
import "math"
func vsiluF32(a, b []float32) {}
func silu(o, d []float64, f32 bool) {
	if f32 { vsiluF32(nil, nil); return }
	//perfscan:ignore alloc-in-loop we accept this one elsewhere
	for i := range o { o[i] = d[i] / (1 + math.Exp(-d[i])) }
}`
	if got := countCat(scanSrc(t, src))["scalar-transcendental-vectorizable"]; got != 1 {
		t.Fatalf("ignoring an unrelated class must leave K live, got %d", got)
	}
}

// A bare //perfscan:ignore (no class) silences all classes at the site — the
// backward-compatible catch-all.
func TestIgnore_BareSilencesAll(t *testing.T) {
	src := `package p
import "math"
func vsiluF32(a, b []float32) {}
func silu(o, d []float64, f32 bool) {
	if f32 { vsiluF32(nil, nil); return }
	//perfscan:ignore
	for i := range o { o[i] = d[i] / (1 + math.Exp(-d[i])) }
}`
	if got := countCat(scanSrc(t, src))["scalar-transcendental-vectorizable"]; got != 0 {
		t.Fatalf("bare ignore should silence all, got %d", got)
	}
}

func TestResolveClass(t *testing.T) {
	if resolveClass("PS4002") != "scalar-transcendental-vectorizable" {
		t.Error("check ID PS4002 should resolve to its category")
	}
	if resolveClass("ps4002") != "scalar-transcendental-vectorizable" {
		t.Error("ID resolution should be case-insensitive")
	}
	if resolveClass("per-element-dispatch") != "per-element-dispatch" {
		t.Error("category should resolve to itself")
	}
	if resolveClass("K") != "" {
		t.Error("the retired single-letter codes must no longer resolve")
	}
	if resolveClass("nonsense") != "" {
		t.Error("unknown token should resolve to empty")
	}
}

// TestCheckRegistry pins the ID scheme: every check has a well-formed PS-prefixed
// 4-digit ID and a category, and both are unique. This is the public contract that
// -checks / -exclude / //perfscan:ignore directives name.
func TestCheckRegistry(t *testing.T) {
	ids, cats := map[string]bool{}, map[string]bool{}
	for _, c := range checks {
		if len(c.id) != 6 || c.id[:2] != "PS" {
			t.Errorf("%q: ID must be PS + 4 digits", c.id)
		}
		for _, r := range c.id[2:] {
			if r < '0' || r > '9' {
				t.Errorf("%q: ID suffix must be 4 digits", c.id)
			}
		}
		if ids[c.id] {
			t.Errorf("duplicate ID %q", c.id)
		}
		if cats[c.category] {
			t.Errorf("duplicate category %q", c.category)
		}
		ids[c.id], cats[c.category] = true, true
		if catToID[c.category] != c.id {
			t.Errorf("catToID[%q] = %q, want %q", c.category, catToID[c.category], c.id)
		}
	}
}

// TestConfigGatesDomainChecks is the core repo-agnostic contract: the domain
// checks (here PS1001) are SILENT with no config — perfscan on an arbitrary module
// reports only language/stdlib patterns — and activate only when a project names
// its own vocabulary. The stdlib checks (PS3002 here) fire regardless.
func TestConfigGatesDomainChecks(t *testing.T) {
	src := `package p

import "sort"

type T struct{ xs []float64 }

func (t T) Numel() int          { return len(t.xs) }
func (t T) AtF64(i int) float64 { return t.xs[i] }

func f(a T, ss []int) {
	for i := 0; i < a.Numel(); i++ {
		_ = a.AtF64(i) // per-element dispatch — PS1001, domain
	}
	sort.Slice(ss, func(i, j int) bool { return ss[i] < ss[j] }) // PS3002, stdlib
}
`
	fset := token.NewFileSet()
	af, err := parser.ParseFile(fset, "p.go", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	// Generic mode: empty config ⇒ the domain check is silent, the stdlib one fires.
	generic := countCat(scanFile(fset, af, Config{}.compile()))
	if generic["per-element-dispatch"] != 0 {
		t.Errorf("empty config: PS1001 must be silent (repo-agnostic), got %d", generic["per-element-dispatch"])
	}
	if generic["closure-comparator-sort"] != 1 {
		t.Errorf("empty config: stdlib PS3002 must still fire, got %d", generic["closure-comparator-sort"])
	}
	// Configured: a project names AtF64/Numel ⇒ PS1001 activates.
	cfg := Config{ElementAccessors: []string{"AtF64"}, ElementCountMethods: []string{"Numel"}}
	if got := countCat(scanFile(fset, af, cfg.compile()))["per-element-dispatch"]; got != 1 {
		t.Errorf("configured: want 1 PS1001, got %d", got)
	}
}

// TestRegexpFix exercises the one auto-fix: PS2005 hoists a loop-invariant
// literal-pattern compile, and the suggested edits transform the source into
// valid, re-scan-clean Go.
func TestRegexpFix(t *testing.T) {
	src := `package p

import "regexp"

func F(ss []string) int {
	n := 0
	for _, s := range ss {
		if regexp.MustCompile(` + "`" + `\d+` + "`" + `).MatchString(s) {
			n++
		}
	}
	return n
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "p.go", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	var fx *suggestedFix
	for _, fd := range scanFile(fset, f, testSets(t)) {
		if fd.category == "regexp-compile-in-loop" {
			fx = fd.fix
		}
	}
	if fx == nil {
		t.Fatal("PS2005 produced no fix for a literal pattern")
	}
	// apply the edits (high offset first) to the source.
	type oe struct {
		s, e int
		txt  string
	}
	var edits []oe
	for _, e := range fx.edits {
		edits = append(edits, oe{fset.Position(e.start).Offset, fset.Position(e.end).Offset, e.newText})
	}
	sort.Slice(edits, func(i, j int) bool { return edits[i].s > edits[j].s })
	out := []byte(src)
	for _, e := range edits {
		out = append(out[:e.s], append([]byte(e.txt), out[e.e:]...)...)
	}
	if _, err := parser.ParseFile(token.NewFileSet(), "p.go", out, 0); err != nil {
		t.Fatalf("fixed source does not parse: %v\n%s", err, out)
	}
	if !bytes.Contains(out, []byte("perfscanRe")) || bytes.Count(out, []byte("regexp.MustCompile")) != 1 {
		t.Fatalf("fix did not hoist the compile:\n%s", out)
	}
}

// TestRegexpFixSkipsDynamicPattern: a pattern that is not a plain string literal
// (it may reference the loop variable) gets NO auto-fix — advisory only.
func TestRegexpFixSkipsDynamicPattern(t *testing.T) {
	src := `package p

import (
	"fmt"
	"regexp"
)

func F(ss []string) {
	for i, s := range ss {
		_ = regexp.MustCompile(fmt.Sprintf("a%d", i)).MatchString(s)
	}
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "p.go", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, fd := range scanFile(fset, f, testSets(t)) {
		if fd.category == "regexp-compile-in-loop" {
			found = true
			if fd.fix != nil {
				t.Error("a computed pattern must not be auto-fixed (may depend on the loop var)")
			}
		}
	}
	if !found {
		t.Fatal("PS2005 should still flag the computed pattern (advisory)")
	}
}

// Detector M fires on a READ of an integer-keyed map inside a loop (the map→slice
// candidate: BPE/GGUF Decode, forest votes).
func TestDetectM_IntKeyMapReadInLoop(t *testing.T) {
	src := `package p
type T struct{ decoder map[int]string }
func (t *T) Decode(ids []int) string {
	var b Builder
	for _, id := range ids {
		b.WriteString(t.decoder[id])
	}
	return b.String()
}`
	got := countCat(scanSrc(t, src))
	if got["int-key-map-in-loop"] != 1 {
		t.Fatalf("want 1 int-key-map-in-loop, got %d (%v)", got["int-key-map-in-loop"], got)
	}
}

// …also matches a rune/byte-keyed local map (the u2b byte-inversion), comma-ok reads
// included.
func TestDetectM_RuneKeyCommaOk(t *testing.T) {
	src := `package p
func Invert(s string) []byte {
	u2b := make(map[rune]byte, 256)
	var out []byte
	for _, r := range s {
		if b, ok := u2b[r]; ok {
			out = append(out, b)
		}
	}
	return out
}`
	if got := countCat(scanSrc(t, src))["int-key-map-in-loop"]; got != 1 {
		t.Fatalf("want 1 int-key-map-in-loop, got %d", got)
	}
}

// …stays silent on the things that are NOT dense-slice candidates: a string-keyed map,
// a slice index, a set-like map[int]bool, and a pure map build (m[k] = v).
func TestDetectM_Silent(t *testing.T) {
	cases := map[string]string{
		"string-key": `package p
func f(m map[string]int, ks []string) (s int) {
	for _, k := range ks { s += m[k] }
	return
}`,
		"slice-index": `package p
func f(a []int, ids []int) (s int) {
	for _, id := range ids { s += a[id] }
	return
}`,
		"set-map-bool": `package p
func f(seen map[int]bool, ids []int) (n int) {
	for _, id := range ids { if seen[id] { n++ } }
	return
}`,
		"map-build-write": `package p
func f(ids []int) map[int]int {
	m := make(map[int]int)
	for i, id := range ids { m[id] = i }
	return m
}`,
	}
	for name, src := range cases {
		if got := countCat(scanSrc(t, src))["int-key-map-in-loop"]; got != 0 {
			t.Fatalf("%s: want 0 int-key-map-in-loop, got %d", name, got)
		}
	}
}

// Detector N fires on a slice make() bound to a non-escaping local inside a per-item
// loop of a pointer-receiver method — the optimizer per-Step scratch shape (Adafactor,
// Cautious, LAMB, …) that GC-churns until hoisted to a reused receiver field.
func TestDetectN_PoolableScratch(t *testing.T) {
	src := `package p
type Opt struct{ Params []int }
func (o *Opt) Step() {
	for _, n := range o.Params {
		u := make([]float64, n)
		for i := range u { u[i] = float64(i) }
		_ = u
	}
}`
	if got := countCat(scanSrc(t, src))["poolable-loop-scratch"]; got != 1 {
		t.Fatalf("want 1 poolable-loop-scratch, got %d", got)
	}
}

// It stays silent when the buffer escapes the iteration — returned, stored into a
// receiver field, or stored into a slot (a ring slot) — since those need a different
// fix than a single reused scratch field; and when the function is not a pointer method.
func TestDetectN_Silent(t *testing.T) {
	cases := map[string]string{
		"returned": `package p
type Opt struct{ Params []int }
func (o *Opt) grad(n int) []float64 {
	for _, m := range o.Params {
		u := make([]float64, n+m)
		return u
	}
	return nil
}`,
		"stored-to-field": `package p
type Opt struct{ Params []int; keep [][]float64 }
func (o *Opt) Step() {
	for i, n := range o.Params {
		u := make([]float64, n)
		o.keep[i] = u
	}
}`,
		"stored-to-slot": `package p
type Opt struct{ Params []int; ring [][]float64; pos int }
func (o *Opt) Step() {
	for _, n := range o.Params {
		flat := make([]float64, n)
		o.ring[o.pos] = flat
	}
}`,
		"stored-in-struct-literal": `package p
type layer struct{ a []float64 }
type M struct{ Layers []int; out []layer }
func (m *M) NewState() {
	for l := range m.Layers {
		a := make([]float64, l)
		m.out[l] = layer{a: a}
	}
}`,
		"stored-in-slice-literal": `package p
type M struct{ Layers []int; keep [][][]float64 }
func (m *M) Step() {
	for l := range m.Layers {
		buf := make([]float64, l)
		m.keep[l] = [][]float64{buf}
	}
}`,
		"not-pointer-method": `package p
type Opt struct{ Params []int }
func (o Opt) Step() {
	for _, n := range o.Params {
		u := make([]float64, n)
		_ = u
	}
}`,
		"free-function": `package p
func Step(params []int) {
	for _, n := range params {
		u := make([]float64, n)
		_ = u
	}
}`,
		"not-in-loop": `package p
type Opt struct{ N int }
func (o *Opt) Step() {
	u := make([]float64, o.N)
	_ = u
}`,
	}
	for name, src := range cases {
		if got := countCat(scanSrc(t, src))["poolable-loop-scratch"]; got != 0 {
			t.Fatalf("%s: want 0 poolable-loop-scratch, got %d", name, got)
		}
	}
}

// Detector O fires on a divide by a loop-invariant scalar in an element-wise arithmetic
// loop — the SoftCap VJP / optimizer bias-correction shape a reciprocal-multiply speeds
// up 1.2–1.5×.
func TestDetectO_LoopInvariantDivide(t *testing.T) {
	src := `package p
func softcapVJP(ys, gs, ds []float64, cap float64) {
	for i := range ys {
		t := ys[i] / cap
		ds[i] = gs[i] * (1 - t*t)
	}
}`
	if got := countCat(scanSrc(t, src))["loop-invariant-divide"]; got != 1 {
		t.Fatalf("want 1 loop-invariant-divide, got %d", got)
	}
}

// It stays silent on: a divide by a REDUCTION accumulated in the function (softmax Σ), a
// loop already dominated by a transcendental (the divide is minor there), an integer
// INDEX division, a divisor that VARIES across iterations, and a non-element-wise loop.
func TestDetectO_Silent(t *testing.T) {
	cases := map[string]string{
		"reduction-divisor": `package p
func softmax(x, o []float64) {
	var sum float64
	for i := range x { sum += x[i] }
	for i := range x { o[i] = x[i] / sum }
}`,
		"transcendental-loop": `package p
import "math"
func f(x, o []float64, sum float64) {
	for i := range x { o[i] = math.Exp(x[i]) / sum }
}`,
		"index-division": `package p
func f(x, o []float64, stride int) {
	for i := range o { o[i] = x[i/stride] }
}`,
		"varying-divisor": `package p
func f(x, y, o []float64) {
	for i := range x { o[i] = x[i] / y[i] }
}`,
		"reassigned-divisor": `package p
func f(x, o []float64, d float64) {
	for i := range x { d = d + x[i]; o[i] = x[i] / d }
}`,
		"scalar-not-elementwise": `package p
func f(total, n float64) float64 {
	var acc float64
	for k := 0; k < 10; k++ { acc += total / n }
	return acc
}`,
	}
	for name, src := range cases {
		if got := countCat(scanSrc(t, src))["loop-invariant-divide"]; got != 0 {
			t.Fatalf("%s: want 0 loop-invariant-divide, got %d", name, got)
		}
	}
}

// PS4004: a counted loop whose only data statement copies one element between two
// slices, with no arithmetic on the value — an element-at-a-time memmove.
func TestDetectPS4004_ScalarCopyLoop(t *testing.T) {
	src := `package p
func fill(dst, src []float64, n int, eff []int, shape []int) {
	ioff := 0
	for pos := 0; pos < n; pos++ {
		dst[pos] = src[ioff]
		for d := len(shape) - 1; d >= 0; d-- {
			ioff += eff[d]
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["scalar-copy-loop"] != 1 {
		t.Fatalf("want 1 scalar-copy-loop, got %d (%v)", got["scalar-copy-loop"], got)
	}
}

// …silent on a rank-sized setup loop. Structurally the same assignment, but it
// ranges over a named container (2-4 iterations), so a bulk copy is noise.
func TestDetectPS4004_SilentOnRankLoop(t *testing.T) {
	src := `package p
func plan(eff []int, strides []int, shape []int) {
	for a := range shape {
		eff[a] = strides[a]
	}
}`
	if got := countCat(scanSrc(t, src)); got["scalar-copy-loop"] != 0 {
		t.Fatalf("rank-sized range loop, want 0 scalar-copy-loop, got %d", got["scalar-copy-loop"])
	}
}

// …silent when the value is transformed rather than copied: that is real
// per-element work, not a memmove.
func TestDetectPS4004_SilentOnArithmetic(t *testing.T) {
	src := `package p
func scale(dst, src []float64, n int, k float64) {
	for i := 0; i < n; i++ {
		dst[i] = src[i] * k
	}
}`
	if got := countCat(scanSrc(t, src)); got["scalar-copy-loop"] != 0 {
		t.Fatalf("arithmetic on the value, want 0 scalar-copy-loop, got %d", got["scalar-copy-loop"])
	}
}

// PS5001 stays silent on an INTEGER divide, proved integer by a modulo sibling over
// the same operand pair: Go rejects % on floats, so `r/hw` alongside `r%hw` cannot be
// floating-point, and the rule's `inv := 1/hw` advice would evaluate to integer zero.
func TestDetectPS5001_SilentOnIntegerDivideWithModuloSibling(t *testing.T) {
	src := `package p
func im2col(dst []float32, src []float32, n, hw int) {
	for r := range n {
		ni, rem := r/hw, r%hw
		dst[r] = src[ni] + src[rem]
	}
}`
	if got := countCat(scanSrc(t, src)); got["loop-invariant-divide"] != 0 {
		t.Fatalf("want 0 loop-invariant-divide on an integer index decomposition, got %d (%v)",
			got["loop-invariant-divide"], got)
	}
}

// The parenthesized form must also be recognized: `r / (ho * wo)` paired with
// `r % (ho * wo)` is the same proof, and comparison is structural so parentheses
// do not defeat it.
func TestDetectPS5001_SilentOnParenthesizedIntegerDivide(t *testing.T) {
	src := `package p
func im2col(dst []float32, src []float32, n, ho, wo int) {
	for r := range n {
		ni := r / (ho * wo)
		rem := r % (ho * wo)
		dst[r] = src[ni] + src[rem]
	}
}`
	if got := countCat(scanSrc(t, src)); got["loop-invariant-divide"] != 0 {
		t.Fatalf("want 0 loop-invariant-divide on a parenthesized integer decomposition, got %d (%v)",
			got["loop-invariant-divide"], got)
	}
}

// FLOOR against over-suppression: a genuine float divide has no modulo sibling —
// one cannot exist, since % is illegal on floats — so it must still report.
func TestDetectPS5001_ReportsFloatDivide(t *testing.T) {
	src := `package p
func mean(out []float32, acc []float32, l float32) {
	for d := range out {
		out[d] = acc[d] / l
	}
}`
	if got := countCat(scanSrc(t, src)); got["loop-invariant-divide"] == 0 {
		t.Fatalf("want ≥1 loop-invariant-divide on a float divide, got 0 (%v)", got)
	}
}

// FLOOR for PS4001: the tree currently contains no genuine bulk-copyable decode, so
// the check reports nothing over ./... . This synthetic true positive is what keeps
// it from being silently dead — a rule that cannot fire is worse than a noisy one,
// because nothing distinguishes it from a rule that is merely quiet.
func TestDetectPS4001_ReportsVerbatimSliceDecode(t *testing.T) {
	src := `package p
func load(dst []uint16, src []byte) {
	for i := range dst {
		dst[i] = binary.LittleEndian.Uint16(src[2*i:])
	}
}`
	if got := countCat(scanSrc(t, src)); got["le-decode-in-loop"] == 0 {
		t.Fatalf("want ≥1 le-decode-in-loop on a verbatim slice decode, got 0 (%v)", got)
	}
}

// Silent on a block SCALE read: one decode per block feeding a conversion, with the
// payload decoded by arithmetic. Nothing here is a memmove (gguf dequantQ8_0Into).
func TestDetectPS4001_SilentOnBlockScaleRead(t *testing.T) {
	src := `package p
func dequant(dst []float32, raw []byte) {
	for b := 0; b*32 < len(dst); b++ {
		blk := raw[b*34 : b*34+34]
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		y, q := dst[b*32:b*32+32], blk[2:34]
		for i := range y {
			y[i] = d * float32(int8(q[i]))
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["le-decode-in-loop"] != 0 {
		t.Fatalf("want 0 le-decode-in-loop on a block-scale read, got %d (%v)",
			got["le-decode-in-loop"], got)
	}
}

// Silent on bits that are COMPUTED rather than read: the IQ sign trick and the radix
// key inversion both store verbatim but assemble their bits arithmetically.
func TestDetectPS4001_SilentOnComputedBits(t *testing.T) {
	for _, src := range []string{`package p
func iq(y []float32, grid []float32, db float32, sbit uint32) {
	for k := range y {
		y[k] = math.Float32frombits(math.Float32bits(db*grid[k]) ^ sbit)
	}
}`, `package p
func invert(col []float64, src []uint64) {
	for i, u := range src {
		if u&(1<<63) != 0 {
			u &^= 1 << 63
		} else {
			u = ^u
		}
		col[i] = math.Float64frombits(u)
	}
}`} {
		if got := countCat(scanSrc(t, src)); got["le-decode-in-loop"] != 0 {
			t.Fatalf("want 0 le-decode-in-loop on computed bits, got %d (%v)",
				got["le-decode-in-loop"], got)
		}
	}
}

// PS1001 must see a loop bounded by a DIMENSION, not only one bounded by an element
// count: `d := t.Shape()[1]; for j := range d` walks d elements exactly as
// `for j := range t.Numel()` does. Without this the rule missed backend/ref/dpo.go,
// whose devirtualization measured 1.57x, and nlp/kvevict.go.
func TestDetectPS1001_ShapeBoundedLoop(t *testing.T) {
	src := `package p
func gather(out, t *T, idx []int) {
	d := t.Shape()[1]
	for r, src := range idx {
		for j := range d {
			out.SetF64(t.AtF64(src, j), r, j)
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["per-element-dispatch"] == 0 {
		t.Fatalf("want ≥1 per-element-dispatch on a shape-bounded accessor loop, got 0 (%v)", got)
	}
}

// PS4006 fires on a [][]T built one row per allocation and then indexed two-deep
// inside a nested loop — the shape measured at 1.5x (cholesky) and 1.2x (SymEig).
func TestDetectPS4006_RowSliceMatrix(t *testing.T) {
	src := `package p
func chol(a []float64, n int) {
	l := make([][]float64, n)
	for i := range n {
		l[i] = make([]float64, n)
	}
	for j := range n {
		for i := range n {
			for k := range j {
				l[i][j] -= l[i][k] * l[j][k]
			}
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["row-slice-matrix"] == 0 {
		t.Fatalf("want ≥1 row-slice-matrix, got 0 (%v)", got)
	}
}

// Silent once flattened: a single [rows*cols] buffer has no two-deep index, so
// applying the rule's own advice removes the finding rather than perpetuating it.
func TestDetectPS4006_SilentOnFlatBuffer(t *testing.T) {
	src := `package p
func chol(a []float64, n int) {
	l := make([]float64, n*n)
	for j := range n {
		for i := range n {
			for k := range j {
				l[i*n+j] -= l[i*n+k] * l[j*n+k]
			}
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["row-slice-matrix"] != 0 {
		t.Fatalf("want 0 row-slice-matrix on a flat buffer, got %d (%v)",
			got["row-slice-matrix"], got)
	}
}

// Silent on a ragged structure that is never indexed two-deep in a nested loop —
// a [][]T is only a defect when the row dereference is paid repeatedly.
func TestDetectPS4006_SilentOnShallowUse(t *testing.T) {
	src := `package p
func rows(n int) [][]float64 {
	m := make([][]float64, n)
	for i := range n {
		m[i] = make([]float64, i+1)
	}
	return m
}`
	if got := countCat(scanSrc(t, src)); got["row-slice-matrix"] != 0 {
		t.Fatalf("want 0 row-slice-matrix on shallow use, got %d (%v)",
			got["row-slice-matrix"], got)
	}
}

// PS6004 reports the dual-arm shape: a comma-ok flat-view guard plus a generic
// accessor fallback, which together assert bit-identity between two code paths.
func TestDetectPS6004_DualPath(t *testing.T) {
	src := `package p
func loss(a *T, n int) float64 {
	var total float64
	as, ok := f64Data(a)
	if ok {
		for i := range n {
			total += as[i]
		}
	} else {
		for i := range n {
			total += a.AtF64(i)
		}
	}
	return total
}`
	if got := countCat(scanSrc(t, src)); got["unverified-dual-path"] == 0 {
		t.Fatalf("want ≥1 unverified-dual-path, got 0 (%v)", got)
	}
}

// Silent with no fallback arm: one path cannot disagree with itself, so there is no
// bit-identity claim to verify.
func TestDetectPS6004_SilentWithoutFallback(t *testing.T) {
	src := `package p
func loss(a *T, n int) float64 {
	var total float64
	as, ok := f64Data(a)
	if !ok {
		return 0
	}
	for i := range n {
		total += as[i]
	}
	return total
}`
	if got := countCat(scanSrc(t, src)); got["unverified-dual-path"] != 0 {
		t.Fatalf("want 0 unverified-dual-path without a fallback, got %d (%v)",
			got["unverified-dual-path"], got)
	}
}

// Silent on a plain single-valued storage accessor. .Storage().F64() is also in
// fastPathHelpers but returns one value, so it makes no success-flag claim; without
// this discrimination the rule matched 193 functions instead of 36.
func TestDetectPS6004_SilentOnBareStorageAccess(t *testing.T) {
	src := `package p
func fill(a *T, n int) {
	xs := a.Storage().F64()
	for i := range n {
		xs[i] = a.AtF64(i)
	}
}`
	if got := countCat(scanSrc(t, src)); got["unverified-dual-path"] != 0 {
		t.Fatalf("want 0 unverified-dual-path on bare storage access, got %d (%v)",
			got["unverified-dual-path"], got)
	}
}

// PS6004 also reports the dtype-SWITCH form of the same dual-arm shape, which has
// no comma-ok: `switch x.Dtype() { case F64: <typed>; default: <accessor> }`.
// blas1 uses this form, and omitting it left the known-kernel floor at 6 of 7.
func TestDetectPS6004_DtypeSwitchForm(t *testing.T) {
	src := `package p
func dot(a, b *T, n int) float64 {
	var acc float64
	switch a.Dtype() {
	case F64:
		as := a.Storage().F64()
		for i := range n {
			acc += as[i]
		}
	default:
		for i := range n {
			acc += a.AtF64(i)
		}
	}
	return acc
}`
	if got := countCat(scanSrc(t, src)); got["unverified-dual-path"] == 0 {
		t.Fatalf("want ≥1 unverified-dual-path on the dtype-switch form, got 0 (%v)", got)
	}
}

// Silent when the dtype switch is exhaustive: with no default clause there is no
// fallback arm, so no two paths claim to agree.
func TestDetectPS6004_SilentOnExhaustiveSwitch(t *testing.T) {
	src := `package p
func dot(a *T, n int) float64 {
	var acc float64
	switch a.Dtype() {
	case F64:
		acc = a.AtF64(0)
	case F32:
		acc = a.AtF64(1)
	}
	return acc
}`
	if got := countCat(scanSrc(t, src)); got["unverified-dual-path"] != 0 {
		t.Fatalf("want 0 unverified-dual-path on an exhaustive switch, got %d (%v)",
			got["unverified-dual-path"], got)
	}
}

// PS4005 fires on an N-D odometer ticked once per ELEMENT — the shape whose fix
// measured 4.49x (ref broadcast), 5.29x (cpu broadcast), 3.14x (tensor gather) and
// 1.78x (ref argmax).
func TestDetectPS4005_PerElementOdometer(t *testing.T) {
	src := `package p
func walk(xs []float64, acc []float64, shape []int, eff []int) {
	nd := len(shape)
	idx := make([]int, nd)
	of := 0
	for pos := range xs {
		acc[of] = combine(acc[of], xs[pos])
		for d := nd - 1; d >= 0; d-- {
			idx[d]++
			of += eff[d]
			if idx[d] < shape[d] {
				break
			}
			idx[d] = 0
			of -= eff[d] * shape[d]
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["per-element-odometer"] == 0 {
		t.Fatalf("want ≥1 per-element-odometer, got 0 (%v)", got)
	}
}

// Silent once the innermost axis is HOISTED: the odometer then starts at nd-2 and
// ticks once per run. Applying the rule's own advice must remove the finding — the
// check PS4005 initially failed, reporting the three sites it had just helped fix.
func TestDetectPS4005_SilentOnHoistedOdometer(t *testing.T) {
	src := `package p
func walk(xs []float64, acc []float64, shape []int, eff []int, inner int) {
	nd := len(shape)
	idx := make([]int, nd)
	of := 0
	for pos := 0; pos < len(xs); pos += inner {
		run := xs[pos : pos+inner]
		for j, v := range run {
			acc[of+j] = combine(acc[of+j], v)
		}
		for d := nd - 2; d >= 0; d-- {
			idx[d]++
			of += eff[d]
			if idx[d] < shape[d] {
				break
			}
			idx[d] = 0
			of -= eff[d] * shape[d]
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["per-element-odometer"] != 0 {
		t.Fatalf("want 0 per-element-odometer on a hoisted odometer, got %d (%v)",
			got["per-element-odometer"], got)
	}
}

// Silent on an ordinary descending loop with no indexed tick — a plain reverse walk
// is not an odometer.
func TestDetectPS4005_SilentOnPlainReverseLoop(t *testing.T) {
	src := `package p
func rev(xs []float64, n int) float64 {
	var total float64
	for i := range xs {
		for d := n - 1; d >= 0; d-- {
			total += xs[i]
		}
	}
	return total
}`
	if got := countCat(scanSrc(t, src)); got["per-element-odometer"] != 0 {
		t.Fatalf("want 0 per-element-odometer on a plain reverse loop, got %d (%v)",
			got["per-element-odometer"], got)
	}
}

// TestDetectN_SilentOnPointerSlice: make([]*T, …) is orchestration scaffolding (a handful
// of pointers overwritten before a concat/reduce reads them), not the numeric value
// scratch the growF64 pool win targets — PS2004 must stay silent.
func TestDetectN_SilentOnPointerSlice(t *testing.T) {
	src := `package p
type M struct{ n int }
func (s *M) Step(ps []*int) {
	for _, p := range ps {
		heads := make([]*int, s.n)
		for i := range heads {
			heads[i] = p
		}
		_ = heads
	}
}`
	if got := countCat(scanSrc(t, src))["poolable-loop-scratch"]; got != 0 {
		t.Fatalf("want 0 (pointer-element slice), got %d", got)
	}
}

// TestDetectO_SilentOnModuloIntDivision: a divisor used in `x % d` is provably integer
// (Go's % requires integer operands), so `i/d` is index arithmetic, not a float
// reciprocal-multiply candidate — PS5001 must stay silent even in an element-wise loop.
func TestDetectO_SilentOnModuloIntDivision(t *testing.T) {
	src := `package p
func f(out []int, m int) {
	for i := range out {
		iy, ix := i/m, i%m
		out[i] = iy*m + ix
	}
}`
	if got := countCat(scanSrc(t, src))["loop-invariant-divide"]; got != 0 {
		t.Fatalf("want 0 (integer modulo arithmetic), got %d", got)
	}
}

func TestDetectPS5002_Gram(t *testing.T) {
	src := `package p
func gram(gm [][]float64, l [][]float64, m, n int) {
	for i := range m {
		for j := range m {
			var acc float64
			for k := range n {
				acc += gm[i][k] * gm[j][k]
			}
			l[i][j] += acc
		}
	}
}`
	if got := countCat(scanSrc(t, src))["symmetric-accumulation"]; got != 1 {
		t.Fatalf("want 1 symmetric-accumulation (gram), got %d", got)
	}
}

// Must stay silent on matmul: the factors have DIFFERENT bases, so it is not symmetric.
func TestDetectPS5002_SilentOnMatmul(t *testing.T) {
	src := `package p
func matmul(a, b, c [][]float64, m, n, k int) {
	for i := range m {
		for j := range n {
			var acc float64
			for kk := range k {
				acc += a[i][kk] * b[kk][j]
			}
			c[i][j] += acc
		}
	}
}`
	if got := countCat(scanSrc(t, src))["symmetric-accumulation"]; got != 0 {
		t.Fatalf("want 0 (matmul, different bases), got %d", got)
	}
}

// PS5002 must stay silent on an already-triangular inner loop (Cholesky j<=i): the
// symmetric product is real but the loop covers only one triangle already.
func TestDetectPS5002_SilentOnTriangular(t *testing.T) {
	src := `package p
func chol(a, l [][]float64, n int) {
	for i := range n {
		for j := 0; j <= i; j++ {
			sum := a[i][j]
			for k := range j {
				sum -= l[i][k] * l[j][k]
			}
			l[i][j] = sum
		}
	}
}`
	if got := countCat(scanSrc(t, src))["symmetric-accumulation"]; got != 0 {
		t.Fatalf("want 0 (already triangular), got %d", got)
	}
}

// PS5002: symmetric-matrix full-accumulation (GMM/PCA class).
func TestDetectPS5002_SymmetricOuterProduct(t *testing.T) {
	src := `package p
func cov(x [][]float64, mean []float64, m [][]float64, d int) {
	c := make([]float64, d)
	for _, row := range x {
		for j := range c {
			c[j] = row[j] - mean[j]
		}
		for i := range d {
			for j := range d {
				m[i][j] += c[i] * c[j]
			}
		}
	}
}`
	if got := countCat(scanSrc(t, src))["symmetric-accumulation"]; got != 1 {
		t.Fatalf("want 1 symmetric-accumulation (outer product), got %d", got)
	}
}

// The exact shape that cost Muon 2.09x: a dot product per output element, its
// accumulator a serial FMADD chain. Taken verbatim from nn.matmulABt as it stood
// before the ikj rewrite.
func TestDetectPS4008_SerialDotMatmul(t *testing.T) {
	src := `package p
func matmulABt(a, b []float64, m, k int) []float64 {
	c := make([]float64, m*m)
	for i := range m {
		ai := a[i*k : i*k+k]
		ci := c[i*m : i*m+m]
		for j := range m {
			bj := b[j*k : j*k+k]
			var s float64
			for p := range ai {
				s += ai[p] * bj[p]
			}
			ci[j] = s
		}
	}
	return c
}`
	if got := countCat(scanSrc(t, src)); got["serial-dot-matmul"] == 0 {
		t.Fatalf("want ≥1 serial-dot-matmul, got 0 (%v)", got)
	}
}

// The short-declaration spelling of the same accumulator must be caught too.
func TestDetectPS4008_ShortDeclAccumulator(t *testing.T) {
	src := `package p
func mm(a, b, c []float64, m, k int) {
	for i := range m {
		for j := range m {
			s := 0.0
			for p := range k {
				s += a[i*k+p] * b[j*k+p]
			}
			c[i*m+j] = s
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["serial-dot-matmul"] == 0 {
		t.Fatalf("want ≥1 serial-dot-matmul, got 0 (%v)", got)
	}
}

// SILENT on the ikj/axpy form — applying the rule's own advice must clear the
// finding, or the rule would keep reporting the code it just helped fix.
func TestDetectPS4008_SilentOnIkjAxpy(t *testing.T) {
	src := `package p
func mm(a, bt, c []float64, m, k int) {
	for i := range m {
		ci := c[i*m : i*m+m]
		for p := range k {
			av := a[i*k+p]
			bp := bt[p*m : p*m+m]
			for j := range ci {
				ci[j] += av * bp[j]
			}
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["serial-dot-matmul"] != 0 {
		t.Fatalf("want 0 serial-dot-matmul on the ikj/axpy rewrite, got %d (%v)",
			got["serial-dot-matmul"], got)
	}
}

// SILENT on a plain reduction: a norm accumulates a product of one slice with
// ITSELF and has no indexed store, so there is no output index to make independent
// and nothing to hoist. Flagging it would be a pure false positive.
func TestDetectPS4008_SilentOnReduction(t *testing.T) {
	src := `package p
func norms(x []float64, out []float64, m, k int) {
	for i := range m {
		for j := range m {
			var s float64
			for p := range k {
				s += x[i*k+p] * x[i*k+p]
			}
			_ = s
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["serial-dot-matmul"] != 0 {
		t.Fatalf("want 0 serial-dot-matmul on a same-base reduction with no store, got %d (%v)",
			got["serial-dot-matmul"], got)
	}
}

// SILENT when the inner loop does more than the dot: the accumulation is then not
// the whole cost and the ikj rewrite is not the fix.
func TestDetectPS4008_SilentOnCompoundInnerBody(t *testing.T) {
	src := `package p
func mm(a, b, c, d []float64, m, k int) {
	for i := range m {
		for j := range m {
			var s float64
			for p := range k {
				s += a[i*k+p] * b[j*k+p]
				d[p] = s
			}
			c[i*m+j] = s
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["serial-dot-matmul"] != 0 {
		t.Fatalf("want 0 serial-dot-matmul when the inner loop has extra work, got %d (%v)",
			got["serial-dot-matmul"], got)
	}
}

// A //perfscan:ignore whose explanation WRAPS onto following lines must still
// suppress the statement the comment block documents. Anchoring the directive to its
// own line + 1 made it silently inert — the comment reads as if it took effect while
// the finding is still reported. Two directives in this repo were dead exactly so.
func TestIgnoreDirective_SpansWrappedCommentBlock(t *testing.T) {
	src := `package p
func mm(a, b, c []float64, m, k int) {
	for i := range m {
		for j := range m {
			s := 0.0
			//perfscan:ignore PS4008 deliberate, see below
			// this explanation wraps onto a second line, and onto a third,
			// which used to push the flagged loop out of the directive's reach
			for p := range k {
				s += a[i*k+p] * b[j*k+p]
			}
			c[i*m+j] = s
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["serial-dot-matmul"] != 0 {
		t.Fatalf("wrapped ignore directive did not suppress: got %d (%v)",
			got["serial-dot-matmul"], got)
	}
}

// The directive may also sit at the END of the block, below the prose.
func TestIgnoreDirective_AtEndOfCommentBlock(t *testing.T) {
	src := `package p
func mm(a, b, c []float64, m, k int) {
	for i := range m {
		for j := range m {
			s := 0.0
			// prose first, explaining why this shape is deliberate here
			// and why the rewrite would not pay off
			//perfscan:ignore PS4008 deliberate
			for p := range k {
				s += a[i*k+p] * b[j*k+p]
			}
			c[i*m+j] = s
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["serial-dot-matmul"] != 0 {
		t.Fatalf("trailing ignore directive did not suppress: got %d (%v)",
			got["serial-dot-matmul"], got)
	}
}

// A directive in an UNRELATED comment block must not leak onto a later statement —
// spanning the block must not become "suppress everything after it".
func TestIgnoreDirective_DoesNotLeakPastItsBlock(t *testing.T) {
	src := `package p
func mm(a, b, c []float64, m, k int) {
	// prose block with a directive that belongs to the declaration below it
	//perfscan:ignore PS4008 belongs to the var, not the loop
	var unrelated int
	_ = unrelated
	for i := range m {
		for j := range m {
			s := 0.0
			for p := range k {
				s += a[i*k+p] * b[j*k+p]
			}
			c[i*m+j] = s
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["serial-dot-matmul"] == 0 {
		t.Fatalf("ignore leaked past its comment block and suppressed a later finding (%v)", got)
	}
}

// The exact shape that cost a 500-token GPT decode 2.21 GB: a cache slot reassigned
// to a concat of itself, per layer, per token.
func TestDetectPS2006_QuadraticCacheAppend(t *testing.T) {
	src := `package p
func (g *M) DecodeStep(cache *C, tok int) error {
	for l := range g.Blocks {
		kt, vt := g.proj(l)
		cache.K[l] = concatRows(cache.K[l], kt)
		cache.V[l] = concatRows(cache.V[l], vt)
	}
	return nil
}`
	if got := countCat(scanSrc(t, src)); got["quadratic-cache-append"] != 2 {
		t.Fatalf("want 2 quadratic-cache-append, got %d (%v)", got["quadratic-cache-append"], got)
	}
}

// SILENT once the amortized row buffer is adopted — applying the rule's own advice
// must clear the finding.
func TestDetectPS2006_SilentOnRowBufAppend(t *testing.T) {
	src := `package p
func (g *M) DecodeStep(cache *C, tok int) error {
	for l := range g.Blocks {
		kt, vt := g.proj(l)
		cache.K[l], cache.V[l] = cache.bufs.appendKV(cache.K, cache.V, l, kt, vt)
	}
	return nil
}`
	if got := countCat(scanSrc(t, src)); got["quadratic-cache-append"] != 0 {
		t.Fatalf("want 0 on the row-buffer append, got %d (%v)", got["quadratic-cache-append"], got)
	}
}

// SILENT outside a per-token step function: the same statement in a one-shot builder
// is an ordinary concatenation, and pooling it would be premature.
func TestDetectPS2006_SilentOutsideStepFunction(t *testing.T) {
	src := `package p
func buildOnce(cache *C, parts []T) {
	for l := range parts {
		cache.K[l] = concatRows(cache.K[l], parts[l])
	}
}`
	if got := countCat(scanSrc(t, src)); got["quadratic-cache-append"] != 0 {
		t.Fatalf("want 0 outside a step function, got %d (%v)", got["quadratic-cache-append"], got)
	}
}

// SILENT when the concat's first operand is a DIFFERENT slot: that is not
// accumulate-into-itself and carries no quadratic growth.
func TestDetectPS2006_SilentOnCrossSlotConcat(t *testing.T) {
	src := `package p
func (g *M) DecodeStep(cache *C, tok int) error {
	for l := range g.Blocks {
		cache.K[l] = concatRows(cache.prefix[l], g.row(l))
	}
	return nil
}`
	if got := countCat(scanSrc(t, src)); got["quadratic-cache-append"] != 0 {
		t.Fatalf("want 0 on a cross-slot concat, got %d (%v)", got["quadratic-cache-append"], got)
	}
}

// SILENT outside a loop: a single concat per call is not quadratic.
func TestDetectPS2006_SilentOutsideLoop(t *testing.T) {
	src := `package p
func (g *M) DecodeStep(cache *C, tok int) error {
	cache.K[0] = concatRows(cache.K[0], g.row(0))
	return nil
}`
	if got := countCat(scanSrc(t, src)); got["quadratic-cache-append"] != 0 {
		t.Fatalf("want 0 outside a loop, got %d (%v)", got["quadratic-cache-append"], got)
	}
}

// The exact shape that cost the T5 decoder 86.8 MB per token: build a square
// [pos+1, pos+1] object, then read row pos out of it.
func TestDetectPS2007_BuildSquareUseOneRow(t *testing.T) {
	src := `package p
func (d *D) biasRow(ctx *C, pos int) []float64 {
	full := d.bias.Bias(ctx, pos+1, pos+1)
	kk := pos + 1
	out := make([]float64, kk)
	for k := range kk {
		out[k] = full[pos*kk+k]
	}
	return out
}`
	if got := countCat(scanSrc(t, src)); got["build-nxn-use-one-row"] == 0 {
		t.Fatalf("want ≥1 build-nxn-use-one-row, got 0 (%v)", got)
	}
}

// SILENT once the row is gathered directly — applying the rule's advice must clear it.
func TestDetectPS2007_SilentOnDirectGather(t *testing.T) {
	src := `package p
func (d *D) biasRow(ctx *C, pos int) []float64 {
	kk := pos + 1
	out := make([]float64, kk)
	for k := range kk {
		out[k] = d.table[bucket(k-pos)]
	}
	return out
}`
	if got := countCat(scanSrc(t, src)); got["build-nxn-use-one-row"] != 0 {
		t.Fatalf("want 0 on the direct gather, got %d (%v)", got["build-nxn-use-one-row"], got)
	}
}

// SILENT on a fixed small square: foo(x, 2, 2) is an ordinary shape, not a growing one.
func TestDetectPS2007_SilentOnConstantSquare(t *testing.T) {
	src := `package p
func (d *D) rot(ctx *C, pos int) []float64 {
	m := d.build(ctx, 2, 2)
	return []float64{m[pos]}
}`
	if got := countCat(scanSrc(t, src)); got["build-nxn-use-one-row"] != 0 {
		t.Fatalf("want 0 on a constant-sized square, got %d (%v)", got["build-nxn-use-one-row"], got)
	}
}

// SILENT when the two size arguments DIFFER: a genuinely rectangular build consumed in
// full is not this pattern.
func TestDetectPS2007_SilentOnRectangularBuild(t *testing.T) {
	src := `package p
func (d *D) block(ctx *C, pos, enc int) []float64 {
	full := d.bias.Bias(ctx, pos+1, enc)
	return full[pos:]
}`
	if got := countCat(scanSrc(t, src)); got["build-nxn-use-one-row"] != 0 {
		t.Fatalf("want 0 on a rectangular build, got %d (%v)", got["build-nxn-use-one-row"], got)
	}
}

// SILENT when the square result is consumed WITHOUT indexing by the driving position —
// then it is genuinely used as a matrix and there is nothing to narrow.
func TestDetectPS2007_SilentWhenWholeMatrixUsed(t *testing.T) {
	src := `package p
func (d *D) full(ctx *C, n int) float64 {
	m := d.bias.Bias(ctx, n+1, n+1)
	var s float64
	for _, v := range m {
		s += v
	}
	return s
}`
	if got := countCat(scanSrc(t, src)); got["build-nxn-use-one-row"] != 0 {
		t.Fatalf("want 0 when the whole matrix is consumed, got %d (%v)", got["build-nxn-use-one-row"], got)
	}
}

// The two false positives the first cut of PS2007 produced, kept as fixtures so the
// precision does not regress: an element accessor given the same index twice
// (a.AtF64(j, j) is a diagonal READ, not a square build), and a square result that IS
// consumed whole as an attention mask while the driving position happens to index
// something else nearby.
func TestDetectPS2007_SilentOnDiagonalElementRead(t *testing.T) {
	src := `package p
func chol(a *M, n int) [][]float64 {
	l := make([][]float64, n)
	for j := range n {
		d := a.AtF64(j, j)
		for k := range j {
			d -= l[j][k] * l[j][k]
		}
		l[j][j] = d
	}
	return l
}`
	if got := countCat(scanSrc(t, src)); got["build-nxn-use-one-row"] != 0 {
		t.Fatalf("want 0 on a diagonal element read, got %d (%v)", got["build-nxn-use-one-row"], got)
	}
}

func TestDetectPS2007_SilentWhenSquareResultUsedWhole(t *testing.T) {
	src := `package p
func (d *D) decode(ctx *C, dseq int, toks []int) *T {
	rb := d.bias.Bias(ctx, dseq, dseq)
	rb = rb.Permute(2, 0, 1)
	first := toks[dseq-1]
	_ = first
	return d.attend(rb)
}`
	if got := countCat(scanSrc(t, src)); got["build-nxn-use-one-row"] != 0 {
		t.Fatalf("want 0 when the square result is consumed whole, got %d (%v)",
			got["build-nxn-use-one-row"], got)
	}
}

// The third false positive: the driving identifier is a LOOP BOUND, so the square
// result is walked in full and the identifier appears in those indices only as a
// stride. That is the T5 full-Decode mask, not the per-token row read.
func TestDetectPS2007_SilentWhenPositionIsALoopBound(t *testing.T) {
	src := `package p
func (d *D) decode(ctx *C, dseq, heads int) []float64 {
	rb := d.bias.Bias(ctx, dseq, dseq)
	sm := rb.Storage()
	for h := range heads {
		for i := 0; i < dseq; i++ {
			for j := i + 1; j < dseq; j++ {
				sm[(h*dseq+i)*dseq+j] = -1
			}
		}
	}
	return sm
}`
	if got := countCat(scanSrc(t, src)); got["build-nxn-use-one-row"] != 0 {
		t.Fatalf("want 0 when the position is a loop bound, got %d (%v)",
			got["build-nxn-use-one-row"], got)
	}
}

// The third PS6004 form: a typed fast path that discriminates by comma-ok TYPE
// ASSERTION on concrete storage and DECLINES to its caller with `return false`. The
// generic arm lives in the caller, so neither the fast-path-helper test nor the
// same-function accessor test can see it — this shape shipped as a 3.19x dual path
// while PS6004 reported nothing.
func TestDetectPS6004_DecliningTypedFastPath(t *testing.T) {
	src := `package p
func gatherHalfTyped(out, t *T, n int) bool {
	su, okS := t.storage.data.([]uint16)
	df, okD := out.storage.data.([]float32)
	if !okS || !okD {
		return false
	}
	for i := range n {
		df[i] = float32(f16ToF32(su[i]))
	}
	return true
}`
	if got := countCat(scanSrc(t, src)); got["unverified-dual-path"] == 0 {
		t.Fatalf("want ≥1 unverified-dual-path on a declining typed fast path, got 0 (%v)", got)
	}
}

// THE FALSE POSITIVE THAT NEARLY SHIPPED: an AST visitor is wall-to-wall
// `x, ok := n.(*ast.Foo)` followed by `return false`, structurally identical to a
// devirtualized kernel and semantically nothing like one. The first cut of this
// widening fired on 13 such functions inside perfscan itself. Asserting a SLICE OF A
// NUMERIC TYPE is the discriminator; asserting a pointer-to-struct is not.
func TestDetectPS6004_SilentOnPointerTypeAssertions(t *testing.T) {
	src := `package p
func visit(n Node) bool {
	id, ok := n.(*Ident)
	if !ok {
		return false
	}
	call, ok2 := n.(*CallExpr)
	if !ok2 {
		return false
	}
	_, _ = id, call
	return true
}`
	if got := countCat(scanSrc(t, src)); got["unverified-dual-path"] != 0 {
		t.Fatalf("want 0 on pointer-to-struct assertions (an AST walker), got %d (%v)",
			got["unverified-dual-path"], got)
	}
}

// SILENT on a single numeric assertion: one cast is an ordinary conversion, not a
// dispatch between arms, so there is no second path whose bit-identity is in question.
func TestDetectPS6004_SilentOnSingleTypedAssertion(t *testing.T) {
	src := `package p
func fill(s *S, n int) bool {
	d, ok := s.data.([]float64)
	if !ok {
		return false
	}
	for i := range n {
		d[i] = 0
	}
	return true
}`
	if got := countCat(scanSrc(t, src)); got["unverified-dual-path"] != 0 {
		t.Fatalf("want 0 on a single typed assertion, got %d (%v)",
			got["unverified-dual-path"], got)
	}
}

// SILENT without the decline: a function that asserts typed slices but never returns
// false has no fallback arm to disagree with, so nothing needs cross-referencing.
func TestDetectPS6004_SilentWithoutDecline(t *testing.T) {
	src := `package p
func both(s *S, d *S, n int) bool {
	a, _ := s.data.([]float32)
	b, _ := d.data.([]float64)
	for i := range n {
		b[i] = float64(a[i])
	}
	return true
}`
	if got := countCat(scanSrc(t, src)); got["unverified-dual-path"] != 0 {
		t.Fatalf("want 0 without a decline path, got %d (%v)",
			got["unverified-dual-path"], got)
	}
}

// The shape behind three measured wins: a comparator dereferencing the sorted index
// into a 2-D structure on every comparison.
func TestDetectPS3005_IndirectKeyComparator(t *testing.T) {
	src := `package p
func presort(x [][]float64, col []int, ff int) {
	sort.Slice(col, func(a, c int) bool { return x[col[a]][ff] < x[col[c]][ff] })
}`
	if got := countCat(scanSrc(t, src)); got["indirect-key-comparator"] == 0 {
		t.Fatalf("want ≥1 indirect-key-comparator, got 0 (%v)", got)
	}
}

// SliceStable and a > comparator are the same shape.
func TestDetectPS3005_StableAndDescending(t *testing.T) {
	src := `package p
func route(scores [][]float64, idx []int, ex int) {
	sort.SliceStable(idx, func(a, b int) bool { return scores[idx[a]][ex] > scores[idx[b]][ex] })
}`
	if got := countCat(scanSrc(t, src)); got["indirect-key-comparator"] == 0 {
		t.Fatalf("want ≥1 on SliceStable/descending, got 0 (%v)", got)
	}
}

// SILENT on the FIXED form — a flat id-indexed key. Applying the rule's own advice must
// clear the finding, or it would keep reporting the code it just helped fix.
func TestDetectPS3005_SilentOnHoistedFlatKey(t *testing.T) {
	src := `package p
func presort(x [][]float64, col []int, key []float64, ff int) {
	for i := range col {
		key[i] = x[i][ff]
	}
	sort.Slice(col, func(a, c int) bool { return key[col[a]] < key[col[c]] })
}`
	if got := countCat(scanSrc(t, src)); got["indirect-key-comparator"] != 0 {
		t.Fatalf("want 0 on the hoisted flat key, got %d (%v)", got["indirect-key-comparator"], got)
	}
}

// SILENT when the comparator reads the sorted elements directly rather than using them
// as indices into something else — there is no indirection to hoist.
func TestDetectPS3005_SilentOnDirectValueSort(t *testing.T) {
	src := `package p
func byDist(cand []nb) {
	sort.Slice(cand, func(a, b int) bool { return cand[a].dist < cand[b].dist })
}`
	if got := countCat(scanSrc(t, src)); got["indirect-key-comparator"] != 0 {
		t.Fatalf("want 0 on a direct value sort, got %d (%v)", got["indirect-key-comparator"], got)
	}
}

// SILENT on a single-level lookup: key[idx[a]] is already the hoisted shape, so only a
// TWO-level dereference through the sorted slice counts.
func TestDetectPS3005_SilentOnSingleLevelLookup(t *testing.T) {
	src := `package p
func byKey(idx []int, key []float64) {
	sort.Slice(idx, func(a, b int) bool { return key[idx[a]] < key[idx[b]] })
}`
	if got := countCat(scanSrc(t, src)); got["indirect-key-comparator"] != 0 {
		t.Fatalf("want 0 on a single-level lookup, got %d (%v)", got["indirect-key-comparator"], got)
	}
}

// SILENT when the two-level dereference goes through a DIFFERENT slice than the one
// being sorted: m[other[a]][f] while sorting idx is not "read this element's key", so
// hoisting a key column indexed by the sorted element would not even be well-defined.
// Without this case the sorted-slice identity check is never exercised — the
// single-level fixture above never reaches it.
func TestDetectPS3005_SilentWhenIndexedThroughAnotherSlice(t *testing.T) {
	src := `package p
func weird(m [][]float64, idx, other []int, f int) {
	sort.Slice(idx, func(a, b int) bool { return m[other[a]][f] < m[other[b]][f] })
}`
	if got := countCat(scanSrc(t, src)); got["indirect-key-comparator"] != 0 {
		t.Fatalf("want 0 when indexed through a different slice, got %d (%v)",
			got["indirect-key-comparator"], got)
	}
}

// SILENT when the outer lookup is not itself indexed — m[idx[a]] alone is a single
// dereference, the cheap case the hoist would not help.
func TestDetectPS3005_SilentOnSingleDereference(t *testing.T) {
	src := `package p
func byRow(m []float64, idx []int) {
	sort.Slice(idx, func(a, b int) bool { return m[idx[a]] < m[idx[b]] })
}`
	if got := countCat(scanSrc(t, src)); got["indirect-key-comparator"] != 0 {
		t.Fatalf("want 0 on a single dereference, got %d (%v)", got["indirect-key-comparator"], got)
	}
}

// PS4006 must be suppressible by a directive written ABOVE THE LOOP, which is where a
// reader puts one. It was not: the finding anchored to the index expression a line
// inside the loop, so the directive's block (which covers itself plus the next
// statement) never reached it. A BARE //perfscan:ignore failing to silence the finding
// is what exposed it — the ID and category were never the problem.
func TestDetectPS4006_SuppressibleAboveTheLoop(t *testing.T) {
	body := `package p
func chol(a []float64, n int) {
	l := make([][]float64, n)
	for i := range n {
		l[i] = make([]float64, n)
	}
	for j := range n {
		for i := range n {
			%s
			for k := range j {
				l[i][j] -= l[i][k] * l[j][k]
			}
		}
	}
}`
	if got := countCat(scanSrc(t, fmt.Sprintf(body, ""))); got["row-slice-matrix"] == 0 {
		t.Fatalf("fixture must produce a finding without the directive, got %v", got)
	}
	if got := countCat(scanSrc(t, fmt.Sprintf(body, "//perfscan:ignore PS4006 deliberate"))); got["row-slice-matrix"] != 0 {
		t.Fatalf("directive above the loop did not suppress: %v", got)
	}
	if got := countCat(scanSrc(t, fmt.Sprintf(body, "//perfscan:ignore"))); got["row-slice-matrix"] != 0 {
		t.Fatalf("bare directive above the loop did not suppress: %v", got)
	}
}

// The shape behind the Mamba2 SSM win: a product varying with the INNER index but not
// the outer, rebuilt on every outer iteration.
func TestDetectPS5003_InnerInvariantRecompute(t *testing.T) {
	src := `package p
func scan(n, m, off int, h, x []float64, a, delta float64) {
	for i := range n {
		for j := range m {
			h[i*m+j] = a*h[i*m+j] + x[i]*(x[off+j]*delta)
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["inner-invariant-recompute"] == 0 {
		t.Fatalf("want ≥1 inner-invariant-recompute, got 0 (%v)", got)
	}
}

// THE UNSOUNDNESS THAT NEARLY SHIPPED: an expression can mention no outer variable and
// still change every outer iteration, because the outer loop REWRITES what it reads. A
// per-row softmax is the canonical case — hoisting p[j]*inv out would be WRONG, not
// merely useless. Every finding sampled before this guard existed was of this kind.
func TestDetectPS5003_SilentWhenOperandIsRewrittenByTheOuterLoop(t *testing.T) {
	src := `package p
func softmaxRows(n, m int, z, p, out []float64, inv float64) {
	for i := range n {
		for j := range m {
			p[j] = z[i*m+j]
		}
		for j := range m {
			out[i*m+j] = 2 * (p[j] * inv)
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["inner-invariant-recompute"] != 0 {
		t.Fatalf("want 0 when the outer loop rewrites the operand, got %d (%v)",
			got["inner-invariant-recompute"], got)
	}
}

// SILENT on a call: hoisting changes how often it runs, which is observable if it is
// not pure, and the rule cannot know that it is.
func TestDetectPS5003_SilentOnCall(t *testing.T) {
	src := `package p
func f(n, m, off int, out, x []float64, d float64) {
	for i := range n {
		for j := range m {
			out[i*m+j] = 2 * (scale(x[off+j]) * d)
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["inner-invariant-recompute"] != 0 {
		t.Fatalf("want 0 on a call, got %d (%v)", got["inner-invariant-recompute"], got)
	}
}

// SILENT when the inner body is not a tight kernel: with more work around it the
// recompute is not plausibly what dominates, and requiring one statement is what took
// this check from 445 findings to a usable number.
func TestDetectPS5003_SilentOnFatInnerBody(t *testing.T) {
	src := `package p
func f(n, m, off int, out, x, tmp []float64, d float64) {
	for i := range n {
		for j := range m {
			tmp[j] = float64(j)
			out[i*m+j] = 2 * (x[off+j] * d)
			tmp[j] += 1
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["inner-invariant-recompute"] != 0 {
		t.Fatalf("want 0 on a fat inner body, got %d (%v)", got["inner-invariant-recompute"], got)
	}
}

// PS5005 — a PURE libm transcendental invariant across the outer loop. The sinusoidal
// positional-encoding shape: 1/Pow(base, 2i/d) depends on the inner index i, not the
// outer position pos, so it is re-evaluated pos-many times for the same value.
func TestDetectPS5005_InvariantTranscendental(t *testing.T) {
	src := `package p
import "math"
func pe(seqLen, dModel int, base float64, out []float64) {
	for pos := range seqLen {
		for i := range dModel / 2 {
			freq := 1.0 / math.Pow(base, float64(2*i)/float64(dModel))
			out[pos*dModel+2*i] = math.Sin(float64(pos) * freq)
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["loop-invariant-transcendental"] == 0 {
		t.Fatalf("want ≥1 loop-invariant-transcendental, got 0 (%v)", got)
	}
}

// SILENT once the Pow is precomputed into a per-inner-index scratch above the outer loop
// (the shipped fix) — the math.Sin that remains genuinely varies with pos, so no finding.
func TestDetectPS5005_SilentWhenHoisted(t *testing.T) {
	src := `package p
import "math"
func pe(seqLen, dModel int, base float64, out []float64) {
	half := dModel / 2
	freqs := make([]float64, half)
	for i := range half {
		freqs[i] = 1.0 / math.Pow(base, float64(2*i)/float64(dModel))
	}
	for pos := range seqLen {
		for i := range half {
			out[pos*dModel+2*i] = math.Sin(float64(pos) * freqs[i])
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["loop-invariant-transcendental"] != 0 {
		t.Fatalf("want 0 once hoisted, got %d (%v)", got["loop-invariant-transcendental"], got)
	}
}

// SILENT when the transcendental's argument IS the outer index — it genuinely varies per
// outer iteration and must not be hoisted.
func TestDetectPS5005_SilentWhenArgMentionsOuter(t *testing.T) {
	src := `package p
import "math"
func f(n, m int, out []float64) {
	for i := range n {
		for j := range m {
			out[i*m+j] = math.Exp(float64(i) + float64(j))
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["loop-invariant-transcendental"] != 0 {
		t.Fatalf("want 0 when arg mentions the outer index, got %d (%v)", got["loop-invariant-transcendental"], got)
	}
}

// SILENT when the outer loop REWRITES an operand the call reads (the same unsoundness
// PS5003 guards against): the value is textually outer-independent yet rebuilt each pass.
func TestDetectPS5005_SilentWhenOperandRewrittenByOuter(t *testing.T) {
	src := `package p
import "math"
func f(n, m int, buf, out []float64) {
	for i := range n {
		for j := range m {
			buf[j] = float64(i * j)
		}
		for j := range m {
			out[i*m+j] = math.Log(buf[j])
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["loop-invariant-transcendental"] != 0 {
		t.Fatalf("want 0 when the outer loop rewrites the operand, got %d (%v)", got["loop-invariant-transcendental"], got)
	}
}

// REGRESSION (the unsoundness that shipped in #488 and was fixed): a transcendental
// whose argument is a LOCAL that carries the outer index defeats a purely textual
// "does not mention the outer var" check. The SSM scan's abar := math.Exp(dt*A[d][n])
// never names t, yet dt := delta[t][d] makes it vary every t — hoisting it above the t
// loop would be WRONG. The guard must taint on any outer-body assignment, not just those
// outside the inner loop.
func TestDetectPS5005_SilentWhenArgIsOuterTaintedLocal(t *testing.T) {
	src := `package p
import "math"
func ssm(L, D, N int, delta, u, A, C [][]float64, h []float64, out [][]float64) {
	for t := range L {
		for d := range D {
			dt := delta[t][d]
			for n := range N {
				abar := math.Exp(dt * A[d][n])
				h[d*N+n] = abar*h[d*N+n] + dt*u[t][d]
				out[t][d] += C[t][n] * h[d*N+n]
			}
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["loop-invariant-transcendental"] != 0 {
		t.Fatalf("want 0 when the arg is an outer-tainted local (dt), got %d (%v)",
			got["loop-invariant-transcendental"], got)
	}
}

// REGRESSION (second PS5005 soundness gap): a slice filled by a CALL is mutated
// invisibly to assignedIn, which only tracks =/:=/++. The distillation KL loop rebuilds
// p via softmaxRowFlat(p, teacherRow) every outer iteration, so math.Log(p[j]) is NOT
// outer-invariant though p never appears on an assignment LHS. Hoisting it would be wrong.
func TestDetectPS5005_SilentWhenSliceFilledByOuterCall(t *testing.T) {
	src := `package p
import "math"
func kl(b, c int, zt, zs []float64, temp float64) float64 {
	p := make([]float64, c)
	q := make([]float64, c)
	var total float64
	for i := 0; i < b; i++ {
		softmaxRow(p, zt[i*c:i*c+c], temp)
		softmaxRow(q, zs[i*c:i*c+c], temp)
		for j := range c {
			total += p[j] * (math.Log(p[j]) - math.Log(q[j]))
		}
	}
	return total
}
func softmaxRow(dst, src []float64, temp float64) {}`
	if got := countCat(scanSrc(t, src)); got["loop-invariant-transcendental"] != 0 {
		t.Fatalf("want 0 when a read slice is filled by an outer call, got %d (%v)",
			got["loop-invariant-transcendental"], got)
	}
}

// PS6005 — the causal-conv single-bound guard: j := t-(K-1)+k; if j>=0 { acc += w[k]*x[j] }.
func TestDetectPS6005_CausalConvBound(t *testing.T) {
	src := `package p
func conv1d(L, D, K int, xs, ws, os []float64) {
	for t := range L {
		for c := range D {
			var acc float64
			for k := range K {
				j := t - (K - 1) + k
				if j >= 0 {
					acc += ws[c*K+k] * xs[j*D+c]
				}
			}
			os[t*D+c] = acc
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["monotone-index-bound"] == 0 {
		t.Fatalf("want ≥1 monotone-index-bound, got 0 (%v)", got)
	}
}

// SILENT once the loop bound is clamped (the shipped fix): no per-tap branch left.
func TestDetectPS6005_SilentWhenHoisted(t *testing.T) {
	src := `package p
func conv1d(L, D, K int, xs, ws, os []float64) {
	for t := range L {
		kStart := 0
		if lo := (K - 1) - t; lo > 0 {
			kStart = lo
		}
		for c := range D {
			var acc float64
			for k := kStart; k < K; k++ {
				acc += ws[c*K+k] * xs[(t-(K-1)+k)*D+c]
			}
			os[t*D+c] = acc
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["monotone-index-bound"] != 0 {
		t.Fatalf("want 0 once hoisted, got %d (%v)", got["monotone-index-bound"], got)
	}
}

// SILENT on a data-dependent value fed to a threshold: qlo is a quantized sample, not a
// monotone address offset — it is compared but never used as an array index in the body.
func TestDetectPS6005_SilentOnDataThreshold(t *testing.T) {
	src := `package p
func q5(blk []byte, qh []byte, u1 byte) {
	for l := range 32 {
		qlo := quant(blk[l])
		if qlo >= 16 {
			qh[l] |= u1
		}
	}
}
func quant(b byte) int { return int(b) }`
	if got := countCat(scanSrc(t, src)); got["monotone-index-bound"] != 0 {
		t.Fatalf("want 0 on a data-threshold guard, got %d (%v)", got["monotone-index-bound"], got)
	}
}

// PS6001 also fires when the sorted slice feeds a FIXED prefix `cand[:k]` used as a set
// (a keep-mask), not just an early-breaking loop — the SnapKV shape.
func TestDetectPS6001_FixedPrefixSliceConsumer(t *testing.T) {
	src := `package p
func keep(scores []float64, n, k int, mask []bool) {
	cand := make([]int, n)
	for i := range cand {
		cand[i] = i
	}
	sortIdxDescByProb(cand, scores)
	for _, i := range cand[:k] {
		mask[i] = true
	}
}
func sortIdxDescByProb(idx []int, key []float64) {}`
	if got := countCat(scanSrc(t, src)); got["full-sort-bounded-prefix"] == 0 {
		t.Fatalf("want ≥1 full-sort-bounded-prefix on a fixed cand[:k] consumer, got 0 (%v)", got)
	}
}

// SILENT once quickselect guards it (the shipped fix).
func TestDetectPS6001_SilentWhenQuickselected(t *testing.T) {
	src := `package p
func keep(scores []float64, n, k int, mask []bool) {
	cand := make([]int, n)
	for i := range cand {
		cand[i] = i
	}
	quickselectIdxDesc(cand, scores, k)
	for _, i := range cand[:k] {
		mask[i] = true
	}
}
func quickselectIdxDesc(idx []int, key []float64, k int) {}`
	if got := countCat(scanSrc(t, src)); got["full-sort-bounded-prefix"] != 0 {
		t.Fatalf("want 0 once quickselect-guarded, got %d (%v)", got["full-sort-bounded-prefix"], got)
	}
}

// PS1005 fires on a manual multi-dimensional tensor walk via AtF64 — the indices are two
// enclosing-loop variables (the VQ-VAE codebook-scan shape PS1001's Numel check misses).
func TestDetectPS1005_ManualWalkDispatch(t *testing.T) {
	src := `package p
func vq(ze, cb *T, batch, k, d int) {
	for i := 0; i < batch; i++ {
		for j := 0; j < k; j++ {
			for c := 0; c < d; c++ {
				_ = ze.AtF64(i, c) - cb.AtF64(j, c)
			}
		}
	}
}`
	if got := countCat(scanSrc(t, src))["manual-walk-dispatch"]; got == 0 {
		t.Fatalf("want ≥1 manual-walk-dispatch, got 0")
	}
}

// SILENT with a single index (a 1-D walk is not the multi-dim dispatch PS1005 targets; a
// single loop var is often a legitimate row/element access).
func TestDetectPS1005_SilentOnSingleIndex(t *testing.T) {
	src := `package p
func f(t_ *T, n int) {
	for i := 0; i < n; i++ {
		_ = t_.AtF64(i)
	}
}`
	if got := countCat(scanSrc(t, src))["manual-walk-dispatch"]; got != 0 {
		t.Fatalf("want 0 on a single-index access, got %d", got)
	}
}

// SILENT when only one index is a loop variable (t.AtF64(0, k) — a fixed row, not a walk).
func TestDetectPS1005_SilentWhenOneIndexIsConstant(t *testing.T) {
	src := `package p
func f(t_ *T, n int) {
	for k := 0; k < n; k++ {
		_ = t_.AtF64(0, k)
	}
}`
	if got := countCat(scanSrc(t, src))["manual-walk-dispatch"]; got != 0 {
		t.Fatalf("want 0 when only one index is a loop var, got %d", got)
	}
}

const ps6003Prelude = `package p
type QT int
const (A QT = iota; B; C; D)
func gen(q QT) int
`

// The gguf.QMatMul shape: a fused path for one format, a switch showing six more.
func TestDetectPS6003_PartialFastPathCoverage(t *testing.T) {
	src := ps6003Prelude + `func f(q QT, m int) int {
	if q == A && m == 1 {
		return 1
	}
	switch q {
	case A:
		return 2
	case B:
		return 3
	case C:
		return 4
	}
	return 0
}`
	if got := countCat(scanSrc(t, src)); got["partial-fast-path-coverage"] != 1 {
		t.Fatalf("want 1 partial-fast-path-coverage, got %d (%v)", got["partial-fast-path-coverage"], got)
	}
}

// THE ONLY FALSE POSITIVE THIS RULE HAD ON THE TREE: a guard INSIDE one case clause of
// the very switch it is judged against. It bypasses nothing — it is a sub-case of the
// dispatch, not a fast path around it. gguf's metadata reader is exactly this shape.
func TestDetectPS6003_SilentWhenGuardIsInsideTheSwitch(t *testing.T) {
	src := ps6003Prelude + `func f(q QT) int {
	switch q {
	case A:
		if q == A {
			return 1
		}
		return 2
	case B:
		return 3
	case C:
		return 4
	}
	return 0
}`
	if got := countCat(scanSrc(t, src)); got["partial-fast-path-coverage"] != 0 {
		t.Fatalf("want 0 when the guard is inside the switch, got %d (%v)",
			got["partial-fast-path-coverage"], got)
	}
}

// SILENT when every variant already has a fast path — there is no gap to report.
func TestDetectPS6003_SilentWhenCoverageIsComplete(t *testing.T) {
	src := ps6003Prelude + `func f(q QT, m int) int {
	if q == A && m == 1 {
		return 1
	}
	if q == B && m == 1 {
		return 2
	}
	if q == C && m == 1 {
		return 3
	}
	switch q {
	case A:
		return 4
	case B:
		return 5
	case C:
		return 6
	}
	return 0
}`
	if got := countCat(scanSrc(t, src)); got["partial-fast-path-coverage"] != 0 {
		t.Fatalf("want 0 when all variants are covered, got %d (%v)",
			got["partial-fast-path-coverage"], got)
	}
}

// SILENT on a two-way switch: that is a branch, not a variant family, and a fast path
// for one of two arms is an if/else written twice.
func TestDetectPS6003_SilentOnTwoWaySwitch(t *testing.T) {
	src := ps6003Prelude + `func f(q QT, m int) int {
	if q == A && m == 1 {
		return 1
	}
	switch q {
	case A:
		return 2
	case B:
		return 3
	}
	return 0
}`
	if got := countCat(scanSrc(t, src)); got["partial-fast-path-coverage"] != 0 {
		t.Fatalf("want 0 on a two-way switch, got %d (%v)", got["partial-fast-path-coverage"], got)
	}
}

// SILENT on literal cases: `switch n { case 1, 2, 3 }` is not a family of formats, and
// admitting literals would bury the real findings.
func TestDetectPS6003_SilentOnLiteralCases(t *testing.T) {
	src := `package p
func f(n, m int) int {
	if n == 1 && m == 1 {
		return 1
	}
	switch n {
	case 1:
		return 2
	case 2:
		return 3
	case 3:
		return 4
	}
	return 0
}`
	if got := countCat(scanSrc(t, src)); got["partial-fast-path-coverage"] != 0 {
		t.Fatalf("want 0 on literal cases, got %d (%v)", got["partial-fast-path-coverage"], got)
	}
}

// The literal exclusion has TWO sides, and the fixture above only reaches one. There the
// guard itself compares against a literal, so it is rejected before the switch is ever
// examined. This case passes the guard side — a named constant — and must still be
// silent because the switch's members are literals, which is what the members-side
// filter is for. Probing found the fixture above did not cover it.
func TestDetectPS6003_SilentOnLiteralCasesWithNamedGuard(t *testing.T) {
	src := ps6003Prelude + `func f(q QT, m int) int {
	if q == A && m == 1 {
		return 1
	}
	switch q {
	case 1:
		return 2
	case 2:
		return 3
	case 3:
		return 4
	}
	return 0
}`
	if got := countCat(scanSrc(t, src)); got["partial-fast-path-coverage"] != 0 {
		t.Fatalf("want 0 when the switch cases are literals, got %d (%v)",
			got["partial-fast-path-coverage"], got)
	}
}

// A switch that MIXES named constants with literals is where the members-side filter
// actually earns its place. Probing showed it suppresses nothing on its own — an
// all-literal switch is already silent because no member matches the guard — so the one
// thing it does is keep a bare literal out of the reported variant list. Without it this
// message names an empty string among the uncovered variants.
func TestDetectPS6003_MixedLiteralAndNamedCasesReportOnlyNamedVariants(t *testing.T) {
	src := ps6003Prelude + `func f(q QT, m int) int {
	if q == A && m == 1 {
		return 1
	}
	switch q {
	case A:
		return 2
	case 7:
		return 3
	case B:
		return 4
	case C:
		return 5
	}
	return 0
}`
	fs := scanSrc(t, src)
	var msg string
	for _, f := range fs {
		if f.category == "partial-fast-path-coverage" {
			msg = f.msg
		}
	}
	if msg == "" {
		t.Fatalf("want a partial-fast-path-coverage finding, got %v", countCat(fs))
	}
	if !strings.Contains(msg, "B, C") {
		t.Errorf("want the uncovered variants reported as %q, got %q", "B, C", msg)
	}
	if strings.Contains(msg, ", ,") || strings.Contains(msg, "; , ") {
		t.Errorf("literal case leaked into the variant list as an empty name: %q", msg)
	}
	if !strings.Contains(msg, "of the 3 ") {
		t.Errorf("want the literal excluded from the family size (3 named members), got %q", msg)
	}
}

// A guard may spell a group of variants as a DISJUNCTION — `(q == A || q == B) && m == 1`
// is how gguf.QMatMul covers the two K-quants that share a helper. Restricting the walk
// to && chains made the rule report those two as uncovered the moment they were fused,
// which is how this case was found.
func TestDetectPS6003_DisjunctionInTheGuardCountsAsCoverage(t *testing.T) {
	src := ps6003Prelude + `func f(q QT, m int) int {
	if (q == A || q == B) && m == 1 {
		return 1
	}
	switch q {
	case A:
		return 2
	case B:
		return 3
	case C:
		return 4
	}
	return 0
}`
	fs := scanSrc(t, src)
	var msg string
	for _, f := range fs {
		if f.category == "partial-fast-path-coverage" {
			msg = f.msg
		}
	}
	if msg == "" {
		t.Fatalf("want a finding naming only the uncovered variant, got %v", countCat(fs))
	}
	if !strings.Contains(msg, "2 of the 3") || strings.Contains(msg, "A") && strings.Contains(msg, "B, ") {
		t.Errorf("want both disjuncts counted as covered and only C reported, got %q", msg)
	}
	if !strings.HasSuffix(strings.TrimSuffix(msg, " still take the general path — benchmark whether they should"), "C") {
		t.Errorf("want C as the sole uncovered variant, got %q", msg)
	}
}

// SILENT when the guard does not return: without the early return the general path
// still runs, so the variant is not short-circuited at all.
func TestDetectPS6003_SilentWhenGuardDoesNotReturn(t *testing.T) {
	src := ps6003Prelude + `func f(q QT, m int) int {
	acc := 0
	if q == A && m == 1 {
		acc++
	}
	switch q {
	case A:
		return acc + 2
	case B:
		return acc + 3
	case C:
		return acc + 4
	}
	return acc
}`
	if got := countCat(scanSrc(t, src)); got["partial-fast-path-coverage"] != 0 {
		t.Fatalf("want 0 when the guard does not return, got %d (%v)",
			got["partial-fast-path-coverage"], got)
	}
}

// PS4007 fires on a *VJP whose hot loop is a single elementwise binop written as a scalar
// Go loop (the pre-fix expVJP shape) — it should dispatch the matching backend op instead.
func TestDetectPS4007_ScalarBinopVJP(t *testing.T) {
	src := `package p
func expVJP(ctx *C, in, out []*T, a A, g *T) ([]*T, error) {
	for i := 0; i < n; i++ {
		ds[i] = gs[i] * ys[i]
	}
	return nil, nil
}`
	if got := countCat(scanSrc(t, src))["vjp-scalar-elementwise-binop"]; got != 1 {
		t.Fatalf("want 1 vjp-scalar-elementwise-binop, got %d", got)
	}
}

// …including the f32 conversion-wrapped form float32(float64(a)*float64(b)).
func TestDetectPS4007_ConversionWrapped(t *testing.T) {
	src := `package p
func expVJP(ctx *C, in, out []*T, a A, g *T) ([]*T, error) {
	for i := 0; i < n; i++ {
		ds[i] = float32(float64(gs[i]) * float64(ys[i]))
	}
	return nil, nil
}`
	if got := countCat(scanSrc(t, src))["vjp-scalar-elementwise-binop"]; got != 1 {
		t.Fatalf("want 1 (conversion-wrapped), got %d", got)
	}
}

// Silent on MULTI-op bodies (tanh g·(1−y²)): they keep f64 intermediates and narrow once,
// so composing f32 backend ops would diverge — they need a fused kernel, not a dispatch.
func TestDetectPS4007_SilentOnMultiOp(t *testing.T) {
	src := `package p
func tanhVJP(ctx *C, in, out []*T, a A, g *T) ([]*T, error) {
	for i := 0; i < n; i++ {
		ds[i] = gs[i] * (1 - ys[i]*ys[i])
	}
	return nil, nil
}`
	if got := countCat(scanSrc(t, src))["vjp-scalar-elementwise-binop"]; got != 0 {
		t.Fatalf("want 0 (multi-op needs a fused kernel), got %d", got)
	}
}

// Silent outside the VJP layer: a backend kernel's scalar loop IS the implementation, not a
// missed dispatch, so the *VJP name scope keeps it from being flagged.
func TestDetectPS4007_SilentOnNonVJP(t *testing.T) {
	src := `package p
func mulKernelCPU(ds, gs, ys []float64, n int) {
	for i := 0; i < n; i++ {
		ds[i] = gs[i] * ys[i]
	}
}`
	if got := countCat(scanSrc(t, src))["vjp-scalar-elementwise-binop"]; got != 0 {
		t.Fatalf("want 0 (non-VJP kernel), got %d", got)
	}
}

// Silent once the VJP already dispatches to the backend (the fixed expVJP shape).
func TestDetectPS4007_SilentOnDispatch(t *testing.T) {
	src := `package p
func expVJP(ctx *C, in, out []*T, a A, g *T) ([]*T, error) {
	res, err := backend.Execute(ctx, backend.OpMul, []*T{g, out[0]}, nil)
	if err != nil {
		return nil, err
	}
	return []*T{res[0]}, nil
}`
	if got := countCat(scanSrc(t, src))["vjp-scalar-elementwise-binop"]; got != 0 {
		t.Fatalf("want 0 (already dispatches), got %d", got)
	}
}

// PS6001 fires on the pre-fix Mirostat shape: fill the whole vocab, sort it all, consume a
// break-bounded prefix — with no quickselect guard.
func TestDetectPS6001_FullSortBoundedPrefix(t *testing.T) {
	src := `package p
func sample(probs []float64, mu float64) int {
	n := len(probs)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sortIdxDescByProb(idx, probs)
	keep := 1
	for keep < n && surpriseBits(probs[idx[keep]]) <= mu {
		keep++
	}
	x := idx[0]
	var cum float64
	for _, i := range idx[:keep] {
		cum += probs[i]
		if cum >= 0.5 {
			x = i
			break
		}
	}
	return x
}`
	if got := countCat(scanSrc(t, src))["full-sort-bounded-prefix"]; got != 1 {
		t.Fatalf("want 1 full-sort-bounded-prefix, got %d", got)
	}
}

// Silent on the optimized quickselect-then-fallback form (nucleusTopP): the retained full-sort
// fallback must NOT be flagged.
func TestDetectPS6001_SilentWithQuickselect(t *testing.T) {
	src := `package p
func nucleus(probs []float64, p float64) {
	n := len(probs)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	quickselectIdxDesc(idx, probs, 512)
	top := idx[:512]
	sortIdxDescByProb(top, probs)
	var cum float64
	for _, i := range top {
		cum += probs[i]
		if cum >= p {
			break
		}
	}
	sortIdxDescByProb(idx, probs)
}`
	if got := countCat(scanSrc(t, src))["full-sort-bounded-prefix"]; got != 0 {
		t.Fatalf("want 0 (quickselect-guarded), got %d", got)
	}
}

// Silent on a pure full-sort helper that just returns the sorted prefix (no in-function break) —
// the caller does the bounded consumption; the helper itself is the guarded fallback.
func TestDetectPS6001_SilentOnSortHelper(t *testing.T) {
	src := `package p
func sortedKeep(probs []float64, n int, mu float64) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sortIdxDescByProb(idx, probs)
	keep := 1
	for keep < n && surpriseBits(probs[idx[keep]]) <= mu {
		keep++
	}
	return idx[:keep]
}`
	if got := countCat(scanSrc(t, src))["full-sort-bounded-prefix"]; got != 0 {
		t.Fatalf("want 0 (no break consumer), got %d", got)
	}
}

// PS6002 fires on an innermost window loop re-testing a compound spatial bounds guard per tap
// (the pre-fix conv2d col2im shape).
func TestDetectPS6002_SpatialBoundsBranch(t *testing.T) {
	src := `package p
func col2im(dXf, dXcols []float64, kw, oy, ox, s, p, h, wd, base, kk, ci, ky int) {
	iy := oy*s + ky - p
	for kx := 0; kx < kw; kx++ {
		ix := ox*s + kx - p
		if iy >= 0 && iy < h && ix >= 0 && ix < wd {
			dXf[ci*h+iy*wd+ix] += dXcols[base+kk]
		}
		kk++
	}
}`
	if got := countCat(scanSrc(t, src))["spatial-bounds-branch"]; got != 1 {
		t.Fatalf("want 1 spatial-bounds-branch, got %d", got)
	}
}

// Silent once hoisted: the contiguous run has no per-tap compound guard.
func TestDetectPS6002_SilentWhenHoisted(t *testing.T) {
	src := `package p
func col2im(dXf, dXcols []float64, kxLo, kxHi, rowBase, base, kk int) {
	for i := kxLo; i < kxHi; i++ {
		dXf[rowBase+i] += dXcols[base+kk+i]
	}
}`
	if got := countCat(scanSrc(t, src))["spatial-bounds-branch"]; got != 0 {
		t.Fatalf("want 0 (hoisted), got %d", got)
	}
}

// Silent on a plain 1-2 term bounds check (not the spatial 4-term window shape).
func TestDetectPS6002_SilentOnSimpleGuard(t *testing.T) {
	src := `package p
func f(a, b []float64, n int) {
	for i := 0; i < n; i++ {
		if i >= 0 && i < n {
			a[i] = b[i]
		}
	}
}`
	if got := countCat(scanSrc(t, src))["spatial-bounds-branch"]; got != 0 {
		t.Fatalf("want 0 (simple 2-term guard), got %d", got)
	}
}

// PS5002 fires on 3+ consecutive same-range loops sharing a buffer (pass-fusion).
func TestDetectPS5002_MultiSweepFusable(t *testing.T) {
	src := `package p
func Step(sum, flat, out []float64, n int) {
	for k := 0; k < n; k++ {
		sum[k] -= flat[k]
	}
	for k := 0; k < n; k++ {
		flat[k] = out[k]
	}
	for k := 0; k < n; k++ {
		sum[k] += flat[k]
	}
}`
	got := countCat(scanSrc(t, src))
	if got["multi-sweep-fusable"] != 1 {
		t.Fatalf("want 1 multi-sweep-fusable, got %d (%v)", got["multi-sweep-fusable"], got)
	}
}

// Two consecutive loops are below the >=3 threshold — no finding (2-pass splits are
// common and often intentional).
func TestDetectPS5002_TwoLoopsSilent(t *testing.T) {
	src := `package p
func Step(sum, flat []float64, n int) {
	for k := 0; k < n; k++ {
		sum[k] -= flat[k]
	}
	for k := 0; k < n; k++ {
		sum[k] += flat[k]
	}
}`
	if got := countCat(scanSrc(t, src))["multi-sweep-fusable"]; got != 0 {
		t.Fatalf("two loops, want 0 multi-sweep-fusable, got %d", got)
	}
}

// Three same-range loops over DISJOINT buffers do not share traffic — no finding.
func TestDetectPS5002_DisjointBuffersSilent(t *testing.T) {
	src := `package p
func Step(a, b, c []float64, n int) {
	for k := 0; k < n; k++ {
		a[k] = a[k] * 2
	}
	for k := 0; k < n; k++ {
		b[k] = b[k] * 2
	}
	for k := 0; k < n; k++ {
		c[k] = c[k] * 2
	}
}`
	if got := countCat(scanSrc(t, src))["multi-sweep-fusable"]; got != 0 {
		t.Fatalf("disjoint buffers, want 0 multi-sweep-fusable, got %d", got)
	}
}

// Different ranges (n vs m) are not consecutive-same-range — no finding.
func TestDetectPS5002_DifferentRangesSilent(t *testing.T) {
	src := `package p
func Step(a []float64, n, m int) {
	for k := 0; k < n; k++ {
		a[k] = a[k] + 1
	}
	for k := 0; k < m; k++ {
		a[k] = a[k] + 1
	}
	for k := 0; k < n; k++ {
		a[k] = a[k] + 1
	}
}`
	if got := countCat(scanSrc(t, src))["multi-sweep-fusable"]; got != 0 {
		t.Fatalf("different ranges, want 0 multi-sweep-fusable, got %d", got)
	}
}

// PS1004 fires on a variadic spread accessor .AtF64(idx...) in a non-Numel loop.
func TestDetectPS1004_SpreadAccessorInLoop(t *testing.T) {
	src := `package p
func Contract(ops []*T, coords [][]int, total int) float64 {
	var acc float64
	for combo := 0; combo < total; combo++ {
		acc += ops[0].AtF64(coords[0]...)
	}
	return acc
}`
	got := countCat(scanSrc(t, src))
	if got["spread-accessor-in-loop"] != 1 {
		t.Fatalf("want 1 spread-accessor-in-loop, got %d (%v)", got["spread-accessor-in-loop"], got)
	}
}

// A non-spread accessor call (explicit args) is not the PS1004 pattern.
func TestDetectPS1004_NonSpreadSilent(t *testing.T) {
	src := `package p
func F(x *T, n int) float64 {
	var a float64
	for i := 0; i < n; i++ {
		a += x.AtF64(i, 0)
	}
	return a
}`
	if got := countCat(scanSrc(t, src))["spread-accessor-in-loop"]; got != 0 {
		t.Fatalf("non-spread call, want 0 spread-accessor-in-loop, got %d", got)
	}
}

// A spread accessor OUTSIDE any loop is not flagged.
func TestDetectPS1004_OutsideLoopSilent(t *testing.T) {
	src := `package p
func F(x *T, idx []int) float64 {
	return x.AtF64(idx...)
}`
	if got := countCat(scanSrc(t, src))["spread-accessor-in-loop"]; got != 0 {
		t.Fatalf("outside a loop, want 0 spread-accessor-in-loop, got %d", got)
	}
}

// PS4010: fires on the butterfly p,q = x+y,x-y (via temps) in a loop.
func TestDetectPS4010_Butterfly(t *testing.T) {
	src := `package p
func fwht(a []float64) {
	n := len(a)
	for h := 1; h < n; h <<= 1 {
		for i := 0; i < n; i += h << 1 {
			for j := i; j < i+h; j++ {
				x, y := a[j], a[j+h]
				a[j], a[j+h] = x+y, x-y
			}
		}
	}
}`
	if got := countCat(scanSrc(t, src))["vectorizable-butterfly"]; got != 1 {
		t.Fatalf("want 1 vectorizable-butterfly, got %d", got)
	}
}

// Must stay SILENT on an ordinary swap / non-add-sub tuple assign.
func TestDetectPS4010_SilentOnSwap(t *testing.T) {
	src := `package p
func swap(a []float64, n int) {
	for i := 0; i < n; i++ {
		a[i], a[n-i] = a[n-i], a[i]
	}
}`
	if got := countCat(scanSrc(t, src))["vectorizable-butterfly"]; got != 0 {
		t.Fatalf("want 0 (swap), got %d", got)
	}
}

// Must stay SILENT when the two RHS are not x+y / x-y of the same pair.
func TestDetectPS4010_SilentOnMixedOps(t *testing.T) {
	src := `package p
func mix(a []float64, n int) {
	for i := 0; i < n; i++ {
		x, y := a[i], a[i+1]
		a[i], a[i+1] = x+y, x*y
	}
}`
	if got := countCat(scanSrc(t, src))["vectorizable-butterfly"]; got != 0 {
		t.Fatalf("want 0 (add/mul, not butterfly), got %d", got)
	}
}

// PS4009: fires on the transposed gram M[k][i]·M[k][j] (k = row/reduction index → column-stride).
func TestDetectPS4009_TransposedGram(t *testing.T) {
	src := `package p
func rgram(gm, r [][]float64, m, n int, b2 float64) {
	for i := range n {
		for j := i; j < n; j++ {
			var acc float64
			for k := range m {
				acc += gm[k][i] * gm[k][j]
			}
			r[i][j] = b2*r[i][j] + (1-b2)*acc
		}
	}
}`
	if got := countCat(scanSrc(t, src))["transposed-gram-colstride"]; got != 1 {
		t.Fatalf("want 1 transposed-gram-colstride, got %d", got)
	}
}

// Must stay SILENT on the cache-friendly L-gram M[i][k]·M[j][k] (k = inner/contiguous index).
func TestDetectPS4009_SilentOnContiguousGram(t *testing.T) {
	src := `package p
func lgram(gm, l [][]float64, m, n int, b2 float64) {
	for i := range m {
		for j := i; j < m; j++ {
			var acc float64
			for k := range n {
				acc += gm[i][k] * gm[j][k]
			}
			l[i][j] = b2*l[i][j] + (1-b2)*acc
		}
	}
}`
	if got := countCat(scanSrc(t, src))["transposed-gram-colstride"]; got != 0 {
		t.Fatalf("want 0 (contiguous gram), got %d", got)
	}
}

// Must stay SILENT on matmul a[i][k]·b[k][j] — different bases, not a symmetric gram.
func TestDetectPS4009_SilentOnMatmul(t *testing.T) {
	src := `package p
func mm(a, b, c [][]float64, m, n, k int) {
	for i := range m {
		for j := range n {
			var acc float64
			for kk := range k {
				acc += a[i][kk] * b[kk][j]
			}
			c[i][j] = acc
		}
	}
}`
	if got := countCat(scanSrc(t, src))["transposed-gram-colstride"]; got != 0 {
		t.Fatalf("want 0 (matmul), got %d", got)
	}
}

// PS5006: fires on the SSDQuadratic-style [j..i] running product recomputed per (i,j).
func TestDetectPS5006_SubrangeRescan(t *testing.T) {
	src := `package p
func ssd(a, y []float64, T int) {
	for i := 0; i < T; i++ {
		for j := 0; j <= i; j++ {
			decay := 1.0
			for k := j + 1; k <= i; k++ {
				decay *= a[k]
			}
			y[i] += decay
		}
	}
}`
	if got := countCat(scanSrc(t, src))["nested-subrange-rescan"]; got != 1 {
		t.Fatalf("want 1 nested-subrange-rescan, got %d", got)
	}
}

// Silent when the innermost loop is a FIXED range (bounds do not span [j..i]).
func TestDetectPS5006_SilentOnFixedRange(t *testing.T) {
	src := `package p
func dot(c, b, y []float64, T, n int) {
	for i := 0; i < T; i++ {
		for j := 0; j <= i; j++ {
			var acc float64
			for k := 0; k < n; k++ {
				acc += c[k] * b[k]
			}
			y[i] += acc
		}
	}
}`
	if got := countCat(scanSrc(t, src))["nested-subrange-rescan"]; got != 0 {
		t.Fatalf("want 0 (fixed range), got %d", got)
	}
}

// Silent when the inner loop does not accumulate a running reduction over k.
func TestDetectPS5006_SilentOnNonReduce(t *testing.T) {
	src := `package p
func f(a, y []float64, T int) {
	for i := 0; i < T; i++ {
		for j := 0; j <= i; j++ {
			for k := j + 1; k <= i; k++ {
				y[k] = a[k] * 2
			}
		}
	}
}`
	if got := countCat(scanSrc(t, src))["nested-subrange-rescan"]; got != 0 {
		t.Fatalf("want 0 (no acc reduction), got %d", got)
	}
}

// PS4011: fires on a recurrence dispatching 2+ backend ops/iter with no flatF64 fast path.
func TestDetectPS4011_OpDispatchRecurrence(t *testing.T) {
	src := `package p
func rec(ctx *C, x T, seq int) T {
	var s T
	for t := 0; t < seq; t++ {
		a := ex(backend.OpMul, s, x)
		s = ex(backend.OpAdd, a, x)
	}
	return s
}`
	if got := countCat(scanSrc(t, src))["op-dispatch-recurrence"]; got != 1 {
		t.Fatalf("want 1 op-dispatch-recurrence, got %d", got)
	}
}

// Silent when the function already has a flatF64 fused fast path.
func TestDetectPS4011_SilentWithFlatF64(t *testing.T) {
	src := `package p
func rec(ctx *C, x T, seq int) T {
	if xs := flatF64(x); xs != nil {
		return x
	}
	var s T
	for t := 0; t < seq; t++ {
		a := ex(backend.OpMul, s, x)
		s = ex(backend.OpAdd, a, x)
	}
	return s
}`
	if got := countCat(scanSrc(t, src))["op-dispatch-recurrence"]; got != 0 {
		t.Fatalf("want 0 (has flatF64), got %d", got)
	}
}

// Silent when the loop dispatches fewer than 2 backend ops.
func TestDetectPS4011_SilentOnSingleDispatch(t *testing.T) {
	src := `package p
func rec(ctx *C, x T, seq int) T {
	var s T
	for t := 0; t < seq; t++ {
		s = ex(backend.OpAdd, s, x)
	}
	return s
}`
	if got := countCat(scanSrc(t, src))["op-dispatch-recurrence"]; got != 0 {
		t.Fatalf("want 0 (single dispatch), got %d", got)
	}
}

func TestDetectPS5007_F32AbsViaF64(t *testing.T) {
	src := `package p
import "math"
func f(x []float32) float32 {
	var m float32
	for _, v := range x {
		if a := float32(math.Abs(float64(v))); a > m {
			m = a
		}
	}
	return m
}`
	if got := countCat(scanSrc(t, src))["f32-abs-via-f64"]; got != 1 {
		t.Fatalf("want 1 f32-abs-via-f64, got %d", got)
	}
}

// PS4012: fires on a serial dot whose accumulator is dequantized (scaled) before the store.
func TestDetectPS4012_ScaledSerialDot(t *testing.T) {
	src := `package p
func qmatmul(qx, qw, sx, sw []float64, y []float64, m, n, k int) {
	for i := range m {
		xi := qx[i*k : i*k+k]
		for j := range n {
			wj := qw[j*k : j*k+k]
			var acc float64
			for c := range xi {
				acc += xi[c] * wj[c]
			}
			deq := acc * sx[i] * sw[j] / (127 * 127)
			y[i*n+j] += deq
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["scaled-serial-dot"] == 0 {
		t.Fatalf("want ≥1 scaled-serial-dot, got 0 (%v)", got)
	}
}

// Must stay SILENT when the accumulator is stored RAW (that is PS4008's job, not PS4012).
func TestDetectPS4012_SilentOnRawStore(t *testing.T) {
	src := `package p
func mm(a, b, c []float64, m, k int) {
	for i := range m {
		ai := a[i*k : i*k+k]
		ci := c[i*m : i*m+m]
		for j := range m {
			bj := b[j*k : j*k+k]
			var s float64
			for p := range ai {
				s += ai[p] * bj[p]
			}
			ci[j] = s
		}
	}
}`
	if got := countCat(scanSrc(t, src))["scaled-serial-dot"]; got != 0 {
		t.Fatalf("want 0 scaled-serial-dot on a raw-store dot, got %d", got)
	}
}

// Must stay SILENT when there is no product accumulator (no serial dot at all).
func TestDetectPS4012_SilentOnNoDot(t *testing.T) {
	src := `package p
func scale(x, s []float64, y []float64, m, n int) {
	for i := range m {
		for j := range n {
			var acc float64
			acc = x[i*n+j]
			y[i*n+j] = acc * s[j]
		}
	}
}`
	if got := countCat(scanSrc(t, src))["scaled-serial-dot"]; got != 0 {
		t.Fatalf("want 0 scaled-serial-dot without a dot loop, got %d", got)
	}
}

// The gguf Q4_0 shape: one activation row shared by every output, one accumulator per
// output, the shared row re-read on every pass.
func TestDetectPS6010_OutputInvariantReload(t *testing.T) {
	src := `package p
func f(n, k int, outf, row, w []float64) {
	for ni := range n {
		wr := w[ni*k:]
		var acc float64
		for i := range k {
			acc += row[i] * wr[i]
		}
		outf[ni] = acc
	}
}`
	if got := countCat(scanSrc(t, src)); got["output-invariant-operand-reload"] != 1 {
		t.Fatalf("want 1 output-invariant-operand-reload, got %d (%v)",
			got["output-invariant-operand-reload"], got)
	}
}

// THE CASE THAT MADE THE FIRST VERSION MISS THE LOOP IT WAS WRITTEN FROM: the per-output
// operand arrives as a RANGE VALUE, never as an index expression. Walking only
// assignments when computing what is output-dependent left `q` unclassified, so the
// check saw no per-output operand and stayed silent on its own motivating case.
func TestDetectPS6010_PerOutputOperandArrivingAsARangeValue(t *testing.T) {
	src := `package p
func f(n, k int, outf, row []float64, w []byte) {
	for ni := range n {
		wr := w[ni*k:]
		var acc float64
		for i, q := range wr {
			acc += row[i] * float64(q)
		}
		outf[ni] = acc
	}
}`
	if got := countCat(scanSrc(t, src)); got["output-invariant-operand-reload"] != 1 {
		t.Fatalf("want 1 when the per-output operand is a range value, got %d (%v)",
			got["output-invariant-operand-reload"], got)
	}
}

// SILENT once blocked: a stride above 1 is the signature of someone having already done
// this, and re-reporting it would make the check noise on exactly the fixed code.
func TestDetectPS6010_SilentOnAnAlreadyBlockedLoop(t *testing.T) {
	src := `package p
func f(n, k int, outf, row, w []float64) {
	for ni := 0; ni+4 <= n; ni += 4 {
		w0, w1 := w[(ni+0)*k:], w[(ni+1)*k:]
		var a0, a1 float64
		for i := range k {
			a0 += row[i] * w0[i]
			a1 += row[i] * w1[i]
		}
		outf[ni] = a0
		outf[ni+1] = a1
	}
}`
	if got := countCat(scanSrc(t, src)); got["output-invariant-operand-reload"] != 0 {
		t.Fatalf("want 0 on an already-blocked loop, got %d (%v)",
			got["output-invariant-operand-reload"], got)
	}
}

// SILENT when EVERY operand is output-invariant: that is a loop-invariant accumulation,
// PS5003's finding, and its fix is to hoist the whole thing out rather than unroll —
// unrolling a computation that should not run n times at all is the wrong advice.
func TestDetectPS6010_SilentWhenNothingVariesWithTheOutput(t *testing.T) {
	src := `package p
func f(n, k int, outf, row, other []float64) {
	for ni := range n {
		var acc float64
		for i := range k {
			acc += row[i] * other[i]
		}
		outf[ni] = acc
	}
}`
	if got := countCat(scanSrc(t, src)); got["output-invariant-operand-reload"] != 0 {
		t.Fatalf("want 0 when nothing varies with the output index, got %d (%v)",
			got["output-invariant-operand-reload"], got)
	}
}

// SILENT when the accumulator never reaches an output index. Without this the check
// fires on every scalar reduction that happens to sit inside some loop — it was 145
// findings tree-wide before this guard and 56 after.
func TestDetectPS6010_SilentWhenTheAccumulatorIsNotStoredPerOutput(t *testing.T) {
	src := `package p
func f(n, k int, row, w []float64) float64 {
	total := 0.0
	for ni := range n {
		wr := w[ni*k:]
		var acc float64
		for i := range k {
			acc += row[i] * wr[i]
		}
		total += acc
	}
	return total
}`
	if got := countCat(scanSrc(t, src)); got["output-invariant-operand-reload"] != 0 {
		t.Fatalf("want 0 when the accumulator is not stored per output, got %d (%v)",
			got["output-invariant-operand-reload"], got)
	}
}

// The classic.GaussianMixture shape: a receiver slice field taken as a local alias, then
// written and read back element-wise within one call.
func TestDetectPS6006_ReceiverScratchViaAlias(t *testing.T) {
	src := `package p
type M struct{ yScratch []float64 }
func (m *M) f(x []float64, d int) float64 {
	y := m.yScratch
	for i := range d {
		s := x[i]
		for j := range i {
			s -= y[j]
		}
		y[i] = s
	}
	var q float64
	for i := range d {
		q += y[i] * y[i]
	}
	return q
}`
	if got := countCat(scanSrc(t, src)); got["receiver-scratch-buffer"] != 1 {
		t.Fatalf("want 1 receiver-scratch-buffer, got %d (%v)", got["receiver-scratch-buffer"], got)
	}
}

// ALIAS TRACKING IS THE LOAD-BEARING PART. The direct spelling is rarer in real code than
// the aliased one, and a version of this check that only matched `m.buf[i]` found nothing
// at all — including the very method it was written from.
func TestDetectPS6006_ReceiverScratchWrittenDirectly(t *testing.T) {
	src := `package p
type M struct{ tmpbuf []float64 }
func (m *M) f(x []float64, d int) float64 {
	for i := range d {
		m.tmpbuf[i] = x[i] * 2
	}
	var q float64
	for i := range d {
		q += m.tmpbuf[i]
	}
	return q
}`
	if got := countCat(scanSrc(t, src)); got["receiver-scratch-buffer"] != 1 {
		t.Fatalf("want 1 for the direct spelling, got %d (%v)", got["receiver-scratch-buffer"], got)
	}
}

// SILENT on persistent state. This is the false positive the first cut produced: a method
// that fills model parameters and reads them back is not using a temporary. Nothing in
// the AST separates the two, so the field NAME carries the intent.
func TestDetectPS6006_SilentOnPersistentState(t *testing.T) {
	src := `package p
type M struct{ Means []float64 }
func (m *M) f(x []float64, d int) float64 {
	for i := range d {
		m.Means[i] = x[i]
	}
	var q float64
	for i := range d {
		q += m.Means[i]
	}
	return q
}`
	if got := countCat(scanSrc(t, src)); got["receiver-scratch-buffer"] != 0 {
		t.Fatalf("want 0 on persistent state, got %d (%v)", got["receiver-scratch-buffer"], got)
	}
}

// SILENT when the field is only WRITTEN here: that is an output being produced, not a
// temporary being reused, and passing it as a parameter would be the wrong advice.
func TestDetectPS6006_SilentWhenOnlyWritten(t *testing.T) {
	src := `package p
type M struct{ outbuf []float64 }
func (m *M) f(x []float64, d int) {
	for i := range d {
		m.outbuf[i] = x[i] * 2
	}
}`
	if got := countCat(scanSrc(t, src)); got["receiver-scratch-buffer"] != 0 {
		t.Fatalf("want 0 when the field is only written, got %d (%v)",
			got["receiver-scratch-buffer"], got)
	}
}

// SILENT on a plain function: with no receiver there is no shared field, so there is
// neither a concurrency hazard nor contention to report.
func TestDetectPS6006_SilentOnNonMethod(t *testing.T) {
	src := `package p
func f(buf, x []float64, d int) float64 {
	for i := range d {
		buf[i] = x[i]
	}
	var q float64
	for i := range d {
		q += buf[i]
	}
	return q
}`
	if got := countCat(scanSrc(t, src)); got["receiver-scratch-buffer"] != 0 {
		t.Fatalf("want 0 on a non-method, got %d (%v)", got["receiver-scratch-buffer"], got)
	}
}

// The structural discriminator: a field INDEXED IN EXACTLY ONE function is a temporary,
// even when its name says nothing. gbmBuilder.vals and gbmBuilder.part are both spelled
// this way, and the name-keyed version of this check missed both.
func TestDetectPS6006_SoleIndexedFieldWithNoScratchishName(t *testing.T) {
	src := `package p
type B struct{ vals []float64 }
func (b *B) alloc(n int) { b.vals = make([]float64, n) }
func (b *B) f(x []float64, n int) float64 {
	v := b.vals
	for i := range n {
		v[i] = x[i] * 2
	}
	var q float64
	for i := range n {
		q += v[i]
	}
	return q
}`
	if got := countCat(scanSrc(t, src)); got["receiver-scratch-buffer"] != 1 {
		t.Fatalf("want 1 for a sole-indexed field with an ordinary name, got %d (%v)",
			got["receiver-scratch-buffer"], got)
	}
}

// SILENT when a second function also indexes the field: that is shared state, and the
// method being examined merely happens to fill and read it.
func TestDetectPS6006_SilentWhenAnotherFunctionAlsoIndexesIt(t *testing.T) {
	src := `package p
type B struct{ vals []float64 }
func (b *B) f(x []float64, n int) float64 {
	for i := range n {
		b.vals[i] = x[i] * 2
	}
	var q float64
	for i := range n {
		q += b.vals[i]
	}
	return q
}
func (b *B) g(i int) float64 { return b.vals[i] }`
	if got := countCat(scanSrc(t, src)); got["receiver-scratch-buffer"] != 0 {
		t.Fatalf("want 0 when a second function indexes the field, got %d (%v)",
			got["receiver-scratch-buffer"], got)
	}
}

// SILENT on an EXPORTED field: it is part of the type's API, so callers outside this file
// can read it and it cannot be a private temporary — whatever indexes it here.
func TestDetectPS6006_SilentOnExportedField(t *testing.T) {
	src := `package p
type M struct{ Means []float64 }
func (m *M) f(x []float64, n int) float64 {
	for i := range n {
		m.Means[i] = x[i]
	}
	var q float64
	for i := range n {
		q += m.Means[i]
	}
	return q
}`
	if got := countCat(scanSrc(t, src)); got["receiver-scratch-buffer"] != 0 {
		t.Fatalf("want 0 on an exported field, got %d (%v)", got["receiver-scratch-buffer"], got)
	}
}

// SILENT on a slice of RECORDS. Optimizer per-parameter state ([]soapState) and fitted
// models ([]*tree) are collections someone keeps, not working space, and they were the
// bulk of the false positives the structural test produced on its own.
func TestDetectPS6006_SilentOnSliceOfRecords(t *testing.T) {
	src := `package p
type entry struct{ v float64 }
type O struct{ st []entry }
func (o *O) step(x []float64, n int) float64 {
	for i := range n {
		o.st[i] = entry{x[i]}
	}
	var q float64
	for i := range n {
		q += o.st[i].v
	}
	return q
}`
	if got := countCat(scanSrc(t, src)); got["receiver-scratch-buffer"] != 0 {
		t.Fatalf("want 0 on a slice of records, got %d (%v)", got["receiver-scratch-buffer"], got)
	}
}

// A SLICE of the field is a read, not only an index. gbmBuilder.part is consumed entirely
// by copy(dst, b.part[:r]) — requiring an indexed read missed it even after the
// structural discriminator was in place.
func TestDetectPS6006_SliceExpressionCountsAsRead(t *testing.T) {
	src := `package p
type B struct{ part []int }
func (b *B) f(src, dst []int, n int) {
	r := 0
	for i := range n {
		b.part[r] = src[i]
		r++
	}
	copy(dst, b.part[:r])
}`
	if got := countCat(scanSrc(t, src)); got["receiver-scratch-buffer"] != 1 {
		t.Fatalf("want 1 when the field is read via a slice expression, got %d (%v)",
			got["receiver-scratch-buffer"], got)
	}
}

// The AQLM k-means shape: an expensive per-item call chooses WHERE to accumulate.
func TestDetectPS6007_SearchFeedsIndexedReduction(t *testing.T) {
	src := `package p
func f(data [][]float64, cent [][]float64, sums [][]float64, cnt []int, dim int) {
	for _, x := range data {
		b := nearest(x, cent)
		cnt[b]++
		for t := range dim {
			sums[b][t] += x[t]
		}
	}
}`
	if got := countCat(scanSrc(t, src)); got["search-feeds-reduction"] != 1 {
		t.Fatalf("want 1 search-feeds-reduction, got %d (%v)", got["search-feeds-reduction"], got)
	}
}

// A BLANK loop key must not hide it. The loop this check was written from is
// `for _, x := range data`, and an earlier version required a NAMED loop variable — so it
// missed its own motivating case. Replaying against the pre-fix revision is what showed
// that; no fixture written from the same mental model would have.
func TestDetectPS6007_BlankLoopKeyStillDetected(t *testing.T) {
	src := `package p
func f(data []float64, cnt []int) {
	for _, x := range data {
		b := bucket(x)
		cnt[b]++
	}
}`
	if got := countCat(scanSrc(t, src)); got["search-feeds-reduction"] != 1 {
		t.Fatalf("want 1 with a blank loop key, got %d (%v)", got["search-feeds-reduction"], got)
	}
}

// SILENT when the accumulation is not indexed by the searched value: an ordinary scalar
// sum fed by a call is every reduction loop ever written, and flagging those would bury
// the shape this check exists for. The distinctive part is that the INDEX makes the loop
// look partitionable when it is not.
func TestDetectPS6007_SilentOnScalarAccumulation(t *testing.T) {
	src := `package p
func f(data []float64) float64 {
	var total float64
	for _, x := range data {
		v := score(x)
		total += v
	}
	return total
}`
	if got := countCat(scanSrc(t, src)); got["search-feeds-reduction"] != 0 {
		t.Fatalf("want 0 on a scalar accumulation, got %d (%v)", got["search-feeds-reduction"], got)
	}
}

// SILENT when the searched value indexes a plain STORE rather than an accumulation. A
// store is idempotent — the last writer wins and no order is being preserved — so the
// loop does not carry the reduction this check is about. The store is indexed BY the
// searched value on purpose, so this fixture isolates the accumulation requirement rather
// than passing on the index check.
func TestDetectPS6007_SilentOnIndexedStore(t *testing.T) {
	src := `package p
func f(data []float64, out []float64) {
	for _, x := range data {
		b := bucket(x)
		out[b] = x
	}
}`
	if got := countCat(scanSrc(t, src)); got["search-feeds-reduction"] != 0 {
		t.Fatalf("want 0 on an indexed store, got %d (%v)", got["search-feeds-reduction"], got)
	}
}

// SILENT when the index is not the result of a call: a loop variable indexing an
// accumulation is an ordinary strided reduction, not a search feeding one.
func TestDetectPS6007_SilentWhenIndexIsNotFromACall(t *testing.T) {
	src := `package p
func f(data []float64, sums []float64, m int) {
	for i, x := range data {
		b := i % m
		sums[b] += x
	}
}`
	if got := countCat(scanSrc(t, src)); got["search-feeds-reduction"] != 0 {
		t.Fatalf("want 0 when the index is not from a call, got %d (%v)",
			got["search-feeds-reduction"], got)
	}
}

// SILENT when an accumulation exists but is indexed by the LOOP variable rather than by
// the searched value. Each item then accumulates into its own slot, so the loop already
// partitions and there is nothing to split. This is the fixture that isolates the
// index-mentions-search requirement: the other silent cases are rejected earlier, by the
// call check or by having no index at all.
func TestDetectPS6007_SilentWhenAccumulationIndexedByLoopVar(t *testing.T) {
	src := `package p
func f(data []float64, sums []float64) {
	for i, x := range data {
		b := score(x)
		sums[i] += b
	}
}`
	if got := countCat(scanSrc(t, src)); got["search-feeds-reduction"] != 0 {
		t.Fatalf("want 0 when the accumulation is indexed by the loop variable, got %d (%v)",
			got["search-feeds-reduction"], got)
	}
}

// The GBM shape: scratch allocated inside the parallel body, once per dispatch.
func TestDetectPS6008_AllocInParallelBody(t *testing.T) {
	src := `package p
func f(d, n int, cols [][]int) {
	parallelFeatures(d, n, func(lo, hi int) {
		vals := make([]float64, n)
		_ = vals
	})
}`
	if got := countCat(scanSrc(t, src)); got["alloc-in-parallel-body"] != 1 {
		t.Fatalf("want 1 alloc-in-parallel-body, got %d (%v)", got["alloc-in-parallel-body"], got)
	}
}

// The dispatch may be reached through the package helper directly.
func TestDetectPS6008_AllocInParallelRowsIdxBody(t *testing.T) {
	src := `package p
func f(n int) {
	parallel.Rows(n, func(lo, hi int) {
		buf := make([]int, n)
		_ = buf
	})
}`
	if got := countCat(scanSrc(t, src)); got["alloc-in-parallel-body"] != 1 {
		t.Fatalf("want 1 for parallel.Rows, got %d (%v)", got["alloc-in-parallel-body"], got)
	}
}

// NO LOCAL LOOP IS REQUIRED, and that is deliberate. The GBM case has none: bestSplit
// contains no loop around its dispatch — bestSplit ITSELF runs once per tree node, one
// call frame up. A check that demanded a visible enclosing loop would have missed the
// only case that mattered, which is why this fixture has a bare function.
func TestDetectPS6008_NoEnclosingLoopStillReported(t *testing.T) {
	src := `package p
func bestSplit(d, n int) {
	parallelFeatures(d, n, func(lo, hi int) {
		vals := make([]float64, n)
		_ = vals
	})
}`
	if got := countCat(scanSrc(t, src)); got["alloc-in-parallel-body"] != 1 {
		t.Fatalf("want 1 without an enclosing loop, got %d (%v)",
			got["alloc-in-parallel-body"], got)
	}
}

// SILENT when the buffer is hoisted out of the body — the fix this check asks for.
func TestDetectPS6008_SilentWhenHoistedOutOfTheBody(t *testing.T) {
	src := `package p
func f(d, n int, scratch [][]float64) {
	parallelFeaturesIdx(d, n, func(ci, lo, hi int) {
		vals := scratch[ci][:n]
		_ = vals
	})
}`
	if got := countCat(scanSrc(t, src)); got["alloc-in-parallel-body"] != 0 {
		t.Fatalf("want 0 when the buffer is per-chunk and hoisted, got %d (%v)",
			got["alloc-in-parallel-body"], got)
	}
}

// SILENT on an allocation in an ordinary closure: this check is about the cost of a
// parallel FAN-OUT repeating an allocation, not about allocation in general, which
// PS2001 and PS2004 already cover.
func TestDetectPS6008_SilentOnNonParallelCallback(t *testing.T) {
	src := `package p
func f(n int) {
	forEach(n, func(lo, hi int) {
		buf := make([]int, n)
		_ = buf
	})
}`
	if got := countCat(scanSrc(t, src)); got["alloc-in-parallel-body"] != 0 {
		t.Fatalf("want 0 on a non-parallel callback, got %d (%v)",
			got["alloc-in-parallel-body"], got)
	}
}

// SILENT on a non-allocating define inside a parallel body. A parallel body naturally
// declares locals — slicing a shared buffer, reading a bound — and none of that repeats
// an allocation. This fixture isolates the make() requirement; without it, relaxing that
// check to accept any define leaves every other PS6008 fixture green.
func TestDetectPS6008_SilentOnNonAllocatingDefineInBody(t *testing.T) {
	src := `package p
func f(d, n int, shared []float64) {
	parallelFeatures(d, n, func(lo, hi int) {
		row := shared[lo:hi]
		total := compute(row)
		_ = total
	})
}`
	if got := countCat(scanSrc(t, src)); got["alloc-in-parallel-body"] != 0 {
		t.Fatalf("want 0 for a non-allocating define, got %d (%v)",
			got["alloc-in-parallel-body"], got)
	}
}

// sort.Slice allocates a reflect swapper on every call.
func TestDetectPS6009_SortSlice(t *testing.T) {
	src := `package p
func f(idx []int, key []float64) {
	sort.Slice(idx, func(a, b int) bool { return key[idx[a]] < key[idx[b]] })
}`
	if got := countCat(scanSrc(t, src)); got["reflect-swapper-sort"] != 1 {
		t.Fatalf("want 1 reflect-swapper-sort, got %d (%v)", got["reflect-swapper-sort"], got)
	}
}

// SliceStable has the same swapper, so it is reported too — with SortStableFunc as its
// counterpart, since swapping a stable sort for an unstable one changes tie order.
func TestDetectPS6009_SortSliceStable(t *testing.T) {
	src := `package p
func f(rows []row) {
	sort.SliceStable(rows, func(a, b int) bool { return rows[a].k < rows[b].k })
}`
	got := scanSrc(t, src)
	if countCat(got)["reflect-swapper-sort"] != 1 {
		t.Fatalf("want 1 for SliceStable, got %v", countCat(got))
	}
	for _, f := range got {
		if f.category == "reflect-swapper-sort" && !strings.Contains(f.msg, "SortStableFunc") {
			t.Errorf("SliceStable must be pointed at SortStableFunc, got %q", f.msg)
		}
	}
}

// SILENT ON ITS OWN FIX — the property PS3002 lacks. That check kept flagging the
// slices.SortFunc replacement it had recommended, so the site could only be silenced with
// a suppression, never cleared. A check that cannot recognize its own remedy cannot tell
// you whether the work is done.
func TestDetectPS6009_SilentOnSlicesSortFunc(t *testing.T) {
	src := `package p
func f(idx []int, key []float64) {
	slices.SortFunc(idx, func(a, b int) int {
		switch {
		case key[a] < key[b]:
			return -1
		case key[a] > key[b]:
			return 1
		}
		return 0
	})
}`
	if got := countCat(scanSrc(t, src)); got["reflect-swapper-sort"] != 0 {
		t.Fatalf("want 0 on slices.SortFunc — the fix must clear the finding, got %d (%v)",
			got["reflect-swapper-sort"], got)
	}
}

// SILENT on sort.Ints and friends: those are concrete, not reflection-based.
func TestDetectPS6009_SilentOnConcreteSorts(t *testing.T) {
	src := `package p
func f(xs []int, ss []string) {
	sort.Ints(xs)
	sort.Strings(ss)
}`
	if got := countCat(scanSrc(t, src)); got["reflect-swapper-sort"] != 0 {
		t.Fatalf("want 0 on concrete sorts, got %d (%v)", got["reflect-swapper-sort"], got)
	}
}

// SILENT on a same-named method that is not the sort package.
func TestDetectPS6009_SilentOnUnrelatedSliceMethod(t *testing.T) {
	src := `package p
func f(db store, xs []int) {
	db.Slice(xs, nil)
}`
	if got := countCat(scanSrc(t, src)); got["reflect-swapper-sort"] != 0 {
		t.Fatalf("want 0 on an unrelated Slice call, got %d (%v)", got["reflect-swapper-sort"], got)
	}
}

// PS6011 fires on the KDA decay shape: the inner loop scales S by the row stride.
func TestDetectPS6011_StridedInnerWalk(t *testing.T) {
	src := `package p
func Decay(S, at []float64, dv, dk int) {
	for c := range dk {
		ac := at[c]
		for r := range dv {
			S[r*dk+c] *= ac
		}
	}
}`
	if got := countCat(scanSrc(t, src))["strided-inner-walk"]; got != 1 {
		t.Fatalf("want 1 strided-inner-walk on the column walk, got %d", got)
	}
}

// The inner loop is often GUARDED rather than a direct child of the outer body — NSA's P*V
// sits inside an `if sum > 0`. The first draft of this check walked outerBody.List only and
// missed exactly this, its own motivating case.
func TestDetectPS6011_InnerLoopBehindGuard(t *testing.T) {
	src := `package p
func PV(vs, scores, orow []float64, dk, dm, off, i int, sum float64) {
	for d := range dk {
		var o float64
		if sum > 0 {
			for j := 0; j <= i; j++ {
				o += scores[j] * vs[j*dm+off+d]
			}
		}
		orow[d] = o
	}
}`
	if got := countCat(scanSrc(t, src))["strided-inner-walk"]; got != 1 {
		t.Fatalf("want 1 strided-inner-walk behind an if guard, got %d", got)
	}
}

// SILENT on correct row-major traversal, where the INNER variable is the additive one.
func TestDetectPS6011_SilentOnRowMajor(t *testing.T) {
	src := `package p
func Scale(S, at []float64, dv, dk int) {
	for r := range dv {
		for c := range dk {
			S[r*dk+c] *= at[c]
		}
	}
}`
	if got := countCat(scanSrc(t, src))["strided-inner-walk"]; got != 0 {
		t.Fatalf("want 0 on contiguous row-major iteration, got %d", got)
	}
}

// SILENT on a transpose: it strides on one side whichever way it is iterated, so the
// interchange this check advises would only move the stride to the other operand.
func TestDetectPS6011_SilentOnTranspose(t *testing.T) {
	src := `package p
func T(x []float64, r, c int) []float64 {
	out := make([]float64, r*c)
	for i := range r {
		for j := range c {
			out[j*r+i] = x[i*c+j]
		}
	}
	return out
}`
	if got := countCat(scanSrc(t, src))["strided-inner-walk"]; got != 0 {
		t.Fatalf("want 0 on a transpose, got %d", got)
	}
}

// SILENT when the two loop variables never meet in one index — the axes are not
// interchangeable and there is nothing to advise.
func TestDetectPS6011_SilentWhenAxesDoNotMeet(t *testing.T) {
	src := `package p
func F(a, b []float64, n, m, stride int) {
	for i := range n {
		for j := range m {
			a[j*stride] += b[i]
		}
	}
}`
	if got := countCat(scanSrc(t, src))["strided-inner-walk"]; got != 0 {
		t.Fatalf("want 0 when only one loop var reaches the index, got %d", got)
	}
}

// SILENT on a permutation copy whose stride was HOISTED out of the inner loop. The
// transpose check above is syntactic and cannot see the mirrored multiplication once
// `row := i*b` moves it, which is exactly how nlp's already-tiled gguf transposes were
// being flagged.
func TestDetectPS6011_SilentOnHoistedStrideTranspose(t *testing.T) {
	src := `package p
func T(dst, src []float64, a, b int) {
	for i := 0; i < a; i++ {
		row := i * b
		for j := 0; j < b; j++ {
			dst[j*a+i] = src[row+j]
		}
	}
}`
	if got := countCat(scanSrc(t, src))["strided-inner-walk"]; got != 0 {
		t.Fatalf("want 0 on a hoisted-stride permutation copy, got %d", got)
	}
}

// SILENT when the source is read through an ACCESSOR rather than an index — the generic
// fallback arms of those same transposes.
func TestDetectPS6011_SilentOnAccessorPermutationCopy(t *testing.T) {
	src := `package p
func T(dst []float64, tc *T2, a, b int) {
	for i := 0; i < a; i++ {
		for j := 0; j < b; j++ {
			dst[j*a+i] = tc.AtF64(i, j)
		}
	}
}`
	if got := countCat(scanSrc(t, src))["strided-inner-walk"]; got != 0 {
		t.Fatalf("want 0 on an accessor permutation copy, got %d", got)
	}
}

// STILL FIRES when the strided write is an ACCUMULATION rather than a copy — the
// suppression must not swallow a genuine reduction that happens to look assignment-shaped.
func TestDetectPS6011_FiresOnStridedAccumulation(t *testing.T) {
	src := `package p
func Acc(dst, a, b []float64, n, m, stride int) {
	for c := range n {
		for r := range m {
			dst[r*stride+c] = dst[r*stride+c] + a[r]*b[c]
		}
	}
}`
	if got := countCat(scanSrc(t, src))["strided-inner-walk"]; got != 1 {
		t.Fatalf("want 1 on a strided accumulation, got %d", got)
	}
}

// PS6012 fires on a product assigned to a NAMED LOCAL and then used in a subtract, in a
// function that pins other products. This is the exact shape that cost three attempts on the
// Titans fused path: naming a subexpression does not stop the compiler inlining and
// contracting it.
func TestDetectPS6012_UnpinnedNamedLocal(t *testing.T) {
	src := `package p
func F(g, s []float64, th, et float64, n int) {
	for i := range n {
		inc := g[i] * th
		s[i] = float64(s[i]*et) - inc
	}
}`
	if got := countCat(scanSrc(t, src))["inconsistent-fma-pinning"]; got != 1 {
		t.Fatalf("want 1 inconsistent-fma-pinning on the named local, got %d", got)
	}
}

// SILENT once the product is pinned — the fix must clear the finding.
func TestDetectPS6012_SilentWhenPinned(t *testing.T) {
	src := `package p
func F(g, s []float64, th, et float64, n int) {
	for i := range n {
		inc := float64(g[i] * th)
		s[i] = float64(s[i]*et) - inc
	}
}`
	if got := countCat(scanSrc(t, src))["inconsistent-fma-pinning"]; got != 0 {
		t.Fatalf("want 0 once pinned, got %d", got)
	}
}

// SILENT in a function that pins nothing: it is not claiming bit-exactness against a
// separately-rounded path, so contraction is none of this check's business.
func TestDetectPS6012_SilentWithoutAnyPinning(t *testing.T) {
	src := `package p
func F(a, b, c []float64, k float64, n int) {
	for i := range n {
		c[i] = a[i]*k + b[i]
	}
}`
	if got := countCat(scanSrc(t, src))["inconsistent-fma-pinning"]; got != 0 {
		t.Fatalf("want 0 without any pinning signal, got %d", got)
	}
}

// SILENT on integer offset arithmetic, whether in a subscript or computed into a call
// argument. FMA has nothing to do with index math, and without types the only tell is that
// it touches no memory.
func TestDetectPS6012_SilentOnIndexArithmetic(t *testing.T) {
	src := `package p
func F(s, d []float64, get func(int) float64, rows, cols, half int, k float64) {
	for p := range rows {
		for e := range half {
			row := p*cols + e
			d[row] = float64(s[row] * k)
			_ = get(p*cols + e)
		}
	}
}`
	if got := countCat(scanSrc(t, src))["inconsistent-fma-pinning"]; got != 0 {
		t.Fatalf("want 0 on integer offset arithmetic, got %d", got)
	}
}

// SILENT when the only conversion is a float32 STORE rounding. float32(a*b) on an F32 path
// is how a result is written, not a declaration that contraction matters; treating it as a
// pinning signal made this check fire in every typed F32 branch in the tree.
func TestDetectPS6012_SilentOnF32StoreRounding(t *testing.T) {
	src := `package p
func F(dst []float32, a, b []float64, k float64, n int) {
	for i := range n {
		dst[i] = float32(a[i] * k)
		dst[i] = float32(a[i]*k + b[i])
	}
}`
	if got := countCat(scanSrc(t, src))["inconsistent-fma-pinning"]; got != 0 {
		t.Fatalf("want 0 when the only conversion is an F32 store rounding, got %d", got)
	}
}

// PS6013 fires when a full sort's only later reader is a counted prefix loop.
func TestDetectPS6013_SortFeedsCountedPrefix(t *testing.T) {
	src := `package p
import "slices"
func f(idx []int, d []bool, k int) {
	slices.SortFunc(idx, func(x, y int) int { return 0 })
	for r := 0; r < k; r++ {
		d[idx[r]] = true
	}
}`
	if got := countCat(scanSrc(t, src))["sort-feeds-counted-prefix"]; got != 1 {
		t.Fatalf("want 1 sort-feeds-counted-prefix, got %d", got)
	}
}

// The sort is often behind a local CLOSURE that captures the slice — the shape the rule was
// built from. A detector matching only direct calls missed its own motivating case.
func TestDetectPS6013_SortBehindClosure(t *testing.T) {
	src := `package p
import "slices"
func f(idx []int, col []float64, d []bool, k int) {
	sortCol := func(c []float64) {
		slices.SortFunc(idx, func(x, y int) int { return 0 })
	}
	sortCol(col)
	for r := 0; r < k; r++ {
		d[idx[r]] = true
	}
}`
	if got := countCat(scanSrc(t, src))["sort-feeds-counted-prefix"]; got != 1 {
		t.Fatalf("want 1 when the sort is behind a closure, got %d", got)
	}
}

// SILENT when something else reads the sorted slice afterwards — then the full order is
// load-bearing and the sort must stay. This is the soundness condition.
func TestDetectPS6013_SilentWhenOrderUsedElsewhere(t *testing.T) {
	src := `package p
import "slices"
func f(idx []int, d []bool, out []int, k int) {
	slices.SortFunc(idx, func(x, y int) int { return 0 })
	for r := 0; r < k; r++ {
		d[idx[r]] = true
	}
	copy(out, idx)
}`
	if got := countCat(scanSrc(t, src))["sort-feeds-counted-prefix"]; got != 0 {
		t.Fatalf("want 0 when the full order is read again, got %d", got)
	}
}

// SILENT when the prefix is the WHOLE slice — len(idx) reads everything, so nothing is
// discarded and a selection buys nothing.
func TestDetectPS6013_SilentWhenPrefixIsWholeSlice(t *testing.T) {
	src := `package p
import "slices"
func f(idx []int, d []bool) {
	slices.SortFunc(idx, func(x, y int) int { return 0 })
	for r := 0; r < len(idx); r++ {
		d[idx[r]] = true
	}
}`
	if got := countCat(scanSrc(t, src))["sort-feeds-counted-prefix"]; got != 0 {
		t.Fatalf("want 0 when the loop covers the whole slice, got %d", got)
	}
}

// SILENT when a same-named method is not the sort package — Sort on a receiver is not this.
func TestDetectPS6013_SilentOnUnrelatedSortMethod(t *testing.T) {
	src := `package p
func f(db store, idx []int, d []bool, k int) {
	db.SortFunc(idx, nil)
	for r := 0; r < k; r++ {
		d[idx[r]] = true
	}
}`
	if got := countCat(scanSrc(t, src))["sort-feeds-counted-prefix"]; got != 0 {
		t.Fatalf("want 0 on an unrelated SortFunc method, got %d", got)
	}
}

// The rl.DQN.learn shape: the same pure forward twice, differing only in the leading
// context argument, with only a fresh tensor written in between.
func TestDetectPS6014_RedundantPureRecompute(t *testing.T) {
	src := `package p
func learn(d *D, states [][]float64, k int) error {
	qPred, err := forward(NewContext(), d.Net, states)
	if err != nil {
		return err
	}
	target := New(F64, Shape{len(states), k})
	for i := range states {
		for a := range k {
			target.SetF64(qPred.AtF64(i, a), i, a)
		}
	}
	q, err := forward(tape.Context(), d.Net, states)
	if err != nil {
		return err
	}
	return use(q, target)
}`
	if got := countCat(scanSrc(t, src))["redundant-pure-recompute"]; got != 1 {
		t.Fatalf("want 1 redundant-pure-recompute, got %d", got)
	}
}

// An assignment to one of the arguments between the two calls means the second genuinely
// asks a different question.
func TestDetectPS6014_SilentWhenAnArgumentIsReassigned(t *testing.T) {
	src := `package p
func f(d *D, states [][]float64) error {
	a, err := forward(NewContext(), d.Net, states)
	if err != nil {
		return err
	}
	states = nextBatch()
	b, err := forward(NewContext(), d.Net, states)
	if err != nil {
		return err
	}
	return use(a, b)
}`
	if got := countCat(scanSrc(t, src))["redundant-pure-recompute"]; got != 0 {
		t.Fatalf("want 0 when an argument is reassigned, got %d", got)
	}
}

// A non-pure call handed one of the arguments may mutate what it points at — the classic
// case being an optimizer step between a preview forward and the real one.
func TestDetectPS6014_SilentWhenAnImpureCallTouchesAnArgument(t *testing.T) {
	src := `package p
func f(d *D, states [][]float64) error {
	a, err := forward(NewContext(), d.Net, states)
	if err != nil {
		return err
	}
	applyGradients(d.Net)
	b, err := forward(NewContext(), d.Net, states)
	if err != nil {
		return err
	}
	return use(a, b)
}`
	if got := countCat(scanSrc(t, src))["redundant-pure-recompute"]; got != 0 {
		t.Fatalf("want 0 when an impure call touches an argument, got %d", got)
	}
}

// A mutation hidden inside a loop between the two calls must still suppress: the
// invalidation scan descends into nested nodes rather than only inspecting top-level
// statements, which is the failure mode three earlier rules in this file shipped with.
func TestDetectPS6014_SilentWhenTheWriteIsNestedInALoop(t *testing.T) {
	src := `package p
func f(d *D, states [][]float64) error {
	a, err := forward(NewContext(), d.Net, states)
	if err != nil {
		return err
	}
	for i := range states {
		if i > 0 {
			states[i] = perturb(states[i])
		}
	}
	b, err := forward(NewContext(), d.Net, states)
	if err != nil {
		return err
	}
	return use(a, b)
}`
	if got := countCat(scanSrc(t, src))["redundant-pure-recompute"]; got != 0 {
		t.Fatalf("want 0 when the write is nested in a loop, got %d", got)
	}
}

// A callee NOT declared pure is never flagged, however identical the two calls look —
// repeated rng draws and repeated env steps are the whole point of those calls.
func TestDetectPS6014_SilentOnAnUndeclaredCallee(t *testing.T) {
	src := `package p
func f(d *D, n int, bound int) int {
	a, _ := sampleIndex(d.rng, n, bound)
	b, _ := sampleIndex(d.rng, n, bound)
	return a + b
}`
	if got := countCat(scanSrc(t, src))["redundant-pure-recompute"]; got != 0 {
		t.Fatalf("want 0 for a callee outside pureComputeFuncs, got %d", got)
	}
}

// Differing arguments are not a recompute.
func TestDetectPS6014_SilentOnDifferentArguments(t *testing.T) {
	src := `package p
func f(d *D, states, nexts [][]float64) error {
	a, err := forward(NewContext(), d.Net, states)
	if err != nil {
		return err
	}
	b, err := forward(NewContext(), d.Net, nexts)
	if err != nil {
		return err
	}
	return use(a, b)
}`
	if got := countCat(scanSrc(t, src))["redundant-pure-recompute"]; got != 0 {
		t.Fatalf("want 0 for different arguments, got %d", got)
	}
}

// The rl.rlRollout critic: a batch-of-one pure call per iteration whose result is only
// appended to a slice consumed after the loop.
func TestDetectPS6015_Batch1FeedsPostloopSlice(t *testing.T) {
	src := `package p
func rollout(critic *Net, ro *R, obs []float64, steps int) error {
	for i := 0; i < steps; i++ {
		v, err := forward(NewContext(), critic, [][]float64{obs})
		if err != nil {
			return err
		}
		ro.values = append(ro.values, v.AtF64(0, 0))
	}
	return nil
}`
	if got := countCat(scanSrc(t, src))["batch1-call-feeds-only-postloop-slice"]; got != 1 {
		t.Fatalf("want 1 batch1-call-feeds-only-postloop-slice, got %d", got)
	}
}

// THE case that must stay silent: the actor in the same loop, whose result feeds the action
// that feeds the environment. It is the same call shape and it cannot be hoisted — PS1003
// covers it with different advice. A rule that flagged this would propose a hoist that
// changes behavior, which is worse than saying nothing.
func TestDetectPS6015_SilentWhenTheResultDrivesTheLoop(t *testing.T) {
	src := `package p
func rollout(actor *Net, ro *R, obs []float64, steps int, k int) error {
	for i := 0; i < steps; i++ {
		logits, err := forward(NewContext(), actor, [][]float64{obs})
		if err != nil {
			return err
		}
		a := sampleAction(logits, k)
		ro.actions = append(ro.actions, a)
		obs = step(a)
	}
	return nil
}`
	if got := countCat(scanSrc(t, src))["batch1-call-feeds-only-postloop-slice"]; got != 0 {
		t.Fatalf("want 0 when the result drives the loop, got %d", got)
	}
}

// A use in a branch condition is loop-carried even though an append is also present.
func TestDetectPS6015_SilentOnABranchUse(t *testing.T) {
	src := `package p
func f(critic *Net, ro *R, obs []float64, steps int) error {
	for i := 0; i < steps; i++ {
		v, err := forward(NewContext(), critic, [][]float64{obs})
		if err != nil {
			return err
		}
		ro.values = append(ro.values, v.AtF64(0, 0))
		if v.AtF64(0, 0) > 1 {
			break
		}
	}
	return nil
}`
	if got := countCat(scanSrc(t, src))["batch1-call-feeds-only-postloop-slice"]; got != 0 {
		t.Fatalf("want 0 when the result is read in a branch, got %d", got)
	}
}

// Appending to a slice declared INSIDE the loop proves nothing outlives the iteration, so
// there is no batched call to hoist to.
func TestDetectPS6015_SilentOnALoopLocalTarget(t *testing.T) {
	src := `package p
func f(critic *Net, obs []float64, steps int) error {
	for i := 0; i < steps; i++ {
		var acc []float64
		v, err := forward(NewContext(), critic, [][]float64{obs})
		if err != nil {
			return err
		}
		acc = append(acc, v.AtF64(0, 0))
		consume(acc)
	}
	return nil
}`
	if got := countCat(scanSrc(t, src))["batch1-call-feeds-only-postloop-slice"]; got != 0 {
		t.Fatalf("want 0 for a loop-local append target, got %d", got)
	}
}

// A callee outside pureComputeFuncs cannot be hoisted across a loop at all — it might
// consume RNG, and moving draws out of the stream changes every later iteration.
func TestDetectPS6015_SilentOnAnUndeclaredCallee(t *testing.T) {
	src := `package p
func f(net *Net, ro *R, obs []float64, steps int) error {
	for i := 0; i < steps; i++ {
		v, err := sampleForward(NewContext(), net, [][]float64{obs})
		if err != nil {
			return err
		}
		ro.values = append(ro.values, v.AtF64(0, 0))
	}
	return nil
}`
	if got := countCat(scanSrc(t, src))["batch1-call-feeds-only-postloop-slice"]; got != 0 {
		t.Fatalf("want 0 for a callee outside pureComputeFuncs, got %d", got)
	}
}

// A multi-element batch is already batched; there is nothing to collapse.
func TestDetectPS6015_SilentOnAMultiElementBatch(t *testing.T) {
	src := `package p
func f(critic *Net, ro *R, a, b []float64, steps int) error {
	for i := 0; i < steps; i++ {
		v, err := forward(NewContext(), critic, [][]float64{a, b})
		if err != nil {
			return err
		}
		ro.values = append(ro.values, v.AtF64(0, 0))
	}
	return nil
}`
	if got := countCat(scanSrc(t, src))["batch1-call-feeds-only-postloop-slice"]; got != 0 {
		t.Fatalf("want 0 for a multi-element batch, got %d", got)
	}
}

// Different receivers, same method and same argument: the three projections at the top of
// every attention block. calleeName collapses a qualified call to its last segment, so
// keying on that alone reported all three as recomputes of each other — and that accounted
// for EVERY hit this rule produced outside the package it was built from. The key carries
// the full callee expression to keep them distinct.
//
// The method is spelled with the name the configured vocabulary actually contains. Spelled
// otherwise the assertion passes because nothing here is a pure call at all — which is how
// this test was first written, and the companion FiresOnTheSameReceiverTwice floor is what
// exposed it.
func TestDetectPS6014_SilentOnDifferentReceivers(t *testing.T) {
	src := `package p
func attn(b *Block, ctx *C, xn T) error {
	q, err := b.Wq.forward(ctx, xn)
	if err != nil {
		return err
	}
	k, err := b.Wk.forward(ctx, xn)
	if err != nil {
		return err
	}
	v, err := b.Wv.forward(ctx, xn)
	if err != nil {
		return err
	}
	return use(q, k, v)
}`
	if got := countCat(scanSrc(t, src))["redundant-pure-recompute"]; got != 0 {
		t.Fatalf("want 0 for three different receivers, got %d", got)
	}
}

// The same receiver and the same argument IS a recompute, so the receiver fix must not
// silence the real case — this is the floor against over-suppression.
func TestDetectPS6014_FiresOnTheSameReceiverTwice(t *testing.T) {
	src := `package p
func f(b *Block, ctx *C, xn T) error {
	a, err := b.Wq.forward(ctx, xn)
	if err != nil {
		return err
	}
	c, err := b.Wq.forward(tape.Context(), xn)
	if err != nil {
		return err
	}
	return use(a, c)
}`
	if got := countCat(scanSrc(t, src))["redundant-pure-recompute"]; got != 1 {
		t.Fatalf("want 1 for the same receiver twice, got %d", got)
	}
}

// The nlp decode shape: attrs literals rebuilt per layer from layer-independent config.
func TestDetectPS6016_LoopInvariantLiteralArg(t *testing.T) {
	src := `package p
func decode(m *M, ctx *C, cfg *Cfg, pos, kv int, q T) error {
	for l, b := range m.Blocks {
		if _, err := exec1(ctx, OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: pos}, q); err != nil {
			return err
		}
		use(l, b)
	}
	return nil
}`
	if got := countCat(scanSrc(t, src))["loop-invariant-literal-arg"]; got != 1 {
		t.Fatalf("want 1 loop-invariant-literal-arg, got %d", got)
	}
}

// A field initializer that reads the loop variable is not invariant — the whole point.
func TestDetectPS6016_SilentWhenAFieldReadsTheLoopVar(t *testing.T) {
	src := `package p
func decode(m *M, ctx *C, cfg *Cfg, q T) error {
	for l, b := range m.Blocks {
		if _, err := exec1(ctx, OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Layer: l}, q); err != nil {
			return err
		}
		use(b)
	}
	return nil
}`
	if got := countCat(scanSrc(t, src))["loop-invariant-literal-arg"]; got != 0 {
		t.Fatalf("want 0 when a field reads the loop variable, got %d", got)
	}
}

// A field reading something the loop ASSIGNS is not invariant either, even though the name
// is not an induction variable.
func TestDetectPS6016_SilentWhenAFieldReadsALoopAssignedName(t *testing.T) {
	src := `package p
func decode(m *M, ctx *C, cfg *Cfg, q T) error {
	pos := 0
	for l, b := range m.Blocks {
		pos = pos + 1
		if _, err := exec1(ctx, OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, PosOffset: pos}, q); err != nil {
			return err
		}
		use(l, b)
	}
	return nil
}`
	if got := countCat(scanSrc(t, src))["loop-invariant-literal-arg"]; got != 0 {
		t.Fatalf("want 0 when a field reads a loop-assigned name, got %d", got)
	}
}

// Appending the literal needs its per-iteration identity: hoisting would make every element
// alias one value, which is a correctness change, not an optimization.
func TestDetectPS6016_SilentOnAnAppendedLiteral(t *testing.T) {
	src := `package p
func f(m *M, cfg *Cfg) []T {
	var out []T
	for l, b := range m.Blocks {
		out = append(out, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads})
		use(l, b)
	}
	return out
}`
	if got := countCat(scanSrc(t, src))["loop-invariant-literal-arg"]; got != 0 {
		t.Fatalf("want 0 for an appended literal, got %d", got)
	}
}

// Slice and map literals are a different question (pooling, PS2001), not hoisting.
func TestDetectPS6016_SilentOnSliceAndMapLiterals(t *testing.T) {
	src := `package p
func f(m *M, ctx *C, a, b2 T) error {
	for l, b := range m.Blocks {
		if _, err := exec(ctx, []*T{a, b2}); err != nil {
			return err
		}
		if _, err := exec2(ctx, map[string]int{"k": 1}); err != nil {
			return err
		}
		use(l, b)
	}
	return nil
}`
	if got := countCat(scanSrc(t, src))["loop-invariant-literal-arg"]; got != 0 {
		t.Fatalf("want 0 for slice/map literals, got %d", got)
	}
}

// Two distinct literals of the SAME type in one loop must both report — the q and k RoPE
// attrs of a decode loop are exactly that, and a per-type dedup hid the second.
func TestDetectPS6016_ReportsTwoLiteralsOfOneType(t *testing.T) {
	src := `package p
func decode(m *M, ctx *C, cfg *Cfg, pos, kv int, q, k T) error {
	for l, b := range m.Blocks {
		if _, err := exec1(ctx, OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: pos}, q); err != nil {
			return err
		}
		if _, err := exec1(ctx, OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv, PosOffset: pos}, k); err != nil {
			return err
		}
		use(l, b)
	}
	return nil
}`
	if got := countCat(scanSrc(t, src))["loop-invariant-literal-arg"]; got != 2 {
		t.Fatalf("want 2 for two distinct literals of one type, got %d", got)
	}
}

// scanSrcPkg runs the package-level pre-pass PS6017 depends on before scanning. scanSrc
// alone cannot exercise this rule: the sibling registry is built across files, so without
// the pre-pass the rule has an empty table and every assertion of zero passes vacuously.
func scanSrcPkg(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	variadicSiblings = map[string]map[string]variadicFamily{}
	collectVariadicSiblings(fset, []*ast.File{f})
	return scanFile(fset, f, testSets(t))
}

const variadicFamilySrc = `
func exec1(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	return nil, nil
}
func exec1a(ctx *backend.Context, op backend.Op, attrs backend.Attrs, a *tensor.Tensor) (*tensor.Tensor, error) {
	return nil, nil
}
func exec3(ctx *backend.Context, op backend.Op, attrs backend.Attrs, a, b, c *tensor.Tensor) (*tensor.Tensor, error) {
	return nil, nil
}
`

// The nlp decode shape: a variadic exec called in a layer loop at an arity a pooled sibling
// covers. Grouped parameters (a, b, c *tensor.Tensor) must expand to three, not one.
func TestDetectPS6017_UnpooledVariadicSibling(t *testing.T) {
	src := `package p` + variadicFamilySrc + `
func decode(m *M, ctx *backend.Context, attn backend.Attrs, q, k, v *tensor.Tensor) error {
	for l, b := range m.Blocks {
		if _, err := exec1(ctx, backend.OpMHA, attn, q, k, v); err != nil {
			return err
		}
		use(l, b)
	}
	return nil
}`
	got := countCat(scanSrcPkg(t, src))["unpooled-variadic-sibling"]
	if got != 1 {
		t.Fatalf("want 1 unpooled-variadic-sibling, got %d", got)
	}
}

// An arity no sibling covers has nothing to switch to.
func TestDetectPS6017_SilentOnAnUncoveredArity(t *testing.T) {
	src := `package p` + variadicFamilySrc + `
func f(m *M, ctx *backend.Context, attn backend.Attrs, a, b2 *tensor.Tensor) error {
	for l, b := range m.Blocks {
		if _, err := exec1(ctx, backend.OpAdd, attn, a, b2); err != nil {
			return err
		}
		use(l, b)
	}
	return nil
}`
	if got := countCat(scanSrcPkg(t, src))["unpooled-variadic-sibling"]; got != 0 {
		t.Fatalf("want 0 when no sibling has that arity (2), got %d", got)
	}
}

// A spread call has no statically known arity.
func TestDetectPS6017_SilentOnASpreadCall(t *testing.T) {
	src := `package p` + variadicFamilySrc + `
func f(m *M, ctx *backend.Context, attn backend.Attrs, ins []*tensor.Tensor) error {
	for l, b := range m.Blocks {
		if _, err := exec1(ctx, backend.OpMHA, attn, ins...); err != nil {
			return err
		}
		use(l, b)
	}
	return nil
}`
	if got := countCat(scanSrcPkg(t, src))["unpooled-variadic-sibling"]; got != 0 {
		t.Fatalf("want 0 for a spread call, got %d", got)
	}
}

// Outside a loop the allocation is once per invocation, which is rarely worth a diff.
func TestDetectPS6017_SilentOutsideALoop(t *testing.T) {
	src := `package p` + variadicFamilySrc + `
func f(ctx *backend.Context, attn backend.Attrs, q, k, v *tensor.Tensor) error {
	_, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
	return err
}`
	if got := countCat(scanSrcPkg(t, src))["unpooled-variadic-sibling"]; got != 0 {
		t.Fatalf("want 0 outside a loop, got %d", got)
	}
}

// A variadic function with NO fixed leading parameters has no shared prefix to make a
// family, and "same trailing types" alone paired concat1D with every two-tensor function in
// its package. Requiring a prefix is what removed the only wrong pairing in the tree.
func TestDetectPS6017_SilentWithoutASharedPrefix(t *testing.T) {
	src := `package p
func concat1D(parts ...*tensor.Tensor) *tensor.Tensor { return nil }
func geGLU(a, b *tensor.Tensor) *tensor.Tensor        { return nil }
func f(m *M, x, y *tensor.Tensor) {
	for l, b := range m.Blocks {
		_ = concat1D(x, y)
		use(l, b)
	}
}`
	if got := countCat(scanSrcPkg(t, src))["unpooled-variadic-sibling"]; got != 0 {
		t.Fatalf("want 0 without a shared leading prefix, got %d", got)
	}
}

// A candidate whose trailing type differs from the variadic element type is not a sibling.
//
// The guard against mis-rendered types is the POSITIVE test above, not this one: exprText has
// no StarExpr case and returns empty for every pointer, which makes the collector skip every
// candidate and the whole rule go silent. A zero-expecting assertion cannot tell that apart
// from correct suppression — swapping typeText for exprText leaves this test green and turns
// TestDetectPS6017_UnpooledVariadicSibling red.
func TestDetectPS6017_SilentOnADifferingTrailingType(t *testing.T) {
	src := `package p
func vexec(ctx *backend.Context, ins ...*tensor.Tensor) (*tensor.Tensor, error) { return nil, nil }
func notASibling(ctx *backend.Context, other *backend.Context) (*tensor.Tensor, error) {
	return nil, nil
}
func f(m *M, ctx *backend.Context, x *tensor.Tensor) error {
	for l, b := range m.Blocks {
		if _, err := vexec(ctx, x); err != nil {
			return err
		}
		use(l, b)
	}
	return nil
}`
	if got := countCat(scanSrcPkg(t, src))["unpooled-variadic-sibling"]; got != 0 {
		t.Fatalf("want 0 when the candidate's trailing type differs, got %d", got)
	}
}

// The partialRoPE shape: seven layout dispatches around one arithmetic op, straight-line code
// with no loop — which is why PS4011, the loop-shaped relative, could not see the largest of
// the three wins this rule was built from.
func TestDetectPS6018_LayoutOpClusterUnfused(t *testing.T) {
	src := `package p
func partialRoPE(ctx *C, x T, heads, rotaryDim int, rope A) (T, error) {
	flat, err := exec1(ctx, backend.OpReshape, reshapeTo(seq*heads, hd), x)
	if err != nil {
		return nil, err
	}
	rot, err := exec1(ctx, backend.OpSlice, sliceTo(0, rotaryDim), flat)
	if err != nil {
		return nil, err
	}
	pass, err := exec1(ctx, backend.OpSlice, sliceTo(rotaryDim, hd), flat)
	if err != nil {
		return nil, err
	}
	rotWide, err := exec1(ctx, backend.OpReshape, reshapeTo(seq, heads*rotaryDim), rot)
	if err != nil {
		return nil, err
	}
	if rotWide, err = exec1a(ctx, backend.OpRoPE, rope, rotWide); err != nil {
		return nil, err
	}
	merged, err := exec1(ctx, backend.OpConcat, concatOn(1), rotWide, pass)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpReshape, reshapeTo(seq, heads*hd), merged)
}`
	if got := countCat(scanSrc(t, src))["layout-op-cluster-unfused"]; got != 1 {
		t.Fatalf("want 1 layout-op-cluster-unfused, got %d", got)
	}
}

// A function that already has the fused path must stop reporting — otherwise the rule flags
// its own successes forever. This is the shape the fix takes.
func TestDetectPS6018_SilentOnceFused(t *testing.T) {
	src := `package p
func partialRoPE(ctx *C, x T, heads, rotaryDim int, rope A) (T, error) {
	if ctx.Recorder == nil {
		xf := x.Contiguous().Storage().F32()
		return gatherScatter(xf, heads, rotaryDim), nil
	}
	flat, err := exec1(ctx, backend.OpReshape, reshapeTo(seq*heads, hd), x)
	if err != nil {
		return nil, err
	}
	rot, err := exec1(ctx, backend.OpSlice, sliceTo(0, rotaryDim), flat)
	if err != nil {
		return nil, err
	}
	merged, err := exec1(ctx, backend.OpConcat, concatOn(1), rot, flat)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpReshape, reshapeTo(seq, heads*hd), merged)
}`
	if got := countCat(scanSrc(t, src))["layout-op-cluster-unfused"]; got != 0 {
		t.Fatalf("want 0 once a fused storage path exists, got %d", got)
	}
}

// Two movement ops around one arithmetic op is often irreducible — transpose then matmul —
// so the threshold is three. This is the floor against the rule firing on ordinary code.
func TestDetectPS6018_SilentOnTwoLayoutOps(t *testing.T) {
	src := `package p
func attn(ctx *C, q, k T) (T, error) {
	kT, err := exec1a(ctx, backend.OpTranspose, nil, k)
	if err != nil {
		return nil, err
	}
	scores, err := exec2(ctx, backend.OpMatMul, nil, q, kT)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpReshape, reshapeTo(1, 4), scores)
}`
	if got := countCat(scanSrc(t, src))["layout-op-cluster-unfused"]; got != 0 {
		t.Fatalf("want 0 for two layout ops, got %d", got)
	}
}

// Arithmetic ops are not movement, however many there are: fusing them raises reassociation
// and FMA questions, which is the whole reason this rule keys on a movement-op list rather
// than on dispatch count.
func TestDetectPS6018_SilentOnArithmeticOps(t *testing.T) {
	src := `package p
func mlp(ctx *C, x, w1, w2, w3 T) (T, error) {
	a, err := exec2(ctx, backend.OpMatMul, nil, x, w1)
	if err != nil {
		return nil, err
	}
	b, err := exec2(ctx, backend.OpMul, nil, a, w2)
	if err != nil {
		return nil, err
	}
	c, err := exec2(ctx, backend.OpAdd, nil, b, w3)
	if err != nil {
		return nil, err
	}
	return exec1a(ctx, backend.OpSoftmax, nil, c)
}`
	if got := countCat(scanSrc(t, src))["layout-op-cluster-unfused"]; got != 0 {
		t.Fatalf("want 0 for arithmetic ops, got %d", got)
	}
}

// The GMM shape that shipped a race: a 4-wide jam whose remainder delegates to a receiver
// method, so the scratch the wide body received as a parameter was not what the tail used.
func TestDetectPS6019_JamTailDelegates(t *testing.T) {
	src := `package p
func (m *M) logGaussianFullBatch(x []float64, ld []float64, y4 *[4][]float64) {
	c := 0
	for ; c+4 <= k; c += 4 {
		y0, y1, y2, y3 := y4[0], y4[1], y4[2], y4[3]
		for i := range d {
			y0[i], y1[i], y2[i], y3[i] = f(x, i), f(x, i), f(x, i), f(x, i)
		}
		ld[c] = y0[0] + y1[0] + y2[0] + y3[0]
	}
	for ; c < k; c++ {
		ld[c], _ = m.logGaussian(x, c)
	}
}`
	if got := countCat(scanSrc(t, src))["jam-tail-delegates"]; got != 1 {
		t.Fatalf("want 1 jam-tail-delegates, got %d", got)
	}
}

// A tail that repeats the wide body INLINE shares every edit by construction — same text — so
// it is not the hazard and must not report. This is the floor keeping the rule off the ordinary
// scalar remainder that nearly every jammed kernel has.
func TestDetectPS6019_SilentOnInlineTail(t *testing.T) {
	src := `package p
func (m *M) batch(x []float64, ld []float64, y4 *[4][]float64) {
	c := 0
	for ; c+4 <= k; c += 4 {
		y0 := y4[0]
		ld[c] = y0[0] * x[c]
	}
	for ; c < k; c++ {
		y0 := y4[0]
		ld[c] = y0[0] * x[c]
	}
}`
	if got := countCat(scanSrc(t, src))["jam-tail-delegates"]; got != 0 {
		t.Fatalf("want 0 for an inline tail, got %d", got)
	}
}

// When the WIDE body also delegates to the receiver, both paths go through the same method and
// share its fixes — the asymmetry is what makes the tail dangerous.
func TestDetectPS6019_SilentWhenBothDelegate(t *testing.T) {
	src := `package p
func (m *M) batch(x []float64, ld []float64) {
	c := 0
	for ; c+4 <= k; c += 4 {
		ld[c] = m.one(x, c) + m.one(x, c+1) + m.one(x, c+2) + m.one(x, c+3)
	}
	for ; c < k; c++ {
		ld[c] = m.one(x, c)
	}
}`
	if got := countCat(scanSrc(t, src))["jam-tail-delegates"]; got != 0 {
		t.Fatalf("want 0 when both paths delegate, got %d", got)
	}
}

// A plain stride-1 loop followed by another loop is not an unroll-and-jam and has no remainder.
func TestDetectPS6019_SilentWithoutAJamHeader(t *testing.T) {
	src := `package p
func (m *M) batch(x []float64, ld []float64) {
	for c := 0; c < k; c++ {
		ld[c] = x[c]
	}
	for c := 0; c < k; c++ {
		ld[c] = m.one(x, c)
	}
}`
	if got := countCat(scanSrc(t, src))["jam-tail-delegates"]; got != 0 {
		t.Fatalf("want 0 without a jam header, got %d", got)
	}
}

// The exact pre-fix signature of knnParallelRows. Three separate wins in this repository
// were blocked by this shape, so the rule's whole value is that it would have fired here.
func TestDetectPS6021_PerItemCallbackNoSeam(t *testing.T) {
	src := `package p
func knnParallelRows(n int, body func(i int)) {
	nw := runtime.GOMAXPROCS(0)
	if nw <= 1 || n < 64 {
		for i := 0; i < n; i++ {
			body(i)
		}
		return
	}
	csz := (n + nw - 1) / nw
	_ = parallelBuild(nw, func(c int) error {
		lo := c * csz
		for i := lo; i < lo+csz && i < n; i++ {
			body(i)
		}
		return nil
	})
}`
	if got := countCat(scanSrc(t, src))["fanout-without-worker-seam"]; got != 1 {
		t.Fatalf("want 1 fanout-without-worker-seam, got %d", got)
	}
}

// The pre-fix nbPredictParallel: extra leading scalars must not matter — only the callback's
// shape does.
func TestDetectPS6021_ExtraLeadingScalars(t *testing.T) {
	src := `package p
func nbPredictParallel(n, feat int, body func(i int)) {
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		go func(start int) {
			defer wg.Done()
			for i := start; i < n; i += 4 {
				body(i)
			}
		}(w)
	}
	wg.Wait()
}`
	if got := countCat(scanSrc(t, src))["fanout-without-worker-seam"]; got != 1 {
		t.Fatalf("want 1 fanout-without-worker-seam, got %d", got)
	}
}

// THE FLOOR THAT MAKES THIS RULE USABLE: a (lo, hi) range callback hands the caller a
// per-chunk closure, which is already a per-worker seam. parallel.Rows and most helpers in
// this repository have this shape, so reporting them would drown the one real hit.
func TestDetectPS6021_SilentOnRangeCallback(t *testing.T) {
	src := `package p
func parallelRows(n int, body func(lo, hi int)) {
	var wg sync.WaitGroup
	chunk := (n + 3) / 4
	for lo := 0; lo < n; lo += chunk {
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, lo+chunk)
	}
	wg.Wait()
}`
	if got := countCat(scanSrc(t, src))["fanout-without-worker-seam"]; got != 0 {
		t.Fatalf("want 0 for a range callback, got %d", got)
	}
}

// A callback already given scratch has the seam. This is the post-fix shape, so the rule must
// go quiet once the fix lands — otherwise it reports forever and gets ignored.
func TestDetectPS6021_SilentWithScratchParam(t *testing.T) {
	src := `package p
func gmmParallelRows(n, per, k int, body func(i int, ldBuf []float64)) {
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		go func(start int) {
			defer wg.Done()
			buf := make([]float64, k)
			for i := start; i < n; i += 4 {
				body(i, buf)
			}
		}(w)
	}
	wg.Wait()
}`
	if got := countCat(scanSrc(t, src))["fanout-without-worker-seam"]; got != 0 {
		t.Fatalf("want 0 with a scratch parameter, got %d", got)
	}
}

// The other post-fix shape: a func() T constructor the helper calls once per worker. This
// passes because the constructor's result must reach the callback, which therefore carries a
// scratch parameter — NOT because of any clause about the constructor itself. A clause for it
// existed and was removed when mutation testing showed no test could reach it.
func TestDetectPS6021_SilentWithScratchConstructor(t *testing.T) {
	src := `package p
func knnParallelRows(n int, newScratch func() *knnScratch, body func(i int, sc *knnScratch)) {
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		go func(start int) {
			defer wg.Done()
			sc := newScratch()
			for i := start; i < n; i += 4 {
				body(i, sc)
			}
		}(w)
	}
	wg.Wait()
}`
	if got := countCat(scanSrc(t, src))["fanout-without-worker-seam"]; got != 0 {
		t.Fatalf("want 0 with a scratch constructor, got %d", got)
	}
}

// A work-queue primitive builds a channel and its callback IS the job; every fix for this
// rule is implemented on top of such a primitive by passing a worker count as the job count.
// Reporting the mechanism would be reporting the cure.
func TestDetectPS6021_SilentOnChannelWorkQueue(t *testing.T) {
	src := `package p
func parallelBuild(n int, work func(t int) error) error {
	jobs := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		go func() {
			defer wg.Done()
			for t := range jobs {
				_ = work(t)
			}
		}()
	}
	for t := 0; t < n; t++ {
		jobs <- t
	}
	close(jobs)
	wg.Wait()
	return nil
}`
	if got := countCat(scanSrc(t, src))["fanout-without-worker-seam"]; got != 0 {
		t.Fatalf("want 0 for a channel work queue, got %d", got)
	}
}

// A serial helper taking a per-item callback is not a fan-out and has no race to avoid — the
// caller can hoist a buffer above the call safely. Without this floor the rule would fire on
// every ordinary callback API in the tree.
func TestDetectPS6021_SilentWithoutFanOut(t *testing.T) {
	src := `package p
func eachRow(n int, body func(i int)) {
	for i := 0; i < n; i++ {
		body(i)
	}
}`
	if got := countCat(scanSrc(t, src))["fanout-without-worker-seam"]; got != 0 {
		t.Fatalf("want 0 without a fan-out, got %d", got)
	}
}

// GIVES THE INTEGER-TYPE CHECK TEETH. Without it a callback taking exactly one NON-index
// parameter would be reported: one parameter, so the count check alone lets it through. That
// shape is a visitor over values, not an index fan-out, and it has no per-item allocation
// problem to report. Mutation testing found this floor missing — the two floors above were
// both passing on the parameter COUNT, leaving the type check unexercised.
func TestDetectPS6021_SilentOnSingleNonIndexParam(t *testing.T) {
	src := `package p
func eachBuffer(n int, body func(buf []float64)) {
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		go func() {
			defer wg.Done()
			body(make([]float64, n))
		}()
	}
	wg.Wait()
}`
	if got := countCat(scanSrc(t, src))["fanout-without-worker-seam"]; got != 0 {
		t.Fatalf("want 0 for a single non-index parameter, got %d", got)
	}
}

// The beam-search shape: a full sort, then a guarded truncation. PS6013 and PS6001 both miss
// it because the consumer is a reslice rather than a loop, which is the whole reason this
// check exists.
func TestDetectPS6022_SortThenGuardedTruncation(t *testing.T) {
	src := `package p
func f(cands []cand, bPrime int) []cand {
	slices.SortFunc(cands, byScore)
	if len(cands) > bPrime {
		cands = cands[:bPrime]
	}
	return cands
}`
	if got := countCat(scanSrc(t, src))["sort-feeds-truncation"]; got != 1 {
		t.Fatalf("want 1 sort-feeds-truncation, got %d", got)
	}
}

// The bare form, and a different sorter, to pin that neither the guard nor the sort function
// is load-bearing for detection.
func TestDetectPS6022_SortThenBareTruncation(t *testing.T) {
	src := `package p
func f(done []Beam, width int) []Beam {
	sort.SliceStable(done, func(i, j int) bool { return done[i].Score > done[j].Score })
	done = done[:width]
	return done
}`
	if got := countCat(scanSrc(t, src))["sort-feeds-truncation"]; got != 1 {
		t.Fatalf("want 1 sort-feeds-truncation, got %d", got)
	}
}

// FLOOR for the len() bound check: reslicing to len(x) keeps every element, so nothing was
// discarded and no comparison was wasted. Without this the check would fire on any sort
// followed by an idiomatic identity reslice.
func TestDetectPS6022_SilentOnFullLengthReslice(t *testing.T) {
	src := `package p
func f(x []int) []int {
	slices.Sort(x)
	x = x[:len(x)]
	return x
}`
	if got := countCat(scanSrc(t, src))["sort-feeds-truncation"]; got != 0 {
		t.Fatalf("want 0 for a full-length reslice, got %d", got)
	}
}

// FLOOR for the prefix requirement: a window that drops a head is not a top-k truncation, and
// a selection does not answer the same question.
func TestDetectPS6022_SilentOnNonPrefixWindow(t *testing.T) {
	src := `package p
func f(x []int, k int) []int {
	slices.Sort(x)
	y := x[2:k]
	return y
}`
	if got := countCat(scanSrc(t, src))["sort-feeds-truncation"]; got != 0 {
		t.Fatalf("want 0 for a non-prefix window, got %d", got)
	}
}

// FLOOR for the intervening-read rule: an indexed read before the truncation may depend on the
// full order, and it also means PS6001 or PS6013 describes the site better.
func TestDetectPS6022_SilentWhenReadBeforeTruncation(t *testing.T) {
	src := `package p
func f(x []int, k int) int {
	slices.Sort(x)
	total := 0
	for i := range x {
		total += x[i]
	}
	x = x[:k]
	return total
}`
	if got := countCat(scanSrc(t, src))["sort-feeds-truncation"]; got != 0 {
		t.Fatalf("want 0 when the slice is read before truncation, got %d", got)
	}
}

// FLOOR for the identity match: truncating a DIFFERENT slice says nothing about the sorted one.
func TestDetectPS6022_SilentOnOtherSliceTruncation(t *testing.T) {
	src := `package p
func f(x, y []int, k int) []int {
	slices.Sort(x)
	y = y[:k]
	return y
}`
	if got := countCat(scanSrc(t, src))["sort-feeds-truncation"]; got != 0 {
		t.Fatalf("want 0 when a different slice is truncated, got %d", got)
	}
}

// GIVES THE len-ARGUMENT CASE TEETH. An indexed read nested inside a len argument is still an
// order-dependent read. An earlier draft special-cased len() and stopped descending into it,
// which would have skipped exactly this; mutation testing showed no floor covered that clause,
// so it was removed and this floor added in its place.
func TestDetectPS6022_SilentOnIndexedReadInsideLen(t *testing.T) {
	src := `package p
func f(x [][]int, k int) int {
	slices.SortFunc(x, byLen)
	n := len(x[0])
	x = x[:k]
	return n
}`
	if got := countCat(scanSrc(t, src))["sort-feeds-truncation"]; got != 0 {
		t.Fatalf("want 0 for an indexed read nested in a len argument, got %d", got)
	}
}

// And the companion: a bare len() guard between the sort and the truncation must NOT suppress
// the finding, since that is how the idiomatic guarded truncation is written across two
// statements.
func TestDetectPS6022_BareLenGuardDoesNotSuppress(t *testing.T) {
	src := `package p
func f(x []int, k int) []int {
	slices.Sort(x)
	if len(x) <= k {
		return x
	}
	x = x[:k]
	return x
}`
	if got := countCat(scanSrc(t, src))["sort-feeds-truncation"]; got != 1 {
		t.Fatalf("want 1 with only a bare len guard in between, got %d", got)
	}
}

// A directive that actually suppresses something must not be reported.
func TestDetectPS0001_SilentOnWorkingDirective(t *testing.T) {
	src := `package p
func f(x []int) {
	//perfscan:ignore PS3002 deliberate
	sort.Slice(x, func(i, j int) bool { return x[i] < x[j] })
}`
	got := countCat(scanSrc(t, src))
	if got["unused-ignore-directive"] != 0 {
		t.Fatalf("want 0 unused-ignore-directive for a working directive, got %d", got["unused-ignore-directive"])
	}
	if got["closure-comparator-sort"] != 0 {
		t.Fatalf("the directive did not suppress: %d closure-comparator-sort remain", got["closure-comparator-sort"])
	}
}

// THE DEFECT THIS CHECK EXISTS FOR: an edit inserts statements between the directive and its
// target, so the directive silently stops suppressing while still reading as though it does.
func TestDetectPS0001_DriftedDirective(t *testing.T) {
	src := `package p
func f(x []int) {
	//perfscan:ignore PS3002 deliberate
	y := len(x)
	_ = y
	sort.Slice(x, func(i, j int) bool { return x[i] < x[j] })
}`
	got := countCat(scanSrc(t, src))
	if got["unused-ignore-directive"] != 1 {
		t.Fatalf("want 1 unused-ignore-directive for a drifted directive, got %d", got["unused-ignore-directive"])
	}
	if got["closure-comparator-sort"] != 1 {
		t.Fatalf("the drifted directive must NOT suppress; got %d closure-comparator-sort", got["closure-comparator-sort"])
	}
}

// FLOOR FOR THE CREDITING SPAN. Two directives stacked above one statement form a two-line
// comment block, so the upper one sits two lines from its target. Crediting only a directive's
// own line and the next reported that working directive as unused — a false report, which is
// the exact failure this check exists to prevent. Both must be credited.
func TestDetectPS0001_StackedDirectivesBothCredited(t *testing.T) {
	src := `package p
func f(x []int) {
	//perfscan:ignore PS3002 first reason
	//perfscan:ignore PS3001 second reason
	sort.Slice(x, func(i, j int) bool { return x[i] < x[j] })
}`
	// PS3001 has nothing to suppress here, so exactly one directive is genuinely unused; the
	// PS3002 one is two lines above its target and must still be credited.
	got := countCat(scanSrc(t, src))
	if got["closure-comparator-sort"] != 0 {
		t.Fatalf("the stacked block did not suppress the sort: %d remain", got["closure-comparator-sort"])
	}
	if got["unused-ignore-directive"] != 1 {
		t.Fatalf("want exactly 1 unused (the PS3001 one), got %d — the PS3002 directive two lines up was not credited",
			got["unused-ignore-directive"])
	}
}

// FLOOR FOR THE DIRECTIVE-VERSUS-MENTION TEST. Prose describing the feature, and indented
// examples inside a doc comment, must not register as live directives. Matching the token
// anywhere in a comment made four of this package's own doc comments report as unused.
func TestDetectPS0001_SilentOnPoseMention(t *testing.T) {
	src := `package p

// f explains that a //perfscan:ignore directive silences a check.
//
//	//perfscan:ignore PS1001 an indented example, not a directive
func f(x []int) {
	_ = x
}`
	if got := countCat(scanSrc(t, src))["unused-ignore-directive"]; got != 0 {
		t.Fatalf("want 0 for prose mentioning the directive, got %d", got)
	}
}

// A bare directive silences every check at its site, and must be credited when it does.
func TestDetectPS0001_SilentOnWorkingBareDirective(t *testing.T) {
	src := `package p
func f(x []int) {
	//perfscan:ignore deliberate, all checks
	sort.Slice(x, func(i, j int) bool { return x[i] < x[j] })
}`
	if got := countCat(scanSrc(t, src))["unused-ignore-directive"]; got != 0 {
		t.Fatalf("want 0 for a working bare directive, got %d", got)
	}
}
