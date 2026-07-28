package main

import (
	"bytes"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
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
