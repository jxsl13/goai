package main

import (
	"bytes"
	"fmt"
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
