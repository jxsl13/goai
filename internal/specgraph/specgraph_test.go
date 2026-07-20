package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/jxsl13/goai/internal/speccheck"
)

const fixtureRoot = "testdata/fixture"

func fixtureGraph(t *testing.T) *Graph {
	t.Helper()
	return buildGraph(fixtureRoot, false)
}

// hasEdge reports whether g contains from -[typ]-> to.
func hasEdge(g *Graph, from, typ, to string) bool {
	for _, i := range g.Out[from] {
		e := g.Edges[i]
		if e.Type == typ && e.To == to {
			return true
		}
	}
	return false
}

// TestFixtureExtraction pins every extractor on the synthetic corpus: node
// kinds, the guard chain, the trusted cites column, touches validation, the
// worker-prose exclusions, and the weak-ref filters.
func TestFixtureExtraction(t *testing.T) {
	g := fixtureGraph(t)

	wantKind := map[string]Kind{
		"G1": KindGoal, "C1": KindConstraint, "C2": KindConstraint,
		"I.L0": KindArchInv, "I1": KindArchInv,
		"R1": KindResearch, "V1": KindVerif, "V2": KindVerif,
		"B1": KindBug, "B2": KindBug,
		"T1": KindTask, "T2": KindTask, "T3": KindTask, "T10": KindTask,
		"ADR-0001": KindADR, "backend/cpu": KindPkg, "nn": KindPkg,
		"CL:T1:2026-01-02": KindChange, "CL:T2:2026-01-03": KindChange,
		"docs/notes.md#Measured section": KindDoc,
	}
	for id, k := range wantKind {
		n := g.Nodes[id]
		if n == nil {
			t.Errorf("missing node %s", id)
			continue
		}
		if n.Kind != k {
			t.Errorf("node %s kind = %s, want %s", id, n.Kind, k)
		}
	}
	// Worker prose ids and doc-prose look-alikes must never become nodes.
	for id := range g.Nodes {
		if strings.Contains(id, "Tw") || id == "Iw8" || id == "V100" || id == "T4" {
			t.Errorf("false-positive node %s", id)
		}
	}
	if got := g.Nodes["V2"].Meta["tag"]; got != "MULTI WORD TAG" {
		t.Errorf("V2 tag = %q", got)
	}
	if !strings.Contains(g.Nodes["C2"].Text, "continuation line") {
		t.Errorf("C2 continuation not folded in: %q", g.Nodes["C2"].Text)
	}

	edges := []struct{ from, typ, to string }{
		{"V1", EdgeGuards, "B1"}, // §B guard chain: "... V1,T2."
		{"V1", EdgeGuards, "B2"}, // "... V1."
		{"B1", EdgeCites, "T2"},  // non-V guard-chain ref
		{"T1", EdgeCites, "C1"},  // trusted cites column
		{"T1", EdgeCites, "V1"},
		{"T1", EdgeCites, "ADR-0001"},
		{"T1", EdgeFixedBy, "abc1234"},     // commit SHA in row text
		{"T1", EdgeTouches, "backend/cpu"}, // validated path token
		{"B2", EdgeTouches, "nn"},          // pkg.Symbol token (nn.Fixture)
		{"T2", EdgeSameClass, "T1"},        // "T1 class" phrase
		{"CL:T1:2026-01-02", EdgeRecords, "T1"},
		{"CL:T2:2026-01-03", EdgeRecords, "T2"}, // em-dash + scope heading
		{"ADR-0001", EdgeRelates, "T1"},         // strong: `- Task:` line
		{"ADR-0001", EdgeRelates, "C1"},         // strong: `- Related:` line
		{"docs/notes.md#Measured section", EdgeMentions, "V1"},
		{"docs/notes.md#Measured section", EdgeMentions, "T2"},
		{"C2", EdgeRefs, "G1"}, // ref on a continuation line
	}
	for _, e := range edges {
		if !hasEdge(g, e.from, e.typ, e.to) {
			t.Errorf("missing edge %s -[%s]-> %s", e.from, e.typ, e.to)
		}
	}
	if hasEdge(g, "ADR-0001", EdgeRelates, "V100") {
		t.Error("weak ADR body ref to nonexistent V100 must be dropped")
	}
	for id := range g.Nodes {
		if strings.HasSuffix(id, "#Empty section") {
			t.Errorf("refless doc section became a node: %s", id)
		}
	}
	// The em-dash CHANGELOG heading names B1 too, but records edges are
	// task-only by design.
	if hasEdge(g, "CL:T2:2026-01-03", EdgeRecords, "B1") {
		t.Error("records edge must only target tasks")
	}
}

// TestParseMatchesSpeccheck pins specgraph's copied defining-occurrence
// regexes to speccheck's behavior without coupling the packages: doubling
// the spec makes speccheck report a duplicate for exactly the ids specgraph
// extracts as T/B/R/V nodes from the same content.
func TestParseMatchesSpeccheck(t *testing.T) {
	content, err := os.ReadFile(filepath.Join(fixtureRoot, "SPEC.md"))
	if err != nil {
		t.Fatal(err)
	}
	g := NewGraph()
	parseSpec(g, "SPEC.md", string(content), loadPkgSet(fixtureRoot))
	mine := map[string]bool{}
	for id, n := range g.Nodes {
		switch n.Kind {
		case KindTask, KindBug, KindResearch, KindVerif:
			mine[id] = true
		}
	}
	theirs := map[string]bool{}
	reDup := regexp.MustCompile(`duplicate id ([TBRV]\d+)`)
	for _, v := range speccheck.Check(string(content)+"\n"+string(content), nil) {
		if m := reDup.FindStringSubmatch(v.Msg); m != nil {
			theirs[m[1]] = true
		}
	}
	if len(theirs) == 0 {
		t.Fatal("speccheck reported no duplicates on a doubled spec — harness broken")
	}
	for id := range mine {
		if !theirs[id] {
			t.Errorf("specgraph extracts %s but speccheck does not track it", id)
		}
	}
	for id := range theirs {
		if !mine[id] {
			t.Errorf("speccheck tracks %s but specgraph does not extract it", id)
		}
	}
}

// TestDeterminism: two independent builds must render byte-identical DOT —
// the guard forcing sorted iteration everywhere.
func TestDeterminism(t *testing.T) {
	a := renderDOT(buildGraph(fixtureRoot, false), nil)
	b := renderDOT(buildGraph(fixtureRoot, false), nil)
	if a != b {
		t.Fatal("two builds rendered different DOT")
	}
}

// TestCacheRoundtrip: save+load must reproduce the graph exactly, and any
// manifest mismatch (here: a touched input) must force a rebuild.
func TestCacheRoundtrip(t *testing.T) {
	root := t.TempDir()
	if err := os.CopyFS(root, os.DirFS(fixtureRoot)); err != nil {
		t.Fatal(err)
	}
	direct := buildGraph(root, false)
	if err := saveCachedGraph(root, false, direct); err != nil {
		t.Fatal(err)
	}
	cached, ok := loadCachedGraph(root, false)
	if !ok {
		t.Fatal("fresh cache did not load")
	}
	if renderDOT(direct, nil) != renderDOT(cached, nil) {
		t.Fatal("cached graph differs from direct build")
	}
	// Any input drift invalidates: rewrite SPEC.md (size change is enough).
	spec := filepath.Join(root, "SPEC.md")
	b, err := os.ReadFile(spec)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(spec, append(b, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, ok := loadCachedGraph(root, false); ok {
		t.Fatal("stale cache (changed SPEC.md) was trusted")
	}
}

// TestQueries covers the traversals on the fixture: guard reverse story,
// path chain, bugs-for prefix matching, impact reachability.
func TestQueries(t *testing.T) {
	g := fixtureGraph(t)
	if got := g.bugsFor("backend"); len(got) != 1 || got[0].ID != "B1" {
		t.Errorf("bugsFor(backend) = %v", ids(got))
	}
	if got := g.bugsFor("nn"); len(got) != 1 || got[0].ID != "B2" {
		t.Errorf("bugsFor(nn) = %v", ids(got))
	}
	if p := g.shortestPath("B2", "T1"); len(p) == 0 {
		t.Error("no path B2..T1 (expected via V1)")
	}
	if _, ok := g.reverseReach("V1", 3)["CL:T1:2026-01-02"]; !ok {
		t.Error("impact V1 should reach the changelog entry via T1")
	}
	if g.Nodes[nearestID(g, "T99")] == nil {
		t.Error("nearestID(T99) found nothing")
	}
}

func ids(ns []*Node) []string {
	out := make([]string, len(ns))
	for i, n := range ns {
		out[i] = n.ID
	}
	return out
}

// TestClusters: B1+B2 share guard V1 AND the "per-element allocation"
// cause keywords, so both groupers must find the pair.
func TestClusters(t *testing.T) {
	g := fixtureGraph(t)
	var haveGuard, haveKeyword bool
	for _, c := range bugClusters(g) {
		members := strings.Join(c.Members, ",")
		if c.Key == "guard V1" && members == "B2,B1" {
			haveGuard = true
			if c.First != "2026-01-02" || c.Last != "2026-01-03" {
				t.Errorf("guard V1 span = %s..%s", c.First, c.Last)
			}
		}
		if strings.HasPrefix(c.Key, "keywords ") && strings.Contains(members, "B1") && strings.Contains(members, "B2") {
			haveKeyword = true
		}
	}
	if !haveGuard {
		t.Error("missing guard V1 cluster {B2,B1}")
	}
	if !haveKeyword {
		t.Error("missing keyword cluster containing B1+B2")
	}
}

// TestBM25 pins lexical search: the NaN-allocation bugs outrank everything.
func TestBM25(t *testing.T) {
	g := fixtureGraph(t)
	hits := buildBM25(g).search("NaN per-element allocation", 3)
	if len(hits) < 2 {
		t.Fatalf("too few hits: %v", hits)
	}
	top := hits[0].ID + " " + hits[1].ID
	if !strings.Contains(top, "B1") || !strings.Contains(top, "B2") {
		t.Errorf("top hits = %s, want B1+B2", top)
	}
}

// fakeEmbedder is deterministic without a model: a token-hash bag vector.
type fakeEmbedder struct{}

func (fakeEmbedder) Embed(text string) ([]float64, error) {
	v := make([]float64, 16)
	for _, tok := range splitTokens(text) {
		h := 0
		for _, r := range tok {
			h = h*31 + int(r)
		}
		v[((h%16)+16)%16]++
	}
	return v, nil
}

// TestEmbedIndex: the fake-embedder index is incremental (second run embeds
// nothing) and vectorSearch ranks the on-topic bug first.
func TestEmbedIndex(t *testing.T) {
	g := fixtureGraph(t)
	path := filepath.Join(t.TempDir(), "embeddings.jsonl")
	embedded, kept, total, err := indexEmbeddings(g, fakeEmbedder{}, path)
	if err != nil {
		t.Fatal(err)
	}
	if embedded == 0 || kept != 0 {
		t.Fatalf("first pass embedded=%d kept=%d", embedded, kept)
	}
	e2, k2, t2, err := indexEmbeddings(g, fakeEmbedder{}, path)
	if err != nil {
		t.Fatal(err)
	}
	if e2 != 0 || k2 != total || t2 != total {
		t.Fatalf("second pass embedded=%d kept=%d total=%d, want 0/%d/%d", e2, k2, t2, total, total)
	}
	qv, _ := fakeEmbedder{}.Embed(g.Nodes["B1"].Text)
	hits, err := vectorSearch(loadEmbeddings(path), qv, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].ID != "B1" {
		t.Errorf("vector top hit = %v, want B1", hits)
	}
}

// TestGuardChain pins the final-sentence extraction the guards edges hang on.
func TestGuardChain(t *testing.T) {
	cases := map[string]string{
		"flat fast path. V1.":                        "V1",
		"fix text. Non-vacuous: red first. V15,V30.": "V15,V30",
		"closed:degenerate-halflife-guarded.":        "closed:degenerate-halflife-guarded",
		"single fragment":                            "single fragment",
	}
	for in, want := range cases {
		if got := guardChain(in); got != want {
			t.Errorf("guardChain(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestRealCorpus is the golden acceptance run against the actual repo spec
// (git off — the git side has its own guarded test): the load-bearing facts
// verified while designing the schema, plus growth-safe lower bounds.
func TestRealCorpus(t *testing.T) {
	root := "../.."
	if _, err := os.Stat(filepath.Join(root, "SPEC.md")); err != nil {
		t.Skip("no repo SPEC.md (fixture-only checkout)")
	}
	g := buildGraph(root, false)

	minima := map[Kind]int{
		KindTask: 900, KindBug: 110, KindResearch: 260,
		KindVerif: 36, KindConstraint: 25, KindADR: 29,
	}
	counts := map[Kind]int{}
	for _, n := range g.Nodes {
		counts[n.Kind]++
	}
	for k, min := range minima {
		if counts[k] < min {
			t.Errorf("%s nodes = %d, want >= %d", k, counts[k], min)
		}
	}
	edges := []struct{ from, typ, to string }{
		{"V36", EdgeGuards, "B96"},  // the §B96 → §V36 backprop chain
		{"B96", EdgeCites, "T886"},  // ... and its spec-lint task
		{"V15", EdgeGuards, "B108"}, // B108 guard chain "V15,V30"
		{"V30", EdgeGuards, "B108"},
		{"T907", EdgeFixedBy, "2d35b3b"},  // SHA embedded in the §T row
		{"ADR-0027", EdgeRelates, "T658"}, // `- Task:` line
		{"ADR-0027", EdgeRelates, "ADR-0026"},
		{"ADR-0027", EdgeRelates, "C2"},
		{"ADR-0027", EdgeRelates, "V10"},
		{"CL:T909:2026-07-20", EdgeRecords, "T909"},
	}
	for _, e := range edges {
		if !hasEdge(g, e.from, e.typ, e.to) {
			t.Errorf("missing corpus edge %s -[%s]-> %s", e.from, e.typ, e.to)
		}
	}
}

// TestRealCorpusGit checks the git side (Commit nodes + implements edges)
// and skips gracefully on shallow or git-less checkouts.
func TestRealCorpusGit(t *testing.T) {
	root := "../.."
	if _, err := os.Stat(filepath.Join(root, "SPEC.md")); err != nil {
		t.Skip("no repo SPEC.md")
	}
	if err := exec.Command("git", "-C", root, "cat-file", "-e", "2d35b3b").Run(); err != nil {
		t.Skip("commit 2d35b3b not in local history (shallow clone?)")
	}
	g := buildGraph(root, true)
	found := false
	for _, i := range g.In["T907"] {
		e := g.Edges[i]
		if e.Type == EdgeImplements && strings.HasPrefix(e.From, "2d35b3b"[:7]) {
			found = true
		}
	}
	if !found {
		t.Error("missing implements edge 2d35b3b -> T907 from git log")
	}
	// The spec-side fixed_by SHA and the git-side node must have converged
	// onto one Commit node (prefix resolution).
	for _, i := range g.Out["T907"] {
		e := g.Edges[i]
		if e.Type == EdgeFixedBy && g.Nodes[e.To] == nil {
			t.Errorf("fixed_by target %s is dangling despite git history", e.To)
		}
	}
}

// TestGoldenOutputs freezes the LLM-facing text format per subcommand on the
// fixture; regenerate with UPDATE_GOLDEN=1 go test ./internal/specgraph.
func TestGoldenOutputs(t *testing.T) {
	g := fixtureGraph(t)
	runs := map[string]func(w *strings.Builder){
		"golden_show.txt":     func(w *strings.Builder) { cmdShow(w, g, "V1", 0, false) },
		"golden_why.txt":      func(w *strings.Builder) { cmdWhy(w, g, "V1", 0, false) },
		"golden_related.txt":  func(w *strings.Builder) { cmdRelated(w, g, "T1", 2, 0, false) },
		"golden_path.txt":     func(w *strings.Builder) { cmdPath(w, g, "B2", "T1", false) },
		"golden_patterns.txt": func(w *strings.Builder) { cmdPatterns(w, g, 0) },
		"golden_dangling.txt": func(w *strings.Builder) { cmdDangling(w, g, 0) },
		"golden_dot.txt":      func(w *strings.Builder) { w.WriteString(renderDOT(g, nil)) },
	}
	for name, run := range runs {
		var w strings.Builder
		run(&w)
		path := filepath.Join("testdata", name)
		if os.Getenv("UPDATE_GOLDEN") != "" {
			if err := os.WriteFile(path, []byte(w.String()), 0o644); err != nil {
				t.Fatal(err)
			}
			continue
		}
		want, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("%s: %v (run UPDATE_GOLDEN=1 go test to generate)", name, err)
		}
		if w.String() != string(want) {
			t.Errorf("%s drifted:\n--- got ---\n%s--- want ---\n%s", name, w.String(), want)
		}
	}
}

// TestMissSuggestion pins the exit-0 "no node" contract with a suggestion.
func TestMissSuggestion(t *testing.T) {
	g := fixtureGraph(t)
	var w strings.Builder
	cmdShow(&w, g, "T99", 0, false)
	out := w.String()
	if !strings.HasPrefix(out, "no node T99") || !strings.Contains(out, "nearest: T10") {
		t.Errorf("miss output = %q", out)
	}
	_ = fmt.Sprint() // keep fmt imported even if assertions change
}
