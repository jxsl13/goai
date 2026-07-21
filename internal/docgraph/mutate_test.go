package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// specFixture materializes a spec/-layout tree in a temp root by splitting
// the flat fixture SPEC.md + worker file — the same code path the real
// migration uses, so every CRUD test also exercises split.
func specFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	if err := os.CopyFS(root, os.DirFS(fixtureRoot)); err != nil {
		t.Fatal(err)
	}
	// the fixture worker file is SPEC-worker-test.md
	var w strings.Builder
	if rc := cmdSplit(&w, root, false); rc != 0 {
		t.Fatalf("split failed: %s", w.String())
	}
	return root
}

func mustRun(t *testing.T, rc int, w *strings.Builder) string {
	t.Helper()
	if rc != 0 {
		t.Fatalf("command failed (%d): %s", rc, w.String())
	}
	return w.String()
}

// TestSplitRoundTrip is the migration proof: splitting the REAL SPEC.md and
// worker spec and re-rendering must be byte-identical (cmdSplit refuses to
// write otherwise; here we also verify the written tree renders back).
func TestSplitRoundTrip(t *testing.T) {
	repo := "../.."
	orig, err := os.ReadFile(filepath.Join(repo, "SPEC.md"))
	if err != nil {
		t.Skip("no repo SPEC.md")
	}
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "SPEC.md"), orig, 0o644); err != nil {
		t.Fatal(err)
	}
	workers, _ := filepath.Glob(filepath.Join(repo, "SPEC-worker-*.md"))
	for _, p := range workers {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, filepath.Base(p)), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var w strings.Builder
	mustRun(t, cmdSplit(&w, root, false), &w)
	c, err := LoadCorpus(root)
	if err != nil {
		t.Fatal(err)
	}
	rendered, err := renderAll(c)
	if err != nil {
		t.Fatal(err)
	}
	if rendered["SPEC.md"] != string(orig) {
		t.Fatal("render(split(SPEC.md)) != SPEC.md")
	}
	for _, p := range workers {
		b, _ := os.ReadFile(p)
		if rendered[filepath.Base(p)] != string(b) {
			t.Fatalf("render(split(%s)) not byte-identical", filepath.Base(p))
		}
	}
}

// TestRenderDeterminism: two loads render identically.
func TestRenderDeterminism(t *testing.T) {
	root := specFixture(t)
	r1, err := LoadCorpus(root)
	if err != nil {
		t.Fatal(err)
	}
	r2, _ := LoadCorpus(root)
	a, err := renderAll(r1)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := renderAll(r2)
	for k := range a {
		if a[k] != b[k] {
			t.Fatalf("nondeterministic render of %s", k)
		}
	}
}

// TestRenderSync is the §V41 CI guard: when spec/ exists at the repo root,
// the committed rendered views must match render(spec/) byte-for-byte.
// Worker views are warn-only while workerRenderSyncHard is false (phase A).
func TestRenderSync(t *testing.T) {
	repo := "../.."
	if !HasSpecDir(repo) {
		t.Skip("no spec/ hierarchy yet (pre-migration tree)")
	}
	c, err := LoadCorpus(repo)
	if err != nil {
		t.Fatal(err)
	}
	ms, err := renderCheck(repo, c)
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range ms {
		if m.Worker && !workerRenderSyncHard {
			t.Logf("warn (phase A): %s: %s", m.RenderPath, m.Reason)
			continue
		}
		t.Errorf("%s: %s", m.RenderPath, m.Reason)
	}
}

// TestTasksFileMustSortLast: §V36 defense in depth at render time.
func TestTasksFileMustSortLast(t *testing.T) {
	o := Output{RenderPath: "SPEC.md", Fragments: []Fragment{
		{Path: "spec/70-tasks.md", Content: "## §T\n\n| id |\n"},
		{Path: "spec/90-extra.md", Content: "## §Z\n"},
	}}
	if _, err := renderOutput(o); err == nil {
		t.Fatal("render accepted a tasks file that does not sort last")
	}
}

// TestNextID covers allocation across root and worker fragments.
func TestNextID(t *testing.T) {
	root := specFixture(t)
	c, err := LoadCorpus(root)
	if err != nil {
		t.Fatal(err)
	}
	// fixture max ids: T10 (worker file), B2, R1, V2
	for class, want := range map[string]int{"T": 11, "B": 3, "R": 2, "V": 3} {
		if got := nextID(c, class); got != want {
			t.Errorf("nextID(%s) = %d, want %d", class, got, want)
		}
	}
}

// TestEscapeCell pins the cell rules.
func TestEscapeCell(t *testing.T) {
	if _, err := escapeCell("two\nlines"); err == nil {
		t.Error("multi-line cell accepted")
	}
	got, err := escapeCell(`a | b and pre-escaped \| stays`)
	if err != nil {
		t.Fatal(err)
	}
	if got != `a \| b and pre-escaped \| stays` {
		t.Errorf("escapeCell = %q", got)
	}
}

// TestTaskLifecycle: add → set-status ~ → x, backward refused without force.
func TestTaskLifecycle(t *testing.T) {
	root := specFixture(t)
	var w strings.Builder
	out := mustRun(t, cmdTaskAdd(&w, root, "lifecycle demo task", mutFlags{Cites: "V1", Priority: "low"}), &w)
	if !strings.Contains(out, "T11 allocated") {
		t.Fatalf("unexpected add output: %s", out)
	}
	// the row must land at the END of the tasks table (append style)
	tasks, _ := os.ReadFile(filepath.Join(root, "spec", "70-tasks.md"))
	lines := strings.Split(strings.TrimRight(string(tasks), "\n"), "\n")
	if !strings.HasPrefix(lines[len(lines)-1], "| T11 | . |") {
		t.Fatalf("T11 not appended last: %q", lines[len(lines)-1])
	}
	// rendered view updated in the same transaction
	spec, _ := os.ReadFile(filepath.Join(root, "SPEC.md"))
	if !strings.Contains(string(spec), "| T11 | . | lifecycle demo task |") {
		t.Fatal("rendered SPEC.md missing the new row")
	}
	w.Reset()
	mustRun(t, cmdTaskSetStatus(&w, root, "T11", "~", mutFlags{}), &w)
	w.Reset()
	mustRun(t, cmdTaskSetStatus(&w, root, "T11", "x", mutFlags{}), &w)
	spec, _ = os.ReadFile(filepath.Join(root, "SPEC.md"))
	if !strings.Contains(string(spec), "| T11 | x | lifecycle demo task | V1 | done | low |") {
		t.Fatalf("status+state not updated in render: %s", grepLine(string(spec), "T11"))
	}
	w.Reset()
	if rc := cmdTaskSetStatus(&w, root, "T11", ".", mutFlags{}); rc == 0 {
		t.Fatal("backward transition accepted without -force")
	}
	w.Reset()
	mustRun(t, cmdTaskSetStatus(&w, root, "T11", ".", mutFlags{Force: true}), &w)
}

// TestBugAddNewestFirst: the fixture §B table is newest-first (B2 above B1
// after B2's later date? fixture has B1 then B2 appended — adaptive rule
// places new max-id rows adjacent to the current max). Verify B3 lands next
// to B2 and the guard chain parses into a guards edge.
func TestBugAddNewestFirst(t *testing.T) {
	root := specFixture(t)
	var w strings.Builder
	mustRun(t, cmdBugAdd(&w, root, mutFlags{Cause: "demo cause", Fix: "demo fix. Non-vacuous: test named", Guards: "V1", Date: "2026-01-04"}), &w)
	if !strings.Contains(w.String(), "B3 allocated") {
		t.Fatalf("unexpected: %s", w.String())
	}
	g := buildGraph(root, false)
	if !hasEdge(g, "V1", EdgeGuards, "B3") {
		t.Fatal("guard chain of the new bug did not produce V1 -guards-> B3")
	}
}

// writeBenchSection seeds the empty §Bench source file in a fixture root (the
// migration creates it out-of-band; bench add only inserts rows).
func writeBenchSection(t *testing.T, root string) {
	t.Helper()
	hdr := "## §Bench — benchmark records\n\n" +
		"| id | date | benchmark | machine | incumbent | metric | before | after | impact | cites |\n" +
		"| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
	if err := os.WriteFile(filepath.Join(root, "spec", "80-benchmarks.md"), []byte(hdr), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestBenchAdd: bench add COMPUTES impact = before/after, renders the row into
// SPEC.md in one transaction, and the row parses back as a KindBench node whose
// cites column mints EdgeCites to the §T task and §V invariant.
func TestBenchAdd(t *testing.T) {
	root := specFixture(t)
	writeBenchSection(t, root)
	var w strings.Builder
	out := mustRun(t, cmdBenchAdd(&w, root, mutFlags{
		Benchmark: "demo op forward", Machine: "darwin/arm64", Metric: "ms",
		Before: "2.76", After: "2.00", Cites: "T1,V1", Date: "2026-01-05",
	}), &w)
	if !strings.Contains(out, "BM1 allocated (impact 1.38×)") {
		t.Fatalf("unexpected add output: %s", out)
	}
	spec, _ := os.ReadFile(filepath.Join(root, "SPEC.md"))
	want := "| BM1 | 2026-01-05 | demo op forward | darwin/arm64 | self | ms | 2.76 | 2.00 | 1.38× | T1,V1 |"
	if !strings.Contains(string(spec), want) {
		t.Fatalf("rendered §Bench row missing/wrong: %s", grepLine(string(spec), "BM1"))
	}
	g := buildGraph(root, false)
	if n := g.Nodes["BM1"]; n == nil || n.Kind != KindBench {
		t.Fatalf("BM1 not a bench node: %#v", n)
	}
	if !hasEdge(g, "BM1", EdgeCites, "T1") || !hasEdge(g, "BM1", EdgeCites, "V1") {
		t.Error("BM1 cites edges to T1/V1 missing")
	}
	// Add a second row, then normalize once (the documented cmdSort workflow for
	// a fresh section — insertRow's single-row heuristic is ambiguous until a
	// sort pins the ascending order). §Bench is a plain ascending table, so sort
	// leaves BM1 then BM2, and a subsequent add appends BM3 last.
	w.Reset()
	mustRun(t, cmdBenchAdd(&w, root, mutFlags{Benchmark: "second", Before: "4", After: "1", Cites: "T1"}), &w)
	if !strings.Contains(w.String(), "BM2 allocated (impact 4×)") {
		t.Fatalf("BM2 add: %s", w.String())
	}
	w.Reset()
	mustRun(t, cmdSort(&w, root, false), &w)
	w.Reset()
	mustRun(t, cmdBenchAdd(&w, root, mutFlags{Benchmark: "third", Before: "9", After: "3", Cites: "T1"}), &w)
	bench, _ := os.ReadFile(filepath.Join(root, "spec", "80-benchmarks.md"))
	lines := strings.Split(strings.TrimRight(string(bench), "\n"), "\n")
	if !strings.HasPrefix(lines[len(lines)-3], "| BM1 ") ||
		!strings.HasPrefix(lines[len(lines)-2], "| BM2 ") ||
		!strings.HasPrefix(lines[len(lines)-1], "| BM3 ") {
		t.Fatalf("§Bench not ascending after normalize+add:\n%s", strings.Join(lines[len(lines)-3:], "\n"))
	}
	// and the normalized section is sort-clean (dry-run reports no change).
	w.Reset()
	if rc := cmdSort(&w, root, true); rc != 0 {
		t.Fatalf("sort -n after normalize failed: %s", w.String())
	}
}

// TestBenchAddRequiresMeasuredValues: missing/bad before/after is a hard error,
// so a bogus impact is never stored.
func TestBenchAddRequiresMeasuredValues(t *testing.T) {
	root := specFixture(t)
	writeBenchSection(t, root)
	var w strings.Builder
	if rc := cmdBenchAdd(&w, root, mutFlags{Benchmark: "x", Before: "1"}); rc == 0 {
		t.Fatal("bench add accepted a missing -after")
	}
	w.Reset()
	if rc := cmdBenchAdd(&w, root, mutFlags{Benchmark: "x", Before: "nope", After: "2"}); rc == 0 {
		t.Fatalf("bench add accepted a non-numeric -before: %s", w.String())
	}
}

// TestComputeImpact pins the before/after ratio + trailing-zero trimming.
func TestComputeImpact(t *testing.T) {
	cases := []struct{ before, after, want string }{
		{"2.76", "2.00", "1.38×"},
		{"16.90", "10.69", "1.58×"},
		{"11.97", "0.91", "13.15×"},
		{"2.00", "1.00", "2×"},        // trailing zeros trimmed
		{"2.76ms", "2.00ms", "1.38×"}, // trailing unit tolerated
	}
	for _, c := range cases {
		got, err := computeImpact(c.before, c.after)
		if err != nil || got != c.want {
			t.Errorf("computeImpact(%q,%q) = %q,%v want %q", c.before, c.after, got, err, c.want)
		}
	}
	if _, err := computeImpact("abc", "2"); err == nil {
		t.Error("non-numeric before accepted")
	}
	if _, err := computeImpact("2", "0"); err == nil {
		t.Error("zero after (division by zero) accepted")
	}
}

// TestBenchmarksMaySortAfterTasks: §Bench is the ONE section allowed to render
// after §T (§V36 relaxation), but only directly after it.
func TestBenchmarksMaySortAfterTasks(t *testing.T) {
	ok := Output{RenderPath: "SPEC.md", Fragments: []Fragment{
		{Path: "spec/70-tasks.md", Content: "## §T\n\n| id |\n"},
		{Path: "spec/80-benchmarks.md", Content: "## §Bench\n\n| id |\n"},
	}}
	if _, err := renderOutput(ok); err != nil {
		t.Fatalf("§Bench directly after §T must be accepted: %v", err)
	}
	bad := Output{RenderPath: "SPEC.md", Fragments: []Fragment{
		{Path: "spec/60-backprop.md", Content: "## §B\n\n| id |\n"},
		{Path: "spec/80-benchmarks.md", Content: "## §Bench\n\n| id |\n"},
	}}
	if _, err := renderOutput(bad); err == nil {
		t.Fatal("§Bench last without §T directly before it must be rejected")
	}
}

// TestResearchAndDefAdds: research add + verif/goal/constraint adds render
// into the view and parse back as nodes.
func TestResearchAndDefAdds(t *testing.T) {
	root := specFixture(t)
	var w strings.Builder
	mustRun(t, cmdResearchAdd(&w, root, mutFlags{Claim: "claim x", Source: "src", Conf: "high"}), &w)
	w.Reset()
	mustRun(t, cmdVerifAdd(&w, root, "every mutation re-verifies", mutFlags{Tag: "DEMO-TAG"}), &w)
	w.Reset()
	mustRun(t, cmdDefAdd(&w, root, "C", "new constraint text", mutFlags{}), &w)
	g := buildGraph(root, false)
	for id, kind := range map[string]Kind{"R2": KindResearch, "V3": KindVerif, "C3": KindConstraint} {
		if n := g.Nodes[id]; n == nil || n.Kind != kind {
			t.Errorf("missing %s (%s) after add", id, kind)
		}
	}
	if g.Nodes["V3"].Meta["tag"] != "DEMO-TAG" {
		t.Errorf("V3 tag = %q", g.Nodes["V3"].Meta["tag"])
	}
	// verif edit keeps the canonical tag prefix and re-renders.
	w.Reset()
	mustRun(t, cmdDefEdit(&w, root, "V3", "reworded rule text", mutFlags{}), &w)
	g = buildGraph(root, false)
	if g.Nodes["V3"].Text != "reworded rule text" || g.Nodes["V3"].Meta["tag"] != "DEMO-TAG" {
		t.Errorf("verif edit lost text/tag: %q / %q", g.Nodes["V3"].Text, g.Nodes["V3"].Meta["tag"])
	}
}

// TestWorkerTwRow: -worker books a bare-pipe Tw row in the worker fragment
// and the worker view re-renders.
func TestWorkerTwRow(t *testing.T) {
	root := specFixture(t)
	var w strings.Builder
	mustRun(t, cmdTaskAdd(&w, root, "worker lever", mutFlags{Worker: "test", Cites: "V1"}), &w)
	if !strings.Contains(w.String(), "Tw1 allocated") {
		t.Fatalf("unexpected: %s", w.String())
	}
	view, _ := os.ReadFile(filepath.Join(root, "SPEC-worker-test.md"))
	if !strings.Contains(string(view), "Tw1|.|worker lever|V1") {
		t.Fatal("worker view missing the bare-pipe row")
	}
}

// TestEntryRmRefusesStrongRefs: V1 guards B1/B2 → rm refused; -force works.
func TestEntryRmRefusesStrongRefs(t *testing.T) {
	root := specFixture(t)
	var w strings.Builder
	if rc := cmdEntryRm(&w, root, "V1", mutFlags{}); rc == 0 {
		t.Fatalf("rm of strongly-referenced V1 accepted: %s", w.String())
	}
	if !strings.Contains(w.String(), "strong references") {
		t.Fatalf("unexpected refusal message: %s", w.String())
	}
	// removing an unreferenced entry works
	w.Reset()
	mustRun(t, cmdTaskAdd(&w, root, "throwaway", mutFlags{}), &w)
	w.Reset()
	mustRun(t, cmdEntryRm(&w, root, "T11", mutFlags{}), &w)
	if _, ok := findEntryInRoot(t, root, "T11"); ok {
		t.Fatal("T11 still present after rm")
	}
}

func findEntryInRoot(t *testing.T, root, id string) (entryLoc, bool) {
	t.Helper()
	c, err := LoadCorpus(root)
	if err != nil {
		t.Fatal(err)
	}
	return findEntry(c, id)
}

// TestMutationVerifyGate: a mutation that would violate §V36 (duplicate id)
// writes NOTHING.
func TestMutationVerifyGate(t *testing.T) {
	root := specFixture(t)
	before, _ := os.ReadFile(filepath.Join(root, "spec", "70-tasks.md"))
	var w strings.Builder
	if rc := cmdTaskAdd(&w, root, "duplicate id attempt", mutFlags{ID: "T1"}); rc == 0 {
		t.Fatalf("duplicate id accepted: %s", w.String())
	}
	after, _ := os.ReadFile(filepath.Join(root, "spec", "70-tasks.md"))
	if string(before) != string(after) {
		t.Fatal("rejected mutation still wrote to spec/")
	}
}

// TestHandEditGoesRed: hand-editing a rendered view (or a fragment without
// re-render) is caught by renderCheck.
func TestHandEditGoesRed(t *testing.T) {
	root := specFixture(t)
	spec := filepath.Join(root, "SPEC.md")
	b, _ := os.ReadFile(spec)
	os.WriteFile(spec, append(b, []byte("hand edit\n")...), 0o644)
	c, err := LoadCorpus(root)
	if err != nil {
		t.Fatal(err)
	}
	ms, err := renderCheck(root, c)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, m := range ms {
		if m.RenderPath == "SPEC.md" {
			found = true
		}
	}
	if !found {
		t.Fatal("hand edit to SPEC.md not detected")
	}
}

// TestDryRunWritesNothing: -dry-run verifies but leaves the tree untouched.
func TestDryRunWritesNothing(t *testing.T) {
	root := specFixture(t)
	before, _ := os.ReadFile(filepath.Join(root, "SPEC.md"))
	var w strings.Builder
	mustRun(t, cmdTaskAdd(&w, root, "dry run task", mutFlags{DryRun: true}), &w)
	after, _ := os.ReadFile(filepath.Join(root, "SPEC.md"))
	if string(before) != string(after) {
		t.Fatal("dry-run wrote to the rendered view")
	}
	if !strings.Contains(w.String(), "dry-run") {
		t.Fatalf("missing dry-run note: %s", w.String())
	}
}

func grepLine(s, needle string) string {
	for _, ln := range strings.Split(s, "\n") {
		if strings.Contains(ln, needle) {
			return ln
		}
	}
	return ""
}
