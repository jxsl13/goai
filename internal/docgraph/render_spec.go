package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/jxsl13/goai/internal/speccheck"
)

// renderOutput joins one output's fragments into the rendered view. Every
// fragment is normalized to end with exactly one \n; the "\n" join separator
// recreates the single blank line between sections, making split/render a
// byte-faithful inverse pair (§V41, proven by TestSplitRoundTrip).
func renderOutput(o Output) (string, error) {
	parts := make([]string, len(o.Fragments))
	for i, f := range o.Fragments {
		c := strings.TrimRight(f.Content, "\n") + "\n"
		if strings.TrimSpace(c) == "" {
			return "", fmt.Errorf("docgraph: %s is empty", f.Path)
		}
		parts[i] = c
	}
	if o.RenderPath == "SPEC.md" {
		// §V36 defense in depth: the tasks fragment must sort last so §T
		// stays the last rendered section.
		last := o.Fragments[len(o.Fragments)-1].Path
		if !strings.Contains(last, "tasks") {
			return "", fmt.Errorf("docgraph: tasks file must sort last in spec/ (got %s last) — §V36/§V41", last)
		}
	}
	return strings.Join(parts, "\n"), nil
}

// renderAll renders every output; the map is keyed by RenderPath.
func renderAll(c *Corpus) (map[string]string, error) {
	out := map[string]string{}
	for _, o := range c.Outputs {
		s, err := renderOutput(o)
		if err != nil {
			return nil, err
		}
		out[o.RenderPath] = s
	}
	return out, nil
}

// mismatch is one out-of-sync rendered view.
type mismatch struct {
	RenderPath string
	Reason     string
	Worker     bool
}

// workerRenderSyncHard gates hard enforcement of render-sync for the
// SPEC-worker-*.md views. False = warn-only while the worker machine has not
// yet pulled the tool and still hand-edits its rendered file (migration
// phase A, see the plan in §T's migration task); flip to true in phase C.
const workerRenderSyncHard = false

// renderCheck compares the in-memory renders against the committed rendered
// files. It never writes.
func renderCheck(root string, c *Corpus) ([]mismatch, error) {
	rendered, err := renderAll(c)
	if err != nil {
		return nil, err
	}
	var out []mismatch
	for path, want := range rendered {
		got, err := os.ReadFile(filepath.Join(root, path))
		worker := strings.HasPrefix(path, "SPEC-worker-")
		switch {
		case err != nil:
			out = append(out, mismatch{path, "rendered view missing: " + err.Error(), worker})
		case string(got) != want:
			out = append(out, mismatch{path, "rendered view out of sync with spec/ — regenerate via `docgraph render`, never hand-edit (§V41)", worker})
		}
	}
	return out, nil
}

// writeRendered writes every rendered view (atomic per file).
func writeRendered(root string, rendered map[string]string) error {
	for path, content := range rendered {
		full := filepath.Join(root, path)
		tmp := full + ".tmp"
		if err := os.WriteFile(tmp, []byte(content), 0o644); err != nil {
			return err
		}
		if err := os.Rename(tmp, full); err != nil {
			return err
		}
	}
	return nil
}

// verifyCorpus runs the full validation suite on the CURRENT in-memory
// corpus state: §V36 speccheck on the rendered strings, GFM table shape on
// every table row, dangling strong refs (§V39), and the tasks-last rule
// (inside renderAll). Returns human-readable problems, empty when clean.
func verifyCorpus(root string, c *Corpus) ([]string, error) {
	var problems []string
	rendered, err := renderAll(c)
	if err != nil {
		return []string{err.Error()}, nil
	}
	// §V36 via the unchanged validator, fed the rendered strings.
	workers := map[string]string{}
	for path, s := range rendered {
		if strings.HasPrefix(path, "SPEC-worker-") {
			workers[path] = s
		}
	}
	for _, v := range speccheck.Check(rendered["SPEC.md"], workers) {
		problems = append(problems, v.String())
	}
	// Table shape: the two mdlint rules §V36 leans on (ragged rows,
	// separator mismatch), applied to the fragments so findings anchor to
	// the editable source. Full mdlint still runs via `make lint-md`.
	for _, o := range c.Outputs {
		for _, f := range o.Fragments {
			problems = append(problems, tableShapeProblems(f.Path, f.Content)...)
		}
	}
	// §V39 strong dangling refs on the graph of the would-be state.
	problems = append(problems, strongDanglingProblems(root, c)...)
	return problems, nil
}

var reDelimRow = regexp.MustCompile(`^\|[ :-]+\|`)

// tableShapeProblems checks GFM tables with mdlint's semantics (the two
// rules §V36 leans on): a table BEGINS at a pipe line whose next line is a
// delimiter row — stray pipe lines outside that shape are not tables (the
// §R section legitimately resumes rows after a blockquote) — and inside a
// table the separator and every row must match the header's column count.
func tableShapeProblems(path, content string) []string {
	var out []string
	lines := strings.Split(content, "\n")
	for i := 0; i < len(lines); i++ {
		if !strings.HasPrefix(lines[i], "|") || i+1 >= len(lines) || !reDelimRow.MatchString(lines[i+1]) {
			continue
		}
		cols := len(splitRow(lines[i]))
		if d := len(splitRow(lines[i+1])); d != cols {
			out = append(out, fmt.Sprintf("%s:%d: table separator has %d columns, header has %d", path, i+2, d, cols))
		}
		j := i + 2
		for ; j < len(lines) && strings.HasPrefix(lines[j], "|"); j++ {
			if n := len(splitRow(lines[j])); n != cols {
				out = append(out, fmt.Sprintf("%s:%d: ragged table row (%d columns, table has %d)", path, j+1, n, cols))
			}
		}
		i = j - 1
	}
	return out
}

// corpusGraph parses the in-memory corpus plus the on-disk ADRs and
// CHANGELOG into a graph — the reference universe for §V39 checks during a
// mutation (ADR-XXXX and changelog records are legitimate cite targets).
func corpusGraph(root string, c *Corpus) *Graph {
	g := NewGraph()
	pkgs := loadPkgSet(root)
	for _, o := range c.Outputs {
		for _, f := range o.Fragments {
			parseSpec(g, f.Path, f.Content, pkgs)
		}
	}
	adrs, _ := filepath.Glob(filepath.Join(root, "docs", "decisions", "ADR-*.md"))
	for _, p := range adrs {
		if b, err := os.ReadFile(p); err == nil {
			rel, _ := filepath.Rel(root, p)
			parseADR(g, filepath.ToSlash(rel), string(b))
		}
	}
	if b, err := os.ReadFile(filepath.Join(root, "CHANGELOG.md")); err == nil {
		parseChangelog(g, "CHANGELOG.md", string(b))
	}
	return g
}

// strongDanglingProblems reports §V39 violations (cites/records/guards to
// nonexistent ids) on the in-memory corpus state.
func strongDanglingProblems(root string, c *Corpus) []string {
	g := corpusGraph(root, c)
	strong := map[string]bool{EdgeCites: true, EdgeRecords: true, EdgeGuards: true}
	var out []string
	for _, e := range g.Dangling() {
		if strong[e.Type] {
			out = append(out, fmt.Sprintf("%s:%d: dangling strong ref %s -[%s]-> %s (§V39)", e.File, e.Line, e.From, e.Type, e.To))
		}
	}
	return out
}

// cmdRender regenerates the views (or with check=true only reports drift).
func cmdRender(w *strings.Builder, root string, check bool) int {
	c, err := LoadCorpus(root)
	if err != nil {
		fmt.Fprintf(w, "render: %v\n", err)
		return 1
	}
	if check {
		ms, err := renderCheck(root, c)
		if err != nil {
			fmt.Fprintf(w, "render: %v\n", err)
			return 1
		}
		return reportMismatches(w, ms)
	}
	rendered, err := renderAll(c)
	if err != nil {
		fmt.Fprintf(w, "render: %v\n", err)
		return 1
	}
	if err := writeRendered(root, rendered); err != nil {
		fmt.Fprintf(w, "render: %v\n", err)
		return 1
	}
	for _, o := range c.Outputs {
		fmt.Fprintf(w, "rendered %s (%d fragments)\n", o.RenderPath, len(o.Fragments))
	}
	return 0
}

// cmdVerify is the read-only full gate: render-sync + verifyCorpus.
func cmdVerify(w *strings.Builder, root string) int {
	c, err := LoadCorpus(root)
	if err != nil {
		fmt.Fprintf(w, "verify: %v\n", err)
		return 1
	}
	ms, err := renderCheck(root, c)
	if err != nil {
		fmt.Fprintf(w, "verify: %v\n", err)
		return 1
	}
	rc := reportMismatches(w, ms)
	problems, err := verifyCorpus(root, c)
	if err != nil {
		fmt.Fprintf(w, "verify: %v\n", err)
		return 1
	}
	for _, p := range problems {
		fmt.Fprintf(w, "%s\n", p)
	}
	if rc == 0 && len(problems) == 0 {
		fmt.Fprintln(w, "verify: clean")
		return 0
	}
	return 1
}

// reportMismatches prints render-sync drift; worker drift is warn-only while
// workerRenderSyncHard is false.
func reportMismatches(w *strings.Builder, ms []mismatch) int {
	rc := 0
	for _, m := range ms {
		if m.Worker && !workerRenderSyncHard {
			fmt.Fprintf(w, "warn: %s: %s (worker enforcement phase A)\n", m.RenderPath, m.Reason)
			continue
		}
		fmt.Fprintf(w, "%s: %s\n", m.RenderPath, m.Reason)
		rc = 1
	}
	return rc
}
