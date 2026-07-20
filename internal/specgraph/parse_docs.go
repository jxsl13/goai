package main

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// pkgSet is the validated set of Go package paths in the repo — the guard
// that keeps touches edges high-precision: a path-like token only becomes a
// Pkg node if a directory with .go files actually exists at that path, so
// prose URLs ("martinfowler.com/bliki") and made-up examples never match.
type pkgSet struct {
	dirs map[string]bool // "backend/cpu" -> true
	tops map[string]bool // "backend" -> true (for pkg.Symbol tokens)
}

var (
	rePathTok = regexp.MustCompile(`\b[a-z][a-z0-9_]*(?:/[a-z0-9_]+)+\b`)
	reDotTok  = regexp.MustCompile(`\b([a-z][a-z0-9_]*)\.[A-Z]\w*`)
)

// loadPkgSet walks root once for directories containing .go files.
func loadPkgSet(root string) *pkgSet {
	ps := &pkgSet{dirs: map[string]bool{}, tops: map[string]bool{}}
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			name := d.Name()
			if strings.HasPrefix(name, ".") && path != root {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		rel, err := filepath.Rel(root, filepath.Dir(path))
		if err != nil || rel == "." {
			return nil
		}
		rel = filepath.ToSlash(rel)
		ps.dirs[rel] = true
		ps.tops[strings.SplitN(rel, "/", 2)[0]] = true
		return nil
	})
	return ps
}

// find returns the validated package paths mentioned in text, sorted:
// multi-segment path tokens that name a real package dir (or a prefix of
// one, so "backend/cpu/vexp.go" resolves to backend/cpu), plus pkg.Symbol
// tokens whose pkg is a real top-level package (nn.WSD -> nn).
func (ps *pkgSet) find(text string) []string {
	seen := map[string]bool{}
	for _, tok := range rePathTok.FindAllString(text, -1) {
		for p := tok; p != ""; {
			if ps.dirs[p] {
				seen[p] = true
				break
			}
			i := strings.LastIndex(p, "/")
			if i < 0 {
				break
			}
			p = p[:i]
		}
	}
	for _, m := range reDotTok.FindAllStringSubmatch(text, -1) {
		if ps.dirs[m[1]] && ps.tops[m[1]] {
			seen[m[1]] = true
		}
	}
	out := make([]string, 0, len(seen))
	for p := range seen {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

// ensurePkgNodes adds a Pkg node for every touches target so they are never
// dangling.
func ensurePkgNodes(g *Graph) {
	for _, e := range g.Edges {
		if e.Type == EdgeTouches {
			g.AddNode(&Node{ID: e.To, Kind: KindPkg})
		}
	}
}

var (
	reADRTitle  = regexp.MustCompile(`^# (ADR-\d{4})[:\s—-]+(.*)`)
	reADRMeta   = regexp.MustCompile(`^- (Status|Date):\s*(.*)`)
	reADRStrong = regexp.MustCompile(`^- (?:Related|Relates|Task):\s*(.*)`)
	reCLHead    = regexp.MustCompile(`^### (.+?) (?:--|—) (.*) \(([^)]*)\)\s*$`)
	reMDHead    = regexp.MustCompile(`^(##+)\s+(.*)`)
	reDate      = regexp.MustCompile(`\b\d{4}-\d{2}-\d{2}\b`)
)

// parseADR extracts one docs/decisions/ADR-*.md file: an ADR node plus
// relates edges — strong from the `- Related:`/`- Task:` header lines, weak
// from body refs.
func parseADR(g *Graph, file, content string) {
	lines := strings.Split(content, "\n")
	var n *Node
	for i, ln := range lines {
		lineNo := i + 1
		if m := reADRTitle.FindStringSubmatch(ln); m != nil && n == nil {
			n = g.AddNode(&Node{
				ID: m[1], Kind: KindADR, File: file, Line: lineNo,
				Text: strings.TrimSpace(m[2]), Meta: map[string]string{},
			})
			continue
		}
		if n == nil {
			continue
		}
		if m := reADRMeta.FindStringSubmatch(ln); m != nil {
			n.Meta[strings.ToLower(m[1])] = strings.TrimSpace(m[2])
			continue
		}
		// Body refs are weak and — unlike the header lines — only accepted
		// when the target node exists: free prose contains bare id
		// look-alikes (GPU names "V100"/"T4") that must not become edges.
		// parseADR/parseDoc therefore run AFTER the spec files.
		weak := true
		text := ln
		if m := reADRStrong.FindStringSubmatch(ln); m != nil {
			weak, text = false, m[1]
		}
		for _, ref := range scanRefs(text) {
			if ref == n.ID {
				continue
			}
			if weak && g.Nodes[ref] == nil {
				continue
			}
			g.AddEdge(Edge{From: n.ID, To: ref, Type: EdgeRelates, File: file, Line: lineNo, Weak: weak})
		}
	}
}

// parseChangelog extracts CHANGELOG.md `### kind -- title (T909, date)`
// headings into Change nodes with records edges to their tasks.
func parseChangelog(g *Graph, file, content string) {
	for i, ln := range strings.Split(content, "\n") {
		m := reCLHead.FindStringSubmatch(ln)
		if m == nil {
			continue
		}
		lineNo := i + 1
		kind, title, paren := m[1], m[2], m[3]
		tasks := regexp.MustCompile(`\bT\d+\b`).FindAllString(paren, -1)
		if len(tasks) == 0 {
			continue
		}
		date := reDate.FindString(paren)
		id := "CL:" + tasks[0]
		if date != "" {
			id += ":" + date
		}
		n := g.AddNode(&Node{
			ID: id, Kind: KindChange, File: file, Line: lineNo,
			Text: title, Meta: map[string]string{"kind": kind, "date": date},
		})
		for _, t := range tasks {
			g.AddEdge(Edge{From: n.ID, To: t, Type: EdgeRecords, File: file, Line: lineNo})
		}
		_ = n
	}
}

// parseDoc extracts one long-form doc page (docs/*.md, README.md,
// BENCHMARKS.md): a Doc node per heading section that references at least
// one spec entity, with mentions edges. Headingless files anchor at line 1.
func parseDoc(g *Graph, file, content string) {
	lines := strings.Split(content, "\n")
	heading, headLine := "", 1
	var pending []Edge
	flush := func() {
		if len(pending) == 0 {
			return
		}
		id := file
		if heading != "" {
			id += "#" + heading
		}
		g.AddNode(&Node{ID: id, Kind: KindDoc, File: file, Line: headLine, Text: heading})
		for _, e := range pending {
			e.From = id
			g.AddEdge(e)
		}
		pending = nil
	}
	for i, ln := range lines {
		lineNo := i + 1
		if m := reMDHead.FindStringSubmatch(ln); m != nil {
			flush()
			heading, headLine = strings.TrimSpace(m[2]), lineNo
			continue
		}
		// Like ADR body refs, doc mentions are only accepted for existing
		// nodes (bare-id look-alikes in prose), so docs parse after specs.
		for _, ref := range scanRefs(ln) {
			if g.Nodes[ref] != nil {
				pending = append(pending, Edge{To: ref, Type: EdgeMentions, File: file, Line: lineNo})
			}
		}
	}
	flush()
}
