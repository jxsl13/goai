package main

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// inputFiles returns the corpus files relative to root, sorted: SPEC.md,
// every SPEC-worker-*.md, CHANGELOG.md, docs/** (ADRs included), README.md,
// BENCHMARKS.md. Missing files are simply skipped so the tool also runs on
// fixture trees.
func inputFiles(root string) []string {
	var files []string
	add := func(rel string) {
		if _, err := os.Stat(filepath.Join(root, rel)); err == nil {
			files = append(files, rel)
		}
	}
	add("SPEC.md")
	workers, _ := filepath.Glob(filepath.Join(root, "SPEC-worker-*.md"))
	for _, w := range workers {
		add(filepath.Base(w))
	}
	add("CHANGELOG.md")
	add("README.md")
	add("BENCHMARKS.md")
	_ = filepath.WalkDir(filepath.Join(root, "docs"), func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, ".md") {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err == nil {
			files = append(files, filepath.ToSlash(rel))
		}
		return nil
	})
	sort.Strings(files)
	return files
}

// buildGraph parses the whole corpus into a fresh graph. Order matters:
// spec files define the id nodes first, so the doc/ADR-body extractors can
// reject bare-id look-alikes; git runs last because it consumes the spec's
// fixed_by SHA citations.
func buildGraph(root string, withGit bool) *Graph {
	g := NewGraph()
	pkgs := loadPkgSet(root)
	files := inputFiles(root)

	classify := func(rel string) string {
		base := filepath.Base(rel)
		switch {
		case strings.HasPrefix(base, "SPEC"):
			return "spec"
		case base == "CHANGELOG.md":
			return "changelog"
		case strings.HasPrefix(rel, "docs/decisions/") && strings.HasPrefix(base, "ADR-"):
			return "adr"
		default:
			return "doc"
		}
	}
	read := func(rel string) (string, bool) {
		b, err := os.ReadFile(filepath.Join(root, rel))
		if err != nil {
			return "", false
		}
		return string(b), true
	}
	for _, pass := range []string{"spec", "adr", "changelog", "doc"} {
		for _, rel := range files {
			if classify(rel) != pass {
				continue
			}
			content, ok := read(rel)
			if !ok {
				continue
			}
			switch pass {
			case "spec":
				parseSpec(g, rel, content, pkgs)
			case "adr":
				parseADR(g, rel, content)
			case "changelog":
				parseChangelog(g, rel, content)
			case "doc":
				parseDoc(g, rel, content)
			}
		}
	}
	if withGit {
		parseGitLog(g, root, pkgs)
	}
	ensurePkgNodes(g)
	return g
}
