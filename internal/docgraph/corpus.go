package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// The spec/ hierarchy (§V41): one caveman-markdown file per section, and the
// committed SPEC.md / SPEC-worker-*.md become deterministically RENDERED
// views. Render = lexicographic filename sort + join with "\n" (each fragment
// ends with exactly one \n, so the join separator recreates the single blank
// line between sections). The number prefix in the filename IS the manifest;
// the tasks file must sort last so §T stays the last rendered section (§V36
// append-at-EOF safety, checked at render time as defense in depth).
const (
	specDirName   = "spec"
	workerDirName = "worker"
)

// Fragment is one section source file of an output.
type Fragment struct {
	Path    string // repo-relative, e.g. spec/70-tasks.md
	Content string // verbatim bytes, ending with exactly one \n
}

// Output is one rendered view: the fragments (sorted) and the render target.
type Output struct {
	RenderPath string // e.g. SPEC.md or SPEC-worker-linux-amd64-cuda.md
	Fragments  []Fragment
}

// Corpus is the whole spec/ tree: the root output plus one per worker.
type Corpus struct {
	Root    string
	Outputs []Output
}

// HasSpecDir reports whether the spec/ hierarchy exists at root.
func HasSpecDir(root string) bool {
	fi, err := os.Stat(filepath.Join(root, specDirName))
	return err == nil && fi.IsDir()
}

// LoadCorpus reads the spec/ tree: spec/*.md → SPEC.md, and each
// spec/worker/<name>/*.md → SPEC-worker-<name>.md.
func LoadCorpus(root string) (*Corpus, error) {
	c := &Corpus{Root: root}
	rootOut, err := loadOutput(root, specDirName, "SPEC.md")
	if err != nil {
		return nil, err
	}
	c.Outputs = append(c.Outputs, rootOut)

	workerBase := filepath.Join(root, specDirName, workerDirName)
	entries, err := os.ReadDir(workerBase)
	if err != nil {
		if os.IsNotExist(err) {
			return c, nil
		}
		return nil, err
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	for _, name := range names {
		o, err := loadOutput(root, filepath.Join(specDirName, workerDirName, name), "SPEC-worker-"+name+".md")
		if err != nil {
			return nil, err
		}
		c.Outputs = append(c.Outputs, o)
	}
	return c, nil
}

// loadOutput reads the *.md fragments of one directory (non-recursive),
// sorted lexicographically.
func loadOutput(root, dir, renderPath string) (Output, error) {
	o := Output{RenderPath: renderPath}
	entries, err := os.ReadDir(filepath.Join(root, dir))
	if err != nil {
		return o, fmt.Errorf("docgraph: %s: %w", dir, err)
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".md") {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	for _, name := range names {
		rel := filepath.ToSlash(filepath.Join(dir, name))
		b, err := os.ReadFile(filepath.Join(root, dir, name))
		if err != nil {
			return o, err
		}
		o.Fragments = append(o.Fragments, Fragment{Path: rel, Content: string(b)})
	}
	if len(o.Fragments) == 0 {
		return o, fmt.Errorf("docgraph: %s has no .md fragments", dir)
	}
	return o, nil
}

// fragmentByPath returns a pointer into the corpus for in-memory mutation.
func (c *Corpus) fragmentByPath(path string) *Fragment {
	for i := range c.Outputs {
		for j := range c.Outputs[i].Fragments {
			if c.Outputs[i].Fragments[j].Path == path {
				return &c.Outputs[i].Fragments[j]
			}
		}
	}
	return nil
}

// output returns the Output whose views the fragment belongs to ("" = root).
func (c *Corpus) output(renderPath string) *Output {
	for i := range c.Outputs {
		if c.Outputs[i].RenderPath == renderPath {
			return &c.Outputs[i]
		}
	}
	return nil
}
