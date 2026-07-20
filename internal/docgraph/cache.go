package main

import (
	"bufio"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// cacheSchemaVersion invalidates every cache on extractor changes — bump it
// whenever parse*/gitlog output shape or semantics change.
const cacheSchemaVersion = 4 // 2: splitRow honors \| escapes (§B114); 3: spec/ hierarchy corpus (§V41); 4: §V/§C/§G/§I are GFM tables

// cacheDir/graph file live under .docgraph/ at repo root (gitignored): the
// cache is pure acceleration, never a source of truth — any doubt about its
// freshness resolves to a rebuild, and a cache failure is never fatal.
const (
	cacheDirName   = ".docgraph"
	graphCacheName = "graph.jsonl"
	embedCacheName = "embeddings.jsonl"
)

// manifest is the first JSONL record: everything that must match for the
// cached graph to be trusted.
type manifest struct {
	Schema int         `json:"schema"`
	Head   string      `json:"head"` // git HEAD ("" when git is unavailable)
	Git    bool        `json:"git"`  // whether the graph includes git nodes
	Inputs []inputStat `json:"inputs"`
}

type inputStat struct {
	Path  string `json:"path"`
	Size  int64  `json:"size"`
	Mtime int64  `json:"mtime"`
}

// cacheRecord is one post-manifest JSONL line: exactly one of Node/Edge set.
type cacheRecord struct {
	Node *Node `json:"node,omitempty"`
	Edge *Edge `json:"edge,omitempty"`
}

// currentManifest stats every input file and git HEAD.
func currentManifest(root string, withGit bool) manifest {
	m := manifest{Schema: cacheSchemaVersion, Git: withGit}
	for _, rel := range inputFiles(root) {
		fi, err := os.Stat(filepath.Join(root, rel))
		if err != nil {
			continue
		}
		m.Inputs = append(m.Inputs, inputStat{Path: rel, Size: fi.Size(), Mtime: fi.ModTime().UnixNano()})
	}
	if withGit {
		if out, err := exec.Command("git", "-C", root, "rev-parse", "HEAD").Output(); err == nil {
			m.Head = strings.TrimSpace(string(out))
		}
	}
	return m
}

func manifestEqual(a, b manifest) bool {
	if a.Schema != b.Schema || a.Head != b.Head || a.Git != b.Git || len(a.Inputs) != len(b.Inputs) {
		return false
	}
	for i := range a.Inputs {
		if a.Inputs[i] != b.Inputs[i] {
			return false
		}
	}
	return true
}

// loadCachedGraph returns the cached graph when the stored manifest matches
// the current corpus state, else (nil, false). Any parse problem = miss.
func loadCachedGraph(root string, withGit bool) (*Graph, bool) {
	f, err := os.Open(filepath.Join(root, cacheDirName, graphCacheName))
	if err != nil {
		return nil, false
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<24)
	if !sc.Scan() {
		return nil, false
	}
	var stored manifest
	if json.Unmarshal(sc.Bytes(), &stored) != nil || !manifestEqual(stored, currentManifest(root, withGit)) {
		return nil, false
	}
	g := NewGraph()
	for sc.Scan() {
		var r cacheRecord
		if json.Unmarshal(sc.Bytes(), &r) != nil {
			return nil, false
		}
		switch {
		case r.Node != nil:
			g.AddNode(r.Node)
		case r.Edge != nil:
			g.AddEdge(*r.Edge)
		}
	}
	if sc.Err() != nil {
		return nil, false
	}
	return g, true
}

// saveCachedGraph writes manifest + nodes + edges atomically (temp+rename),
// sorted so the file is deterministic. Errors are swallowed by the caller —
// worst case is the next invocation rebuilding.
func saveCachedGraph(root string, withGit bool, g *Graph) error {
	dir := filepath.Join(root, cacheDirName)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	path := filepath.Join(dir, graphCacheName)
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w)
	if err := enc.Encode(currentManifest(root, withGit)); err != nil {
		f.Close()
		return err
	}
	for _, id := range g.NodeIDs() {
		if err := enc.Encode(cacheRecord{Node: g.Nodes[id]}); err != nil {
			f.Close()
			return err
		}
	}
	for i := range g.Edges {
		if err := enc.Encode(cacheRecord{Edge: &g.Edges[i]}); err != nil {
			f.Close()
			return err
		}
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// loadGraph is the cache-aware entry point: cached graph when fresh, else
// build + best-effort save. noCache skips both directions (hermetic tests);
// refresh forces a rebuild but still writes.
func loadGraph(root string, withGit, noCache, refresh bool) *Graph {
	if !noCache && !refresh {
		if g, ok := loadCachedGraph(root, withGit); ok {
			return g
		}
	}
	g := buildGraph(root, withGit)
	if !noCache {
		_ = saveCachedGraph(root, withGit, g) // never fatal: worst case is a rebuild
	}
	return g
}
