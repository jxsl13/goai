// Package main (docgraph) is the query layer over the SPEC.md/docs citation
// graph: it deterministically parses the spec corpus (SPEC.md, worker specs,
// ADRs, CHANGELOG.md, docs/*.md) plus git history into a typed in-memory
// multigraph and answers the multi-hop questions the SDD workflow needs —
// "which bugs touched this package", "which invariant guards this bug class",
// "which PERF patterns were already measured" — as fast CLI subcommands with
// file:line-anchored output.
//
// internal/speccheck stays the §V36 validator; docgraph never mutates the
// spec. Extraction is regex/table-based (no LLM, no external deps beyond the
// module's own packages): the corpus is already typed via T/B/R/V/C/G/I/ADR
// ids and the §T cites column, so a parser-derived graph is reproducible and
// testable where an extracted one would not be. Tables are assumed to be
// clean GFM — `make lint-md` guarantees no ragged rows or bare pipes.
//
// Subcommands: show, related, why, bugs-for, impact, path, patterns, search,
// index, stats, dangling. Run `go run ./internal/docgraph help` or
// `make spec-graph ARGS=...`.
package main

import "sort"

// Kind classifies a node. The spec id namespaces map 1:1 onto kinds; the
// remaining kinds come from the surrounding corpus (git, docs, CHANGELOG).
type Kind string

const (
	KindTask       Kind = "task"       // §T rows: T907
	KindBug        Kind = "bug"        // §B rows: B108
	KindResearch   Kind = "research"   // §R rows: R42
	KindVerif      Kind = "verif"      // §V defs: V22
	KindConstraint Kind = "constraint" // §C defs: C3
	KindGoal       Kind = "goal"       // §G defs: G5
	KindArchInv    Kind = "archinv"    // §I defs: I.L0, I4
	KindBench      Kind = "bench"      // §Bench rows: BM1 (benchmark records)
	KindADR        Kind = "adr"        // docs/decisions: ADR-0027
	KindCommit     Kind = "commit"     // git log: 2d35b3b
	KindDoc        Kind = "doc"        // docs/x.md#heading
	KindChange     Kind = "change"     // CHANGELOG headings: CL:T909:2026-07-20
	KindPkg        Kind = "pkg"        // Go package paths: format/gguf
)

// Edge types. Direction is written From→To; queries use both adjacencies.
const (
	EdgeCites      = "cites"      // Task/Bug → id (the trusted §T cites column)
	EdgeGuards     = "guards"     // Verif → Bug (the §B fix guard chain)
	EdgeFixedBy    = "fixed_by"   // Task/Bug → Commit (SHA in row text)
	EdgeImplements = "implements" // Commit → id (spec refs in commit subjects)
	EdgeRecords    = "records"    // Change → Task (CHANGELOG heading)
	EdgeRelates    = "relates"    // ADR → id (header lines strong, body weak)
	EdgeMentions   = "mentions"   // Doc → id
	EdgeTouches    = "touches"    // Task/Bug/Commit → Pkg
	EdgeRefs       = "refs"       // fallback for inline § refs in row text
	EdgeSameClass  = "same_class" // Task → Task ("T870/T905 class" phrases)
)

// Node is one graph vertex. Text is the full extracted text (render
// truncates); Meta carries the kind-specific columns (status, date, ...).
// File:Line anchor the node so output is jump-to-able.
type Node struct {
	ID   string            `json:"id"`
	Kind Kind              `json:"kind"`
	File string            `json:"file,omitempty"`
	Line int               `json:"line,omitempty"`
	Text string            `json:"text,omitempty"`
	Meta map[string]string `json:"meta,omitempty"`
}

// Edge is one typed, directed reference with provenance. Weak marks
// low-confidence extractions (e.g. ADR body refs vs header refs).
type Edge struct {
	From string `json:"from"`
	To   string `json:"to"`
	Type string `json:"type"`
	File string `json:"file,omitempty"`
	Line int    `json:"line,omitempty"`
	Weak bool   `json:"weak,omitempty"`
}

// Graph is the in-memory multigraph: nodes by id, the edge list, and both
// adjacencies as edge-index slices. Edges may reference ids with no node —
// those are the `dangling` subcommand's material, never an error.
type Graph struct {
	Nodes map[string]*Node
	Edges []Edge
	Out   map[string][]int
	In    map[string][]int
}

// NewGraph returns an empty graph.
func NewGraph() *Graph {
	return &Graph{
		Nodes: map[string]*Node{},
		Out:   map[string][]int{},
		In:    map[string][]int{},
	}
}

// AddNode inserts n, keeping the first definition on duplicate ids (the spec
// guarantees uniqueness via §V36; duplicates here would come from re-parsing).
func (g *Graph) AddNode(n *Node) *Node {
	if have, ok := g.Nodes[n.ID]; ok {
		return have
	}
	g.Nodes[n.ID] = n
	return n
}

// AddEdge appends e and indexes it in both adjacencies. Self-loops and exact
// duplicates (same from/to/type) are dropped to keep neighborhoods readable.
func (g *Graph) AddEdge(e Edge) {
	if e.From == e.To || e.From == "" || e.To == "" {
		return
	}
	for _, i := range g.Out[e.From] {
		h := g.Edges[i]
		if h.To == e.To && h.Type == e.Type {
			return
		}
	}
	g.Edges = append(g.Edges, e)
	i := len(g.Edges) - 1
	g.Out[e.From] = append(g.Out[e.From], i)
	g.In[e.To] = append(g.In[e.To], i)
}

// NodeIDs returns every node id in sorted order (all iteration in docgraph
// is sorted so output — and the DOT export — is byte-deterministic).
func (g *Graph) NodeIDs() []string {
	ids := make([]string, 0, len(g.Nodes))
	for id := range g.Nodes {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// Dangling returns the edges whose To (or From) id has no node, sorted by
// provenance. These are cites/refs to entities that do not exist — a spec
// hygiene signal speccheck does not cover.
func (g *Graph) Dangling() []Edge {
	var out []Edge
	for _, e := range g.Edges {
		if _, ok := g.Nodes[e.To]; !ok {
			out = append(out, e)
			continue
		}
		if _, ok := g.Nodes[e.From]; !ok {
			out = append(out, e)
		}
	}
	sort.Slice(out, func(a, b int) bool {
		if out[a].File != out[b].File {
			return out[a].File < out[b].File
		}
		if out[a].Line != out[b].Line {
			return out[a].Line < out[b].Line
		}
		return out[a].To < out[b].To
	})
	return out
}
