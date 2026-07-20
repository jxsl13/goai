package main

import (
	"fmt"
	"sort"
	"strings"
)

// truncate caps s at max runes for the one-record-per-line output contract.
func truncate(s string, max int) string {
	s = strings.Join(strings.Fields(s), " ")
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max-3]) + "..."
}

// anchor renders the file:line jump target (or the id's kind for anchorless
// nodes like commits and packages).
func anchor(n *Node) string {
	if n == nil {
		return "?"
	}
	if n.File == "" {
		return string(n.Kind)
	}
	return fmt.Sprintf("%s:%d", n.File, n.Line)
}

// nodeLine renders one node as a single output record: id, anchor, the
// kind-specific columns, then the truncated text.
func nodeLine(n *Node) string {
	if n == nil {
		return "?"
	}
	parts := []string{n.ID, anchor(n)}
	for _, k := range []string{"status", "state", "priority", "date", "tag", "conf", "kind"} {
		if v := n.Meta[k]; v != "" {
			parts = append(parts, v)
		}
	}
	if t := truncate(n.Text, 140); t != "" {
		parts = append(parts, t)
	}
	return strings.Join(parts, "  ")
}

// renderGroups prints hops grouped by "direction edgetype", each group with
// its count in the header, capped at limit records per group (0 = no cap).
func renderGroups(w *strings.Builder, g *Graph, hops []hop, limit int) {
	groups := map[string][]hop{}
	for _, h := range hops {
		key := h.Edge.Type + " ->"
		if !h.Forward {
			key = "<- " + h.Edge.Type
		}
		groups[key] = append(groups[key], h)
	}
	keys := make([]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		hs := groups[k]
		fmt.Fprintf(w, "%s (%d):\n", k, len(hs))
		for i, h := range hs {
			if limit > 0 && i == limit {
				fmt.Fprintf(w, "  ... %d more (raise -n)\n", len(hs)-limit)
				break
			}
			fmt.Fprintf(w, "  %s\n", nodeLine(g.Nodes[h.ID]))
		}
	}
}

// nearestID suggests the closest existing id for a miss ("no node T99" →
// try the same-prefix id with the nearest number).
func nearestID(g *Graph, id string) string {
	want := idNum(id)
	prefix := strings.TrimRightFunc(id, func(r rune) bool { return r >= '0' && r <= '9' })
	best, bestDelta := "", 1<<62
	for _, cand := range g.NodeIDs() {
		if !strings.HasPrefix(cand, prefix) || cand == id {
			continue
		}
		d := idNum(cand) - want
		if d < 0 {
			d = -d
		}
		if d < bestDelta {
			best, bestDelta = cand, d
		}
	}
	return best
}

// renderDOT emits the subgraph induced by ids (nil = whole graph) as
// deterministic Graphviz DOT.
func renderDOT(g *Graph, ids map[string]bool) string {
	var w strings.Builder
	w.WriteString("digraph specgraph {\n  rankdir=LR;\n  node [shape=box, fontsize=10];\n")
	include := func(id string) bool { return ids == nil || ids[id] }
	for _, id := range g.NodeIDs() {
		if !include(id) {
			continue
		}
		n := g.Nodes[id]
		fmt.Fprintf(&w, "  %q [label=%q];\n", id, id+"\n"+string(n.Kind))
	}
	edges := make([]Edge, 0, len(g.Edges))
	for _, e := range g.Edges {
		if include(e.From) && include(e.To) && g.Nodes[e.From] != nil && g.Nodes[e.To] != nil {
			edges = append(edges, e)
		}
	}
	sort.Slice(edges, func(a, b int) bool {
		if edges[a].From != edges[b].From {
			return edges[a].From < edges[b].From
		}
		if edges[a].To != edges[b].To {
			return edges[a].To < edges[b].To
		}
		return edges[a].Type < edges[b].Type
	})
	for _, e := range edges {
		fmt.Fprintf(&w, "  %q -> %q [label=%q];\n", e.From, e.To, e.Type)
	}
	w.WriteString("}\n")
	return w.String()
}
