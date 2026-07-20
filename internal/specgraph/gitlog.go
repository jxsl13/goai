package main

import (
	"os/exec"
	"strings"
)

// parseGitLog adds Commit nodes and implements/touches edges from one
// `git log` pass over every commit subject. Only commits that cite a spec id
// — or whose SHA the spec cites (resolved by prefix, since spec rows embed
// 7-char short SHAs while %h length follows core.abbrev) — become nodes; the
// rest of history stays out of the graph. Errors degrade to a no-git graph:
// the spec-side fixed_by edges still carry the story.
func parseGitLog(g *Graph, root string, pkgs *pkgSet) {
	out, err := exec.Command("git", "-C", root, "log", "--format=%H%x09%h%x09%ad%x09%s", "--date=short").Output()
	if err != nil {
		return
	}
	// SHA prefixes the spec cites via fixed_by, so those commits get nodes
	// even without a spec id in their subject.
	wanted := map[string]bool{}
	for _, e := range g.Edges {
		if e.Type == EdgeFixedBy {
			wanted[e.To] = true
		}
	}
	resolve := map[string]string{} // cited prefix -> canonical short SHA
	for _, ln := range strings.Split(strings.TrimRight(string(out), "\n"), "\n") {
		parts := strings.SplitN(ln, "\t", 4)
		if len(parts) != 4 {
			continue
		}
		full, short, date, subject := parts[0], parts[1], parts[2], parts[3]
		refs := scanRefs(subject)
		cited := ""
		for p := range wanted {
			if strings.HasPrefix(full, p) {
				cited = p
				break
			}
		}
		if len(refs) == 0 && cited == "" {
			continue
		}
		n := g.AddNode(&Node{
			ID: short, Kind: KindCommit, Text: subject,
			Meta: map[string]string{"date": date, "sha": full},
		})
		if cited != "" && cited != short {
			resolve[cited] = short
		}
		for _, ref := range refs {
			g.AddEdge(Edge{From: n.ID, To: ref, Type: EdgeImplements})
		}
		// The `pkg: subject` convention is high-precision for touches.
		if i := strings.Index(subject, ":"); i > 0 {
			for _, p := range pkgs.find(subject[:i]) {
				g.AddEdge(Edge{From: n.ID, To: p, Type: EdgeTouches})
			}
		}
	}
	// Rewrite fixed_by targets whose cited prefix resolved to a different
	// canonical short SHA, so both sides land on one Commit node.
	rewrote := false
	for i, e := range g.Edges {
		if e.Type == EdgeFixedBy {
			if to, ok := resolve[e.To]; ok && to != e.To {
				g.Edges[i].To = to
				rewrote = true
			}
		}
	}
	if rewrote {
		reindex(g)
	}
}

// reindex rebuilds both adjacencies after in-place edge rewrites.
func reindex(g *Graph) {
	g.Out = map[string][]int{}
	g.In = map[string][]int{}
	for i, e := range g.Edges {
		g.Out[e.From] = append(g.Out[e.From], i)
		g.In[e.To] = append(g.In[e.To], i)
	}
}
