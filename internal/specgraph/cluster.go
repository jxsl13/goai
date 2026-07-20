package main

import (
	"fmt"
	"sort"
	"strings"
)

// cluster is one recurrence class: bugs (or PERF tasks) grouped by what they
// share — a guarding invariant, a package, or cause-keyword overlap.
type cluster struct {
	Key     string // "guard V36", "pkg backend/cpu", "keywords per-element allocation"
	Members []string
	First   string // earliest date among members ("" when undated)
	Last    string
}

// bugClusters groups §B rows three deterministic ways. LLM summarization of
// a cluster is the consumer's job (the skills say so); this never calls a
// model.
func bugClusters(g *Graph) []cluster {
	var out []cluster
	out = append(out, groupBy(g, KindBug, func(id string) []string {
		var keys []string
		for _, i := range g.In[id] {
			if e := g.Edges[i]; e.Type == EdgeGuards {
				keys = append(keys, "guard "+e.From)
			}
		}
		return keys
	})...)
	out = append(out, groupBy(g, KindBug, func(id string) []string {
		var keys []string
		for _, i := range g.Out[id] {
			if e := g.Edges[i]; e.Type == EdgeTouches {
				keys = append(keys, "pkg "+e.To)
			}
		}
		return keys
	})...)
	out = append(out, keywordClusters(g, KindBug)...)
	sort.Slice(out, func(a, b int) bool {
		if len(out[a].Members) != len(out[b].Members) {
			return len(out[a].Members) > len(out[b].Members)
		}
		return out[a].Key < out[b].Key
	})
	return out
}

// perfClusters does the same over done PERF §T rows — the measured
// optimization patterns.
func perfClusters(g *Graph) []cluster {
	isPerf := func(n *Node) bool {
		return n.Kind == KindTask && strings.Contains(n.Text, "PERF") && n.Meta["state"] == "done"
	}
	var out []cluster
	out = append(out, groupBy(g, KindTask, func(id string) []string {
		if !isPerf(g.Nodes[id]) {
			return nil
		}
		var keys []string
		for _, i := range g.Out[id] {
			switch e := g.Edges[i]; e.Type {
			case EdgeTouches:
				keys = append(keys, "pkg "+e.To)
			case EdgeSameClass:
				keys = append(keys, "class "+e.To)
			}
		}
		return keys
	})...)
	out = append(out, keywordClustersWhere(g, KindTask, isPerf)...)
	sort.Slice(out, func(a, b int) bool {
		if len(out[a].Members) != len(out[b].Members) {
			return len(out[a].Members) > len(out[b].Members)
		}
		return out[a].Key < out[b].Key
	})
	return out
}

// groupBy buckets nodes of kind by the keys keyfn yields; buckets with ≥2
// members become clusters.
func groupBy(g *Graph, kind Kind, keyfn func(id string) []string) []cluster {
	buckets := map[string][]string{}
	for _, id := range g.NodeIDs() {
		if g.Nodes[id].Kind != kind {
			continue
		}
		for _, k := range keyfn(id) {
			buckets[k] = append(buckets[k], id)
		}
	}
	var out []cluster
	for k, members := range buckets {
		if len(members) < 2 {
			continue
		}
		out = append(out, finishCluster(g, k, members))
	}
	return out
}

// stopwords for cause tokenization — the caveman prose's structural filler,
// not content.
var stopwords = map[string]bool{
	"the": true, "a": true, "an": true, "and": true, "or": true, "of": true,
	"to": true, "in": true, "on": true, "for": true, "with": true, "was": true,
	"is": true, "are": true, "so": true, "at": true, "by": true, "its": true,
	"it": true, "not": true, "but": true, "as": true, "that": true, "this": true,
	"every": true, "only": true, "into": true, "from": true, "new": true,
	"finding": true, "read": true, "audit": true, "round": true, "which": true,
}

// splitTokens lowercases and splits text, keeping id tokens (v22) and path
// tokens (format/gguf) whole, dropping stopwords and 1–2 char noise.
// Duplicates are kept (BM25 term frequency); tokens() is the set view.
func splitTokens(s string) []string {
	var out []string
	for _, f := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '/' || r == '-' || r == '.')
	}) {
		f = strings.Trim(f, "./-")
		if len(f) < 3 || stopwords[f] {
			continue
		}
		out = append(out, f)
	}
	return out
}

// tokens is the deduplicated set view of splitTokens.
func tokens(s string) map[string]bool {
	out := map[string]bool{}
	for _, f := range splitTokens(s) {
		out[f] = true
	}
	return out
}

// keywordClusters greedily merges same-kind nodes whose cause/task token
// sets overlap at Jaccard ≥ 0.25 — deterministic (sorted seeds, fixed
// threshold), no embeddings.
func keywordClusters(g *Graph, kind Kind) []cluster {
	return keywordClustersWhere(g, kind, func(*Node) bool { return true })
}

func keywordClustersWhere(g *Graph, kind Kind, keep func(*Node) bool) []cluster {
	var ids []string
	toks := map[string]map[string]bool{}
	for _, id := range g.NodeIDs() {
		n := g.Nodes[id]
		if n.Kind != kind || !keep(n) {
			continue
		}
		ids = append(ids, id)
		toks[id] = tokens(n.Text)
	}
	used := map[string]bool{}
	var out []cluster
	for _, seed := range ids {
		if used[seed] {
			continue
		}
		members := []string{seed}
		shared := toks[seed]
		for _, other := range ids {
			if other == seed || used[other] {
				continue
			}
			if jaccard(toks[seed], toks[other]) >= 0.25 {
				members = append(members, other)
				shared = intersect(shared, toks[other])
			}
		}
		if len(members) < 2 {
			continue
		}
		for _, m := range members {
			used[m] = true
		}
		out = append(out, finishCluster(g, "keywords "+topTokens(shared, 4), members))
	}
	return out
}

func jaccard(a, b map[string]bool) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	inter := 0
	for t := range a {
		if b[t] {
			inter++
		}
	}
	return float64(inter) / float64(len(a)+len(b)-inter)
}

func intersect(a, b map[string]bool) map[string]bool {
	out := map[string]bool{}
	for t := range a {
		if b[t] {
			out[t] = true
		}
	}
	return out
}

func topTokens(set map[string]bool, n int) string {
	ts := make([]string, 0, len(set))
	for t := range set {
		ts = append(ts, t)
	}
	sort.Strings(ts)
	if len(ts) > n {
		ts = ts[:n]
	}
	if len(ts) == 0 {
		return "(mixed)"
	}
	return strings.Join(ts, ",")
}

// finishCluster sorts members newest-first and computes the date span.
func finishCluster(g *Graph, key string, members []string) cluster {
	sort.Slice(members, func(a, b int) bool { return idNum(members[a]) > idNum(members[b]) })
	c := cluster{Key: key, Members: members}
	for _, m := range members {
		d := g.Nodes[m].Meta["date"]
		if d == "" {
			continue
		}
		if c.First == "" || d < c.First {
			c.First = d
		}
		if c.Last == "" || d > c.Last {
			c.Last = d
		}
	}
	return c
}

// renderClusters prints clusters one per block: key, span, member lines.
func renderClusters(w *strings.Builder, g *Graph, cs []cluster, limit int) {
	for i, c := range cs {
		if limit > 0 && i == limit {
			fmt.Fprintf(w, "... %d more clusters (raise -n)\n", len(cs)-limit)
			return
		}
		span := ""
		if c.First != "" {
			span = "  " + c.First
			if c.Last != c.First {
				span += ".." + c.Last
			}
		}
		fmt.Fprintf(w, "%s (%d)%s\n", c.Key, len(c.Members), span)
		for _, m := range c.Members {
			fmt.Fprintf(w, "  %s\n", nodeLine(g.Nodes[m]))
		}
	}
}
