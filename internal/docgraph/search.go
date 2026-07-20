package main

import (
	"math"
	"sort"
)

// bm25Index is a plain inverted index over node texts — the always-available
// lexical entry point into the graph (the vector side is optional and
// model-gated; see embed.go). ~2k docs, so everything stays in memory and
// unpersisted.
type bm25Index struct {
	ids      []string
	docLen   map[string]int
	postings map[string]map[string]int // term -> doc id -> tf
	avgLen   float64
}

// buildBM25 indexes every node that has text (id + kind + text, so a query
// for "gguf" also hits the format/gguf Pkg node).
func buildBM25(g *Graph) *bm25Index {
	idx := &bm25Index{docLen: map[string]int{}, postings: map[string]map[string]int{}}
	total := 0
	for _, id := range g.NodeIDs() {
		n := g.Nodes[id]
		doc := id + " " + string(n.Kind) + " " + n.Text + " " + n.Meta["fix"] + " " + n.Meta["tag"]
		ts := tokensList(doc)
		if len(ts) == 0 {
			continue
		}
		idx.ids = append(idx.ids, id)
		idx.docLen[id] = len(ts)
		total += len(ts)
		for _, t := range ts {
			if idx.postings[t] == nil {
				idx.postings[t] = map[string]int{}
			}
			idx.postings[t][id]++
		}
	}
	if len(idx.ids) > 0 {
		idx.avgLen = float64(total) / float64(len(idx.ids))
	}
	return idx
}

// tokensList is splitTokens — tokens() with duplicates kept, because BM25
// needs term frequency.
func tokensList(s string) []string { return splitTokens(s) }

// scored is one search hit.
type scored struct {
	ID    string
	Score float64
}

// search runs BM25 (k1=1.2, b=0.75) over the query terms.
func (idx *bm25Index) search(query string, k int) []scored {
	const k1, b = 1.2, 0.75
	n := float64(len(idx.ids))
	scores := map[string]float64{}
	for _, term := range splitTokens(query) {
		post := idx.postings[term]
		if len(post) == 0 {
			continue
		}
		idf := math.Log(1 + (n-float64(len(post))+0.5)/(float64(len(post))+0.5))
		for id, tf := range post {
			d := float64(idx.docLen[id])
			scores[id] += idf * float64(tf) * (k1 + 1) / (float64(tf) + k1*(1-b+b*d/idx.avgLen))
		}
	}
	out := make([]scored, 0, len(scores))
	for id, s := range scores {
		out = append(out, scored{ID: id, Score: s})
	}
	sort.Slice(out, func(a, b int) bool {
		if out[a].Score != out[b].Score {
			return out[a].Score > out[b].Score
		}
		return out[a].ID < out[b].ID
	})
	if k > 0 && len(out) > k {
		out = out[:k]
	}
	return out
}

// rrf fuses ranked lists by Reciprocal Rank Fusion (k=60, the standard
// constant): score = Σ 1/(60+rank). Deterministic tie-break on id.
func rrf(lists ...[]scored) []scored {
	fused := map[string]float64{}
	for _, list := range lists {
		for rank, s := range list {
			fused[s.ID] += 1.0 / float64(60+rank+1)
		}
	}
	out := make([]scored, 0, len(fused))
	for id, s := range fused {
		out = append(out, scored{ID: id, Score: s})
	}
	sort.Slice(out, func(a, b int) bool {
		if out[a].Score != out[b].Score {
			return out[a].Score > out[b].Score
		}
		return out[a].ID < out[b].ID
	})
	return out
}
