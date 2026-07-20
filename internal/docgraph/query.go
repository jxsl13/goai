package main

import "sort"

// hop is one traversal step: the neighbor id, the edge that got there, and
// whether it was followed forward (From→To) or in reverse.
type hop struct {
	ID      string
	Edge    Edge
	Forward bool
}

// neighbors returns id's one-hop neighborhood over both adjacencies, sorted
// by neighbor id then edge type — deterministic like all iteration here.
func (g *Graph) neighbors(id string) []hop {
	var out []hop
	for _, i := range g.Out[id] {
		out = append(out, hop{ID: g.Edges[i].To, Edge: g.Edges[i], Forward: true})
	}
	for _, i := range g.In[id] {
		out = append(out, hop{ID: g.Edges[i].From, Edge: g.Edges[i], Forward: false})
	}
	sort.Slice(out, func(a, b int) bool {
		if out[a].ID != out[b].ID {
			return out[a].ID < out[b].ID
		}
		return out[a].Edge.Type < out[b].Edge.Type
	})
	return out
}

// khop returns every node within depth hops of id (excluding id), mapped to
// its distance, undirected BFS.
func (g *Graph) khop(id string, depth int) map[string]int {
	dist := map[string]int{id: 0}
	frontier := []string{id}
	for d := 1; d <= depth && len(frontier) > 0; d++ {
		var next []string
		for _, cur := range frontier {
			for _, h := range g.neighbors(cur) {
				if _, ok := dist[h.ID]; !ok {
					dist[h.ID] = d
					next = append(next, h.ID)
				}
			}
		}
		frontier = next
	}
	delete(dist, id)
	return dist
}

// reverseReach returns everything that transitively points AT id (or at a
// Pkg via touches) within depth hops — the impact set: tasks, bugs, commits
// and docs whose story depends on the argument.
func (g *Graph) reverseReach(id string, depth int) map[string]int {
	dist := map[string]int{id: 0}
	frontier := []string{id}
	for d := 1; d <= depth && len(frontier) > 0; d++ {
		var next []string
		for _, cur := range frontier {
			for _, i := range g.In[cur] {
				from := g.Edges[i].From
				if _, ok := dist[from]; !ok {
					dist[from] = d
					next = append(next, from)
				}
			}
		}
		frontier = next
	}
	delete(dist, id)
	return dist
}

// shortestPath returns one shortest undirected path from a to b as hops
// (empty when unreachable). Ties resolve to the lexicographically smallest
// neighbor thanks to neighbors()'s sorted order.
func (g *Graph) shortestPath(a, b string) []hop {
	if a == b {
		return nil
	}
	prev := map[string]hop{}
	seen := map[string]bool{a: true}
	frontier := []string{a}
	for len(frontier) > 0 && !seen[b] {
		var next []string
		for _, cur := range frontier {
			for _, h := range g.neighbors(cur) {
				if !seen[h.ID] {
					seen[h.ID] = true
					prev[h.ID] = hop{ID: cur, Edge: h.Edge, Forward: h.Forward}
					next = append(next, h.ID)
				}
			}
		}
		frontier = next
	}
	if !seen[b] {
		return nil
	}
	var path []hop
	for cur := b; cur != a; {
		p := prev[cur]
		path = append(path, hop{ID: cur, Edge: p.Edge, Forward: p.Forward})
		cur = p.ID
	}
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	return path
}

// bugsFor returns the Bug nodes with a touches edge into the package prefix,
// newest-id first (§B ids are monotonic, so higher id = newer).
func (g *Graph) bugsFor(pkgPrefix string) []*Node {
	var out []*Node
	for _, id := range g.NodeIDs() {
		n := g.Nodes[id]
		if n.Kind != KindBug {
			continue
		}
		for _, i := range g.Out[id] {
			e := g.Edges[i]
			if e.Type == EdgeTouches && hasPkgPrefix(e.To, pkgPrefix) {
				out = append(out, n)
				break
			}
		}
	}
	sort.Slice(out, func(a, b int) bool { return idNum(out[a].ID) > idNum(out[b].ID) })
	return out
}

// hasPkgPrefix reports whether pkg equals prefix or lives under it.
func hasPkgPrefix(pkg, prefix string) bool {
	return pkg == prefix || len(pkg) > len(prefix) && pkg[:len(prefix)] == prefix && pkg[len(prefix)] == '/'
}

// idNum extracts the numeric suffix of a spec id (T907 -> 907; 0 otherwise).
func idNum(id string) int {
	n, started := 0, false
	for i := 0; i < len(id); i++ {
		if id[i] >= '0' && id[i] <= '9' {
			n, started = n*10+int(id[i]-'0'), true
		} else if started {
			return 0 // digits must be the suffix (rules out SHAs like 2d35b3b)
		}
	}
	if !started {
		return 0
	}
	return n
}
