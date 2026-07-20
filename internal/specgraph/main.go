package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const usage = `specgraph — query the SPEC.md/docs citation graph.

usage: go run ./internal/specgraph [flags] <command> [args]

commands:
  show <id>            node + one-hop neighborhood, grouped by edge type
  related <id>         k-hop neighborhood (-d depth, default 2)
  why <Vn>             reverse story of an invariant: guarded bugs, citing tasks/commits
  bugs-for <pkg>       §B bugs touching a package prefix, newest first
  impact <id|pkg>      everything transitively citing/guarding/touching the argument (-d, default 3)
  path <id> <id>       shortest reference chain between two entities
  patterns             recurring §B bug classes + done-PERF §T optimization patterns
  search <query...>    hybrid BM25 + vector search (vectors when indexed; see index)
  index                (re-)embed node texts into .specgraph/ (needs -embed-model or $SPECGRAPH_EMBED_MODEL)
  stats                node/edge counts by kind/type
  dangling             references whose target does not exist (spec hygiene)
  help                 this text

mutation commands (§V41 — spec/ is the source, SPEC.md a generated view;
every mutation re-renders and fully re-verifies before writing anything):
  next-id T|B|R|V|G|C|I                  print the next free id of a class
  task add [flags] <text>                new §T row (-cites, -priority, -status, -id; -worker <n> books a Tw row)
  task set-status [flags] <id> .|~|x     forward-only lifecycle (backward needs -force); keeps state in step
  task edit [flags] <id>                 rewrite cells (-text, -cites, -state, -priority)
  bug add -cause .. -fix .. [flags]      new §B row, newest-first (-date; -guards V<n> appends the guard chain)
  research add -claim .. -source .. [-conf high|med|low|ref]
  verif add -tag TAG <text>              new §V def (canonical "V<n> TAG:" shape)
  goal|constraint|archinv add <text> | edit <id> <text> | rm <id>
  entry get <id>                         raw source line + spec/<file>:<line>
  entry rm [-force] <id>                 remove an entry (refuses when strong refs point at it)
  NOTE: mutation flags come BEFORE positional text (Go flag convention).
  render [-check]               regenerate SPEC.md + SPEC-worker-*.md (-check: report drift, write nothing)
  sort [-n]                     reorder every section by id (asc; §B newest-first), then re-render (-n: dry-run)
  verify                        render-sync + §V36 speccheck + table shapes + §V39 dangling strong refs
  split [-force]                one-time migration SPEC.md -> spec/ (also re-import during worker phase A)

flags:
  -d N            traversal depth (related: 2, impact: 3)
  -n N            max records per group (default 20; 0 = all)
  -dot            emit the result subgraph as Graphviz DOT instead of text
  -no-git         skip git history (hermetic)
  -no-cache       neither read nor write .specgraph/ caches
  -refresh        force rebuild (still writes the cache)
  -embed-model D  HF checkpoint dir (config.json+model.safetensors+tokenizer.json)

The graph is rebuilt from SPEC.md, SPEC-worker-*.md, CHANGELOG.md, docs/**
and git log; .specgraph/ (gitignored) only accelerates repeat calls.`

func main() {
	var (
		depth      = flag.Int("d", 0, "traversal depth")
		limit      = flag.Int("n", 20, "max records per group (0 = all)")
		dot        = flag.Bool("dot", false, "emit Graphviz DOT")
		noGit      = flag.Bool("no-git", false, "skip git history")
		noCache    = flag.Bool("no-cache", false, "bypass .specgraph/ caches")
		refresh    = flag.Bool("refresh", false, "force rebuild")
		embedModel = flag.String("embed-model", os.Getenv("SPECGRAPH_EMBED_MODEL"), "HF embedding checkpoint dir")
		check      = flag.Bool("check", false, "render: report drift instead of writing")
	)
	flag.Usage = func() { fmt.Fprintln(os.Stderr, usage) }
	flag.Parse()
	args := flag.Args()
	if len(args) == 0 || args[0] == "help" {
		fmt.Println(usage)
		return
	}
	root, err := repoRoot()
	if err != nil {
		fmt.Fprintln(os.Stderr, "specgraph:", err)
		os.Exit(1)
	}
	out := &strings.Builder{}
	cmd, rest := args[0], args[1:]
	if handled, rc := dispatchMutation(out, root, cmd, rest, *check); handled {
		fmt.Print(out.String())
		os.Exit(rc)
	}
	g := loadGraph(root, !*noGit, *noCache, *refresh)
	switch cmd {
	case "show":
		cmdShow(out, g, one(rest), *limit, *dot)
	case "related":
		cmdRelated(out, g, one(rest), orDefault(*depth, 2), *limit, *dot)
	case "why":
		cmdWhy(out, g, one(rest), *limit, *dot)
	case "bugs-for":
		cmdBugsFor(out, g, one(rest), *limit)
	case "impact":
		cmdImpact(out, g, one(rest), orDefault(*depth, 3), *limit, *dot)
	case "path":
		if len(rest) < 2 {
			fmt.Fprintln(os.Stderr, "specgraph: path needs two ids")
			os.Exit(2)
		}
		cmdPath(out, g, rest[0], rest[1], *dot)
	case "patterns":
		cmdPatterns(out, g, *limit)
	case "search":
		cmdSearch(out, g, root, strings.Join(rest, " "), *embedModel, *limit)
	case "index":
		cmdIndex(out, g, root, *embedModel)
	case "stats":
		cmdStats(out, g)
	case "dangling":
		cmdDangling(out, g, *limit)
	default:
		fmt.Fprintf(os.Stderr, "specgraph: unknown command %q (try help)\n", cmd)
		os.Exit(2)
	}
	fmt.Print(out.String())
}

// newMutFlagSet builds the per-subcommand flag set: mutation flags come
// right after the subcommand words, positional text follows them
// (`specgraph verif add -tag TAG the text …`, Go flag convention).
func newMutFlagSet() (*flag.FlagSet, *mutFlags) {
	fs := flag.NewFlagSet("mutation", flag.ContinueOnError)
	fl := &mutFlags{}
	fs.StringVar(&fl.Cites, "cites", "", "comma-joined cites list")
	fs.StringVar(&fl.Priority, "priority", "", "task priority high|med|low")
	fs.StringVar(&fl.Status, "status", "", "initial task status (default .)")
	fs.StringVar(&fl.ID, "id", "", "explicit id (default: allocate next)")
	fs.StringVar(&fl.Worker, "worker", "", "worker name for worker task rows")
	fs.StringVar(&fl.Cause, "cause", "", "bug cause cell")
	fs.StringVar(&fl.Fix, "fix", "", "bug fix cell")
	fs.StringVar(&fl.Date, "date", "", "bug date (default today UTC)")
	fs.StringVar(&fl.Guards, "guards", "", "guard chain ids appended to the fix cell")
	fs.StringVar(&fl.Claim, "claim", "", "research claim cell")
	fs.StringVar(&fl.Source, "source", "", "research source cell")
	fs.StringVar(&fl.Conf, "conf", "", "research confidence high|med|low|ref")
	fs.StringVar(&fl.Tag, "tag", "", "verification invariant TAG")
	fs.StringVar(&fl.Text, "text", "", "replacement text for edit")
	fs.StringVar(&fl.State, "state", "", "task state cell")
	fs.BoolVar(&fl.Force, "force", false, "override guarded transitions/removals")
	fs.BoolVar(&fl.DryRun, "dry-run", false, "verify and report, write nothing")
	return fs, fl
}

// dispatchMutation routes the §V41 mutation/render commands; returns false
// when cmd is a query for the graph-backed switch in main. rest[0] is the
// subcommand; flags follow it, positionals come last.
func dispatchMutation(out *strings.Builder, root, cmd string, rest []string, check bool) (handled bool, rc int) {
	sub := ""
	if len(rest) > 0 {
		sub = rest[0]
	}
	fs, flp := newMutFlagSet()
	parse := func() ([]string, bool) {
		if len(rest) < 2 {
			return nil, true
		}
		if err := fs.Parse(rest[1:]); err != nil {
			fmt.Fprintf(out, "%v\n", err)
			return nil, false
		}
		return fs.Args(), true
	}
	switch cmd {
	case "next-id":
		return true, cmdNextID(out, root, sub)
	case "task", "bug", "research", "verif", "goal", "constraint", "archinv", "entry":
		pos, ok := parse()
		if !ok {
			return true, 2
		}
		fl := *flp
		text := strings.Join(pos, " ")
		switch cmd + " " + sub {
		case "task add":
			return true, cmdTaskAdd(out, root, text, fl)
		case "task set-status":
			if len(pos) < 2 {
				fmt.Fprintln(out, "task set-status: [flags] <id> <.|~|x>")
				return true, 2
			}
			return true, cmdTaskSetStatus(out, root, pos[0], pos[1], fl)
		case "task edit":
			if len(pos) < 1 {
				fmt.Fprintln(out, "task edit: [flags] <id>")
				return true, 2
			}
			return true, cmdTaskEdit(out, root, pos[0], fl)
		case "bug add":
			return true, cmdBugAdd(out, root, fl)
		case "research add":
			return true, cmdResearchAdd(out, root, fl)
		case "verif add":
			return true, cmdVerifAdd(out, root, text, fl)
		case "verif edit":
			if len(pos) < 2 {
				fmt.Fprintln(out, "verif edit: [flags] <id> <text>")
				return true, 2
			}
			return true, cmdDefEdit(out, root, pos[0], strings.Join(pos[1:], " "), fl)
		case "goal add", "constraint add", "archinv add":
			class := map[string]string{"goal": "G", "constraint": "C", "archinv": "I"}[cmd]
			return true, cmdDefAdd(out, root, class, text, fl)
		case "goal edit", "constraint edit", "archinv edit":
			if len(pos) < 2 {
				fmt.Fprintf(out, "%s edit: [flags] <id> <text>\n", cmd)
				return true, 2
			}
			return true, cmdDefEdit(out, root, pos[0], strings.Join(pos[1:], " "), fl)
		case "goal rm", "constraint rm", "archinv rm", "entry rm":
			if len(pos) < 1 {
				fmt.Fprintf(out, "%s rm: [flags] <id>\n", cmd)
				return true, 2
			}
			return true, cmdEntryRm(out, root, pos[0], fl)
		case "entry get":
			if len(pos) < 1 {
				fmt.Fprintln(out, "entry get: <id>")
				return true, 2
			}
			return true, cmdEntryGet(out, root, pos[0])
		}
		fmt.Fprintf(out, "%s: unknown subcommand %q (try help)\n", cmd, sub)
		return true, 2
	case "render":
		for _, a := range rest {
			if a == "-check" || a == "--check" {
				check = true
			}
		}
		return true, cmdRender(out, root, check)
	case "sort":
		dry := false
		for _, a := range rest {
			if a == "-n" || a == "-dry-run" || a == "--dry-run" {
				dry = true
			}
		}
		return true, cmdSort(out, root, dry)
	case "verify":
		return true, cmdVerify(out, root)
	case "split":
		force := false
		for _, a := range rest {
			if a == "-force" || a == "--force" {
				force = true
			}
		}
		return true, cmdSplit(out, root, force)
	}
	return false, 0
}

func one(args []string) string {
	if len(args) == 0 {
		return ""
	}
	return args[0]
}

func orDefault(v, d int) int {
	if v == 0 {
		return d
	}
	return v
}

// repoRoot walks up from cwd to the directory holding go.mod.
func repoRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("no go.mod above %s — run from inside the repo", dir)
		}
		dir = parent
	}
}

// miss prints the not-found line with the nearest-id suggestion; exit 0 by
// contract (an LLM consumer treats "no node" as an answer, not a failure).
func miss(w *strings.Builder, g *Graph, id string) {
	fmt.Fprintf(w, "no node %s", id)
	if near := nearestID(g, id); near != "" {
		fmt.Fprintf(w, " — nearest: %s", near)
	}
	fmt.Fprintln(w)
}

func cmdShow(w *strings.Builder, g *Graph, id string, limit int, dot bool) {
	n := g.Nodes[id]
	if n == nil {
		miss(w, g, id)
		return
	}
	hops := g.neighbors(id)
	if dot {
		ids := map[string]bool{id: true}
		for _, h := range hops {
			ids[h.ID] = true
		}
		w.WriteString(renderDOT(g, ids))
		return
	}
	fmt.Fprintln(w, nodeLine(n))
	renderGroups(w, g, hops, limit)
}

func cmdRelated(w *strings.Builder, g *Graph, id string, depth, limit int, dot bool) {
	n := g.Nodes[id]
	if n == nil {
		miss(w, g, id)
		return
	}
	dist := g.khop(id, depth)
	if dot {
		ids := map[string]bool{id: true}
		for other := range dist {
			ids[other] = true
		}
		w.WriteString(renderDOT(g, ids))
		return
	}
	fmt.Fprintln(w, nodeLine(n))
	byKind := map[Kind][]string{}
	for other := range dist {
		if g.Nodes[other] != nil {
			byKind[g.Nodes[other].Kind] = append(byKind[g.Nodes[other].Kind], other)
		}
	}
	kinds := make([]string, 0, len(byKind))
	for k := range byKind {
		kinds = append(kinds, string(k))
	}
	sort.Strings(kinds)
	for _, k := range kinds {
		ids := byKind[Kind(k)]
		sort.Slice(ids, func(a, b int) bool {
			if dist[ids[a]] != dist[ids[b]] {
				return dist[ids[a]] < dist[ids[b]]
			}
			return idNum(ids[a]) > idNum(ids[b]) || idNum(ids[a]) == idNum(ids[b]) && ids[a] < ids[b]
		})
		fmt.Fprintf(w, "%s (%d):\n", k, len(ids))
		for i, other := range ids {
			if limit > 0 && i == limit {
				fmt.Fprintf(w, "  ... %d more (raise -n)\n", len(ids)-limit)
				break
			}
			fmt.Fprintf(w, "  d%d %s\n", dist[other], nodeLine(g.Nodes[other]))
		}
	}
}

// cmdWhy tells an entity's reverse story — for an invariant: which bugs it
// guards (the reason it exists), then who cites/implements it.
func cmdWhy(w *strings.Builder, g *Graph, id string, limit int, dot bool) {
	n := g.Nodes[id]
	if n == nil {
		miss(w, g, id)
		return
	}
	if dot {
		cmdShow(w, g, id, limit, true)
		return
	}
	fmt.Fprintln(w, nodeLine(n))
	type group struct {
		title string
		ids   []string
	}
	var groups []group
	pick := func(title string, match func(e Edge, forward bool) bool) {
		var ids []string
		seen := map[string]bool{}
		for _, h := range g.neighbors(id) {
			if match(h.Edge, h.Forward) && !seen[h.ID] {
				seen[h.ID] = true
				ids = append(ids, h.ID)
			}
		}
		sort.Slice(ids, func(a, b int) bool {
			return idNum(ids[a]) > idNum(ids[b]) || idNum(ids[a]) == idNum(ids[b]) && ids[a] < ids[b]
		})
		if len(ids) > 0 {
			groups = append(groups, group{title, ids})
		}
	}
	pick("guards bugs", func(e Edge, fwd bool) bool { return e.Type == EdgeGuards && fwd })
	pick("guarded by", func(e Edge, fwd bool) bool { return e.Type == EdgeGuards && !fwd })
	pick("cited by", func(e Edge, fwd bool) bool {
		return (e.Type == EdgeCites || e.Type == EdgeRefs) && !fwd && g.kindOf(e.From) != KindCommit && g.kindOf(e.From) != KindDoc
	})
	pick("implemented by commits", func(e Edge, fwd bool) bool { return e.Type == EdgeImplements && !fwd })
	pick("related ADRs", func(e Edge, fwd bool) bool { return e.Type == EdgeRelates && !fwd && !e.Weak })
	pick("documented in", func(e Edge, fwd bool) bool { return e.Type == EdgeMentions && !fwd })
	pick("cites", func(e Edge, fwd bool) bool { return (e.Type == EdgeCites || e.Type == EdgeRefs) && fwd })
	for _, grp := range groups {
		fmt.Fprintf(w, "%s (%d):\n", grp.title, len(grp.ids))
		for i, other := range grp.ids {
			if limit > 0 && i == limit {
				fmt.Fprintf(w, "  ... %d more (raise -n)\n", len(grp.ids)-limit)
				break
			}
			fmt.Fprintf(w, "  %s\n", nodeLine(g.Nodes[other]))
		}
	}
}

func cmdBugsFor(w *strings.Builder, g *Graph, pkg string, limit int) {
	bugs := g.bugsFor(pkg)
	fmt.Fprintf(w, "bugs touching %s (%d):\n", pkg, len(bugs))
	for i, b := range bugs {
		if limit > 0 && i == limit {
			fmt.Fprintf(w, "  ... %d more (raise -n)\n", len(bugs)-limit)
			break
		}
		fmt.Fprintf(w, "  %s\n", nodeLine(b))
	}
}

func cmdImpact(w *strings.Builder, g *Graph, id string, depth, limit int, dot bool) {
	if g.Nodes[id] == nil {
		miss(w, g, id)
		return
	}
	dist := g.reverseReach(id, depth)
	if dot {
		ids := map[string]bool{id: true}
		for other := range dist {
			ids[other] = true
		}
		w.WriteString(renderDOT(g, ids))
		return
	}
	fmt.Fprintln(w, nodeLine(g.Nodes[id]))
	var hops []hop
	for other, d := range dist {
		hops = append(hops, hop{ID: other, Edge: Edge{Type: fmt.Sprintf("d%d", d)}, Forward: false})
	}
	sort.Slice(hops, func(a, b int) bool {
		if hops[a].Edge.Type != hops[b].Edge.Type {
			return hops[a].Edge.Type < hops[b].Edge.Type
		}
		return idNum(hops[a].ID) > idNum(hops[b].ID) || idNum(hops[a].ID) == idNum(hops[b].ID) && hops[a].ID < hops[b].ID
	})
	fmt.Fprintf(w, "impacted (%d):\n", len(hops))
	for i, h := range hops {
		if limit > 0 && i == limit {
			fmt.Fprintf(w, "  ... %d more (raise -n)\n", len(hops)-limit)
			break
		}
		fmt.Fprintf(w, "  %s %s\n", h.Edge.Type, nodeLine(g.Nodes[h.ID]))
	}
}

func cmdPath(w *strings.Builder, g *Graph, a, b string, dot bool) {
	if g.Nodes[a] == nil {
		miss(w, g, a)
		return
	}
	if g.Nodes[b] == nil {
		miss(w, g, b)
		return
	}
	path := g.shortestPath(a, b)
	if path == nil {
		fmt.Fprintf(w, "no path %s .. %s\n", a, b)
		return
	}
	if dot {
		ids := map[string]bool{a: true}
		for _, h := range path {
			ids[h.ID] = true
		}
		w.WriteString(renderDOT(g, ids))
		return
	}
	fmt.Fprintln(w, nodeLine(g.Nodes[a]))
	for _, h := range path {
		arrow := "-[" + h.Edge.Type + "]->"
		if !h.Forward {
			arrow = "<-[" + h.Edge.Type + "]-"
		}
		fmt.Fprintf(w, "  %s %s\n", arrow, nodeLine(g.Nodes[h.ID]))
	}
}

func cmdPatterns(w *strings.Builder, g *Graph, limit int) {
	fmt.Fprintln(w, "== recurring §B bug classes ==")
	renderClusters(w, g, bugClusters(g), limit)
	fmt.Fprintln(w, "== done-PERF §T optimization patterns ==")
	renderClusters(w, g, perfClusters(g), limit)
}

func cmdSearch(w *strings.Builder, g *Graph, root, query, embedModel string, limit int) {
	if strings.TrimSpace(query) == "" {
		fmt.Fprintln(w, "search: empty query")
		return
	}
	lists := [][]scored{buildBM25(g).search(query, 50)}
	mode := "bm25"
	embPath := filepath.Join(root, cacheDirName, embedCacheName)
	if recs := loadEmbeddings(embPath); len(recs) > 0 && embedModel != "" {
		if emb, err := newBertEmbedder(embedModel); err == nil {
			if qv, err := emb.Embed(query); err == nil {
				if vs, err := vectorSearch(recs, qv, 50); err == nil {
					lists = append(lists, vs)
					mode = "bm25+vector"
				}
			}
		} else {
			fmt.Fprintf(w, "note: embed model unusable (%v), falling back to bm25\n", err)
		}
	}
	hits := rrf(lists...)
	if limit > 0 && len(hits) > limit {
		hits = hits[:limit]
	}
	fmt.Fprintf(w, "hits (%d, %s):\n", len(hits), mode)
	for _, h := range hits {
		fmt.Fprintf(w, "  %.4f %s\n", h.Score, nodeLine(g.Nodes[h.ID]))
	}
	if mode == "bm25" {
		fmt.Fprintln(w, "note: bm25-only — run `specgraph index` with -embed-model (or $SPECGRAPH_EMBED_MODEL) for hybrid vector search")
	}
}

func cmdIndex(w *strings.Builder, g *Graph, root, embedModel string) {
	if embedModel == "" {
		fmt.Fprintln(w, "index: no embedding model — set -embed-model or $SPECGRAPH_EMBED_MODEL to a HF checkpoint dir (config.json + model.safetensors + tokenizer.json, e.g. all-MiniLM-L6-v2)")
		return
	}
	emb, err := newBertEmbedder(embedModel)
	if err != nil {
		fmt.Fprintf(w, "index: %v\n", err)
		return
	}
	embedded, kept, total, err := indexEmbeddings(g, emb, filepath.Join(root, cacheDirName, embedCacheName))
	if err != nil {
		fmt.Fprintf(w, "index: %v\n", err)
		return
	}
	fmt.Fprintf(w, "indexed %d nodes (%d embedded, %d unchanged)\n", total, embedded, kept)
}

func cmdStats(w *strings.Builder, g *Graph) {
	nodeCounts := map[Kind]int{}
	for _, n := range g.Nodes {
		nodeCounts[n.Kind]++
	}
	edgeCounts := map[string]int{}
	for _, e := range g.Edges {
		edgeCounts[e.Type]++
	}
	fmt.Fprintf(w, "nodes (%d):\n", len(g.Nodes))
	for _, k := range sortedKeys(nodeCounts) {
		fmt.Fprintf(w, "  %-10s %d\n", k, nodeCounts[Kind(k)])
	}
	fmt.Fprintf(w, "edges (%d):\n", len(g.Edges))
	for _, t := range sortedKeysS(edgeCounts) {
		fmt.Fprintf(w, "  %-10s %d\n", t, edgeCounts[t])
	}
}

func cmdDangling(w *strings.Builder, g *Graph, limit int) {
	es := g.Dangling()
	fmt.Fprintf(w, "dangling references (%d):\n", len(es))
	for i, e := range es {
		if limit > 0 && i == limit {
			fmt.Fprintf(w, "  ... %d more (raise -n)\n", len(es)-limit)
			break
		}
		fmt.Fprintf(w, "  %s:%d  %s -[%s]-> %s\n", e.File, e.Line, e.From, e.Type, e.To)
	}
}

// kindOf is a nil-safe node kind lookup.
func (g *Graph) kindOf(id string) Kind {
	if n := g.Nodes[id]; n != nil {
		return n.Kind
	}
	return ""
}

func sortedKeys(m map[Kind]int) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, string(k))
	}
	sort.Strings(out)
	return out
}

func sortedKeysS(m map[string]int) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
