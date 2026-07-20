package main

import (
	"regexp"
	"strings"
)

// Defining-occurrence and reference regexes. reTRow/reBRow/reRRow/reVDef are
// speccheck's defining-occurrence rules (the row regexes verbatim; reVDef
// additionally captures the TAG, falling back to speccheck's bare `^V\d+ `
// via reVDefBare) — TestParseMatchesSpeccheck pins the copies to identical id
// extraction on a shared fixture instead of coupling the packages, since
// speccheck is the §V36 CI guard and documents its self-containment. The
// digit-after-letter rule keeps worker prose ids like "Tw-FLASHATTN" or
// "Iw8" from ever matching. reSection widens speccheck's single-letter rule
// to the worker specs' multi-letter sections (§RUN, §Tw, §CPU).
var (
	reTRow     = regexp.MustCompile(`^\| (T\d+) `)
	reBRow     = regexp.MustCompile(`^\| (B\d+) `)
	reRRow     = regexp.MustCompile(`^\| (R\d+) `)
	reVDef     = regexp.MustCompile(`^(V\d+) ([A-Z][A-Z0-9+/ -]*):`)
	reVDefBare = regexp.MustCompile(`^(V\d+) `)
	reSection  = regexp.MustCompile(`^## §?([A-Za-z]+)\b`)

	reGCDef = regexp.MustCompile(`^([GC]\d+)(?: \([^)]*\))?:`)
	reIDef  = regexp.MustCompile(`^(I\.L\d+[a-z]?|I\d+)[: ]`)

	// Reference forms. Bare T/B/V/R ids are the corpus norm (cites cells,
	// commit subjects, §B guard chains); C/G/I need the § prefix in prose
	// because bare "C3"/"I1" are too ambiguous — except inside the trusted
	// §T cites column, where refCites applies.
	reRefTBVR  = regexp.MustCompile(`§?\b([TBVR]\d+)\b`)
	reRefCGI   = regexp.MustCompile(`§(C\d+|G\d+|I\.L\d+[a-z]?|I\d+)\b`)
	reRefADR   = regexp.MustCompile(`\b(ADR-\d{4})\b`)
	reRefSHA   = regexp.MustCompile(`\bcommit ([0-9a-f]{7,40})\b`)
	reCitesRef = regexp.MustCompile(`^(T\d+|B\d+|R\d+|V\d+|C\d+|G\d+|I\.L\d+[a-z]?|I\d+|ADR-\d{4})$`)

	// "T870/T905 class" phrases inside §T rows become same_class edges.
	reClass = regexp.MustCompile(`((?:T\d+[/, ]+)*T\d+) class\b`)
)

// scanRefs returns every spec-entity reference in s, in order, deduplicated:
// bare/§ T/B/V/R ids, §-prefixed C/G/I ids, and ADR ids.
func scanRefs(s string) []string {
	seen := map[string]bool{}
	var out []string
	add := func(id string) {
		if !seen[id] {
			seen[id] = true
			out = append(out, id)
		}
	}
	for _, m := range reRefTBVR.FindAllStringSubmatch(s, -1) {
		add(m[1])
	}
	for _, m := range reRefCGI.FindAllStringSubmatch(s, -1) {
		add(m[1])
	}
	for _, m := range reRefADR.FindAllStringSubmatch(s, -1) {
		add(m[1])
	}
	return out
}

// scanSHAs returns the "commit <sha>" citations in s.
func scanSHAs(s string) []string {
	var out []string
	for _, m := range reRefSHA.FindAllStringSubmatch(s, -1) {
		out = append(out, m[1])
	}
	return out
}

// splitRow splits a clean GFM table row into trimmed cells. lint-md
// guarantees every table is well-formed and cells never contain a bare pipe
// (FORMAT.md: logical-or is ∨, literal pipes are escaped \|) — so the scan
// treats \| as a literal pipe INSIDE the cell and only a bare | as the
// separator. A strings.Split on every pipe silently shifted all columns
// after an escaped one (§B114: B113's cause tail became its fix cell and
// the guard-chain extractor minted strong edges out of prose).
func splitRow(line string) []string {
	line = strings.TrimSpace(line)
	line = strings.TrimPrefix(line, "|")
	line = strings.TrimSuffix(line, "|")
	var cells []string
	var cur strings.Builder
	for i := 0; i < len(line); i++ {
		switch {
		case line[i] == '\\' && i+1 < len(line) && line[i+1] == '|':
			cur.WriteByte('|')
			i++
		case line[i] == '|':
			cells = append(cells, strings.TrimSpace(cur.String()))
			cur.Reset()
		default:
			cur.WriteByte(line[i])
		}
	}
	return append(cells, strings.TrimSpace(cur.String()))
}

// kindOfID maps a spec id to its node kind, or "" for unknown shapes.
func kindOfID(id string) Kind {
	switch {
	case strings.HasPrefix(id, "ADR-"):
		return KindADR
	case strings.HasPrefix(id, "I.L"):
		return KindArchInv
	}
	if len(id) < 2 {
		return ""
	}
	switch id[0] {
	case 'T':
		return KindTask
	case 'B':
		return KindBug
	case 'R':
		return KindResearch
	case 'V':
		return KindVerif
	case 'C':
		return KindConstraint
	case 'G':
		return KindGoal
	case 'I':
		return KindArchInv
	}
	return ""
}

// parseSpec extracts nodes and edges from one spec file (SPEC.md or a worker
// spec — the section walk is shared; worker sections like §Tw/§CPU simply
// contribute their table rows and defs under their own file). pkgs is the
// validated Go package-path set for touches edges.
func parseSpec(g *Graph, file, content string, pkgs *pkgSet) {
	lines := strings.Split(content, "\n")
	section := ""
	// Multi-line defs (§C entries continue on indented lines): remember the
	// last def node so continuation lines extend its text.
	var open *Node
	flush := func() { open = nil }

	for i, ln := range lines {
		lineNo := i + 1
		if m := reSection.FindStringSubmatch(ln); m != nil {
			section = m[1]
			flush()
			continue
		}
		switch {
		case reTRow.MatchString(ln):
			flush()
			parseTaskRow(g, file, lineNo, ln, pkgs)
		case reBRow.MatchString(ln):
			flush()
			parseBugRow(g, file, lineNo, ln, pkgs)
		case reRRow.MatchString(ln):
			flush()
			cells := splitRow(ln)
			n := g.AddNode(&Node{ID: cells[0], Kind: KindResearch, File: file, Line: lineNo, Meta: map[string]string{}})
			if len(cells) >= 4 {
				n.Text = cells[1]
				n.Meta["source"] = cells[2]
				n.Meta["conf"] = cells[3]
			}
		case reVDefBare.MatchString(ln):
			flush()
			id, tag, rest := "", "", ""
			if m := reVDef.FindStringSubmatch(ln); m != nil {
				id, tag, rest = m[1], m[2], strings.TrimSpace(ln[len(m[0]):])
			} else {
				m := reVDefBare.FindStringSubmatch(ln)
				id, rest = m[1], strings.TrimSpace(ln[len(m[0]):])
			}
			n := g.AddNode(&Node{
				ID: id, Kind: KindVerif, File: file, Line: lineNo,
				Text: rest,
				Meta: map[string]string{"tag": tag},
			})
			for _, ref := range scanRefs(n.Text) {
				if ref != n.ID {
					g.AddEdge(Edge{From: n.ID, To: ref, Type: EdgeRefs, File: file, Line: lineNo})
				}
			}
		case section == "G" || section == "C":
			if m := reGCDef.FindStringSubmatch(ln); m != nil {
				open = g.AddNode(&Node{
					ID: m[1], Kind: kindOfID(m[1]), File: file, Line: lineNo,
					Text: strings.TrimSpace(ln[len(m[0]):]),
				})
				addRefEdges(g, open, file, lineNo, open.Text)
			} else if open != nil && strings.TrimSpace(ln) != "" {
				open.Text += " " + strings.TrimSpace(ln)
				addRefEdges(g, open, file, lineNo, ln)
			} else {
				flush()
			}
		case section == "I":
			if m := reIDef.FindStringSubmatch(ln); m != nil {
				open = g.AddNode(&Node{
					ID: m[1], Kind: KindArchInv, File: file, Line: lineNo,
					Text: strings.TrimSpace(ln[len(m[1]):]),
				})
				addRefEdges(g, open, file, lineNo, open.Text)
			} else if open != nil && strings.TrimSpace(ln) != "" {
				open.Text += " " + strings.TrimSpace(ln)
				addRefEdges(g, open, file, lineNo, ln)
			} else {
				flush()
			}
		}
	}
}

// addRefEdges adds a refs edge from n to every reference found in text.
func addRefEdges(g *Graph, n *Node, file string, line int, text string) {
	for _, ref := range scanRefs(text) {
		if ref != n.ID {
			g.AddEdge(Edge{From: n.ID, To: ref, Type: EdgeRefs, File: file, Line: line})
		}
	}
}

// parseTaskRow handles a §T/§Tw row: `| id | status | task | cites | state | priority |`
// (old rows may have fewer columns — FORMAT.md allows `| id | status | task | cites |`).
func parseTaskRow(g *Graph, file string, lineNo int, ln string, pkgs *pkgSet) {
	cells := splitRow(ln)
	n := g.AddNode(&Node{ID: cells[0], Kind: KindTask, File: file, Line: lineNo, Meta: map[string]string{}})
	if len(cells) > 1 {
		n.Meta["status"] = cells[1]
	}
	if len(cells) > 2 {
		n.Text = cells[2]
	}
	if len(cells) > 4 {
		n.Meta["state"] = cells[4]
	}
	if len(cells) > 5 {
		n.Meta["priority"] = cells[5]
	}
	// The cites column is a trusted context: bare comma-separated ids.
	if len(cells) > 3 {
		for _, c := range strings.Split(cells[3], ",") {
			c = strings.TrimSpace(c)
			if reCitesRef.MatchString(c) && c != n.ID {
				g.AddEdge(Edge{From: n.ID, To: c, Type: EdgeCites, File: file, Line: lineNo})
			}
		}
	}
	// Prose refs, commit SHAs, package paths, and "T…/T… class" phrases.
	for _, ref := range scanRefs(n.Text) {
		if ref != n.ID {
			g.AddEdge(Edge{From: n.ID, To: ref, Type: EdgeRefs, File: file, Line: lineNo})
		}
	}
	for _, sha := range scanSHAs(n.Text) {
		g.AddEdge(Edge{From: n.ID, To: sha, Type: EdgeFixedBy, File: file, Line: lineNo})
	}
	for _, p := range pkgs.find(n.Text) {
		g.AddEdge(Edge{From: n.ID, To: p, Type: EdgeTouches, File: file, Line: lineNo})
	}
	for _, m := range reClass.FindAllStringSubmatch(n.Text, -1) {
		for _, peer := range regexp.MustCompile(`T\d+`).FindAllString(m[1], -1) {
			if peer != n.ID {
				g.AddEdge(Edge{From: n.ID, To: peer, Type: EdgeSameClass, File: file, Line: lineNo})
			}
		}
	}
}

// parseBugRow handles a §B row: `| id | date | cause | fix |`. The fix cell's
// final sentence is the guard chain ("… Non-vacuous: …. V15,V30."): V refs
// there become guards edges (Verif→Bug), other refs become cites — the
// bug→invariant backprop link the whole workflow hinges on.
func parseBugRow(g *Graph, file string, lineNo int, ln string, pkgs *pkgSet) {
	cells := splitRow(ln)
	n := g.AddNode(&Node{ID: cells[0], Kind: KindBug, File: file, Line: lineNo, Meta: map[string]string{}})
	if len(cells) > 1 {
		n.Meta["date"] = cells[1]
	}
	cause, fix := "", ""
	if len(cells) > 2 {
		cause = cells[2]
	}
	if len(cells) > 3 {
		fix = cells[3]
	}
	n.Text = cause
	n.Meta["fix"] = fix

	for _, ref := range scanRefs(guardChain(fix)) {
		if ref == n.ID {
			continue
		}
		if kindOfID(ref) == KindVerif {
			g.AddEdge(Edge{From: ref, To: n.ID, Type: EdgeGuards, File: file, Line: lineNo})
		} else {
			g.AddEdge(Edge{From: n.ID, To: ref, Type: EdgeCites, File: file, Line: lineNo})
		}
	}
	for _, ref := range scanRefs(cause + " " + fix) {
		if ref != n.ID {
			g.AddEdge(Edge{From: n.ID, To: ref, Type: EdgeRefs, File: file, Line: lineNo})
		}
	}
	for _, sha := range scanSHAs(cause + " " + fix) {
		g.AddEdge(Edge{From: n.ID, To: sha, Type: EdgeFixedBy, File: file, Line: lineNo})
	}
	for _, p := range pkgs.find(cause + " " + fix) {
		g.AddEdge(Edge{From: n.ID, To: p, Type: EdgeTouches, File: file, Line: lineNo})
	}
}

// guardChain returns the final sentence of a §B fix cell — the conventional
// position of the guarding invariant list ("…green. V15,V30.").
func guardChain(fix string) string {
	fix = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(fix), "."))
	if i := strings.LastIndex(fix, ". "); i >= 0 {
		return fix[i+2:]
	}
	return fix
}
