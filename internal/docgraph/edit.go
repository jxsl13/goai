package main

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// Entity mutation layer (§V41): every id-bearing entry — §T/§B/§R rows,
// §V/§G/§C/§I defs, worker §Tw bare-pipe rows — is created, edited and
// removed through these helpers, never by hand. All functions operate on
// the in-memory Corpus; applyMutation (cmd_mutate.go) is the only writer.

// classFile maps an id class to its root fragment path.
var classFile = map[string]string{
	"T":  "spec/70-tasks.md",
	"B":  "spec/60-backprop.md",
	"BM": "spec/80-benchmarks.md",
	"R":  "spec/40-research.md",
	"V":  "spec/50-verification.md",
	"G":  "spec/10-goals.md",
	"C":  "spec/20-constraints.md",
	"I":  "spec/30-arch-invariants.md",
}

// nextID allocates the next free id of a class: max over every corpus
// fragment PLUS any unmigrated rendered worker files (T/B/R share the
// number space across files, §V36), +1.
func nextID(c *Corpus, class string) int {
	// def ids end with ":" (C1:, G1:), V defs with " TAG:", table rows with
	// " |" — one terminator class covers all shapes.
	re := regexp.MustCompile(`(?m)^\|? ?` + class + `(\d+)[ :|]`)
	max := 0
	scan := func(s string) {
		for _, m := range re.FindAllStringSubmatch(s, -1) {
			if n, err := strconv.Atoi(m[1]); err == nil && n > max {
				max = n
			}
		}
	}
	for _, o := range c.Outputs {
		for _, f := range o.Fragments {
			scan(f.Content)
		}
	}
	return max + 1
}

// nextWorkerID allocates the next Tw<n> id for a worker output.
func nextWorkerID(o *Output) int {
	re := regexp.MustCompile(`(?m)^Tw(\d+)\|`)
	max := 0
	for _, f := range o.Fragments {
		for _, m := range re.FindAllStringSubmatch(f.Content, -1) {
			if n, err := strconv.Atoi(m[1]); err == nil && n > max {
				max = n
			}
		}
	}
	return max + 1
}

// escapeCell prepares one table cell: single-line only, bare pipes escaped
// (FORMAT.md: prefer ∨ for logical-or; \| for a true literal).
func escapeCell(s string) (string, error) {
	if strings.ContainsAny(s, "\n\r") {
		return "", fmt.Errorf("cell must be single-line (FORMAT.md table shape)")
	}
	s = strings.TrimSpace(s)
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		if s[i] == '|' && (i == 0 || s[i-1] != '\\') {
			b.WriteString(`\|`)
			continue
		}
		b.WriteByte(s[i])
	}
	return b.String(), nil
}

// reAnyID matches any full entity id (BM before B: the two-letter class is
// disjoint but listed first so the intent is explicit).
var reAnyID = regexp.MustCompile(`^(T\d+|BM\d+|B\d+|R\d+|V\d+|G\d+|C\d+|I\.L\d+[a-z]?|I\d+|Tw\d+)$`)

// validCites checks a comma-joined cites list.
func validCites(cites string) error {
	if cites == "" {
		return nil
	}
	for _, c := range strings.Split(cites, ",") {
		c = strings.TrimSpace(c)
		if !reCitesRef.MatchString(c) {
			return fmt.Errorf("invalid cite %q (want T/B/R/V/C/G/I ids or ADR-NNNN)", c)
		}
	}
	return nil
}

// entryLoc is where an entity lives in the corpus.
type entryLoc struct {
	Frag *Fragment
	Line int // 0-based index into the fragment's lines
}

// findEntry locates an id's defining line in any fragment: a table row
// (`| T918 | …`), a def line (`V39 TAG: …`, `G1: …`, `I.L0 …`), or a worker
// bare-pipe row (`Tw3|…`).
func findEntry(c *Corpus, id string) (entryLoc, bool) {
	prefixes := []string{"| " + id + " ", id + ":", id + " ", id + "|"}
	for i := range c.Outputs {
		for j := range c.Outputs[i].Fragments {
			f := &c.Outputs[i].Fragments[j]
			for li, ln := range strings.Split(f.Content, "\n") {
				for _, p := range prefixes {
					if strings.HasPrefix(ln, p) {
						return entryLoc{f, li}, true
					}
				}
			}
		}
	}
	return entryLoc{}, false
}

// entryLine returns the raw defining line at loc.
func (l entryLoc) entryLine() string {
	return strings.Split(l.Frag.Content, "\n")[l.Line]
}

// replaceLine swaps loc's line for repl ("" removes the line, together with
// a directly following blank line for def-style entries so spacing stays
// canonical).
func replaceLine(l entryLoc, repl string) {
	lines := strings.Split(l.Frag.Content, "\n")
	if repl == "" {
		rest := lines[l.Line+1:]
		if len(rest) > 0 && rest[0] == "" && l.Line > 0 && lines[l.Line-1] == "" {
			rest = rest[1:] // def entries are blank-line separated
		}
		lines = append(lines[:l.Line], rest...)
	} else {
		lines[l.Line] = repl
	}
	l.Frag.Content = strings.Join(lines, "\n")
}

// insertRow places a new table row using the adaptive rule that reproduces
// every observed table style with one mechanism: the row lands adjacent to
// the row holding the current max id of its class — BEFORE it when that row
// is the table's first data row (newest-first, §B), AFTER it when it is the
// last (append, §T). An empty table gets the row right below the delimiter.
func insertRow(f *Fragment, class, row string) error {
	lines := strings.Split(f.Content, "\n")
	reRow := regexp.MustCompile(`^\|? ?` + class + `(\d+)[ |]`)
	first, last, maxIdx, maxN := -1, -1, -1, -1
	for i, ln := range lines {
		m := reRow.FindStringSubmatch(ln)
		if m == nil {
			continue
		}
		if first < 0 {
			first = i
		}
		last = i
		if n, _ := strconv.Atoi(m[1]); n > maxN {
			maxN, maxIdx = n, i
		}
	}
	insertAt := -1
	switch {
	case maxIdx < 0: // empty table: after the delimiter row
		for i, ln := range lines {
			if reDelimRow.MatchString(ln) {
				insertAt = i + 1
				break
			}
		}
		if insertAt < 0 {
			return fmt.Errorf("%s: no table delimiter row to insert under", f.Path)
		}
	case maxIdx == first: // newest-first table
		insertAt = maxIdx
	case maxIdx == last: // append table
		insertAt = maxIdx + 1
	default:
		// max id mid-table (historic reordering): keep locality, insert after it.
		insertAt = maxIdx + 1
	}
	lines = append(lines[:insertAt], append([]string{row}, lines[insertAt:]...)...)
	f.Content = strings.Join(lines, "\n")
	return nil
}

// Status transitions for §T rows: forward-only unless forced. The paired
// `state` column follows the status cell.
var statusState = map[string]string{".": ".", "~": "wip", "x": "done"}

func statusRank(s string) int {
	switch s {
	case ".":
		return 0
	case "~":
		return 1
	case "x":
		return 2
	}
	return -1
}

// setStatus rewrites a task row's status (cell 2) and state (cell 5).
func setStatus(l entryLoc, to string, force bool) (string, error) {
	if statusRank(to) < 0 {
		return "", fmt.Errorf("status must be one of . ~ x")
	}
	cells := splitRow(l.entryLine())
	if len(cells) < 3 {
		return "", fmt.Errorf("%s is not a task table row", cells[0])
	}
	from := cells[1]
	if statusRank(to) < statusRank(from) && !force {
		return "", fmt.Errorf("backward transition %s -> %s needs -force (forward is . -> ~ -> x)", from, to)
	}
	cells[1] = to
	if len(cells) >= 5 {
		cells[4] = statusState[to]
	}
	row, err := joinRow(cells)
	if err != nil {
		return "", err
	}
	replaceLine(l, row)
	return from, nil
}

// joinRow rebuilds a GFM row from cells, re-escaping pipes.
func joinRow(cells []string) (string, error) {
	esc := make([]string, len(cells))
	for i, c := range cells {
		e, err := escapeCell(c)
		if err != nil {
			return "", err
		}
		esc[i] = e
	}
	return "| " + strings.Join(esc, " | ") + " |", nil
}

// buildTaskRow constructs a root §T row.
func buildTaskRow(id, status, text, cites, priority string) (string, error) {
	if err := validCites(cites); err != nil {
		return "", err
	}
	return joinRow([]string{id, status, text, cites, statusState[status], priority})
}

// buildBugRow constructs a §B row; guards are appended as the fix cell's
// final sentence so parseBugRow's guard chain mints the V->B edges.
func buildBugRow(id, date, cause, fix, guards string) (string, error) {
	if guards != "" {
		if err := validCites(guards); err != nil {
			return "", err
		}
		fix = strings.TrimRight(strings.TrimSpace(fix), ".") + ". " + guards + "."
	}
	return joinRow([]string{id, date, cause, fix})
}

// buildResearchRow constructs a §R row.
func buildResearchRow(id, claim, source, conf string) (string, error) {
	switch conf {
	case "high", "med", "low", "ref":
	default:
		return "", fmt.Errorf("conf must be high|med|low|ref")
	}
	return joinRow([]string{id, claim, source, conf})
}

// buildBenchRow constructs a §Bench row:
// `| id | date | benchmark | machine | incumbent | metric | before | after | impact | cites |`.
// impact is computed by the caller (computeImpact) so it can never drift from
// before/after; cites is the trusted §T-task + §V-invariant list.
func buildBenchRow(id, date, benchmark, machine, incumbent, metric, before, after, impact, cites string) (string, error) {
	if err := validCites(cites); err != nil {
		return "", err
	}
	return joinRow([]string{id, date, benchmark, machine, incumbent, metric, before, after, impact, cites})
}

// reLeadingNumber captures the leading (optionally signed/decimal) number of a
// before/after cell, tolerating a trailing unit ("2.76ms" → 2.76) so impact is
// computable whether or not the unit rides in the value cell.
var reLeadingNumber = regexp.MustCompile(`^[-+]?[0-9]*\.?[0-9]+`)

// computeImpact returns the pre/post speedup as before/after, formatted "1.38×"
// (trailing zeros trimmed: "2.00"→"2×", "13.15"→"13.15×"). before/after must be
// numeric (§Bench records a MEASURED ratio, never a hand-typed one, §V22); a
// non-numeric or zero-after cell is a hard error so a bogus impact is never
// stored.
func computeImpact(before, after string) (string, error) {
	b, err1 := leadingNumber(before)
	a, err2 := leadingNumber(after)
	if err1 != nil || err2 != nil {
		return "", fmt.Errorf("bench add: -before (%q) and -after (%q) must be numeric to compute impact = before/after", before, after)
	}
	if a == 0 {
		return "", fmt.Errorf("bench add: -after is zero — impact = before/after is undefined")
	}
	s := strconv.FormatFloat(b/a, 'f', 2, 64)
	if strings.Contains(s, ".") {
		s = strings.TrimRight(strings.TrimRight(s, "0"), ".")
	}
	return s + "×", nil
}

func leadingNumber(s string) (float64, error) {
	m := reLeadingNumber.FindString(strings.TrimSpace(s))
	if m == "" {
		return 0, fmt.Errorf("no leading number in %q", s)
	}
	return strconv.ParseFloat(m, 64)
}

// buildTwRow constructs a worker §Tw bare-pipe row (verbatim legacy shape).
func buildTwRow(id, status, text, cites string) (string, error) {
	for _, cell := range []string{text, cites} {
		if strings.ContainsAny(cell, "\n\r") {
			return "", fmt.Errorf("cell must be single-line")
		}
	}
	return id + "|" + status + "|" + text + "|" + cites, nil
}

// strongRefsTo lists ids of entries whose STRONG edges point at id — the
// §V39 guard entry-removal must respect.
func strongRefsTo(root string, c *Corpus, id string) []string {
	g := corpusGraph(root, c)
	strong := map[string]bool{EdgeCites: true, EdgeRecords: true, EdgeGuards: true}
	var out []string
	seen := map[string]bool{}
	for _, i := range g.In[id] {
		e := g.Edges[i]
		if strong[e.Type] && !seen[e.From] {
			seen[e.From] = true
			out = append(out, e.From)
		}
	}
	for _, i := range g.Out[id] {
		e := g.Edges[i]
		if e.Type == EdgeGuards && !seen[e.To] {
			seen[e.To] = true
			out = append(out, e.To) // removing a V that guards bugs orphans them
		}
	}
	return out
}
