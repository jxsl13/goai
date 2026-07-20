package main

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// reEntry matches the start of a NUMBERED spec entry — a table row (`| T922 | …`)
// or a def line (`V41 TAG: …`, `C3 (thr): …`) — capturing its class, number and any
// lowercase sub-id suffix (`T11b`). It does NOT match a table header (`| id | … |`, no
// digit), a delimiter (`| --- |`), a blockquote annotation (`> …`), or a named worker
// id (`Tw-FLASHATTN`, no numeric part), so those are never mistaken for entries.
var reEntry = regexp.MustCompile(`^(?:\|\s*)?([A-Za-z]+(?:\.[A-Za-z]+)?)(\d+)([a-z]*)(?:[ |:.]|$)`)

// entryKey is the sort key of an entry: numeric id first, then any lowercase sub-id
// suffix (so T11 < T11a < T11b < T12).
type entryKey struct {
	num    int
	suffix string
}

// keyOf extracts the entryKey from an entry's first line (zero key if not an entry).
func keyOf(line string) entryKey {
	m := reEntry.FindStringSubmatch(line)
	if m == nil {
		return entryKey{}
	}
	n, _ := strconv.Atoi(m[2])
	return entryKey{num: n, suffix: m[3]}
}

func lessKey(a, b entryKey) bool {
	if a.num != b.num {
		return a.num < b.num
	}
	return a.suffix < b.suffix
}

// sortFragmentEntries reorders the entries of a single-section fragment by id —
// ASCENDING for every class except B, which stays newest-first (descending), the
// documented §B convention that insertRow relies on. Once sorted, insertRow keeps it
// so: a new max id appends (ascending) or prepends (§B), adjacent to the current max.
//
// It is BLOCK-based: an entry owns its id-line plus every following line up to the next
// id-line, so a multi-line §C def, a §R blockquote annotation, or a wrapped cell travels
// with its entry — nothing is dropped or split. It is also CONSERVATIVE: it acts only
// when every entry in the fragment shares one class (a mixed fragment like §I's I.L*+I*
// is left untouched), and it asserts the result is a pure permutation of the original
// lines before committing (applyMutation's verifyCorpus is the further backstop). The
// spec is the source of truth; a sort must never lose a byte.
func sortFragmentEntries(f *Fragment) bool {
	lines := strings.Split(strings.TrimRight(f.Content, "\n"), "\n")
	var starts []int
	for i, ln := range lines {
		if reEntry.MatchString(ln) {
			starts = append(starts, i)
		}
	}
	if len(starts) < 2 {
		return false
	}
	class := reEntry.FindStringSubmatch(lines[starts[0]])[1]
	for _, i := range starts {
		if reEntry.FindStringSubmatch(lines[i])[1] != class {
			return false // mixed classes (e.g. §I's I.L* + I*) — leave untouched
		}
	}
	// Partition [firstStart, end) into per-entry blocks; content before the first entry
	// (section header, table header+delimiter) is preserved verbatim as scaffolding.
	// Each block keeps its continuation lines but sheds trailing blank separators, which
	// are regenerated between blocks so a moved entry can't carry a stray blank.
	head := lines[:starts[0]]
	type block struct {
		key   entryKey
		lines []string
	}
	blocks := make([]block, len(starts))
	for k, s := range starts {
		e := len(lines)
		if k+1 < len(starts) {
			e = starts[k+1]
		}
		bl := lines[s:e]
		for len(bl) > 0 && strings.TrimSpace(bl[len(bl)-1]) == "" {
			bl = bl[:len(bl)-1]
		}
		blocks[k] = block{key: keyOf(lines[s]), lines: bl}
	}
	ascending := class != "B"
	sort.SliceStable(blocks, func(a, b int) bool {
		if ascending {
			return lessKey(blocks[a].key, blocks[b].key)
		}
		return lessKey(blocks[b].key, blocks[a].key)
	})
	// Defs are blank-line separated; table rows are contiguous.
	isTable := strings.HasPrefix(strings.TrimSpace(blocks[0].lines[0]), "|")
	out := append([]string{}, head...)
	for k, bl := range blocks {
		if k > 0 && !isTable {
			out = append(out, "")
		}
		out = append(out, bl.lines...)
	}
	next := strings.Join(out, "\n") + "\n"
	if next == f.Content {
		return false
	}
	if !sameContentLines(f.Content, next) {
		return false // defensive: a block bug would drop/dupe content — never write it
	}
	f.Content = next
	return true
}

// sameContentLines reports whether a and b hold the same multiset of NON-BLANK lines.
// Blank lines are separators the sort regenerates, so their count may change; every
// line carrying actual content must survive untouched.
func sameContentLines(a, b string) bool {
	m := map[string]int{}
	for s := range strings.SplitSeq(a, "\n") {
		if strings.TrimSpace(s) != "" {
			m[s]++
		}
	}
	for s := range strings.SplitSeq(b, "\n") {
		if strings.TrimSpace(s) != "" {
			m[s]--
		}
	}
	for _, v := range m {
		if v != 0 {
			return false
		}
	}
	return true
}

// cmdSort sorts every section of every source fragment by id and re-renders the views
// (§V41). Run once to normalize the corpus; thereafter every CLI mutation preserves the
// order. -n/-dry-run prints the fragments it would rewrite without touching disk.
func cmdSort(w *strings.Builder, root string, dryRun bool) int {
	return applyMutation(w, root, dryRun, func(c *Corpus) error {
		n := 0
		for oi := range c.Outputs {
			for fi := range c.Outputs[oi].Fragments {
				if sortFragmentEntries(&c.Outputs[oi].Fragments[fi]) {
					fmt.Fprintf(w, "sorted %s\n", c.Outputs[oi].Fragments[fi].Path)
					n++
				}
			}
		}
		if n == 0 {
			fmt.Fprintln(w, "all sections already numerically sorted")
		}
		return nil
	})
}
