// Package speccheck is the mechanical guard for SPEC.md's structural invariant §V36
// (prevents the §B96 duplicate-id and §B97 headerless-fragment classes from
// recurring). It parses SPEC.md — and any worker spec files for cross-file id
// uniqueness — as text and reports every violation, so a CI test can fail on a
// malformed backlog the way internal/mdlint fails on malformed markdown.
//
// The three checks (§V36):
//
//   - id UNIQUENESS: every §T/§B/§R/§V id is defined at most once, within SPEC.md
//     and across the SPEC-worker-*.md files (§B96: parallel agents numbering from
//     different bases both booked T870; the byte-identical T791–T798 re-append).
//   - §T MEMBERSHIP: every task row (a line beginning "| T<n> |") lives inside the
//     §T section, never as a headerless fragment spliced into another section
//     (§B97: 148 rows appended after §B fed the 4-col §B table 6-col rows).
//   - §T LAST: the §T section is the last "## " section of SPEC.md, so the
//     append-at-EOF habit stays structurally correct.
//
// The id regexes require a DIGIT immediately after the class letter, so a worker's
// prose ids like "Tw-FLASHATTN" or "Iw8" never match and never false-positive.
package speccheck

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// Violation is one spec-integrity problem: a 1-based line number (0 = file-level)
// in the file named by File, and a human-readable message naming the rule.
type Violation struct {
	File string
	Line int
	Msg  string
}

func (v Violation) String() string {
	if v.Line == 0 {
		return fmt.Sprintf("%s: %s", v.File, v.Msg)
	}
	return fmt.Sprintf("%s:%d: %s", v.File, v.Line, v.Msg)
}

var (
	// Defining occurrences: a §T/§B/§R/§V id in the LEADING cell of a GFM table
	// row. Every id-bearing section is now a clean GFM table (§V/§C/§G/§I joined
	// §T/§B/§R), so the shape is uniform: `| <id> | …`. A digit must follow the
	// letter.
	reTRow    = regexp.MustCompile(`^\| (T\d+) `)
	reBRow    = regexp.MustCompile(`^\| (B\d+) `)
	reRRow    = regexp.MustCompile(`^\| (R\d+) `)
	reVDef    = regexp.MustCompile(`^\| (V\d+) `)
	reSection = regexp.MustCompile(`^## §?([A-Z])\b`) // "## §T — …" or "## T. …"
)

// Check runs the §V36 SPEC-integrity checks. spec is SPEC.md's content; workers maps
// each worker-spec filename to its content (for cross-file id uniqueness — pass nil
// or empty to check SPEC.md alone). Returns the violations, empty when clean.
func Check(spec string, workers map[string]string) []Violation {
	var vs []Violation
	lines := strings.Split(spec, "\n")

	// (c) locate every "## " section and the §T section; §T must be last.
	lastLetter, lastLine, tLine := "", 0, 0
	for i, ln := range lines {
		if m := reSection.FindStringSubmatch(ln); m != nil {
			lastLetter, lastLine = m[1], i+1
			if m[1] == "T" {
				tLine = i + 1
			}
		}
	}
	switch {
	case tLine == 0:
		vs = append(vs, Violation{"SPEC.md", 0, "no §T section header found (§V36)"})
	case lastLetter != "T":
		vs = append(vs, Violation{"SPEC.md", lastLine,
			fmt.Sprintf("§T is not the last section — §%s follows it here; append-at-EOF is only safe when §T is last (§V36)", lastLetter)})
	}

	// (a) id uniqueness within SPEC.md + (b) §T membership.
	seen := map[string]int{} // id -> first line in SPEC.md
	for i, ln := range lines {
		lineNo := i + 1
		for _, re := range []*regexp.Regexp{reTRow, reBRow, reRRow, reVDef} {
			if m := re.FindStringSubmatch(ln); m != nil {
				id := m[1]
				if first, ok := seen[id]; ok {
					vs = append(vs, Violation{"SPEC.md", lineNo,
						fmt.Sprintf("duplicate id %s (first defined at line %d) — ids are monotonic and never reused (§V36/§B96)", id, first)})
				} else {
					seen[id] = lineNo
				}
			}
		}
		if m := reTRow.FindStringSubmatch(ln); m != nil && tLine > 0 && lineNo < tLine {
			vs = append(vs, Violation{"SPEC.md", lineNo,
				fmt.Sprintf("task row %s is OUTSIDE the §T section (which starts at line %d) — a headerless fragment in another section (§V36/§B97)", m[1], tLine)})
		}
	}

	// (a) cross-file: a §T/§B/§R id defined in SPEC.md must not also be defined in a
	// worker spec (§B96 — the two backlogs share the T/B/R number space). Iterate the
	// worker files in sorted name order so violations are deterministic.
	names := make([]string, 0, len(workers))
	for name := range workers {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		for i, ln := range strings.Split(workers[name], "\n") {
			for _, re := range []*regexp.Regexp{reTRow, reBRow, reRRow} {
				if m := re.FindStringSubmatch(ln); m != nil {
					if first, ok := seen[m[1]]; ok {
						vs = append(vs, Violation{name, i + 1,
							fmt.Sprintf("id %s collides with SPEC.md line %d — the shared T/B/R id space is cross-file unique (§V36/§B96)", m[1], first)})
					}
				}
			}
		}
	}
	return vs
}
