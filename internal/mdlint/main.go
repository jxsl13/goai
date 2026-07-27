// Command mdlint is the repository's dependency-free markdown linter (SPEC
// T612, C19). It deliberately implements a small RULE SET instead of a full
// CommonMark parser: the failure modes that actually break our rendered docs
// are structural (unclosed code fences, ragged tables, skipped heading
// levels, malformed links, stray-tilde strikethrough, mid-document h1
// headings) and are detectable line-wise. goldmark — the Go
// reference CommonMark implementation — served as the semantics reference
// when designing the rules; depending on it would end the zero-dependency
// module, and the evaluation showed the rule set suffices (SPEC T612).
//
// In plain terms: it catches the markdown mistakes that make GitHub render a
// page wrong — a code block that never ends, a table row with a missing
// column, a heading that jumps two levels — without needing any library.
//
// Usage:
//
//	go run ./internal/mdlint ./...            lint every *.md under the repo
//	go run ./internal/mdlint <dir|file.md …>  lint specific paths (dirs recurse)
//
// Exit 1 on findings. Discovery skips .git, .venv and the server-owned
// .spectackle/ spec bundles; no file list to maintain.
package main

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

type finding struct {
	file string
	line int
	rule string
	msg  string
}

// skipDir reports whether a directory is excluded from markdown discovery. It is
// the SINGLE source of that list: both the CLI walker and the repo-wide test walk
// through it, so the two can never drift apart on what counts as lintable.
//
// .spectackle/ holds server-owned spec bundles with their own linter
// (`spectackle lint`); that markdown is generated rather than hand-authored for
// GitHub, so the human-audience style rules here do not apply to it.
func skipDir(name string) bool {
	return name == ".git" || strings.HasPrefix(name, ".venv") ||
		name == "node_modules" || name == ".spectackle"
}

func main() {
	args := os.Args[1:]
	rewrite := false
	if len(args) > 0 && args[0] == "-w" {
		rewrite = true
		args = args[1:]
	}
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: mdlint [-w] ./... | <dir|file.md ...>")
		os.Exit(2)
	}
	var files []string
	for _, arg := range args {
		root := arg
		if arg == "./..." {
			root = "."
		}
		info, err := os.Stat(root)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		if !info.IsDir() {
			files = append(files, root)
			continue
		}
		err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				if skipDir(d.Name()) {
					return filepath.SkipDir
				}
				return nil
			}
			if strings.HasSuffix(path, ".md") {
				files = append(files, path)
			}
			return nil
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
	}
	var all []finding
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		all = append(all, Lint(f, string(raw))...)
		lines := strings.Split(string(raw), "\n")
		gf, fixed := lintGoBlocks(f, lines, rewrite)
		all = append(all, gf...)
		if rewrite {
			joined := strings.Join(fixed, "\n")
			if joined != string(raw) {
				if err := os.WriteFile(f, []byte(joined), 0o644); err != nil {
					fmt.Fprintln(os.Stderr, err)
					os.Exit(2)
				}
				fmt.Println("reformatted:", f)
			}
		}
	}
	for _, fd := range all {
		fmt.Printf("%s:%d: [%s] %s\n", fd.file, fd.line, fd.rule, fd.msg)
	}
	if len(all) > 0 {
		os.Exit(1)
	}
}

// Lint runs every rule over one document.
func Lint(file, src string) []finding {
	var out []finding
	lines := strings.Split(src, "\n")

	// fence-balance: every ``` opener needs a closer; report the opener's line.
	fenceOpen := -1
	inFence := func() bool { return fenceOpen >= 0 }
	prevHeading := 0
	tableWidth := 0 // >0 while inside a table block
	tableStart := 0
	for i, ln := range lines {
		n := i + 1
		trimmed := strings.TrimSpace(ln)

		if strings.HasPrefix(trimmed, "```") {
			if inFence() {
				fenceOpen = -1
			} else {
				fenceOpen = n
			}
			tableWidth = 0
			continue
		}
		if inFence() {
			continue // raw code: no other rule applies
		}

		// table-consistency: a table block = header row starting with | followed
		// by a |---| separator; every row must keep the header's cell count.
		if strings.HasPrefix(trimmed, "|") {
			cells := tableCells(trimmed)
			if tableWidth == 0 {
				if i+1 < len(lines) && isTableSeparator(strings.TrimSpace(lines[i+1])) {
					tableWidth = cells
					tableStart = n
					// header/separator column-count must match: GFM renders a table
					// whose delimiter row has a different column count than the header
					// with dropped/blank columns. Report against the separator line.
					if sep := tableCells(strings.TrimSpace(lines[i+1])); sep != cells {
						out = append(out, finding{file, n + 1, "table-separator-mismatch",
							fmt.Sprintf("delimiter row has %d columns, the header at line %d has %d", sep, n, cells)})
					}
				}
			} else if cells != tableWidth && !isTableSeparator(trimmed) {
				out = append(out, finding{file, n, "table-ragged",
					fmt.Sprintf("row has %d cells, the table starting at line %d has %d", cells, tableStart, tableWidth)})
			}
			continue
		}
		tableWidth = 0

		// heading-jump: levels may only increase by one.
		if lvl := headingLevel(trimmed); lvl > 0 {
			if prevHeading > 0 && lvl > prevHeading+1 {
				out = append(out, finding{file, n, "heading-jump",
					fmt.Sprintf("heading level %d follows level %d (skips %d)", lvl, prevHeading, lvl-prevHeading-1)})
			}
			// multiple-h1: a document should have ONE h1 (its title). A second `# ` after
			// the first heading renders LARGER than the `##` sections around it (inverted
			// hierarchy) — the caveman `# comment` lines that read as giant headings. Use
			// `##`+, bold, or a blockquote for mid-document annotations.
			if lvl == 1 && prevHeading > 0 {
				out = append(out, finding{file, n, "multiple-h1",
					"a second `# ` heading after the first — renders larger than the section headers; use ##+, bold, or a blockquote"})
			}
			prevHeading = lvl
		}

		// tilde-strikethrough: GitHub GFM renders a single-tilde span ~like this~ as
		// struck-through. Two stray `~` (e.g. the "approx" shorthand ~4× … ~16×) on one line
		// silently strike everything between them. Intentional ~~strikethrough~~ (double
		// tildes) is fine. Report so `~` gets replaced with ≈ or escaped.
		if unintendedStrikethrough(ln) {
			out = append(out, finding{file, n, "tilde-strikethrough",
				"2+ single `~` on a line render as GitHub strikethrough — use ≈ for \"approx\" or \\~ to escape"})
		}

		// code-span balance is checked per PARAGRAPH below (spans may wrap lines).

		// swallowed-html: GFM treats <word> as an HTML tag and silently drops
		// unknown ones — <unk>, <t0004>, <s> vanish from the rendered page
		// unless wrapped in backticks.
		if m := htmlish.FindString(stripSpans(ln)); m != "" {
			out = append(out, finding{file, n, "swallowed-html",
				fmt.Sprintf("%q renders as an HTML tag and disappears — wrap it in backticks", m)})
		}

		// malformed-link: a "](" join needs balanced [] before and () after.
		if idx := strings.Index(ln, "]("); idx >= 0 {
			if strings.Count(ln, "[") != strings.Count(ln, "]") ||
				!closesParen(ln[idx+2:]) {
				out = append(out, finding{file, n, "malformed-link", "unbalanced link brackets/parens"})
			}
		}
	}
	if inFence() {
		out = append(out, finding{file, fenceOpen, "unclosed-fence", "``` code fence never closes"})
	}

	// paragraph-level backtick parity: a code span may wrap lines, but an odd
	// count over a whole blank-line-delimited paragraph leaves the rest of the
	// page rendered as code.
	count, start, fenced := 0, 0, false
	flush := func(end int) {
		if count%2 == 1 {
			out = append(out, finding{file, start, "odd-backticks",
				fmt.Sprintf("paragraph (lines %d-%d) has an odd number of ` — a code span never closes", start, end)})
		}
		count = 0
	}
	for i, ln := range lines {
		t := strings.TrimSpace(ln)
		if strings.HasPrefix(t, "```") {
			fenced = !fenced
			flush(i)
			continue
		}
		if fenced {
			continue
		}
		if t == "" {
			flush(i)
			continue
		}
		if count == 0 {
			start = i + 1
		}
		count += strings.Count(ln, "`")
	}
	flush(len(lines))
	return out
}

// unintendedStrikethrough reports whether a line has 2+ single-tilde delimiters that GitHub GFM
// would render as strikethrough. Code spans are stripped (tildes in `code` are literal) and
// intentional ~~double-tilde~~ strikethrough is removed first; if 2+ single `~` remain, they pair
// into an unintended struck span.
func unintendedStrikethrough(s string) bool {
	s = stripSpans(s)                    // tildes inside `code` are literal
	s = strings.ReplaceAll(s, "\\~", "") // escaped tildes never strike
	s = strings.ReplaceAll(s, "~~", "")  // intentional double-tilde strikethrough is fine
	return strings.Count(s, "~") >= 2
}

var htmlish = regexp.MustCompile(`</?[a-zA-Z][a-zA-Z0-9]*/?>`)

// stripSpans removes `code spans` so their content is exempt from prose rules.
func stripSpans(s string) string {
	for {
		i := strings.IndexByte(s, '`')
		if i < 0 {
			return s
		}
		j := strings.IndexByte(s[i+1:], '`')
		if j < 0 {
			return s
		}
		s = s[:i] + s[i+j+2:]
	}
}

// tableCells counts the cells of a |-delimited row (leading/trailing | stripped).
func tableCells(s string) int {
	s = strings.TrimPrefix(s, "|")
	s = strings.TrimSuffix(s, "|")
	// escaped \| does not split; our docs never use it, but stay correct.
	return len(strings.Split(strings.ReplaceAll(s, `\|`, ""), "|"))
}

func isTableSeparator(s string) bool {
	if !strings.HasPrefix(s, "|") {
		return false
	}
	for _, r := range s {
		switch r {
		case '|', '-', ':', ' ':
		default:
			return false
		}
	}
	return strings.Contains(s, "-")
}

func headingLevel(s string) int {
	n := 0
	for n < len(s) && s[n] == '#' {
		n++
	}
	if n == 0 || n > 6 || n == len(s) || s[n] != ' ' {
		return 0
	}
	return n
}

// closesParen reports whether the link target's parens balance on this line.
func closesParen(s string) bool {
	depth := 1
	for _, r := range s {
		switch r {
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return true
			}
		}
	}
	return false
}
