package main

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os/exec"
	"strings"
)

// DocsOnly and Code are the two verdicts Classify can print.
const (
	DocsOnly = "docs-only"
	Code     = "code"
)

// Classify inspects the diff between two revisions of the git repository at
// dir and returns DocsOnly when EVERY change is provably documentation-only,
// Code otherwise (§V26: every uncertainty fails open to Code). What counts as
// documentation comes from cfg (§T584) — by default docs/ and
// root-level markdown/text.
func Classify(cfg *config, dir, base, head string) string {
	out, err := gitRun(dir, "diff", "--name-status", "-z", base, head)
	if err != nil {
		return Code // base unavailable (force push, shallow clone) → run CI
	}
	fields := strings.Split(strings.TrimRight(string(out), "\x00"), "\x00")
	if len(fields) == 1 && fields[0] == "" {
		return DocsOnly // empty diff: nothing changed at all
	}
	for i := 0; i < len(fields); {
		status := fields[i]
		if status == "" || i+1 >= len(fields) {
			return Code
		}
		path := fields[i+1]
		i += 2
		// renames/copies carry a second path.
		newPath := path
		if status[0] == 'R' || status[0] == 'C' {
			if i >= len(fields) {
				return Code
			}
			newPath = fields[i]
			i++
		}
		switch {
		case cfg.ignored(path) && cfg.ignored(newPath):
			continue // configured documentation paths, regardless of status
		case status[0] == 'M' && strings.HasSuffix(path, ".go"):
			if goCodeUnchanged(dir, base, head, path) {
				continue // only comments moved
			}
			return Code
		default:
			// added/deleted/renamed .go files, workflows, build files, binaries,
			// testdata, anything unknown: code (§V26).
			return Code
		}
	}
	return DocsOnly
}

// What is PROVABLY documentation is configuration (§T584, config.ignored); the
// defaults cover docs/ and root-level .md/.txt (LICENSE.md included, §T585). Markdown or text files
// deeper in the tree deliberately do NOT default to documentation — a .txt
// inside a package directory can be //go:embed'ed into the binary (§B50), so
// they classify as package assets / code instead (§V26 fail-open).

// goCodeUnchanged proves that the .go file at path is semantically identical
// between the two revisions once every non-directive comment is stripped.
func goCodeUnchanged(dir, base, head, path string) bool {
	a, err := gitRun(dir, "show", base+":"+path)
	if err != nil {
		return false
	}
	b, err := gitRun(dir, "show", head+":"+path)
	if err != nil {
		return false
	}
	sa, da, err := stripComments(a)
	if err != nil {
		return false
	}
	sb, db, err := stripComments(b)
	if err != nil {
		return false
	}
	// directive comments (//go:build, //go:embed, //go:generate, //line, cgo
	// preambles, +build) CHANGE COMPILATION — they must match exactly.
	return bytes.Equal(sa, sb) && da == db
}

// stripComments parses src and returns (a) the file rendered WITHOUT any
// comments and (b) the concatenation of all DIRECTIVE comments, which count
// as code (§V26).
func stripComments(src []byte) (stripped []byte, directives string, err error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		return nil, "", err
	}
	var dirs []string
	cgo := usesCgo(f)
	for _, cg := range f.Comments {
		for _, c := range cg.List {
			// in a cgo file EVERY comment is significant (the preamble is C
			// source); otherwise only toolchain directives are (§V26).
			if cgo || isDirective(c.Text) {
				dirs = append(dirs, c.Text)
			}
		}
	}
	// render without comments: re-parse with comments dropped entirely.
	f2, err := parser.ParseFile(token.NewFileSet(), "x.go", src, 0)
	if err != nil {
		return nil, "", err
	}
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, token.NewFileSet(), f2); err != nil {
		return nil, "", err
	}
	return buf.Bytes(), strings.Join(dirs, "\n"), nil
}

// isDirective reports compiler/toolchain directive comments.
func isDirective(text string) bool {
	t := strings.TrimPrefix(text, "//")
	if t == text { // block comment: cgo preambles are handled separately
		return false
	}
	return strings.HasPrefix(t, "go:") || strings.HasPrefix(t, "line ") ||
		strings.HasPrefix(strings.TrimSpace(t), "+build") ||
		strings.HasPrefix(t, "export ")
}

// usesCgo reports whether the file imports "C".
func usesCgo(f *ast.File) bool {
	for _, imp := range f.Imports {
		if imp.Path.Value == `"C"` {
			return true
		}
	}
	return false
}

// gitRun executes git with args in dir and returns stdout.
func gitRun(dir string, args ...string) ([]byte, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	return cmd.Output()
}
