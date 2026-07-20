package main

import (
	"fmt"
	"go/format"
	"go/scanner"
	"go/token"
	"strings"
)

// Go-block formatting (§T615): every ```go fence in a markdown file must be
// gofmt-clean. Snippets are usually statement lists, not whole files, so they
// are wrapped in a synthetic func body before formatting and unwrapped after;
// a block that does not even parse wrapped is itself a finding — documentation
// code must at least be syntactically plausible Go.
//
// In plain terms: the Go examples readers copy from our docs are held to the
// same formatting standard as the code itself.

// goBlock is one fenced ```go region: line span and raw body.
type goBlock struct {
	startLine int // 1-based line of the opening fence
	body      []string
}

// goBlocks extracts the ```go fences of a document.
func goBlocks(lines []string) []goBlock {
	var out []goBlock
	for i := 0; i < len(lines); i++ {
		t := strings.TrimSpace(lines[i])
		if t != "```go" {
			continue
		}
		j := i + 1
		for j < len(lines) && strings.TrimSpace(lines[j]) != "```" {
			j++
		}
		if j >= len(lines) {
			break // unclosed fence: the fence rule reports it
		}
		out = append(out, goBlock{startLine: i + 1, body: lines[i+1 : j]})
		i = j
	}
	return out
}

// hasCodeTokens reports whether the snippet contains any token beyond
// comments. A comment-only block — e.g. a lone `//go:build vulkan`
// constraint (ADR-0029) — must be left verbatim: format.Source's fragment
// heuristic "formats" it destructively (`//go:build vulkan` →
// "vulkan\n\npackage p", the constraint prefix stripped and a package
// clause invented — §B115/§T889(b)).
func hasCodeTokens(src string) bool {
	fset := token.NewFileSet()
	var s scanner.Scanner
	s.Init(fset.AddFile("", fset.Base(), len(src)), []byte(src), nil, 0)
	for {
		_, tok, _ := s.Scan()
		switch tok {
		case token.EOF:
			return false
		case token.SEMICOLON: // auto-inserted, not code
		default:
			return true
		}
	}
}

// formatSnippet gofmt-formats a statement-level snippet. It first tries the
// snippet as a whole file (for blocks with package/func decls), then wrapped
// in a synthetic func body. Returns the formatted snippet lines.
func formatSnippet(body []string) ([]string, error) {
	src := strings.Join(body, "\n")
	if out, err := format.Source([]byte(src)); err == nil {
		return strings.Split(strings.TrimRight(string(out), "\n"), "\n"), nil
	}
	wrapped := "package p\n\nfunc _() {\n" + src + "\n}\n"
	out, err := format.Source([]byte(wrapped))
	if err != nil {
		return nil, fmt.Errorf("snippet does not parse as Go (even wrapped): %v", err)
	}
	s := string(out)
	i := strings.Index(s, "func _() {\n")
	if i < 0 {
		return nil, fmt.Errorf("internal: wrapper vanished")
	}
	s = s[i+len("func _() {\n"):]
	s = strings.TrimSuffix(strings.TrimRight(s, "\n"), "}")
	s = strings.TrimRight(s, "\n")
	var res []string
	for _, ln := range strings.Split(s, "\n") {
		res = append(res, strings.TrimPrefix(ln, "\t")) // drop the func-body indent
	}
	return res, nil
}

// lintGoBlocks reports unformatted or unparseable go fences; with rewrite=true
// it returns the corrected document lines instead.
func lintGoBlocks(file string, lines []string, rewrite bool) ([]finding, []string) {
	var out []finding
	res := lines
	// iterate from the bottom so line spans stay valid while rewriting
	blocks := goBlocks(lines)
	for bi := len(blocks) - 1; bi >= 0; bi-- {
		b := blocks[bi]
		if !hasCodeTokens(strings.Join(b.body, "\n")) {
			continue // comment-only block: verbatim by contract (§B115)
		}
		formatted, err := formatSnippet(b.body)
		if err != nil {
			out = append(out, finding{file, b.startLine, "go-block-parse", err.Error()})
			continue
		}
		if strings.Join(formatted, "\n") == strings.Join(b.body, "\n") {
			continue
		}
		if rewrite {
			res = append(res[:b.startLine], append(append([]string{}, formatted...), res[b.startLine+len(b.body):]...)...)
		} else {
			out = append(out, finding{file, b.startLine, "go-block-fmt", "go code block is not gofmt-formatted (run: go run ./internal/mdlint -w <file>)"})
		}
	}
	return out, res
}
