package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRepoMarkdownIsClean (§T612): every committed human-audience markdown file
// passes the linter — the CI-side twin of the pre-commit hook.
func TestRepoMarkdownIsClean(t *testing.T) {
	root := "../.."
	var files []string
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if n := d.Name(); n == ".git" || n == ".venv" || n == "node_modules" {
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
		t.Fatal(err)
	}
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		for _, fd := range Lint(f, string(raw)) {
			t.Errorf("%s:%d [%s] %s", fd.file, fd.line, fd.rule, fd.msg)
		}
	}
}

// TestLintRules: each rule fires on a minimal bad snippet and stays silent on
// the corrected form.
func TestLintRules(t *testing.T) {
	cases := []struct{ name, bad, good, rule string }{
		{"fence", "```go\ncode\n", "```go\ncode\n```\n", "unclosed-fence"},
		{"table", "| a | b |\n|---|---|\n| 1 |\n", "| a | b |\n|---|---|\n| 1 | 2 |\n", "table-ragged"},
		{"heading", "# t\n\n### deep\n", "# t\n\n## deep\n", "heading-jump"},
		{"html", "the <unk> token\n", "the `<unk>` token\n", "swallowed-html"},
		{"backtick", "one ` dangling\n", "one `` pair\n", "odd-backticks"},
		{"link", "[text](broken\n", "[text](ok)\n", "malformed-link"},
		{"tilde", "gains of ~4x and ~16x here\n", "gains of ≈4x, and a ~~properly struck~~ span\n", "tilde-strikethrough"},
		{"h1", "# Title\n\n# a mid-doc comment\n", "# Title\n\n## a section\n", "multiple-h1"},
	}
	for _, tc := range cases {
		found := false
		for _, fd := range Lint("x.md", tc.bad) {
			if fd.rule == tc.rule {
				found = true
			}
		}
		if !found {
			t.Errorf("%s: rule %s did not fire on the bad snippet", tc.name, tc.rule)
		}
		for _, fd := range Lint("x.md", tc.good) {
			if fd.rule == tc.rule {
				t.Errorf("%s: rule %s fired on the good snippet: %s", tc.name, tc.rule, fd.msg)
			}
		}
	}
}

var _ = strings.TrimSpace // silence unused-import drift if rules change

// §T615: the go-block rules — unformatted snippets are findings, -w fixes them,
// and non-Go content in a go fence is flagged.
func TestGoBlockRules(t *testing.T) {
	bad := "# t\n\n```go\nx:=1\nif x==1 { y := 2; _ = y }\n```\n"
	fs, _ := lintGoBlocks("x.md", strings.Split(bad, "\n"), false)
	if len(fs) != 1 || fs[0].rule != "go-block-fmt" {
		t.Fatalf("want one go-block-fmt finding, got %v", fs)
	}
	_, fixed := lintGoBlocks("x.md", strings.Split(bad, "\n"), true)
	joined := strings.Join(fixed, "\n")
	if !strings.Contains(joined, "x := 1") {
		t.Errorf("rewrite did not format: %q", joined)
	}
	fs2, _ := lintGoBlocks("x.md", fixed, false)
	if len(fs2) != 0 {
		t.Errorf("formatted block still flagged: %v", fs2)
	}
	garbage := "```go\nthis is not go at all {{{\n```\n"
	fs3, _ := lintGoBlocks("x.md", strings.Split(garbage, "\n"), false)
	if len(fs3) != 1 || fs3[0].rule != "go-block-parse" {
		t.Errorf("want go-block-parse, got %v", fs3)
	}
}
