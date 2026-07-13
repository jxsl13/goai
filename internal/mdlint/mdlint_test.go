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
