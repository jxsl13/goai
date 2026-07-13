package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// scratchRepo builds a throwaway git repo with a base commit containing files,
// applies changes, commits head, and returns (dir, baseRev, headRev).
func scratchRepo(t *testing.T, base, head map[string]string) (string, string, string) {
	t.Helper()
	dir := t.TempDir()
	run := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return string(out)
	}
	write := func(files map[string]string) {
		for p, content := range files {
			full := filepath.Join(dir, p)
			if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
				t.Fatal(err)
			}
			if content == "<delete>" {
				if err := os.Remove(full); err != nil {
					t.Fatal(err)
				}
				continue
			}
			if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	run("init", "-q")
	write(base)
	run("add", "-A")
	run("commit", "-qm", "base")
	write(head)
	run("add", "-A")
	run("commit", "-qm", "head", "--allow-empty")
	return dir, "HEAD~1", "HEAD"
}

const goBase = `package p

// Add adds two ints.
func Add(a, b int) int { return a + b }
`

func TestClassify(t *testing.T) {
	cases := []struct {
		name string
		base map[string]string
		head map[string]string
		want string
	}{
		{"md only", map[string]string{"README.md": "a", "x.go": goBase},
			map[string]string{"README.md": "b"}, DocsOnly},
		{"docs dir", map[string]string{"docs/a.md": "a", "x.go": goBase},
			map[string]string{"docs/a.md": "b"}, DocsOnly},
		{"comment-only go edit", map[string]string{"x.go": goBase},
			map[string]string{"x.go": "package p\n\n// Add returns the sum of two ints (better doc).\nfunc Add(a, b int) int { return a + b }\n"}, DocsOnly},
		{"comment whitespace", map[string]string{"x.go": goBase},
			map[string]string{"x.go": "package p\n\n// Add adds two    ints.\nfunc Add(a, b int) int { return a + b }\n"}, DocsOnly},
		{"real code edit", map[string]string{"x.go": goBase},
			map[string]string{"x.go": "package p\n\n// Add adds two ints.\nfunc Add(a, b int) int { return a - b }\n"}, Code},
		{"mixed md and code", map[string]string{"README.md": "a", "x.go": goBase},
			map[string]string{"README.md": "b", "x.go": "package p\n\nfunc Add(a, b int) int { return a * b }\n"}, Code},
		{"build tag change is code", map[string]string{"x.go": "//go:build linux\n\npackage p\n"},
			map[string]string{"x.go": "//go:build darwin\n\npackage p\n"}, Code},
		{"added go file", map[string]string{"x.go": goBase},
			map[string]string{"y.go": "package p\n"}, Code},
		{"deleted go file", map[string]string{"x.go": goBase, "y.go": "package p\n"},
			map[string]string{"y.go": "<delete>"}, Code},
		{"workflow edit", map[string]string{".github/workflows/ci.yml": "a", "x.go": goBase},
			map[string]string{".github/workflows/ci.yml": "b"}, Code},
		{"go.mod edit", map[string]string{"go.mod": "module m\n", "x.go": goBase},
			map[string]string{"go.mod": "module m\n\ngo 1.26\n"}, Code},
		{"parse error fails open", map[string]string{"x.go": goBase},
			map[string]string{"x.go": "package p\n\nfunc {{{ broken\n"}, Code},
		{"cgo comment change is code", map[string]string{"x.go": "package p\n\n// #include <math.h>\nimport \"C\"\n"},
			map[string]string{"x.go": "package p\n\n// #include <stdio.h>\nimport \"C\"\n"}, Code},
		{"no change at all", map[string]string{"x.go": goBase},
			map[string]string{}, DocsOnly},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir, base, head := scratchRepo(t, tc.base, tc.head)
			if got := Classify(defaultConfig(dir), dir, base, head); got != tc.want {
				t.Fatalf("Classify = %q, want %q", got, tc.want)
			}
		})
	}
}

// Missing base revision (shallow clone / force push) must fail open to Code.
func TestClassifyMissingBaseFailsOpen(t *testing.T) {
	dir, _, head := scratchRepo(t, map[string]string{"x.go": goBase}, map[string]string{"README.md": "b"})
	if got := Classify(defaultConfig(dir), dir, "deadbeef", head); got != Code {
		t.Fatalf("missing base = %q, want %q", got, Code)
	}
}
