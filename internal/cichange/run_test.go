package main

import (
	"bytes"
	"strings"
	"testing"
)

// §V16 tier-1 (§T587): the transparent report names the per-file rule, gives every
// selected package a reason and every skipped package its reason, lists test
// functions, prints the exact go test command with individual packages, streams the
// go test output, and exits with go test's code.
func TestRunTransparentReport(t *testing.T) {
	head := map[string]string{
		"a/a.go":      "package a\n\nfunc Add(x, y int) int { return x - y }\n",
		"a/a_test.go": "package a\n\nimport \"testing\"\n\nfunc TestAdd(t *testing.T) {}\n",
	}
	dir, base, headRev := scratchRepo(t, modBase, head)
	var buf bytes.Buffer
	code := Run(defaultConfig(dir), dir, base, headRev, []string{"-count=1"}, &buf)
	out := buf.String()
	if code != 0 {
		t.Fatalf("exit %d, output:\n%s", code, out)
	}
	for _, want := range []string{
		"a/a.go: code → a",
		"a/a_test.go: test-only code → a (test files are not importable: no propagation)",
		"run \texample.com/m/a\tchanged",
		"run \texample.com/m/b\tdepends on a changed package",
		"skip\texample.com/m/c\tno dependency path from any changed package",
		"example.com/m/a: TestAdd",
		"go test -count=1 example.com/m example.com/m/a example.com/m/b example.com/m/e",
		"ok  ", // real go test output streamed through
	} {
		if !strings.Contains(out, want) {
			t.Errorf("report lacks %q\n---\n%s", want, out)
		}
	}
}

// §V16 tier-1 (§T587): a failing selected test propagates go test's exit code — the
// CLI never masks a failure.
func TestRunPropagatesFailure(t *testing.T) {
	head := map[string]string{
		"c/c.go":      "package c\n\nvar X = 1\n",
		"c/c_test.go": "package c\n\nimport \"testing\"\n\nfunc TestBoom(t *testing.T) { t.Fatal(\"boom\") }\n",
	}
	dir, base, headRev := scratchRepo(t, modBase, head)
	var buf bytes.Buffer
	code := Run(defaultConfig(dir), dir, base, headRev, []string{"-count=1"}, &buf)
	if code == 0 {
		t.Fatalf("failing test must yield nonzero exit, output:\n%s", buf.String())
	}
	if !strings.Contains(buf.String(), "boom") {
		t.Error("go test failure output must be streamed, not swallowed")
	}
}

// §V16 tier-1 (§T587): docs-only → explicit "no tests to run", exit 0; and the
// pre-execution report is byte-identical across runs (deterministic ordering).
func TestRunNoneAndDeterminism(t *testing.T) {
	dir, base, headRev := scratchRepo(t, modBase, map[string]string{"docs/guide.md": "v2\n"})
	var buf bytes.Buffer
	if code := Run(defaultConfig(dir), dir, base, headRev, nil, &buf); code != 0 {
		t.Fatalf("docs-only must exit 0:\n%s", buf.String())
	}
	if !strings.Contains(buf.String(), "no tests to run") {
		t.Errorf("missing no-tests marker:\n%s", buf.String())
	}
	head := map[string]string{
		"a/a.go": "package a\n\nfunc Add(x, y int) int { return x * y }\n",
		"c/c.go": "package c\n\nvar Z = 3\n",
	}
	dir2, base2, head2 := scratchRepo(t, modBase, head)
	report := func() string {
		var b bytes.Buffer
		Run(defaultConfig(dir2), dir2, base2, head2, []string{"-count=1", "-run=NONE"}, &b)
		s := b.String()
		return s[:strings.Index(s, "== go test output ==")]
	}
	first := report()
	for range 3 {
		if got := report(); got != first {
			t.Fatalf("non-deterministic report:\n--- first ---\n%s\n--- got ---\n%s", first, got)
		}
	}
}
