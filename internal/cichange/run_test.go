package main

import (
	"bytes"
	"os"
	"path/filepath"
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
	code := Run(defaultConfig(dir), dir, base, headRev, []string{"-count=1", "-run=TestAdd"}, &buf)
	out := buf.String()
	if code != 0 {
		t.Fatalf("exit %d, output:\n%s", code, out)
	}
	for _, want := range []string{
		"a/a.go: code → a",
		"a/a_test.go: test-only code → a (test files are not importable: no propagation)",
		"run \texample.com/m/a\t[changed]",
		"run \texample.com/m/b\t[depends on a changed package]",
		"?   \texample.com/m/c\t[not affected by this change]",
		"example.com/m/a: TestAdd",
		"go test -count=1 -run=TestAdd example.com/m example.com/m/a example.com/m/b example.com/m/e",
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

// The SAME propagation guarantee, invoked the way CI actually invokes it — with the
// leading "--" separator. TestRunPropagatesFailure above passes the go-test args bare,
// which is why it stayed green through the entire outage this test exists to prevent.
//
// flag.Parse stops at the first positional (base), so it never consumes the separator;
// it reached Run as a literal argument and was forwarded into the command line. And
// `go test -- -count=1 <pkgs>` does not test <pkgs>: everything after -- goes to the
// test binary, the package list is swallowed, go test falls back to the package in the
// working directory, and it EXITS 0. Every "run tests for the affected packages" step
// therefore tested one package and reported success no matter what was broken — which
// is how nn sat red on main across many green PRs (Muon F32 panic, MARS fast-path
// drift, DyT FMA contraction, and the apicheck doc gaps).
//
// The failure mode is silence, so assert on both halves: a real nonzero exit, and the
// separator absent from the emitted command line.
func TestRunPropagatesFailureWithCIArgSeparator(t *testing.T) {
	head := map[string]string{
		"c/c.go":      "package c\n\nvar X = 1\n",
		"c/c_test.go": "package c\n\nimport \"testing\"\n\nfunc TestBoom(t *testing.T) { t.Fatal(\"boom\") }\n",
	}
	dir, base, headRev := scratchRepo(t, modBase, head)
	var buf bytes.Buffer
	code := Run(defaultConfig(dir), dir, base, headRev, []string{"--", "-count=1"}, &buf)
	out := buf.String()
	if code == 0 {
		t.Fatalf("a failing test must yield a nonzero exit even when the go-test args carry CI's leading %q separator; got 0, output:\n%s", "--", out)
	}
	if !strings.Contains(out, "boom") {
		t.Error("go test failure output must be streamed, not swallowed")
	}
	if strings.Contains(out, "go test -- ") {
		t.Errorf("the %q separator must be stripped before it reaches the go test command line, or the package list is silently ignored\n---\n%s", "--", out)
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

// §T594: without a -run/-skip filter, whole-package selections print NO per-function
// listing; with one, the listing appears.
func TestRunFunctionListingOnlyWhenFiltered(t *testing.T) {
	head := map[string]string{
		"a/a.go":      "package a\n\nfunc Add(x, y int) int { return x - y }\n",
		"a/a_test.go": "package a\n\nimport \"testing\"\n\nfunc TestAdd(t *testing.T) {}\n",
	}
	dir, base, headRev := scratchRepo(t, modBase, head)
	var plain bytes.Buffer
	Run(defaultConfig(dir), dir, base, headRev, []string{"-count=1"}, &plain)
	if strings.Contains(plain.String(), "test functions in selected packages") {
		t.Error("no filter → no function listing")
	}
	var filtered bytes.Buffer
	Run(defaultConfig(dir), dir, base, headRev, []string{"-count=1", "-run=TestAdd"}, &filtered)
	if !strings.Contains(filtered.String(), "test functions in selected packages") ||
		!strings.Contains(filtered.String(), "example.com/m/a: TestAdd") {
		t.Error("-run filter → function listing expected")
	}
}

// TestBuildablePkgsDropsConstraintExcluded covers the failure that turned every pure-go CI lane red
// while `go test ./...` stayed green locally.
//
// The two forms disagree: the wildcard silently skips a package whose build constraints exclude all
// its files, while naming that package EXPLICITLY is a hard "build constraints exclude all Go files"
// error. The selective runner names packages explicitly, so a cgo-only package (internal/gpudecode)
// failed the pure-go lanes with a failure no local check could see — `go test ./...` cannot reproduce
// it by construction.
//
// The filter must also run `go list` in the module under test rather than the process's working
// directory; getting that wrong made every package look unbuildable and silently emptied the run.
func TestBuildablePkgsDropsConstraintExcluded(t *testing.T) {
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Dir(filepath.Dir(dir)) // .../internal/cichange -> module root
	const self = "github.com/jxsl13/goai/internal/cichange"
	got := buildablePkgs(root, []string{self})
	if len(got) != 1 || got[0] != self {
		t.Fatalf("a buildable package must survive the filter: got %v", got)
	}
	// A package that exists but has no files under this build config must be dropped, not passed
	// through to `go test` where it would be a setup failure.
	const nofiles = "github.com/jxsl13/goai/internal/cichange/testdata/nonexistent"
	got = buildablePkgs(root, []string{self, nofiles})
	for _, p := range got {
		if p == nofiles {
			t.Errorf("unbuildable package %q survived the filter; `go test` would fail on it", p)
		}
	}
}
