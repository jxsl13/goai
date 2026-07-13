package main

import (
	"strings"
	"testing"
)

// modBase is a three-package module: root imports a; b imports a; c is independent.
// Layout mirrors the real repo's shapes (root register file, nested non-Go assets).
var modBase = map[string]string{
	"go.mod":        "module example.com/m\n\ngo 1.26\n",
	"root.go":       "package m\n\nimport _ \"example.com/m/a\"\n",
	"a/a.go":        "package a\n\n// Add adds.\nfunc Add(x, y int) int { return x + y }\n",
	"a/a_test.go":   "package a\n",
	"b/b.go":        "package b\n\nimport \"example.com/m/a\"\n\nvar _ = a.Add\n",
	"c/c.go":        "package c\n",
	"c/gen.s":       "// asm placeholder\n",
	"c/testdata/f":  "fixture\n",
	"b/ext_test.go": "package b\n\nimport _ \"example.com/m/c\"\n",
	"e/e.go":        "package e\n\nimport _ \"example.com/m/b\"\n",
	"docs/guide.md": "hi\n",
}

func impactOf(t *testing.T, head map[string]string) string {
	t.Helper()
	dir, base, headRev := scratchRepo(t, modBase, head)
	return Impact(dir, base, headRev)
}

// §V16 tier-1 (§T579): the reverse closure — a leaf change selects the leaf and every
// transitive importer, nothing else; an independent package selects only itself.
func TestImpactReverseClosure(t *testing.T) {
	got := impactOf(t, map[string]string{
		"a/a.go": "package a\n\nfunc Add(x, y int) int { return x - y }\n",
	})
	if got != ". ./a ./b ./e" {
		t.Errorf("leaf change: got %q, want %q", got, ". ./a ./b ./e")
	}
	// c is imported ONLY by b's _test.go: b's tests link c (select b), but b's
	// compiled code does not — so e (importing b) must NOT be selected.
	got = impactOf(t, map[string]string{"c/c.go": "package c\n\nvar X = 1\n"})
	if got != "./b ./c" {
		t.Errorf("test-edge change: got %q, want %q", got, "./b ./c")
	}
}

// §V16 tier-1: _test.go-only changes select just their package — test files cannot be
// imported, so dependents are provably unaffected.
func TestImpactTestOnlyChange(t *testing.T) {
	got := impactOf(t, map[string]string{
		"a/a_test.go": "package a\n\nimport \"testing\"\n\nfunc TestX(t *testing.T) {}\n",
	})
	if got != "./a" {
		t.Errorf("test-only change: got %q, want %q", got, "./a")
	}
}

// §V16 tier-1: comment-only .go edits and documentation are invisible → none.
func TestImpactNone(t *testing.T) {
	if got := impactOf(t, map[string]string{
		"a/a.go":        "package a\n\n// Add adds two ints together.\nfunc Add(x, y int) int { return x + y }\n",
		"docs/guide.md": "changed\n",
	}); got != None {
		t.Errorf("got %q, want none", got)
	}
	if got := impactOf(t, map[string]string{}); got != None {
		t.Errorf("empty diff: got %q, want none", got)
	}
}

// §V16 tier-1: non-Go assets (assembly, testdata) attach to the nearest ancestor
// package and propagate like code — cgo/embed reach is confined to the package subtree.
func TestImpactNonGoAssets(t *testing.T) {
	if got := impactOf(t, map[string]string{"c/gen.s": "// new asm\n"}); got != "./b ./c" {
		t.Errorf("assembly change: got %q, want ./b ./c", got)
	}
	if got := impactOf(t, map[string]string{"c/testdata/f": "new fixture\n"}); got != "./b ./c" {
		t.Errorf("testdata change: got %q, want ./b ./c", got)
	}
}

// §V16 tier-1 (§V26 fail-open): build-steering files and files outside every package
// force the full suite; a root-package change with dependents behaves like any package.
func TestImpactGlobalAndRoot(t *testing.T) {
	for _, tc := range []map[string]string{
		{"go.mod": "module example.com/m\n\ngo 1.27\n"},
		{"Makefile": "all:\n"},
		{".github/workflows/ci.yml": "name: ci\n"},
		{"orphan.c": "int x;\n"}, // root IS a package here, so attach... see below
	} {
		got := impactOf(t, tc)
		want := All
		if _, hasOrphan := tc["orphan.c"]; hasOrphan {
			want = "." // root package owns root-level non-Go files
		}
		if got != want {
			t.Errorf("%v: got %q, want %q", tc, got, want)
		}
	}
	// a change to the root package itself: nothing imports root → only root runs.
	if got := impactOf(t, map[string]string{"root.go": "package m\n\nimport _ \"example.com/m/a\"\n\nvar X = 1\n"}); got != "." {
		t.Errorf("root change: got %q, want .", got)
	}
}

// §V16 tier-1: a new package is only itself; adding a file to an existing package
// propagates; a deleted package fails open (its importers would break unnoticed).
func TestImpactAddDelete(t *testing.T) {
	if got := impactOf(t, map[string]string{"d/d.go": "package d\n"}); got != "./d" {
		t.Errorf("new package: got %q, want ./d", got)
	}
	if got := impactOf(t, map[string]string{"a/extra.go": "package a\n\nvar Y = 2\n"}); got != ". ./a ./b ./e" {
		t.Errorf("added file: got %q, want . ./a ./b ./e", got)
	}
	if got := impactOf(t, map[string]string{"c/c.go": "<delete>", "c/gen.s": "<delete>", "c/testdata/f": "<delete>"}); got != All {
		t.Errorf("deleted package: got %q, want all", got)
	}
}

// §V16 tier-1 (§V26): a file whose package clause cannot be parsed breaks the graph →
// all; a file with a broken BODY still parses in ImportsOnly mode, counts as a code
// change and propagates — the selected `go test` run then surfaces the compile error.
func TestImpactParseErrorFailsOpen(t *testing.T) {
	if got := impactOf(t, map[string]string{"c/c.go": "func {{{ no package clause\n"}); got != All {
		t.Errorf("header parse error: got %q, want all", got)
	}
	if got := impactOf(t, map[string]string{"c/c.go": "package c\n\nfunc {{{\n"}); got != "./b ./c" {
		t.Errorf("body parse error: got %q, want ./b ./c (selected, compile fails in CI)", got)
	}
}

// §V16 tier-1: determinism — the output list is sorted, so identical inputs produce
// byte-identical selector output for CI.
func TestImpactDeterministicOrder(t *testing.T) {
	head := map[string]string{
		"a/a_test.go": "package a\n\nimport \"testing\"\n\nfunc TestY(t *testing.T) {}\n",
		"c/c.go":      "package c\n\nvar Z = 3\n",
	}
	first := impactOf(t, head)
	if first != "./a ./b ./c" {
		t.Fatalf("unexpected selection: %q, want ./a ./b ./c", first)
	}
	for range 3 {
		if got := impactOf(t, head); got != first {
			t.Fatalf("non-deterministic output: %q vs %q", got, first)
		}
	}
	if sorted := strings.Fields(first); !sort_StringsAreSorted(sorted) {
		t.Errorf("output not sorted: %q", first)
	}
}

func sort_StringsAreSorted(s []string) bool {
	for i := 1; i < len(s); i++ {
		if s[i-1] > s[i] {
			return false
		}
	}
	return true
}
