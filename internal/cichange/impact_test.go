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
	return Impact(defaultConfig(dir), dir, base, headRev)
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

// §V16 tier-1 (§T581): a build-tag-gated import still counts — the graph is the UNION
// over all tags, so platform- or tag-specific edges (register_darwin.go, vulkan) are
// permanent and can never be missed by running the selector on a different platform.
func TestImpactBuildTagImportCounted(t *testing.T) {
	base := map[string]string{}
	for k, v := range modBase {
		base[k] = v
	}
	base["f/f.go"] = "//go:build sometag\n\npackage f\n\nimport _ \"example.com/m/c\"\n"
	dir, baseRev, headRev := scratchRepo(t, base, map[string]string{"c/c.go": "package c\n\nvar Tagged = 1\n"})
	got := Impact(defaultConfig(dir), dir, baseRev, headRev)
	if got != "./b ./c ./f" {
		t.Errorf("tag-gated import: got %q, want ./b ./c ./f", got)
	}
}

// §V16 tier-1 (§T581): a file MOVED between packages changes both — the old package
// lost code, the new one gained it; both closures join.
func TestImpactRenameAcrossPackages(t *testing.T) {
	got := impactOf(t, map[string]string{
		"c/c.go":     "<delete>",
		"c/moved.go": "package c\n",
		"a/a.go":     "<delete>",
		"c/fromA.go": "package c\n\n// Add adds.\nfunc Add(x, y int) int { return x + y }\n",
	})
	// a/a.go left package a: a changed (and its importers . b e); c gained files.
	if got != All && got != ". ./a ./b ./c ./e" {
		t.Errorf("cross-package move: got %q, want closure of both (or all if a died)", got)
	}
}

// §V16 tier-1 (§T581): a change to an embedded data file selects the embedding
// package — //go:embed patterns cannot reach outside the package directory, so the
// ancestor-attach rule covers embeds by construction.
func TestImpactEmbeddedFileChange(t *testing.T) {
	base := map[string]string{}
	for k, v := range modBase {
		base[k] = v
	}
	base["g/g.go"] = "package g\n\nimport _ \"embed\"\n\n//go:embed data.txt\nvar Data string\n"
	base["g/data.txt"] = "v1\n"
	dir, baseRev, headRev := scratchRepo(t, base, map[string]string{"g/data.txt": "v2\n"})
	if got := Impact(defaultConfig(dir), dir, baseRev, headRev); got != "./g" {
		t.Errorf("embed change: got %q, want ./g", got)
	}
}

// §V16 tier-1 (§T581): closure monotonicity — changing a dependency affects at least
// everything that changing its dependent affects (b imports a ⇒ affected(b) ⊆ affected(a)).
func TestImpactClosureMonotonic(t *testing.T) {
	affectedA := strings.Fields(impactOf(t, map[string]string{
		"a/a.go": "package a\n\nfunc Add(x, y int) int { return x - y }\n"}))
	affectedB := strings.Fields(impactOf(t, map[string]string{
		"b/b.go": "package b\n\nimport \"example.com/m/a\"\n\nvar _ = a.Add\nvar Q = 1\n"}))
	inA := map[string]bool{}
	for _, p := range affectedA {
		inA[p] = true
	}
	for _, p := range affectedB {
		if !inA[p] {
			t.Errorf("monotonicity violated: %q affected by dependent-change but not by dependency-change", p)
		}
	}
}

// depBase extends modBase with an external dependency: c imports dep.example/lib
// (code), a imports it from a _test.go only.
func depBase() map[string]string {
	base := map[string]string{}
	for k, v := range modBase {
		base[k] = v
	}
	base["go.mod"] = "module example.com/m\n\ngo 1.26\n\nrequire dep.example/lib v1.0.0\n"
	base["c/c.go"] = "package c\n\nimport _ \"dep.example/lib/sub\"\n"
	base["a/a_test.go"] = "package a\n\nimport _ \"dep.example/lib\"\n"
	return base
}

// §V16 tier-1 (§T582): a dependency VERSION BUMP selects exactly the importers of that
// module — code importers with their closure, test-only importers without propagation.
func TestImpactDependencyVersionBump(t *testing.T) {
	dir, baseRev, headRev := scratchRepo(t, depBase(), map[string]string{
		"go.mod": "module example.com/m\n\ngo 1.26\n\nrequire dep.example/lib v1.1.0\n",
	})
	// c imports the dep (code) → c + b (test edge on c); a's _test.go imports it → a.
	if got := Impact(defaultConfig(dir), dir, baseRev, headRev); got != "./a ./b ./c" {
		t.Errorf("dep bump: got %q, want ./a ./b ./c", got)
	}
}

// §V16 tier-1 (§T582): non-require go.mod changes (go directive, replace) still force
// the full suite; an added-but-unimported dependency selects nothing; a removed
// dependency seeds nothing beyond the import-dropping file changes themselves.
func TestImpactDependencyEdgeCases(t *testing.T) {
	dir, baseRev, headRev := scratchRepo(t, depBase(), map[string]string{
		"go.mod": "module example.com/m\n\ngo 1.26\n\nrequire dep.example/lib v1.0.0\n\nreplace dep.example/lib => ../local\n",
	})
	if got := Impact(defaultConfig(dir), dir, baseRev, headRev); got != All {
		t.Errorf("replace directive: got %q, want all", got)
	}
	dir, baseRev, headRev = scratchRepo(t, depBase(), map[string]string{
		"go.mod": "module example.com/m\n\ngo 1.26\n\nrequire (\n\tdep.example/lib v1.0.0\n\tother.example/x v2.0.0\n)\n",
	})
	if got := Impact(defaultConfig(dir), dir, baseRev, headRev); got != None {
		t.Errorf("added unimported dep: got %q, want none", got)
	}
	dir, baseRev, headRev = scratchRepo(t, depBase(), map[string]string{
		"go.mod":      "module example.com/m\n\ngo 1.26\n",
		"c/c.go":      "package c\n",
		"a/a_test.go": "package a\n",
	})
	// removal: the import-dropping .go edits attribute normally (c code, a test-only).
	if got := Impact(defaultConfig(dir), dir, baseRev, headRev); got != "./a ./b ./c" {
		t.Errorf("removed dep: got %q, want ./a ./b ./c", got)
	}
}
