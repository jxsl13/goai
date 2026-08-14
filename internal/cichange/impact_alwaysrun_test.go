package main

import (
	"bytes"
	"strings"
	"testing"
)

// §T893 / §B98: the whole-tree meta-tests (internal/apicheck walks every package's
// source, internal/mdlint every *.md) have no import edge to what they check, so the
// reverse closure can never reach them. The always-run set adds every configured
// package that EXISTS to a non-empty selection, so a started pipeline runs them; a
// pure-docs diff stays zero-runner (§C16 EXC2).

// A module shaped like the real repo: an ordinary package plus the two meta-test
// packages, plus untouched filler packages so a small diff selects a PROPER SUBSET
// (otherwise selecting nlp + both meta-tests would be every package → all).
var modAlwaysRun = map[string]string{
	"go.mod":                 "module example.com/m\n\ngo 1.26\n",
	"nlp/nlp.go":             "package nlp\n\n// F does.\nfunc F() int { return 1 }\n",
	"internal/apicheck/a.go": "package apicheck\n",
	"internal/mdlint/m.go":   "package mdlint\n",
	"nn/nn.go":               "package nn\n",
	"tensor/tensor.go":       "package tensor\n",
	"format/format.go":       "package format\n",
	"docs/guide.md":          "hi\n",
}

// TestImpactAlwaysRunSelectsMetaTests is the B98 regression: an nlp-only CODE diff must
// select internal/apicheck AND internal/mdlint (the default always-run set) even though
// nothing imports them — the seam that let §V19/mdlint rot red while CI stayed green.
func TestImpactAlwaysRunSelectsMetaTests(t *testing.T) {
	dir, base, head := scratchRepo(t, modAlwaysRun, map[string]string{
		"nlp/nlp.go": "package nlp\n\nfunc F() int { return 2 }\n",
	})
	// The default alwaysRun is empty until the gates are green (§T893); test the
	// mechanism with an explicit config that always-runs the two meta-tests.
	ig, igRe, fRe, pRe, _ := defaultRules()
	cfg, err := newConfig(dir, ig, igRe, fRe, pRe, []string{"internal/apicheck", "internal/mdlint"})
	if err != nil {
		t.Fatal(err)
	}
	got := Impact(cfg, dir, base, head)
	for _, want := range []string{"example.com/m/nlp", "example.com/m/internal/apicheck", "example.com/m/internal/mdlint"} {
		if !strings.Contains(got, want) {
			t.Errorf("nlp-only diff: %q is missing %q (the meta-test the closure cannot reach)", got, want)
		}
	}
}

// TestRunAlwaysRunSelectsMetaTests is the §B98 regression for the EXECUTION path: CI runs
// tests via `cichange -run` (ci.yml), not Impact(), so the always-run set must be forced
// into a non-empty RUN selection too — else the meta-tests are padded into the -impact
// AFFECTED string but never actually execute on an nlp/nn push. Mirrors the Impact test.
func TestRunAlwaysRunSelectsMetaTests(t *testing.T) {
	dir, base, head := scratchRepo(t, modAlwaysRun, map[string]string{
		"nlp/nlp.go": "package nlp\n\n// F does.\nfunc F() int { return 2 }\n",
	})
	ig, igRe, fRe, pRe, _ := defaultRules()
	cfg, err := newConfig(dir, ig, igRe, fRe, pRe, []string{"internal/apicheck", "internal/mdlint"})
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	code := Run(cfg, dir, base, head, []string{"-count=1"}, &buf)
	out := buf.String()
	if code != 0 {
		t.Fatalf("exit %d, output:\n%s", code, out)
	}
	for _, want := range []string{
		"run \texample.com/m/internal/apicheck\t[always-run meta-test (§T893)]",
		"run \texample.com/m/internal/mdlint\t[always-run meta-test (§T893)]",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("run-mode selection lacks %q (the seam that let meta-tests skip execution)\n---\n%s", want, out)
		}
	}
}

// TestRunAlwaysRunSkippedOnDocsOnly: the run path, like Impact, must NOT resurrect the
// meta-tests on a docs-only diff — the None verdict returns before the always-run loop.
func TestRunAlwaysRunSkippedOnDocsOnly(t *testing.T) {
	dir, base, head := scratchRepo(t, modAlwaysRun, map[string]string{
		"docs/guide.md": "changed\n",
	})
	ig, igRe, fRe, pRe, _ := defaultRules()
	cfg, err := newConfig(dir, ig, igRe, fRe, pRe, []string{"internal/apicheck", "internal/mdlint"})
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if code := Run(cfg, dir, base, head, []string{"-count=1"}, &buf); code != 0 {
		t.Fatalf("exit %d, output:\n%s", code, buf.String())
	}
	if out := buf.String(); strings.Contains(out, "always-run meta-test") {
		t.Errorf("docs-only diff must not select any always-run meta-test\n---\n%s", out)
	}
}

// TestImpactAlwaysRunSkippedOnDocsOnly: a pure-documentation diff returns None and adds
// NO meta-test — always-run applies only once a pipeline starts (§C16 EXC2), so it never
// resurrects work on a docs-only push.
func TestImpactAlwaysRunSkippedOnDocsOnly(t *testing.T) {
	dir, base, head := scratchRepo(t, modAlwaysRun, map[string]string{
		"docs/guide.md": "changed\n",
	})
	ig, igRe, fRe, pRe, _ := defaultRules()
	cfg, err := newConfig(dir, ig, igRe, fRe, pRe, []string{"internal/apicheck", "internal/mdlint"})
	if err != nil {
		t.Fatal(err)
	}
	if got := Impact(cfg, dir, base, head); got != None {
		t.Errorf("docs-only diff: got %q, want none (always-run must not resurrect a docs-only pipeline)", got)
	}
}

// TestImpactAlwaysRunCustomAndMissing: a custom always-run package that EXISTS is added;
// one ABSENT from the checkout contributes nothing (no bogus go-test target, no spurious
// All) — the tolerate-missing rule that keeps temp modules and renamed packages safe.
func TestImpactAlwaysRunCustomAndMissing(t *testing.T) {
	base := map[string]string{
		"go.mod":    "module example.com/m\n\ngo 1.26\n",
		"a/a.go":    "package a\n\nfunc A() {}\n",
		"meta/m.go": "package meta\n",
		"x/x.go":    "package x\n", // untouched filler so {a,meta} is a proper subset
		"y/y.go":    "package y\n",
	}
	dir, b, h := scratchRepo(t, base, map[string]string{"a/a.go": "package a\n\nfunc A() int { return 1 }\n"})
	cfg, err := newConfig(dir, nil, nil, nil, nil, []string{"meta", "does/not/exist"})
	if err != nil {
		t.Fatal(err)
	}
	if got := Impact(cfg, dir, b, h); got != "example.com/m/a example.com/m/meta" {
		t.Errorf("custom always-run: got %q, want %q (meta added, ghost skipped)", got, "example.com/m/a example.com/m/meta")
	}
}

// TestDefaultAlwaysRunExcludesExternalPerfscan pins T984's execution boundary:
// perfscan is an external CLI with its own CI job, not a Go package cichange may
// pass to `go test`. Reintroducing the deleted package name would make every
// selective test run fail before the external scanner even starts.
func TestDefaultAlwaysRunExcludesExternalPerfscan(t *testing.T) {
	_, _, _, _, alwaysRun := defaultRules()
	for _, pkg := range alwaysRun {
		if pkg == "internal/perfscan" {
			t.Fatal("internal/perfscan must stay out of alwaysRun; external perfscan has a dedicated CI job")
		}
	}
}
