package main

import (
	"path/filepath"
	"strings"
	"testing"
)

// cfgWith builds a config rooted at dir with the defaults plus extra rules.
func cfgWith(t *testing.T, dir string, ignore, ignoreRe, fullRe, pkgRe []string) *config {
	t.Helper()
	dIg, dIgRe, dFull, dPkg, dAlways := defaultRules()
	c, err := newConfig(dir,
		append(dIg, ignore...), append(dIgRe, ignoreRe...),
		append(dFull, fullRe...), append(dPkg, pkgRe...), dAlways)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// §V16 tier-1 (§T584): -ignore rules out files and whole directories, and relative and
// absolute spellings are interchangeable (step 1 absolutizes everything).
func TestConfigIgnorePathsRelAndAbs(t *testing.T) {
	head := map[string]string{"c/gen.s": "// new asm\n"}
	dir, base, headRev := scratchRepo(t, modBase, head)
	if got := Impact(cfgWith(t, dir, []string{"c"}, nil, nil, nil), dir, base, headRev); got != None {
		t.Errorf("relative dir ignore: got %q, want none", got)
	}
	abs := filepath.Join(dir, "c", "gen.s")
	if got := Impact(cfgWith(t, dir, []string{abs}, nil, nil, nil), dir, base, headRev); got != None {
		t.Errorf("absolute file ignore: got %q, want none", got)
	}
	// prefix safety: ignoring "c" must not swallow a sibling like "cc".
	dir2, base2, head2 := scratchRepo(t, modBase, map[string]string{"cc/x.go": "package cc\n"})
	if got := Impact(cfgWith(t, dir2, []string{"c"}, nil, nil, nil), dir2, base2, head2); got != "example.com/m/cc" {
		t.Errorf("sibling of ignored dir: got %q, want ./cc", got)
	}
}

// §V16 tier-1 (§T584): -ignore-regex matches repo-relative paths; -full-rerun-regex
// forces all; -pkg-rerun-regex forces the owning package even for changes the content
// analysis would dismiss (comment-only .go edits).
func TestConfigRegexRules(t *testing.T) {
	commentOnly := map[string]string{
		"a/a.go": "package a\n\n// Add adds two ints (reworded).\nfunc Add(x, y int) int { return x + y }\n"}
	dir, base, head := scratchRepo(t, modBase, commentOnly)
	if got := Impact(defaultConfig(dir), dir, base, head); got != None {
		t.Fatalf("baseline: comment-only should be none, got %q", got)
	}
	if got := Impact(cfgWith(t, dir, nil, nil, nil, []string{`^a/`}), dir, base, head); got != "example.com/m example.com/m/a example.com/m/b example.com/m/e" {
		t.Errorf("pkg-rerun-regex: got %q, want . ./a ./b ./e", got)
	}
	if got := Impact(cfgWith(t, dir, nil, nil, []string{`\.go$`}, nil), dir, base, head); got != All {
		t.Errorf("full-rerun-regex: got %q, want all", got)
	}
	dir, base, head = scratchRepo(t, modBase, map[string]string{"c/notes.gen": "x\n"})
	if got := Impact(cfgWith(t, dir, nil, []string{`\.gen$`}, nil, nil), dir, base, head); got != None {
		t.Errorf("ignore-regex: got %q, want none", got)
	}
}

// §V16 tier-1 (§T584): precedence full-rerun > pkg-rerun > ignore — the same path
// matched by all three rules forces the full suite; matched by pkg+ignore it still
// reruns its package.
func TestConfigPrecedence(t *testing.T) {
	head := map[string]string{"c/data.bin": "v2\n"}
	dir, base, headRev := scratchRepo(t, modBase, head)
	all3 := cfgWith(t, dir, []string{"c"}, nil, []string{`^c/`}, []string{`^c/`})
	if got := Impact(all3, dir, base, headRev); got != All {
		t.Errorf("full>pkg>ignore: got %q, want all", got)
	}
	pkgAndIgnore := cfgWith(t, dir, []string{"c"}, nil, nil, []string{`^c/`})
	if got := Impact(pkgAndIgnore, dir, base, headRev); got != "example.com/m/b example.com/m/c" {
		t.Errorf("pkg>ignore: got %q, want ./b ./c", got)
	}
}

// §V16 tier-1 (§T584): invalid regexes are configuration errors, not silent no-ops.
func TestConfigInvalidRegex(t *testing.T) {
	if _, err := newConfig("example.com/m", nil, []string{"["}, nil, nil, nil); err == nil || !strings.Contains(err.Error(), "ignore-regex") {
		t.Errorf("invalid ignore-regex must error with the flag name, got %v", err)
	}
	if _, err := newConfig("example.com/m", nil, nil, []string{"("}, nil, nil); err == nil {
		t.Error("invalid full-rerun-regex must error")
	}
}

// §V16 tier-1 (§T584): the binary Classify mode honors the same configuration.
func TestConfigClassifyHonorsRules(t *testing.T) {
	head := map[string]string{"c/gen.s": "// new asm\n"}
	dir, base, headRev := scratchRepo(t, modBase, head)
	if got := Classify(defaultConfig(dir), dir, base, headRev); got != Code {
		t.Fatalf("baseline: got %q, want code", got)
	}
	if got := Classify(cfgWith(t, dir, []string{"c/gen.s"}, nil, nil, nil), dir, base, headRev); got != DocsOnly {
		t.Errorf("ignored path in classify: got %q, want docs-only", got)
	}
}
