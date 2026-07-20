package main

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

// config makes the CLI generic (§T584): nothing about a particular repository is
// hardcoded — which paths are documentation, which force a full rerun, and which force
// their package's tests are all rules, and every rule is a flag. The built-in defaults
// reproduce this repository's behavior and are expressed in the SAME vocabulary, so
// -no-default-rules plus explicit flags configures the tool for any other codebase.
//
// Precedence per changed path: full-rerun > pkg-rerun > ignore > content analysis.
//
// STEP 1 of every run absolutizes all configured -ignore paths (relative paths resolve
// against the repository root) and every changed path before comparing, so relative
// and absolute spellings are interchangeable. The regex flags match the
// slash-separated repository-RELATIVE path instead — patterns stay portable across
// machines.
type config struct {
	root        string           // absolute repository root
	ignoreParts []string         // absolutized -ignore entries (files or whole dirs)
	ignoreRe    []*regexp.Regexp // -ignore-regex: relative path matches → no impact
	fullRe      []*regexp.Regexp // -full-rerun-regex: match → verdict all
	pkgRe       []*regexp.Regexp // -pkg-rerun-regex: match → owning package reruns
	alwaysRun   []string         // -always-run: module-relative package dirs added to every NON-EMPTY selection (§T893: source-walking meta-tests the import closure can never reach)
}

// stringList is a repeatable flag.Value collecting strings in order.
type stringList []string

func (s *stringList) String() string { return strings.Join(*s, ",") }
func (s *stringList) Set(v string) error {
	*s = append(*s, v)
	return nil
}

// defaultRules returns this repository's rules expressed as config vocabulary —
// exactly what the hardcoded classifier used to do (§T584).
func defaultRules() (ignore, ignoreRe, fullRe, pkgRe, alwaysRun []string) {
	ignore = []string{"docs", ".claude", "spec"} // + .claude/** (skills/workflows/memory) + spec/** (the §V40 source tree — pure markdown, never embedded; §T585 root-markdown covers LICENSE.md)
	ignoreRe = []string{`^[^/]+\.(md|txt)$`}     // root-level markdown/text (deeper ones can be embedded, §B50)
	fullRe = []string{`^go\.sum$`, `^Makefile$`, `^\.github/`}
	// The whole-tree meta-tests walk SOURCE/markdown they have no import edge to, so
	// the reverse closure can never select them (§B98: they rot red while CI stays
	// green on nlp/nn/… pushes). The -always-run mechanism forces configured packages
	// into every non-empty selection to close that seam (a pure-docs diff still stays
	// zero-runner, §C16 EXC2). A package may only be listed here once it is GREEN on
	// the committed tree, else it fails CI on the FIRST push.
	//   - internal/speccheck (§V36 SPEC-integrity: id-uniqueness, §T-membership) is
	//     GREEN today — SPEC.md is clean — so it is enabled and now gates every push
	//     (§T886/§T893).
	//   - internal/perfscan (T920 per-element hot-loop finder) is GREEN — its detector
	//     tests + TestScanWholeModule (parses every first-party .go, asserts the scan
	//     completes and still finds candidates) pass on the committed tree — so it is
	//     enabled. It gates the TOOL, not the codebase: the scan stays ADVISORY (no
	//     candidate-count assertion), so a legitimately cold per-element loop never
	//     reddens CI, but a detector that panics/regresses on real source does.
	//   - internal/apicheck (§V19 doc/example gate) is now GREEN — the per-package
	//     documentation pass added the missing godocs/Examples and justified-allowlisted
	//     the internal transformer blocks/caches — so it is enabled and gates every push
	//     (a new undocumented exported public symbol reddens CI). Note apicheck parses
	//     source (no compile), so it is orthogonal to the cgo/cuda build lanes.
	//   - internal/mdlint (every *.md) stays OUT until green: it fails on
	//     .claude/memory + SPEC-worker-*.md tilde debt (T889). Add it here once green.
	//   - internal/docgraph (§V39 dangling refs + §V41 render-sync: spec/ is the
	//     source, SPEC.md a generated view) is GREEN and hermetic — enabled so a
	//     hand-edited rendered view goes red on the next non-empty selection.
	alwaysRun = []string{"internal/speccheck", "internal/perfscan", "internal/docgraph", "internal/apicheck"}
	return ignore, ignoreRe, fullRe, pkgRe, alwaysRun
}

// newConfig absolutizes the path rules against root and compiles the regex rules.
// An invalid regex or unresolvable path is a configuration error (usage, exit 2).
func newConfig(root string, ignore, ignoreRe, fullRe, pkgRe, alwaysRun []string) (*config, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("cichange: resolve root %q: %w", root, err)
	}
	c := &config{root: absRoot}
	for _, ar := range alwaysRun {
		if ar = strings.TrimSpace(filepath.ToSlash(ar)); ar != "" {
			c.alwaysRun = append(c.alwaysRun, ar)
		}
	}
	for _, p := range ignore {
		ap := p
		if !filepath.IsAbs(ap) {
			ap = filepath.Join(absRoot, ap)
		}
		c.ignoreParts = append(c.ignoreParts, filepath.Clean(ap))
	}
	compile := func(pats []string, dst *[]*regexp.Regexp, flagName string) error {
		for _, pat := range pats {
			re, err := regexp.Compile(pat)
			if err != nil {
				return fmt.Errorf("cichange: -%s %q: %w", flagName, pat, err)
			}
			*dst = append(*dst, re)
		}
		return nil
	}
	if err := compile(ignoreRe, &c.ignoreRe, "ignore-regex"); err != nil {
		return nil, err
	}
	if err := compile(fullRe, &c.fullRe, "full-rerun-regex"); err != nil {
		return nil, err
	}
	if err := compile(pkgRe, &c.pkgRe, "pkg-rerun-regex"); err != nil {
		return nil, err
	}
	return c, nil
}

// defaultConfig is the zero-flag configuration rooted at dir.
func defaultConfig(dir string) *config {
	ig, igRe, fRe, pRe, aRun := defaultRules()
	c, err := newConfig(dir, ig, igRe, fRe, pRe, aRun)
	if err != nil {
		panic(err) // defaults are constants: cannot fail
	}
	return c
}

// ignoredBy reports whether the repo-relative changed path is ruled out of impact —
// under an absolutized -ignore entry, or matching an -ignore-regex — and WHICH rule
// did it (§T587: every decision in the log names its rule).
func (c *config) ignoredBy(rel string) (string, bool) {
	abs := filepath.Clean(filepath.Join(c.root, filepath.FromSlash(rel)))
	for _, ip := range c.ignoreParts {
		if abs == ip || strings.HasPrefix(abs, ip+string(filepath.Separator)) {
			return "-ignore " + ip, true
		}
	}
	for _, re := range c.ignoreRe {
		if re.MatchString(rel) {
			return "-ignore-regex '" + re.String() + "'", true
		}
	}
	return "", false
}

// ignored is ignoredBy without the rule name.
func (c *config) ignored(rel string) bool {
	_, ok := c.ignoredBy(rel)
	return ok
}

// fullRerunBy reports whether the changed path forces the full suite, and which rule.
func (c *config) fullRerunBy(rel string) (string, bool) {
	for _, re := range c.fullRe {
		if re.MatchString(rel) {
			return "-full-rerun-regex '" + re.String() + "'", true
		}
	}
	return "", false
}

// pkgRerunBy reports whether the changed path forces its owning package's tests
// regardless of content analysis (comment-only Go edits included), and which rule.
func (c *config) pkgRerunBy(rel string) (string, bool) {
	for _, re := range c.pkgRe {
		if re.MatchString(rel) {
			return "-pkg-rerun-regex '" + re.String() + "'", true
		}
	}
	return "", false
}
