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
func defaultRules() (ignore, ignoreRe, fullRe, pkgRe []string) {
	ignore = []string{"docs", ".claude"}     // + .claude/** (skills/workflows/memory — no Go, never affects build/tests; §T585 root-markdown covers LICENSE.md)
	ignoreRe = []string{`^[^/]+\.(md|txt)$`} // root-level markdown/text (deeper ones can be embedded, §B50)
	fullRe = []string{`^go\.sum$`, `^Makefile$`, `^\.github/`}
	return ignore, ignoreRe, fullRe, pkgRe
}

// newConfig absolutizes the path rules against root and compiles the regex rules.
// An invalid regex or unresolvable path is a configuration error (usage, exit 2).
func newConfig(root string, ignore, ignoreRe, fullRe, pkgRe []string) (*config, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("cichange: resolve root %q: %w", root, err)
	}
	c := &config{root: absRoot}
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
	ig, igRe, fRe, pRe := defaultRules()
	c, err := newConfig(dir, ig, igRe, fRe, pRe)
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
