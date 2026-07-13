package main

import (
	"fmt"
	"os/exec"
	"sort"
	"strings"
)

// Validate cross-checks the -impact selection against an INDEPENDENT oracle: the Go
// toolchain's own dependency graph (§T583, §V27). `go list -test -deps` resolves build
// tags, test binaries and synthesized test-variant packages exactly as the compiler
// will — swept over GOOS linux/darwin/windows so tag-gated edges join the union.
//
// The oracle recomputes which packages' test binaries (transitively) depend on the
// changed packages and compares:
//
//   - selected ⊉ oracle → UNDER-SELECTION: the selector would skip tests that link
//     changed code. Release-blocking selector bug — ok=false, missing listed.
//   - selected ⊃ oracle → over-selection: sound but wasteful; listed as excess, ok=true.
//
// Scope: the oracle replaces the GRAPH and CLOSURE (the risky, hand-built parts); the
// file→package attribution and the None/All verdicts are shared with -impact and are
// covered by unit tests instead. All/None short-circuit: All is a superset of every
// oracle answer; None is valid iff no package changed at all.
func Validate(cfg *config, dir, base, head string) (report string, ok bool) {
	selected := Impact(cfg, dir, base, head)
	if selected == All {
		return "validate: impact=all — trivially sound (superset of any oracle set)", true
	}
	g, err := buildGraph(dir)
	if err != nil {
		return "validate: graph unbuildable — impact fell open to all? got: " + selected, false
	}
	propagate, testOnly, verdict := g.changedSets(cfg, dir, base, head)
	if verdict == All {
		return "validate: attribution says all but impact printed " + selected, selected == All
	}
	changed := map[string]bool{}
	for p := range propagate {
		changed[p] = true
	}
	if selected == None {
		if len(changed) == 0 && len(testOnly) == 0 {
			return "validate: impact=none and no package changed — sound", true
		}
		return "validate: impact=none but packages changed — UNDER-SELECTION", false
	}
	oracle, err := g.oracleAffected(dir, changed, testOnly)
	if err != nil {
		return fmt.Sprintf("validate: oracle unavailable (%v) — cannot judge", err), false
	}
	sel := map[string]bool{}
	for _, p := range strings.Fields(selected) {
		sel[strings.TrimPrefix(p, "./")] = true
	}
	var missing, excess []string
	for p := range oracle {
		if !sel[p] {
			missing = append(missing, p)
		}
	}
	for p := range sel {
		if !oracle[p] {
			excess = append(excess, p)
		}
	}
	sort.Strings(missing)
	sort.Strings(excess)
	switch {
	case len(missing) > 0:
		return fmt.Sprintf("validate: UNDER-SELECTION — missing %d: %s (excess %d)",
			len(missing), strings.Join(missing, " "), len(excess)), false
	case len(excess) > 0:
		return fmt.Sprintf("validate: sound, over-selection of %d: %s", len(excess), strings.Join(excess, " ")), true
	}
	return fmt.Sprintf("validate: exact — %d packages, 0 missing, 0 excess", len(oracle)), true
}

// oracleAffected asks the toolchain which packages' test binaries depend on the
// changed set: for every GOOS, `go list -test -deps` yields each package (and its
// synthesized test variants) with its full transitive Deps; a package is affected iff
// any changed package appears there. Test-only-changed packages join directly.
func (g *moduleGraph) oracleAffected(dir string, changed, testOnly map[string]bool) (map[string]bool, error) {
	out := map[string]bool{}
	for p := range changed {
		out[p] = true
	}
	for p := range testOnly {
		out[p] = true
	}
	sawOS := false
	for _, goos := range []string{"linux", "darwin", "windows"} {
		deps, err := g.goListDeps(dir, goos)
		if err != nil {
			continue // a GOOS that cannot load (exotic constraint) must not kill the union
		}
		sawOS = true
		for pkg, ds := range deps {
			if out[pkg] {
				continue
			}
			for d := range ds {
				if changed[d] {
					out[pkg] = true
					break
				}
			}
		}
	}
	if !sawOS {
		return nil, fmt.Errorf("go list failed on every GOOS")
	}
	return out, nil
}

// goListDeps runs `go list -test -deps -f {{.ImportPath}}|{{join .Deps ","}}` under
// one GOOS and folds test-variant entries ("pkg.test", "pkg [pkg.test]") onto their
// base package, returning rel-package → set of rel-package transitive deps.
func (g *moduleGraph) goListDeps(dir, goos string) (map[string]map[string]bool, error) {
	cmd := exec.Command("go", "list", "-e", "-test", "-deps",
		"-f", "{{.ImportPath}}|{{range .Deps}}{{.}},{{end}}", "./...")
	cmd.Dir = dir
	cmd.Env = append(cmd.Environ(), "GOOS="+goos, "CGO_ENABLED=0")
	raw, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	deps := map[string]map[string]bool{}
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		name, list, found := strings.Cut(line, "|")
		if !found {
			continue
		}
		pkg, ok := g.relOf(baseImportPath(name))
		if !ok {
			continue
		}
		if deps[pkg] == nil {
			deps[pkg] = map[string]bool{}
		}
		for _, d := range strings.Split(list, ",") {
			if rel, ok := g.relOf(baseImportPath(d)); ok && rel != pkg {
				deps[pkg][rel] = true
			}
		}
	}
	return deps, nil
}

// baseImportPath folds go list -test variants onto their base package:
// "pkg.test" and "pkg [pkg.test]" → "pkg"; "pkg_test [pkg.test]" → "pkg".
func baseImportPath(name string) string {
	if i := strings.Index(name, " ["); i >= 0 {
		name = name[:i]
	}
	name = strings.TrimSuffix(name, ".test")
	name = strings.TrimSuffix(name, "_test")
	return name
}

// relOf maps a module-internal import path to its rel package dir ("." for root).
func (g *moduleGraph) relOf(ip string) (string, bool) {
	if ip == g.modPath {
		return ".", true
	}
	if rest, ok := strings.CutPrefix(ip, g.modPath+"/"); ok {
		return rest, true
	}
	return "", false
}
