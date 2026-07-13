package main

import (
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Impact verdicts besides a concrete package list.
const (
	// None means no test needs to run (every change is documentation).
	None = "none"
	// All means the full suite must run (a change with global or unknowable reach).
	All = "all"
)

// Impact computes WHICH packages' tests are affected by the diff between two revisions
// of the git repository at dir (§T579, the hyper-optimized CI selector). It returns:
//
//   - None: every change is documentation — nothing to test;
//   - All: a change with global or unknowable reach (go.mod, workflows, Makefile,
//     parse failures, files outside any package, deleted packages) — fail open, §V26;
//   - otherwise a space-separated, sorted list of package paths relative to the module
//     root ("." for the root package), ready for `go test`.
//
// Method (§R235, govulncheck's -scan package mode applied to test selection): parse
// EVERY .go file of the head checkout with go/parser in ImportsOnly mode — including
// _test.go files and files behind build tags (tag-UNION, so register_darwin.go's blank
// backend imports count on every platform) — into a package import graph. Map each
// changed file to its owning package (non-Go files — C, assembly, Metal shaders,
// SPIR-V, testdata, embedded data — attach to the nearest ancestor package). Take the
// REVERSE transitive closure of the code-changed packages; a package whose change is
// confined to _test.go files joins the set WITHOUT propagating (test files cannot be
// imported). Comment-only .go changes are excluded by the same AST comparison the
// docs-only gate uses. Soundness at package granularity is inherited from the import
// relation: dynamic interface dispatch requires the implementation to be linked, and
// linkage requires an import path to it.
func Impact(cfg *config, dir, base, head string) string {
	g, err := buildGraph(dir)
	if err != nil {
		return All
	}
	propagate, testOnly, verdict := g.changedSets(cfg, dir, base, head)
	if verdict != "" {
		return verdict
	}
	affected := g.closure(propagate)
	for p := range testOnly {
		affected[p] = true
	}
	if len(affected) == 0 {
		return None
	}
	if len(affected) == len(g.pkgs) {
		return All
	}
	list := make([]string, 0, len(affected))
	for p := range affected {
		if p == "." {
			list = append(list, ".")
		} else {
			list = append(list, "./"+p)
		}
	}
	sort.Strings(list)
	return strings.Join(list, " ")
}

// changedSets attributes every file of the base..head diff: propagate = packages whose
// non-test code changed, testOnly = packages with only _test.go changes. verdict is ""
// for a normal result, or None/All when the diff resolves without a closure.
func (g *moduleGraph) changedSets(cfg *config, dir, base, head string) (propagate, testOnly map[string]bool, verdict string) {
	p, t, v, _ := g.changedSetsTraced(cfg, dir, base, head)
	return p, t, v
}

// changedSetsTraced is changedSets plus a human-readable per-file decision trace
// (§T587): one line per changed path naming the decision AND the rule that made it,
// in git-diff order (deterministic). On an All verdict the trace ends with the line
// that forced it.
func (g *moduleGraph) changedSetsTraced(cfg *config, dir, base, head string) (propagate, testOnly map[string]bool, verdict string, trace []string) {
	out, err := gitRun(dir, "diff", "--name-status", "-z", base, head)
	if err != nil {
		return nil, nil, All, []string{"(diff unavailable — force push or shallow clone → all)"}
	}
	fields := strings.Split(strings.TrimRight(string(out), "\x00"), "\x00")
	if len(fields) == 1 && fields[0] == "" {
		return nil, nil, None, []string{"(empty diff)"}
	}
	propagate = map[string]bool{}
	testOnly = map[string]bool{}
	for i := 0; i < len(fields); {
		status := fields[i]
		if status == "" || i+1 >= len(fields) {
			return nil, nil, All, append(trace, "(malformed diff record → all)")
		}
		paths := []string{fields[i+1]}
		i += 2
		if status[0] == 'R' || status[0] == 'C' { // renames/copies carry a second path
			if i >= len(fields) {
				return nil, nil, All, append(trace, "(malformed rename record → all)")
			}
			paths = append(paths, fields[i])
			i++
		}
		for _, p := range paths {
			outcome, why := g.classifyChanged(cfg, dir, base, head, status, p, propagate, testOnly)
			trace = append(trace, p+": "+why)
			if outcome == impactGlobal {
				return nil, nil, All, trace
			}
		}
	}
	return propagate, testOnly, "", trace
}

// classifyChanged outcomes.
const (
	impactHandled = iota // attributed to a package (or ignorable)
	impactGlobal         // forces All
)

// go.mod is not a configured full-rerun rule — it gets precise dependency-diff
// handling (§T582); go.sum stays a default full-rerun pattern (a sum change
// without a matching go.mod change is anomalous).

// depChangeSeeds handles a go.mod change (§T582): if only require VERSIONS changed or
// modules were added, the packages importing those modules seed the closure — code
// importers propagate, test-only importers are selected without propagating. Any other
// difference (module line, go/toolchain, replace, exclude, malformed) → global. The
// returned reason names the changed modules and their seeds, sorted (§T587).
func (g *moduleGraph) depChangeSeeds(dir, base, head string, propagate, testOnly map[string]bool) (int, string) {
	baseSrc, err := gitRun(dir, "show", base+":go.mod")
	if err != nil {
		return impactGlobal, "FULL RERUN (go.mod unreadable at base)"
	}
	headSrc, err := gitRun(dir, "show", head+":go.mod")
	if err != nil {
		return impactGlobal, "FULL RERUN (go.mod unreadable at head)"
	}
	baseReq, baseRest := parseGoMod(string(baseSrc))
	headReq, headRest := parseGoMod(string(headSrc))
	if strings.Join(baseRest, "\n") != strings.Join(headRest, "\n") {
		return impactGlobal, "FULL RERUN (go.mod changed beyond require versions: module/go/replace/exclude)"
	}
	mods := changedModules(baseReq, headReq)
	sort.Strings(mods)
	var parts []string
	for _, mod := range mods {
		var seeds []string
		for pkg, ext := range g.extImports {
			for ip := range ext {
				if modOwns(mod, ip) {
					propagate[pkg] = true
					seeds = append(seeds, pkg)
					break
				}
			}
		}
		for pkg, ext := range g.extTestImports {
			for ip := range ext {
				if modOwns(mod, ip) {
					testOnly[pkg] = true
					seeds = append(seeds, pkg+" (test-only)")
					break
				}
			}
		}
		sort.Strings(seeds)
		parts = append(parts, mod+" → seeds: "+strings.Join(seeds, " "))
	}
	if len(parts) == 0 {
		return impactHandled, "go.mod dependency diff: no changed require entries (removals seed nothing)"
	}
	return impactHandled, "go.mod dependency diff: " + strings.Join(parts, "; ")
}

// classifyChanged attributes one changed file to the propagate/testOnly sets per the
// configured rules (§T584; precedence full-rerun > pkg-rerun > ignore > analysis),
// reports impactGlobal for anything with unbounded reach (§V26), and names the
// decision plus the rule that made it (§T587).
func (g *moduleGraph) classifyChanged(cfg *config, dir, base, head, status, p string, propagate, testOnly map[string]bool) (int, string) {
	if rule, ok := cfg.fullRerunBy(p); ok {
		return impactGlobal, "FULL RERUN (rule: " + rule + ")"
	}
	if rule, ok := cfg.pkgRerunBy(p); ok {
		pkg, found := g.pkgFor(p)
		if !found {
			return impactGlobal, "FULL RERUN (rule: " + rule + ", but no owning package → all)"
		}
		propagate[pkg] = true
		return impactHandled, "package rerun → " + pkg + " (rule: " + rule + ")"
	}
	if rule, ok := cfg.ignoredBy(p); ok {
		return impactHandled, "ignored (rule: " + rule + ")"
	}
	switch {
	case p == "go.mod":
		outcome, why := g.depChangeSeeds(dir, base, head, propagate, testOnly)
		return outcome, why
	case strings.HasSuffix(p, ".go"):
		// a .go file's owning package is EXACTLY its directory — no ancestor walk,
		// else a deleted package would silently attribute to its parent.
		pkg := filepath.ToSlash(filepath.Dir(p))
		if !g.pkgs[pkg] {
			return impactGlobal, "FULL RERUN (package " + pkg + " gone at head: importer breakage unknowable)"
		}
		if status[0] == 'M' && goCodeUnchanged(dir, base, head, p) {
			return impactHandled, "ignored (comment-only Go change: AST identical after comment strip)"
		}
		if strings.HasSuffix(p, "_test.go") {
			testOnly[pkg] = true
			return impactHandled, "test-only code → " + pkg + " (test files are not importable: no propagation)"
		}
		propagate[pkg] = true
		return impactHandled, "code → " + pkg
	default:
		// C, assembly, shaders, testdata, embedded data: the nearest ancestor
		// package owns it (cgo/embed cannot reach outside the package subtree).
		pkg, ok := g.pkgFor(p)
		if !ok {
			return impactGlobal, "FULL RERUN (outside every package: reach unknowable)"
		}
		propagate[pkg] = true
		return impactHandled, "package asset → " + pkg
	}
}

// moduleGraph is the package import graph of the head checkout: package dirs relative
// to the module root ("." for the root package) and their module-internal imports.
// Imports from _test.go files are tracked SEPARATELY: they make the importing package's
// tests depend on the target, but not its compiled code — so in the reverse closure a
// test edge selects the importer without propagating further through it.
type moduleGraph struct {
	modPath        string
	pkgs           map[string]bool
	imports        map[string]map[string]bool // edges from non-test files
	testImports    map[string]map[string]bool // edges from _test.go files
	extImports     map[string]map[string]bool // pkg → external import paths (non-test)
	extTestImports map[string]map[string]bool // pkg → external import paths (_test.go)
}

// buildGraph walks the module at root and parses every .go file (ImportsOnly,
// tag-union: build constraints are deliberately ignored) into the import graph.
func buildGraph(root string) (*moduleGraph, error) {
	modBytes, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		return nil, err
	}
	modPath := ""
	for _, line := range strings.Split(string(modBytes), "\n") {
		if rest, ok := strings.CutPrefix(strings.TrimSpace(line), "module "); ok {
			modPath = strings.TrimSpace(rest)
			break
		}
	}
	if modPath == "" {
		return nil, fmt.Errorf("cichange: no module line in go.mod")
	}
	g := &moduleGraph{
		modPath:        modPath,
		pkgs:           map[string]bool{},
		imports:        map[string]map[string]bool{},
		testImports:    map[string]map[string]bool{},
		extImports:     map[string]map[string]bool{},
		extTestImports: map[string]map[string]bool{},
	}
	fset := token.NewFileSet()
	err = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		name := d.Name()
		if d.IsDir() {
			if path != root && (name == ".git" || name == "testdata" || strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_")) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(name, ".go") {
			return nil
		}
		rel, err := filepath.Rel(root, filepath.Dir(path))
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		f, err := parser.ParseFile(fset, path, nil, parser.ImportsOnly)
		if err != nil {
			return fmt.Errorf("cichange: parse %s: %w", path, err)
		}
		g.pkgs[rel] = true
		edges, ext := g.imports, g.extImports
		if strings.HasSuffix(name, "_test.go") {
			edges, ext = g.testImports, g.extTestImports
		}
		if edges[rel] == nil {
			edges[rel] = map[string]bool{}
		}
		if ext[rel] == nil {
			ext[rel] = map[string]bool{}
		}
		for _, imp := range f.Imports {
			ip := strings.Trim(imp.Path.Value, `"`)
			if ip == g.modPath {
				edges[rel]["."] = true
			} else if rest, ok := strings.CutPrefix(ip, g.modPath+"/"); ok {
				edges[rel][rest] = true
			} else if strings.Contains(strings.SplitN(ip, "/", 2)[0], ".") {
				ext[rel][ip] = true // dotted first element = external module, not std
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return g, nil
}

// pkgFor maps a changed file path (slash-separated, module-relative) to its owning
// package: the file's directory if that is a package, else the nearest ancestor
// package (C sources, shaders, testdata, embedded data live below their package).
func (g *moduleGraph) pkgFor(file string) (string, bool) {
	d := filepath.ToSlash(filepath.Dir(file))
	for {
		if g.pkgs[d] {
			return d, true
		}
		if d == "." {
			return "", false
		}
		parent := filepath.ToSlash(filepath.Dir(d))
		if parent == d {
			return "", false
		}
		d = parent
	}
}

// closure returns changed plus every package that transitively imports one of them
// through CODE edges, plus — one hop, non-propagating — every package whose _test.go
// files import a member of that set (their test binaries link the changed code, their
// compiled packages do not, so their own importers stay unaffected).
func (g *moduleGraph) closure(changed map[string]bool) map[string]bool {
	rev := map[string][]string{}
	for from, tos := range g.imports {
		for to := range tos {
			rev[to] = append(rev[to], from)
		}
	}
	out := map[string]bool{}
	var stack []string
	for p := range changed {
		stack = append(stack, p)
	}
	for len(stack) > 0 {
		p := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if out[p] {
			continue
		}
		out[p] = true
		stack = append(stack, rev[p]...)
	}
	for from, tos := range g.testImports {
		if out[from] {
			continue
		}
		for to := range tos {
			if out[to] {
				out[from] = true
				break
			}
		}
	}
	return out
}
