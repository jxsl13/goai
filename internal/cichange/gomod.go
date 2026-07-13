package main

import (
	"bufio"
	"strings"
)

// go.mod dependency-change selection (§T582): a version bump of an external module
// must not trigger the full suite — only the packages that (transitively) depend on
// that module are affected. parseGoMod splits a go.mod into its REQUIRE set
// (module → version) and the REST (module line, go/toolchain directives, replace,
// exclude, retract — normalized, comment-free). Two revisions whose rest differs
// force a full run (§V26: replace/exclude/toolchain semantics are not modeled);
// otherwise the changed/added require entries seed the impact closure via the
// external-import index.
func parseGoMod(src string) (require map[string]string, rest []string) {
	require = map[string]string{}
	inBlock := false
	sc := bufio.NewScanner(strings.NewReader(src))
	for sc.Scan() {
		line := sc.Text()
		if i := strings.Index(line, "//"); i >= 0 {
			line = line[:i]
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		switch {
		case inBlock:
			if line == ")" {
				inBlock = false
				continue
			}
			if f := strings.Fields(line); len(f) == 2 {
				require[f[0]] = f[1]
			} else {
				rest = append(rest, line) // malformed entry: make it visible in rest
			}
		case line == "require (":
			inBlock = true
		case strings.HasPrefix(line, "require "):
			if f := strings.Fields(strings.TrimPrefix(line, "require ")); len(f) == 2 {
				require[f[0]] = f[1]
			} else {
				rest = append(rest, line)
			}
		default:
			rest = append(rest, line)
		}
	}
	return require, rest
}

// changedModules diffs two require sets and returns the module paths whose version
// changed or that were added in head. Removed-only modules seed nothing: their former
// importers must have dropped the import in the same diff (otherwise the build breaks
// and the always-full `go vet ./...` backstop catches it).
func changedModules(base, head map[string]string) []string {
	var out []string
	for mod, v := range head {
		if bv, ok := base[mod]; !ok || bv != v {
			out = append(out, mod)
		}
	}
	return out
}

// modOwns reports whether importPath belongs to the module at modPath.
func modOwns(modPath, importPath string) bool {
	return importPath == modPath || strings.HasPrefix(importPath, modPath+"/")
}
