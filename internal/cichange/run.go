package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

// Run is the transparent end-to-end mode (§T587): it reports every decision the
// selector makes — per changed file WITH the rule that decided it, per skipped package
// WITH the reason, the test functions each selected package contributes, and the EXACT
// `go test` command — and then, as its LAST STEP, executes that command streaming the
// full go test output unswallowed. The returned code is go test's exit code (0 when
// nothing needs to run). Every printed sequence is explicitly sorted, so identical
// inputs produce byte-identical reports — no map-iteration order anywhere.
func Run(cfg *config, dir, base, head string, goTestArgs []string, w io.Writer) int {
	fmt.Fprintf(w, "== cichange run: %s..%s ==\n", base, head)
	g, err := buildGraph(dir)
	if err != nil {
		fmt.Fprintf(w, "graph unbuildable (%v) → FULL RERUN (§V26)\n", err)
		return execGoTest(dir, goTestArgs, []string{"./..."}, w)
	}
	propagate, testOnly, verdict, trace := g.changedSetsTraced(cfg, dir, base, head)

	fmt.Fprintln(w, "\n== changed files (decision + rule) ==")
	for _, line := range trace {
		fmt.Fprintln(w, "  "+line)
	}

	allPkgs := make([]string, 0, len(g.pkgs))
	for p := range g.pkgs {
		allPkgs = append(allPkgs, p)
	}
	sort.Strings(allPkgs)

	var selected []string
	reasons := map[string]string{}
	switch verdict {
	case None:
		fmt.Fprintln(w, "\n== selection ==")
		fmt.Fprintf(w, "  affected: none — all %d packages skipped (documentation-only change)\n", len(allPkgs))
		fmt.Fprintln(w, "\n== command ==\n  (no tests to run)")
		return 0
	case All:
		selected = allPkgs
		for _, p := range allPkgs {
			reasons[p] = "selected: full rerun"
		}
	default:
		affected := g.closure(propagate)
		for p := range affected {
			reasons[p] = "selected: depends on a changed package"
		}
		for p := range propagate {
			reasons[p] = "selected: changed"
		}
		for p := range testOnly {
			if _, ok := reasons[p]; !ok {
				reasons[p] = "selected: its own tests changed"
			}
			affected[p] = true
		}
		// refine: test-edge selections (in closure but only via _test.go imports)
		for p := range affected {
			if !propagate[p] && !testOnly[p] && reasons[p] == "selected: depends on a changed package" {
				if !reachesViaCode(g, p, propagate) {
					reasons[p] = "selected: its _test.go files import a changed package"
				}
			}
			selected = append(selected, p)
		}
		sort.Strings(selected)
	}

	fmt.Fprintln(w, "\n== selection ==")
	fmt.Fprintf(w, "  affected: %d, skipped: %d\n", len(selected), len(allPkgs)-len(selected))
	for _, p := range selected { // go-test-shaped: status column, absolute import path
		fmt.Fprintf(w, "run \t%s\t%s\n", g.importPath(p), strings.TrimPrefix(reasons[p], "selected: "))
	}
	for _, p := range allPkgs {
		if _, ok := reasons[p]; !ok {
			fmt.Fprintf(w, "skip\t%s\t%s\n", g.importPath(p), "no dependency path from any changed package")
		}
	}

	if len(selected) == 0 {
		fmt.Fprintln(w, "\n== command ==\n  (no tests to run — nothing selected)")
		return 0
	}

	fmt.Fprintln(w, "\n== test functions in selected packages ==")
	for _, p := range selected {
		funcs := testFuncs(filepath.Join(dir, filepath.FromSlash(p)))
		if len(funcs) == 0 {
			fmt.Fprintf(w, "  %s: (no test functions)\n", g.importPath(p))
			continue
		}
		fmt.Fprintf(w, "  %s: %s\n", g.importPath(p), strings.Join(funcs, " "))
	}

	paths := make([]string, len(selected))
	for i, p := range selected {
		paths[i] = g.importPath(p) // absolute import paths, as go test prints them
	}
	return execGoTest(dir, goTestArgs, paths, w)
}

// reachesViaCode reports whether pkg reaches any member of changed through NON-test
// import edges (used only to word the selection reason precisely).
func reachesViaCode(g *moduleGraph, pkg string, changed map[string]bool) bool {
	seen := map[string]bool{}
	stack := []string{pkg}
	for len(stack) > 0 {
		p := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[p] {
			continue
		}
		seen[p] = true
		for dep := range g.imports[p] {
			if changed[dep] {
				return true
			}
			stack = append(stack, dep)
		}
	}
	return false
}

// testFuncs lists the Test/Benchmark/Fuzz/Example functions declared in a package
// directory's _test.go files, sorted — the log shows exactly what can run (§T587).
func testFuncs(pkgDir string) []string {
	entries, err := os.ReadDir(pkgDir)
	if err != nil {
		return nil
	}
	fset := token.NewFileSet()
	var out []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, filepath.Join(pkgDir, e.Name()), nil, parser.SkipObjectResolution)
		if err != nil {
			continue
		}
		for _, d := range f.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok || fn.Recv != nil {
				continue
			}
			name := fn.Name.Name
			for _, prefix := range []string{"Test", "Benchmark", "Fuzz", "Example"} {
				if strings.HasPrefix(name, prefix) {
					out = append(out, name)
					break
				}
			}
		}
	}
	sort.Strings(out)
	return out
}

// execGoTest prints the exact command and then runs it with stdout/stderr streamed
// through unmodified — the CLI never swallows test output (§T587).
func execGoTest(dir string, args, pkgs []string, w io.Writer) int {
	full := append(append([]string{"test"}, args...), pkgs...)
	fmt.Fprintln(w, "\n== command ==\n  go "+strings.Join(full, " "))
	fmt.Fprintln(w, "\n== go test output ==")
	cmd := exec.Command("go", full...)
	cmd.Dir = dir
	cmd.Stdout = w
	cmd.Stderr = w
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		fmt.Fprintf(w, "cichange: go test could not start: %v\n", err)
		return 1
	}
	return 0
}
