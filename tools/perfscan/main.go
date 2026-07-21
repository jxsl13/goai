// Command perfscan is a static scanner for the recurring performance
// anti-patterns that repeatedly turned up in GoAI's hot paths. Each finding
// points at a concrete, benchmark-verified fix that has already paid off
// elsewhere in the tree (see PATTERNS.md for the evidence and speedups).
//
// It is deliberately a LINT, not a fixer: every hit is a CANDIDATE that must be
// confirmed with a pre/post benchmark before changing anything — some hits are
// on cold paths (one-time init, eval-only) where the fix is not worth it. The
// scanner exists so the patterns are institutional knowledge, not something we
// have to rediscover by profiling each time.
//
// Usage:
//
//	go run ./tools/perfscan [packages...]   # default: ./...
//	go run ./tools/perfscan -tests          # also scan _test.go files
//	go run ./tools/perfscan -json           # machine-readable output
//
// Exit status is 0 regardless of findings (advisory tool); use -strict to exit
// non-zero when any finding is reported (for opt-in CI gating).
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// finding is one reported candidate site.
type finding struct {
	File    string `json:"file"`
	Line    int    `json:"line"`
	Pattern string `json:"pattern"`
	Detail  string `json:"detail"`
	Fix     string `json:"fix"`
}

// perElemVisitors are helpers that invoke their function-literal argument ONCE
// PER ELEMENT. Passing a closure to them means an indirect call per element even
// on their contiguous fast branch — the exact cost that made SWA/EMA/GradAccum/
// GradFn 1.6–4.2× slower than a raw-slice loop (PATTERNS.md P1). Extend this list
// whenever a new per-element visitor helper is introduced.
var perElemVisitors = map[string]bool{
	"readGen":  true,
	"fillGen":  true,
	"visitGen": true,
	"forEach":  true,
	"eachElem": true,
}

// closureSorters are sort entry points whose comparator is a function value. On
// a large slice keyed by a monotonic float/int, the per-comparison indirect call
// dominates — replacing it with a radix sort on the key bits made top-p / typical
// sampling 1.9–2.25× faster (PATTERNS.md P2).
var closureSorters = map[string]bool{
	"SortFunc":       true, // slices.SortFunc
	"SortStableFunc": true, // slices.SortStableFunc
	"Slice":          true, // sort.Slice
	"SliceStable":    true, // sort.SliceStable
	"Sort":           true, // sort.Sort (sort.Interface — Less is a method call)
}

func main() {
	var (
		withTests bool
		asJSON    bool
		strict    bool
	)
	flag.BoolVar(&withTests, "tests", false, "also scan _test.go files")
	flag.BoolVar(&asJSON, "json", false, "emit findings as JSON")
	flag.BoolVar(&strict, "strict", false, "exit non-zero if any finding is reported")
	flag.Parse()

	roots := flag.Args()
	if len(roots) == 0 {
		roots = []string{"."}
	}

	var files []string
	for _, root := range roots {
		root = strings.TrimSuffix(strings.TrimSuffix(root, "..."), "/")
		if root == "" {
			root = "."
		}
		_ = filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.IsDir() {
				// skip vendored / hidden / build-artifact dirs
				base := d.Name()
				if base == "vendor" || base == "testdata" || (strings.HasPrefix(base, ".") && base != ".") {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(p, ".go") {
				return nil
			}
			if !withTests && strings.HasSuffix(p, "_test.go") {
				return nil
			}
			files = append(files, p)
			return nil
		})
	}
	sort.Strings(files)

	fset := token.NewFileSet()
	var findings []finding
	for _, f := range files {
		af, err := parser.ParseFile(fset, f, nil, parser.ParseComments)
		if err != nil {
			continue // unparseable file: skip, not our job to report syntax errors
		}
		ignore := ignoredLines(fset, af)
		for _, fd := range scanFile(fset, f, af) {
			if ignore[fd.Line] {
				continue // explicitly marked handled/intentional
			}
			findings = append(findings, fd)
		}
	}

	if asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(findings)
	} else {
		printText(findings)
	}
	if strict && len(findings) > 0 {
		os.Exit(1)
	}
}

// ignoredLines returns the set of line numbers carrying a `//perfscan:ignore`
// comment — used to suppress sites that are intentionally the way they are (a
// small-slice sort, or a per-element closure kept only as the guarded fallback
// behind a raw-slice fast path). Both the comment's own line and the next line
// are honoured, so the marker may sit above the flagged statement.
func ignoredLines(fset *token.FileSet, af *ast.File) map[int]bool {
	out := map[int]bool{}
	for _, cg := range af.Comments {
		for _, c := range cg.List {
			if strings.Contains(c.Text, "perfscan:ignore") {
				ln := fset.Position(c.Pos()).Line
				out[ln] = true
				out[ln+1] = true
			}
		}
	}
	return out
}

func scanFile(fset *token.FileSet, path string, af *ast.File) []finding {
	var out []finding
	ast.Inspect(af, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		name := calleeName(call.Fun)
		if name == "" {
			return true
		}
		pos := fset.Position(call.Pos())
		// P1: per-element visitor called with a closure argument.
		if perElemVisitors[name] && hasFuncLitArg(call) {
			out = append(out, finding{
				File: path, Line: pos.Line, Pattern: "P1-per-element-closure",
				Detail: fmt.Sprintf("%s(...) invokes a closure per element", name),
				Fix:    "add a contiguous raw-slice fast path for F32/F64 (IsContiguous && Offset==0); keep the closure only as the non-contiguous/half-float fallback. Verify bit-identical + benchmark.",
			})
		}
		// P2: closure-comparator sort.
		if closureSorters[name] {
			// Sort (sort.Interface) has no FuncLit but is still an indirect Less;
			// the *Func / Slice variants take a comparator value.
			if name == "Sort" || hasFuncArg(call) {
				out = append(out, finding{
					File: path, Line: pos.Line, Pattern: "P2-closure-comparator-sort",
					Detail: fmt.Sprintf("%s uses an indirect comparator", name),
					Fix:    "if the key is a monotonic float/int over a large slice, replace with an LSD radix sort on the key bits (for non-negative float64, math.Float64bits is monotonic). Verify identical order + benchmark.",
				})
			}
		}
		return true
	})
	return out
}

// calleeName returns the identifier being called: "f" for f(), "pkg.F"/"x.M" →
// "F"/"M" for selector calls (we match on the final segment).
func calleeName(fun ast.Expr) string {
	switch e := fun.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.SelectorExpr:
		return e.Sel.Name
	}
	return ""
}

func hasFuncLitArg(call *ast.CallExpr) bool {
	for _, a := range call.Args {
		if _, ok := a.(*ast.FuncLit); ok {
			return true
		}
	}
	return false
}

// hasFuncArg reports whether any argument is a function value (a literal, or an
// identifier/selector that plausibly names a func — we accept idents to catch
// `slices.SortFunc(s, cmp)` where cmp is a named comparator).
func hasFuncArg(call *ast.CallExpr) bool {
	for _, a := range call.Args {
		switch a.(type) {
		case *ast.FuncLit, *ast.Ident, *ast.SelectorExpr:
			return true
		}
	}
	return false
}

func printText(findings []finding) {
	if len(findings) == 0 {
		fmt.Println("perfscan: no candidate hot-path anti-patterns found")
		return
	}
	byPat := map[string][]finding{}
	for _, f := range findings {
		byPat[f.Pattern] = append(byPat[f.Pattern], f)
	}
	pats := make([]string, 0, len(byPat))
	for p := range byPat {
		pats = append(pats, p)
	}
	sort.Strings(pats)
	total := 0
	for _, p := range pats {
		fs := byPat[p]
		total += len(fs)
		fmt.Printf("\n## %s  (%d candidate%s)\n", p, len(fs), plural(len(fs)))
		if len(fs) > 0 {
			fmt.Printf("   %s\n", fs[0].Fix)
		}
		for _, f := range fs {
			fmt.Printf("   %s:%d  %s\n", f.File, f.Line, f.Detail)
		}
	}
	fmt.Printf("\nperfscan: %d candidate%s across %d pattern%s — each is a CANDIDATE; confirm with a pre/post benchmark before changing.\n",
		total, plural(total), len(pats), plural(len(pats)))
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}
