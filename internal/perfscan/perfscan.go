// Command perfscan is a static finder for the per-element hot-loop anti-patterns
// this codebase repeatedly optimizes away (§base-perf-sweep). It parses Go source
// with go/ast — it never builds or imports the packages, so cgo/build-tagged
// backends are scanned too — and reports the STRUCTURAL SHAPE of three patterns:
//
//	A. per-element .AtF64/.SetF64 dispatch in a Numel/Unravel loop with no
//	   flatF64/flatF32 contiguous fast path in the enclosing function
//	   (the dominant win: optimizers 3.5–15×, dropout/droppath 1.6×/13×).
//	B. an allocation (tensor.New/FromFloat64/Zeros/…/.Cast) INSIDE a per-element
//	   loop — worse than dispatch (the AMP roundHalf case, 50×).
//	C. a batch API called with a single-element slice literal inside a loop, e.g.
//	   tree.Predict([][]float64{row}) (forest predict, 80001→1 alloc, 3×).
//
// IMPORTANT — these are CANDIDATES, not confirmed wins. A static check sees the
// shape of a hot loop, never its temperature: a per-element write in a one-time
// constructor is fine (§C3, measure don't assume). Each hit still needs an A/B
// measurement and a bit-identity proof before it ships (§V22). perfscan replaces
// the ad-hoc awk scans in docs/perf-notes-training.md with an AST-accurate,
// comment/string-safe, CI-wirable pass.
//
// Usage:
//
//	go run ./internal/perfscan ./...        # scan the whole module (advisory)
//	go run ./internal/perfscan ./nn/...     # scan one subtree
//	go run ./internal/perfscan -strict ./nn # exit 1 if any candidate is found
//	go run ./internal/perfscan -tests ./... # include _test.go files
package main

import (
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

// finding is one candidate anti-pattern occurrence, anchored to the enclosing loop.
type finding struct {
	pos      token.Position
	category string // "per-element-dispatch" | "alloc-in-loop" | "batch-single-elt"
	msg      string
}

// allocCallees are constructors/converters that allocate a fresh tensor; called
// per element they turn an O(n) loop into O(n) allocations (§base-perf §2).
var allocCallees = map[string]bool{
	"New": true, "NewOn": true, "FromFloat64": true, "FromFloat32": true,
	"Zeros": true, "Ones": true, "Full": true, "Scalar": true,
	"scalarTensor": true, "Cast": true,
}

// calleeName returns the trailing identifier of a call target: the func name for a
// bare call (Unravel), or the method/selector name for x.Method() (SetF64). Empty
// when the callee is not a plain ident/selector.
func calleeName(fun ast.Expr) string {
	switch f := fun.(type) {
	case *ast.Ident:
		return f.Name
	case *ast.SelectorExpr:
		return f.Sel.Name
	}
	return ""
}

// isSingleEltNestedSliceLit reports whether e is a one-element NESTED-slice literal
// such as [][]float64{row} — the shape of wrapping one ROW to feed a batch API
// (detector C, the forest-predict case). The element type must itself be a
// slice/array: that excludes []*tensor.Tensor{x}, the canonical single-INPUT
// op-argument list (backend.Execute), which is not a batch-wrap and was the sole
// source of false positives on the first module-wide run.
func isSingleEltNestedSliceLit(e ast.Expr) bool {
	cl, ok := e.(*ast.CompositeLit)
	if !ok || len(cl.Elts) != 1 {
		return false
	}
	at, ok := cl.Type.(*ast.ArrayType)
	if !ok || at.Len != nil { // Len==nil ⇒ slice, not a fixed array
		return false
	}
	_, nested := at.Elt.(*ast.ArrayType) // element itself a slice/array ⇒ a wrapped row
	return nested
}

// scanFile runs every detector over one parsed file and returns deduplicated
// findings (one per enclosing loop per category).
func scanFile(fset *token.FileSet, f *ast.File) []finding {
	var out []finding
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		out = append(out, scanFunc(fset, fn)...)
	}
	return dedup(out)
}

// scanFunc analyzes a single function body. Fast-path presence (flatF64/flatF32)
// and Numel-derived loop bounds are function-scoped facts, so they are gathered up
// front; then each trigger call is attributed to its nearest enclosing loop.
func scanFunc(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	// Pass 1: function-scoped facts + a child→parent map for ancestor walks.
	hasFlat := false
	numelIdents := map[string]bool{}
	parent := map[ast.Node]ast.Node{}
	var stack []ast.Node
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		if len(stack) > 0 {
			parent[n] = stack[len(stack)-1]
		}
		stack = append(stack, n)
		switch x := n.(type) {
		case *ast.CallExpr:
			if name := calleeName(x.Fun); name == "flatF64" || name == "flatF32" {
				hasFlat = true
			}
		case *ast.AssignStmt:
			// n := x.Numel()  /  n = x.Numel()
			for i, rhs := range x.Rhs {
				call, ok := rhs.(*ast.CallExpr)
				if !ok || calleeName(call.Fun) != "Numel" || i >= len(x.Lhs) {
					continue
				}
				if id, ok := x.Lhs[i].(*ast.Ident); ok {
					numelIdents[id.Name] = true
				}
			}
		}
		return true
	})

	// Pass 2: classify every loop as per-element (a full-tensor walk) or not.
	perElemLoop := map[ast.Node]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch loop := n.(type) {
		case *ast.RangeStmt:
			perElemLoop[loop] = isNumelRange(loop, numelIdents) || directlyHasUnravel(loop.Body)
		case *ast.ForStmt:
			perElemLoop[loop] = isNumelForCond(loop, numelIdents) || directlyHasUnravel(loop.Body)
		}
		return true
	})

	// Pass 3: attribute triggers to their nearest enclosing loop.
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		loop := nearestLoop(parent, call)
		if loop == nil {
			return true
		}
		perElem := anyAncestorPerElem(parent, perElemLoop, call)
		name := calleeName(call.Fun)

		// A: per-element dispatch with no fast path in this function.
		if perElem && (name == "AtF64" || name == "SetF64") && !hasFlat {
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "per-element-dispatch",
				msg: fmt.Sprintf("per-element .%s in a Numel/Unravel loop, no flatF64/flatF32 fast path in %s()"+
					" — add a typed contiguous walk (docs/perf-notes-training.md §1)", name, fn.Name.Name),
			})
		}
		// B: allocation inside a per-element loop.
		if perElem && allocCallees[name] {
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "alloc-in-loop",
				msg: fmt.Sprintf("allocation %q inside a per-element loop — hoist/bulk it out of the loop"+
					" (docs/perf-notes-training.md §2, AMP roundHalf 50×)", name),
			})
		}
		// C: batch API fed a single-element nested-slice literal (a wrapped row), in any loop.
		for _, arg := range call.Args {
			if isSingleEltNestedSliceLit(arg) {
				out = append(out, finding{
					pos:      fset.Position(loop.Pos()),
					category: "batch-single-elt",
					msg: fmt.Sprintf("%q called with a single-element slice literal inside a loop"+
						" — use the single-item path (T917, forest predict 80001→1 alloc)", name),
				})
				break
			}
		}
		return true
	})
	return out
}

// isNumelRange reports whether a range loop iterates a tensor's element count:
// `for … range x.Numel()` or `for … range n` with n bound from a .Numel() call.
func isNumelRange(r *ast.RangeStmt, numelIdents map[string]bool) bool {
	switch x := r.X.(type) {
	case *ast.CallExpr:
		return calleeName(x.Fun) == "Numel"
	case *ast.Ident:
		return numelIdents[x.Name]
	}
	return false
}

// isNumelForCond reports whether a 3-clause for loop is bounded by a Numel-derived
// count: `for i := 0; i < n; i++` with n from a .Numel() call (either operand).
func isNumelForCond(f *ast.ForStmt, numelIdents map[string]bool) bool {
	bin, ok := f.Cond.(*ast.BinaryExpr)
	if !ok {
		return false
	}
	for _, side := range []ast.Expr{bin.X, bin.Y} {
		if id, ok := side.(*ast.Ident); ok && numelIdents[id.Name] {
			return true
		}
		if call, ok := side.(*ast.CallExpr); ok && calleeName(call.Fun) == "Numel" {
			return true
		}
	}
	return false
}

// directlyHasUnravel reports whether a loop body calls tensor.Unravel in ITS OWN
// scope — the tell-tale of a flat-index → multi-index walk over a whole tensor. It
// deliberately does NOT descend into a nested Range/For/FuncLit: an Unravel there
// belongs to that inner loop, so counting it would misclassify an outer per-row or
// per-parameter loop as per-element (the TIESMerge/soup false positive: a
// once-per-parameter tensor.New sitting above an inner per-element Unravel loop).
func directlyHasUnravel(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}
		switch n.(type) {
		case *ast.RangeStmt, *ast.ForStmt, *ast.FuncLit:
			return false // nested scope: its Unravel belongs to that loop, not this one
		}
		if call, ok := n.(*ast.CallExpr); ok && calleeName(call.Fun) == "Unravel" {
			found = true
			return false
		}
		return true
	})
	return found
}

// nearestLoop walks parent links up from n to the innermost enclosing Range/For
// (nil if n is not inside a loop).
func nearestLoop(parent map[ast.Node]ast.Node, n ast.Node) ast.Node {
	for p := parent[n]; p != nil; p = parent[p] {
		switch p.(type) {
		case *ast.RangeStmt, *ast.ForStmt:
			return p
		}
	}
	return nil
}

// anyAncestorPerElem reports whether any enclosing loop of n is per-element, so an
// AtF64/SetF64 nested one level down (inside a fixed inner loop, say) still counts.
func anyAncestorPerElem(parent map[ast.Node]ast.Node, perElem map[ast.Node]bool, n ast.Node) bool {
	for p := parent[n]; p != nil; p = parent[p] {
		if perElem[p] {
			return true
		}
	}
	return false
}

// dedup collapses identical (position, category) findings — one loop can hold both
// an AtF64 and a SetF64, which are the same candidate.
func dedup(in []finding) []finding {
	seen := map[string]bool{}
	var out []finding
	for _, f := range in {
		key := f.pos.String() + "\x00" + f.category
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, f)
	}
	return out
}

// --- CLI ---

func main() {
	strict := flag.Bool("strict", false, "exit 1 if any candidate is found (for optional CI gating)")
	tests := flag.Bool("tests", false, "include _test.go files")
	flag.Parse()
	roots := flag.Args()
	if len(roots) == 0 {
		roots = []string{"./..."}
	}

	files, err := goFiles(roots, *tests)
	if err != nil {
		fmt.Fprintln(os.Stderr, "perfscan:", err)
		os.Exit(2)
	}

	fset := token.NewFileSet()
	var all []finding
	for _, path := range files {
		f, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			fmt.Fprintf(os.Stderr, "perfscan: parse %s: %v\n", path, err)
			continue
		}
		all = append(all, scanFile(fset, f)...)
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].pos.Filename != all[j].pos.Filename {
			return all[i].pos.Filename < all[j].pos.Filename
		}
		return all[i].pos.Line < all[j].pos.Line
	})

	byCat := map[string]int{}
	for _, f := range all {
		fmt.Printf("%s: [%s] %s\n", f.pos, f.category, f.msg)
		byCat[f.category]++
	}
	if len(all) == 0 {
		fmt.Println("perfscan: no candidate anti-patterns found")
		return
	}
	fmt.Printf("\nperfscan: %d candidate(s) — dispatch=%d alloc-in-loop=%d batch-single-elt=%d\n",
		len(all), byCat["per-element-dispatch"], byCat["alloc-in-loop"], byCat["batch-single-elt"])
	fmt.Println("NOTE: candidates, not confirmed wins — measure hotness (§C3) + prove bit-identity (§V22) before shipping.")
	if *strict {
		os.Exit(1)
	}
}

// goFiles expands root arguments (a file, a dir, or a `dir/...` recursive pattern)
// into the set of .go files to scan, skipping vendored, generated, and tooling
// trees that would only add noise.
func goFiles(roots []string, includeTests bool) ([]string, error) {
	var files []string
	add := func(path string) {
		if !strings.HasSuffix(path, ".go") {
			return
		}
		if !includeTests && strings.HasSuffix(path, "_test.go") {
			return
		}
		files = append(files, path)
	}
	for _, root := range roots {
		recursive := strings.HasSuffix(root, "...")
		base := strings.TrimSuffix(strings.TrimSuffix(root, "..."), "/")
		if base == "" || base == "." {
			base = "."
		}
		info, err := os.Stat(base)
		if err != nil {
			return nil, err
		}
		if !info.IsDir() {
			add(base)
			continue
		}
		if !recursive {
			entries, err := os.ReadDir(base)
			if err != nil {
				return nil, err
			}
			for _, e := range entries {
				if !e.IsDir() {
					add(filepath.Join(base, e.Name()))
				}
			}
			continue
		}
		err = filepath.WalkDir(base, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() && skipDir(d.Name()) {
				return filepath.SkipDir
			}
			if !d.IsDir() {
				add(path)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	sort.Strings(files)
	return files, nil
}

// skipDir lists directory names whose contents are not first-party runtime code.
func skipDir(name string) bool {
	switch name {
	case "vendor", "testdata", ".git", ".claude", "graphify-out", "node_modules":
		return true
	}
	return false
}
