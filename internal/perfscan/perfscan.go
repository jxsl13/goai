// Command perfscan is a REPO-AGNOSTIC static finder for hot-loop performance
// anti-patterns. It parses Go source with go/ast — it never builds or imports the
// packages, so cgo/build-tagged files are scanned too — and reports the STRUCTURAL
// SHAPE of fifteen patterns. The language/stdlib-shape checks run on any module
// with no configuration; the four DOMAIN checks that key on a project's own
// element-access / allocation vocabulary (per-element dispatch/closure, alloc in a
// per-element loop, a scalar transcendental beside a vectorized sibling) are driven
// entirely by a Config (see -config; empty by default, so they stay silent until a
// project names its functions — the engine hard-codes nothing project-specific).
// The illustrative names below (AtF64, flatF64, …) are just one project's config;
// substitute your own.
//
//	PS1001. per-element .AtF64/.SetF64 dispatch in a Numel/Unravel loop with no
//	   flatF64/flatF32 contiguous fast path in the enclosing function
//	   (the dominant win: optimizers 3.5–15×, dropout/droppath 1.6×/13×).
//	PS2001. an allocation (tensor.New/FromFloat64/Zeros/…/.Cast) INSIDE a per-element
//	   loop — worse than dispatch (the AMP roundHalf case, 50×).
//	PS1003. a batch API called with a single-element slice literal inside a loop, e.g.
//	   tree.Predict([][]float64{row}) (forest predict, 80001→1 alloc, 3×).
//	PS3001. a reflection-based fmt SCAN call (fmt.Sscanf/Sscan/Fscanf/…) in a loop —
//	   parses a format string + reflects over varargs every call, allocating
//	   (SPM.Decode ran fmt.Sscanf per token, 1.55M allocs → 20× once gated, T931).
//	PS2002. a strings.Builder/bytes.Buffer written in a loop with no .Grow — WriteX
//	   doubles the backing buffer log(n) times (BPE/SPM/Unigram Decode pre-size, T929).
//	PS2003. an allocating strings transform (ReplaceAll/Replace/Map/Repeat) in a loop —
//	   a fresh string per iteration; write it in place (T934, 52k→1 alloc, 1.65×).
//	PS4001. a per-element little-endian bit decode (binary.LittleEndian.UintN /
//	   math.FloatNfrombits) in a loop with no rawCopyLE fast path — a memcpy in
//	   disguise on LE hosts for VERBATIM-bit dtypes (T720/T907 readers, 2–5×).
//	PS2005. a regexp.Compile/MustCompile inside a loop — recompiles the pattern every
//	   iteration; hoist it out.
//	PS1002. a per-element visitor (readGen/fillGen/visitGen/…) fed a CLOSURE — an
//	   indirect call per element even on its contiguous branch; add a raw-slice
//	   fast path (SWA/EMA/GradAccum 1.6–4.2×). (Ex tools/perfscan P1.)
//	PS3002. a package-qualified sort (sort.Slice/SliceStable/Sort, slices.SortFunc/…)
//	   with a comparator — a per-comparison indirect call; radix on the key bits
//	   when it is a monotonic float/int over a large slice (top-p 1.9–2.25×).
//	   (Ex tools/perfscan P2.)
//	PS4002. a scalar libm transcendental (math.Exp/Tanh/Erf/Log/…) in a loop while the
//	   SAME kernel calls a vectorized v*F32/F64 sibling for another dtype — the
//	   scalar dtype branch is a SIMD candidate (F64 SwiGLU SiLU vexp = 1.52× Llama
//	   prefill, T667). CAVEAT: first check the op is not under a bit-exact CPU==Ref
//	   invariant (f64 exp/log/tanh/sigmoid/gelu are locked; OpSiLU/F64 is not).
//	PS4003. a loop calls a package-local elementwise helper that WRAPS a libm
//	   transcendental (softplus/mish/swish/…) — K sees only a DIRECT math.X, so this
//	   hides the scalar cost one call deep (F64 OpSoftplus vsoftplus = 1.62× Mamba).
//	PS3003. a READ of an integer-keyed map (map[int]/map[rune]/…, sets excluded) inside a
//	   loop — when the keys are dense over [0,N) a []T indexed by the key drops the
//	   per-lookup hash (BPE Decode 2.85×, GGUF Decode 3.67×, forest votes T312).
//	PS2004. a slice make() bound to a NON-escaping local inside a per-item loop of a
//	   pointer-receiver method — per-call scratch reallocated every call; hoist it to a
//	   reused receiver field (Adafactor/Cautious/LAMB/Grokfast Step, 1.2–1.6×, →0 allocs).
//	PS5001. a divide by a loop-invariant SCALAR on every element of an element-wise arithmetic
//	   loop — hoist inv:=1/D and multiply (SoftCap VJP 1.28×, optimizer bias-corrections
//	   1.1–1.3×). SAFE ONLY for continuous outputs (½ulp), NEVER feeding round/quantize.
//
// IMPORTANT — these are CANDIDATES, not confirmed wins. A static check sees the
// shape of a hot loop, never its temperature: a per-element write in a one-time
// constructor is fine (measure, don't assume). Each hit still needs an A/B
// measurement and a bit-identity proof before it ships. perfscan is an
// AST-accurate, comment/string-safe, CI-wirable pass.
//
// CONFIG (making it repo-agnostic). Ten checks are pure language/stdlib shapes and
// need no configuration: PS2002 unsized-builder, PS2003 strings-alloc, PS2004
// poolable-scratch, PS2005 regexp, PS3001 reflection, PS3002 closure-comparator-
// sort, PS3003 int-key-map, PS4001 le-decode, PS4003 transcendental-wrapper, PS5001
// loop-invariant-divide. The four DOMAIN checks — PS1001, PS1002, PS2001, PS4002 —
// key on a project's own vocabulary (its element accessors, allocators, fast-path
// helpers and vectorized kernels), which lives in a JSON Config, NOT in the engine.
// With no config those four stay silent, so perfscan on an arbitrary module reports
// only the language/stdlib patterns. Point it at a config with `-config file.json`
// (fields: elementAccessors, fastPathHelpers, elementCountMethods,
// indexDecomposeFuncs, allocatorFuncs, perElementVisitors, bulkCopyHelpers,
// vectorizedSiblingFuncs), or drop a perfscan.json / .perfscan.json in a parent
// directory to have it discovered automatically.
//
// CHECK IDs. Every detector has a stable, staticcheck-style ID: the prefix PS
// plus a four-digit number whose thousands digit groups the detector — PS1xxx
// per-element access, PS2xxx allocation, PS3xxx indirection/reflection, PS4xxx
// vectorization, PS5xxx arithmetic. The ID is what a report line, -checks,
// -exclude and a //perfscan:ignore directive all name; the descriptive category
// (per-element-dispatch, …) is accepted everywhere as an alias. `-list` prints
// the full table (ID, category, whether -fix can rewrite it, and a one-line title).
//
// Usage:
//
//	go run ./internal/perfscan ./...                  # scan the whole module (advisory)
//	go run ./internal/perfscan ./nn/...               # scan one subtree
//	go run ./internal/perfscan -list                  # list the checks (ID, category, fixable, title)
//	go run ./internal/perfscan -strict ./nn           # exit 1 if any finding is reported (CI gating)
//	go run ./internal/perfscan -tests ./...           # include _test.go files
//	go run ./internal/perfscan -checks=PS1001 ./...    # run ONLY these checks (allow-list)
//	go run ./internal/perfscan -exclude=PS4002 ./...   # silence these checks repo-wide
//	go run ./internal/perfscan -fix ./...             # apply the safe mechanical fixes in place
//	go run ./internal/perfscan -json ./...            # emit findings + fixes as JSON (editor / VS Code)
//
// AUTO-FIX (-fix) and EDITOR INTEGRATION (-json). Only checks whose rewrite is
// deterministic AND bit-identical carry a fix (see the FIX column of -list; today
// PS2005 regexp-compile-in-loop hoists a literal-pattern compile out of the loop).
// Everything else is advisory by design — a per-element→typed rewrite needs an A/B
// measurement + a bit-identity proof (§C3/§V22) a static tool cannot give, so
// perfscan reports the shape and leaves the transform to a human. -fix applies the
// safe fixes in place (review the diff — even these want an A/B before shipping);
// -json emits every finding, with any fix's text edits (line:col ranges + byte
// offsets + replacement text), for a VS Code task/extension to surface as a
// quick-fix.
//
// SUPPRESSING FINDINGS (staticcheck-style, per-check). Silencing one check never
// hides another — so a site you deliberately accept for PS1001 still reports a
// NEW, unrelated PS2002. Name a check by ID (precise — the "ignore only this
// explicit detection" path) or by its category alias:
//
//	//perfscan:ignore PS1001 reason        // silence ONLY PS1001 on the next (or same) line
//	//perfscan:ignore PS1001,PS3003 reason  // silence several checks; IDs or categories
//	//perfscan:ignore per-element-dispatch reason  // by category alias
//	//perfscan:ignore                      // bare: silence ALL checks at that site
//	-exclude=PS4002,per-element-closure    // silence checks for the whole run
//	-checks=PS1001,PS2002                  // run ONLY these (allow-list; -exclude still subtracts)
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// finding is one candidate anti-pattern occurrence, anchored to the enclosing loop.
type finding struct {
	pos      token.Position
	end      token.Position // end of the flagged node (for -json ranges / editor highlight); zero ⇒ unset
	id       string         // PS1001… — filled from category at report time
	category string         // one of checks[].category (see below)
	msg      string
	fix      *suggestedFix // nil ⇒ advisory (no safe mechanical rewrite)
}

// suggestedFix is a staticcheck/gopls-style rewrite: a human title plus the byte
// edits that realise it. -fix applies the edits in place; -json emits them so an
// editor (VS Code) can offer the quick-fix. Only checks with a DETERMINISTIC,
// bit-identical rewrite carry one (see checks[].fixable); everything else is
// advisory by design — most perf patterns need an A/B + bit-identity proof a
// static tool cannot give (§C3/§V22).
type suggestedFix struct {
	title string
	edits []textEdit
}

// textEdit replaces the half-open range [start,end) with newText. Positions are
// token.Pos values resolved through the FileSet, so both a byte offset (for -fix)
// and a line:col range (for -json / an editor) derive from the same source.
type textEdit struct {
	start, end token.Pos
	newText    string
}

// check is one detector. The stable public ID is a PREFIX (PS) + a four-digit
// number whose THOUSANDS digit groups the detector (staticcheck-style, e.g.
// SA1xxx/S1xxx): PS1xxx per-element access, PS2xxx allocation, PS3xxx
// indirection/reflection, PS4xxx vectorization, PS5xxx arithmetic. The ID is
// what -checks and //perfscan:ignore name; the descriptive category is accepted
// as an alias. fixable marks checks that -fix can rewrite automatically.
type check struct {
	id       string
	category string
	title    string
	fixable  bool
}

// checks is the detector registry (order = report/-list order = ID order).
var checks = []check{
	// PS1xxx — per-element access in hot loops
	{"PS1001", "per-element-dispatch", "per-element AtF64/SetF64 dispatch in a Numel/Unravel loop with no typed fast path", false},
	{"PS1002", "per-element-closure", "per-element visitor fed a closure (an indirect call per element)", false},
	{"PS1003", "batch-single-elt", "a batch API called with a single-element slice literal inside a loop", false},
	// PS2xxx — allocation inside loops
	{"PS2001", "alloc-in-loop", "a tensor allocation inside a per-element loop", false},
	{"PS2002", "unsized-builder", "a strings.Builder/bytes.Buffer written in a loop with no .Grow", false},
	{"PS2003", "strings-alloc-in-loop", "an allocating strings transform (Replace/Map/Repeat) in a loop", false},
	{"PS2004", "poolable-loop-scratch", "per-call scratch make() bound to a non-escaping local in a pointer-method loop", false},
	{"PS2005", "regexp-compile-in-loop", "a regexp.Compile/MustCompile inside a loop", true},
	// PS3xxx — indirection / reflection overhead
	{"PS3001", "reflection-in-loop", "a reflection-based fmt scan (Sscanf/Sscan/Fscanf) in a loop", false},
	{"PS3002", "closure-comparator-sort", "a package sort (sort.Slice/SliceStable) with a comparator closure", false},
	{"PS3003", "int-key-map-in-loop", "a read of an integer-keyed map inside a loop", false},
	// PS4xxx — vectorization candidates
	{"PS4001", "le-decode-in-loop", "a per-element little-endian bit decode in a loop with no bulk-copy fast path", false},
	{"PS4002", "scalar-transcendental-vectorizable", "a scalar libm transcendental in a loop while a vectorized sibling is called", false},
	{"PS4003", "transcendental-wrapper-in-loop", "a loop calls a helper that wraps a libm transcendental", false},
	{"PS4004", "scalar-copy-loop", "an element-by-element slice copy in a loop where a bulk copy would do", false},
	{"PS4005", "per-element-odometer", "an N-D coordinate odometer ticked once per element instead of once per run", false},
	{"PS4006", "row-slice-matrix", "a [][]T matrix built row-by-row and then indexed inside a nested loop", false},
	// PS5xxx — arithmetic
	{"PS5001", "loop-invariant-divide", "a divide by a loop-invariant scalar on every element", false},
}

// catToID indexes the registry for O(1) id lookup at report time.
var catToID = func() map[string]string {
	m := make(map[string]string, len(checks))
	for _, c := range checks {
		m[c.category] = c.id
	}
	return m
}()

// Config makes perfscan repo-agnostic. The language/stdlib-shape checks — PS2002
// unsized-builder, PS2003 strings-alloc, PS2004 poolable-scratch, PS2005 regexp,
// PS3001 reflection, PS3002 closure-comparator-sort, PS3003 int-key-map, PS4001
// le-decode, PS4003 transcendental-wrapper, PS5001 loop-invariant-divide — run on
// ANY Go module with no configuration. The four DOMAIN checks — PS1001
// per-element-dispatch, PS1002 per-element-closure, PS2001 alloc-in-loop, PS4002
// scalar-transcendental-vectorizable — key on a project's own vocabulary (the
// element accessors, allocators and fast-path helpers of its tensor/array type),
// which lives ENTIRELY here rather than hard-coded in the engine. Every field is
// empty by default, so those four checks stay silent until a project names its
// functions. Supply them with `-config <file.json>` or a perfscan.json /
// .perfscan.json discovered upward from the working directory.
type Config struct {
	// PS1001/PS1002 — per-element read/write methods (e.g. a tensor's AtF64/SetF64).
	ElementAccessors []string `json:"elementAccessors,omitempty"`
	// PS1001 — bulk/typed accessors whose presence in a function proves its
	// per-element loop is only a fallback, so the dispatch is NOT reported (the
	// "fast path" helpers, e.g. flatF64/flatF32 or a Storage().F64() slice grab).
	FastPathHelpers []string `json:"fastPathHelpers,omitempty"`
	// PS1001 — element-count methods; a loop bounded by x.Method() over one reads as
	// per-element (e.g. Numel).
	ElementCountMethods []string `json:"elementCountMethods,omitempty"`
	// ShapeMethods name calls whose INDEXED result is a dimension (Shape()[1]).
	ShapeMethods []string `json:"shapeMethods,omitempty"`
	// PS1001 — flat→multi-index calls; one in a loop body marks it per-element
	// (e.g. Unravel).
	IndexDecomposeFuncs []string `json:"indexDecomposeFuncs,omitempty"`
	// PS2001 — allocation constructors/converters flagged inside a per-element loop
	// (e.g. a tensor package's New/Zeros/FromFloat64/Cast).
	AllocatorFuncs []string `json:"allocatorFuncs,omitempty"`
	// PS1002 — helpers that invoke a per-element closure argument (e.g. readGen/fillGen).
	PerElementVisitors []string `json:"perElementVisitors,omitempty"`
	// PS4001 — bulk little-endian copy helpers whose presence proves an intentional
	// big-endian / genuine-decode path, silencing the per-element-decode report.
	BulkCopyHelpers []string `json:"bulkCopyHelpers,omitempty"`
	// PS4002 — hand-vectorized SIMD kernel names; a call to one in a file marks a
	// scalar math.X in a loop there as a vectorize candidate (e.g. vsiluF32).
	VectorizedSiblingFuncs []string `json:"vectorizedSiblingFuncs,omitempty"`
}

// nameSets is Config compiled to maps for O(1) lookup during a scan.
type nameSets struct {
	accessors, fastPath, elemCount, indexDecompose map[string]bool
	shapeMethods                                   map[string]bool
	allocators, visitors, bulkCopy, vectorized     map[string]bool
}

func toSet(xs []string) map[string]bool {
	m := make(map[string]bool, len(xs))
	for _, x := range xs {
		m[x] = true
	}
	return m
}

func (c Config) compile() nameSets {
	return nameSets{
		accessors:      toSet(c.ElementAccessors),
		fastPath:       toSet(c.FastPathHelpers),
		elemCount:      toSet(c.ElementCountMethods),
		shapeMethods:   toSet(c.ShapeMethods),
		indexDecompose: toSet(c.IndexDecomposeFuncs),
		allocators:     toSet(c.AllocatorFuncs),
		visitors:       toSet(c.PerElementVisitors),
		bulkCopy:       toSet(c.BulkCopyHelpers),
		vectorized:     toSet(c.VectorizedSiblingFuncs),
	}
}

// loadConfig reads a JSON Config from path.
func loadConfig(path string) (Config, error) {
	var c Config
	b, err := os.ReadFile(path)
	if err != nil {
		return c, err
	}
	if err := json.Unmarshal(b, &c); err != nil {
		return c, fmt.Errorf("%s: %w", path, err)
	}
	return c, nil
}

// discoverConfig walks up from the working directory looking for perfscan.json or
// .perfscan.json (golangci-lint style). It returns the empty Config — the fully
// generic, stdlib-only mode — when none is found.
func discoverConfig() (Config, string) {
	dir, err := os.Getwd()
	if err != nil {
		return Config{}, ""
	}
	for {
		for _, name := range []string{"perfscan.json", ".perfscan.json"} {
			p := filepath.Join(dir, name)
			if _, err := os.Stat(p); err == nil {
				if c, err := loadConfig(p); err == nil {
					return c, p
				}
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return Config{}, ""
		}
		dir = parent
	}
}

// loopAssignedIdents returns the set of identifier names that a loop MUTATES — the LHS
// of every assignment/inc-dec in its body, plus the loop's own iteration variables. A
// divisor drawn from this set varies across iterations and is NOT a hoistable invariant.
func loopAssignedIdents(loop ast.Node) map[string]bool {
	m := map[string]bool{}
	mark := func(e ast.Expr) {
		switch x := e.(type) {
		case *ast.Ident:
			m[x.Name] = true
		case *ast.SelectorExpr: // recv.field = … : the field (and its root) may change
			if id, ok := x.X.(*ast.Ident); ok {
				m[id.Name] = true
			}
		}
	}
	var body ast.Node
	switch l := loop.(type) {
	case *ast.RangeStmt:
		mark(l.Key)
		mark(l.Value)
		body = l.Body
	case *ast.ForStmt:
		if a, ok := l.Init.(*ast.AssignStmt); ok {
			for _, lhs := range a.Lhs {
				mark(lhs)
			}
		}
		if p, ok := l.Post.(*ast.IncDecStmt); ok {
			mark(p.X)
		}
		body = l.Body
	default:
		return m
	}
	ast.Inspect(body, func(n ast.Node) bool {
		switch s := n.(type) {
		case *ast.AssignStmt:
			for _, lhs := range s.Lhs {
				mark(lhs)
			}
		case *ast.IncDecStmt:
			mark(s.X)
		}
		return true
	})
	return m
}

// loopBodyNode returns a loop's body block, or nil for a non-loop node.
func loopBodyNode(loop ast.Node) ast.Node {
	switch l := loop.(type) {
	case *ast.RangeStmt:
		return l.Body
	case *ast.ForStmt:
		return l.Body
	}
	return nil
}

// loopHasIndexAccess reports whether a loop's body indexes a slice/array (a[i]) — the
// element-wise shape that makes a per-iteration divide worth vectorizing (and keeps
// scalar/integer bookkeeping loops out of detector PS5001's signal).
func loopHasIndexAccess(loop ast.Node) bool {
	body := loopBodyNode(loop)
	if body == nil {
		return false
	}
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if _, ok := n.(*ast.IndexExpr); ok {
			found = true
			return false
		}
		return true
	})
	return found
}

// oExpensiveOps are per-element costs that DWARF a divide — a loop already dominated by
// one of these is not a reciprocal-multiply candidate (the divide is in the noise, and
// it is already the K/L detectors' territory). math.Sqrt is deliberately absent: a
// divide-after-sqrt (norm/RMSNorm) is exactly where the reciprocal-multiply pays.
var oExpensiveOps = map[string]bool{
	"Exp": true, "Log": true, "Log1p": true, "Log2": true, "Log10": true,
	"Tanh": true, "Sinh": true, "Cosh": true, "Erf": true, "Erfc": true,
	"Pow": true, "Sin": true, "Cos": true, "Tan": true, "Atan": true, "Atan2": true,
	"Exp2": true, "Expm1": true, "Cbrt": true, "Gamma": true,
}

// loopHasExpensiveTranscendental reports whether the loop body calls a transcendental
// (math.X or a package-local wrapper) that dominates a per-element divide.
func loopHasExpensiveTranscendental(loop ast.Node, wrappers map[string]bool) bool {
	body := loopBodyNode(loop)
	if body == nil {
		return false
	}
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if name, ok := pkgFuncCall(call.Fun, "math", oExpensiveOps); ok {
			_ = name
			found = true
			return false
		}
		if wrappers[calleeName(call.Fun)] {
			found = true
			return false
		}
		return true
	})
	return found
}

// invariantDivisorName reports the name of a divisor that is a bare identifier or a
// simple recv.field selector (the shape of a scalar constant/parameter/attr), for the
// loop-invariance check. Calls (float64(n), len(x)) and indexed operands (a[i]) are NOT
// simple divisors and return false — they are either type conversions or per-element.
func invariantDivisorName(e ast.Expr) (string, bool) {
	switch x := e.(type) {
	case *ast.Ident:
		return x.Name, true
	case *ast.SelectorExpr:
		if id, ok := x.X.(*ast.Ident); ok {
			return id.Name, true // invariance keyed on the receiver/struct root
		}
	}
	return "", false
}

// isSliceMake reports whether call is make([]T, …) — a slice allocation, not a map or
// channel or a fixed array (an array type carries a non-nil Len).
func isSliceMake(call *ast.CallExpr) bool {
	id, ok := call.Fun.(*ast.Ident)
	if !ok || id.Name != "make" || len(call.Args) == 0 {
		return false
	}
	at, ok := call.Args[0].(*ast.ArrayType)
	return ok && at.Len == nil
}

// isPointerMethod reports whether fn is a method with a pointer receiver (the shape of
// a reusable stateful object — an optimizer, a builder — whose per-call scratch can be
// hoisted to a receiver field and reused across calls).
func isPointerMethod(fn *ast.FuncDecl) bool {
	if fn.Recv == nil || len(fn.Recv.List) != 1 {
		return false
	}
	_, ok := fn.Recv.List[0].Type.(*ast.StarExpr)
	return ok
}

// integerKeyType reports whether e names a Go integer type — the map key that, when
// dense over [0,N), a []T indexed by the key can replace (detector PS3003).
// intTypeReg maps a package name to the NAMED integer types it declares
// (`type Op int`). curPkg is the package of the file being scanned. Both are
// package-level because perfscan is a single-pass, single-threaded CLI and
// threading them through scanFile/scanFunc/intKeyMapNames would be a wide diff
// for no behavioural gain.
//
// They exist because PS3003 was blind to every enum-keyed dispatch table: a key
// spelled `backend.Op` is an *ast.SelectorExpr and a locally named `Op` is a
// non-builtin *ast.Ident, so the old builtin-name switch rejected both on sight.
// In a library whose dispatch is entirely enum-keyed that is a systemic miss.
// Resolving it needs to know the underlying type — but perfscan is AST-only
// (no go/types, no packages.Load), so instead the declarations are harvested
// from the scanned source itself, which is where they already live.
var (
	intTypeReg = map[string]map[string]bool{}
	curPkg     string
)

// collectIntTypes pre-scans parsed files for `type <Name> <integer kind>` and
// records them per package. It must run over ALL files before any is judged,
// since a key can name a type declared in another package.

// intMapReg maps a package name to the PACKAGE-LEVEL integer-keyed map vars it
// declares. Dispatch registries are declared in one file and read in another —
// vjps lives in autograd/vjp.go, its hot read in autograd/autograd.go — and
// intKeyMapNames is file-scoped, so the name never entered the set for the file
// that reads it. That, not the key-type predicate, is why PS3003 never saw a
// single enum-keyed dispatch table.
//
// Deliberately package-level declarations ONLY. Collecting function-local maps
// package-wide would let a `size` in one file mark an unrelated `size` in
// another, which is a false positive the file-scoped pass already handles
// correctly on its own.
var intMapReg = map[string]map[string]bool{}

// collectIntKeyMaps pre-scans for package-level integer-keyed map vars. It must
// run after collectIntTypes, since deciding whether a key is an integer type
// consults that registry.
func collectIntKeyMaps(files []*ast.File) {
	for _, f := range files {
		if f.Name == nil {
			continue
		}
		curPkg = f.Name.Name
		for _, decl := range f.Decls {
			gd, ok := decl.(*ast.GenDecl)
			if !ok || gd.Tok != token.VAR {
				continue
			}
			for _, spec := range gd.Specs {
				vs, ok := spec.(*ast.ValueSpec)
				if !ok {
					continue
				}
				for i, nm := range vs.Names {
					t := vs.Type
					if t == nil && i < len(vs.Values) {
						if cl, ok := vs.Values[i].(*ast.CompositeLit); ok {
							t = cl.Type
						}
					}
					mt, ok := t.(*ast.MapType)
					if !ok || !integerKeyType(mt.Key) {
						continue
					}
					if intMapReg[curPkg] == nil {
						intMapReg[curPkg] = map[string]bool{}
					}
					intMapReg[curPkg][nm.Name] = true
				}
			}
		}
	}
}

func collectIntTypes(files []*ast.File) {
	for _, f := range files {
		if f.Name == nil {
			continue
		}
		pkg := f.Name.Name
		for _, decl := range f.Decls {
			gd, ok := decl.(*ast.GenDecl)
			if !ok || gd.Tok != token.TYPE {
				continue
			}
			for _, spec := range gd.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok {
					continue
				}
				if under, ok := ts.Type.(*ast.Ident); ok && builtinIntName(under.Name) {
					if intTypeReg[pkg] == nil {
						intTypeReg[pkg] = map[string]bool{}
					}
					intTypeReg[pkg][ts.Name.Name] = true
				}
			}
		}
	}
}

func builtinIntName(n string) bool {
	switch n {
	case "int", "int8", "int16", "int32", "int64",
		"uint", "uint8", "uint16", "uint32", "uint64",
		"byte", "rune", "uintptr":
		return true
	}
	return false
}

func integerKeyType(e ast.Expr) bool {
	switch k := e.(type) {
	case *ast.Ident:
		// A builtin, or a named integer type declared in this package.
		return builtinIntName(k.Name) || intTypeReg[curPkg][k.Name]
	case *ast.SelectorExpr:
		// pkg.Name — resolvable only when the qualifier is a plain identifier.
		// An alias or dot-import is left UNREPORTED rather than guessed: a
		// detector that guesses is how this class of bug arises.
		if q, ok := k.X.(*ast.Ident); ok {
			return intTypeReg[q.Name][k.Sel.Name]
		}
	}
	return false
}

// intKeyMapNames collects the local-variable and struct-field names declared with an
// integer-keyed map type (map[int]…, map[rune]…, map[int32]…, …) anywhere in the file —
// the lookups a hot loop can turn into a dense-slice index. Both a bare name (m[k]) and a
// selector (t.field[k]) match by this name.
func intKeyMapNames(f *ast.File) map[string]bool {
	names := map[string]bool{}
	add := func(typ ast.Expr, name string) {
		mt, ok := typ.(*ast.MapType)
		if !ok || !integerKeyType(mt.Key) {
			return
		}
		// Skip set-like maps (map[int]bool / map[int]struct{}) — almost always a sparse
		// membership set, not a dense [0,N) lookup a slice would replace.
		if id, ok := mt.Value.(*ast.Ident); ok && id.Name == "bool" {
			return
		}
		if st, ok := mt.Value.(*ast.StructType); ok && st.Fields != nil && len(st.Fields.List) == 0 {
			return
		}
		names[name] = true
	}
	ast.Inspect(f, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.Field: // struct field: `decoder map[int]string`
			for _, nm := range x.Names {
				add(x.Type, nm.Name)
			}
		case *ast.ValueSpec: // var m map[int]V  /  var m = map[int]V{}
			for i, nm := range x.Names {
				add(x.Type, nm.Name)
				// With no explicit type the map type lives on the RHS composite
				// literal (`var vjps = map[backend.Op]VJP{}`), so x.Type is nil and
				// the line above is a no-op. The AssignStmt arm below already
				// handles that shape; this mirrors it for package-level vars, which
				// is exactly where dispatch registries are declared.
				if x.Type == nil && i < len(x.Values) {
					if cl, ok := x.Values[i].(*ast.CompositeLit); ok {
						add(cl.Type, nm.Name)
					}
				}
			}
		case *ast.AssignStmt: // m := make(map[int]V) / m := map[int]V{}
			for i, rhs := range x.Rhs {
				if i >= len(x.Lhs) {
					break
				}
				id, ok := x.Lhs[i].(*ast.Ident)
				if !ok {
					continue
				}
				switch r := rhs.(type) {
				case *ast.CallExpr:
					if calleeName(r.Fun) == "make" && len(r.Args) > 0 {
						add(r.Args[0], id.Name)
					}
				case *ast.CompositeLit:
					add(r.Type, id.Name)
				}
			}
		}
		return true
	})
	return names
}

// indexedMapName returns the base name an index expression indexes: m → "m",
// t.decoder → "decoder"; "" for anything else.
func indexedMapName(e ast.Expr) string {
	switch v := e.(type) {
	case *ast.Ident:
		return v.Name
	case *ast.SelectorExpr:
		return v.Sel.Name
	}
	return ""
}

// resolveClass maps a user token — a check ID (PS1001) OR a category string
// (per-element-dispatch), case-insensitive — to its canonical category, or ""
// if it names no check. This is what makes -checks/-exclude and //perfscan:ignore
// name a check PRECISELY by its ID (the "ignore only this explicit detection"
// path) while still accepting the readable category as an alias.
func resolveClass(tok string) string {
	tok = strings.TrimSpace(tok)
	for _, c := range checks {
		if strings.EqualFold(tok, c.id) || strings.EqualFold(tok, c.category) {
			return c.category
		}
	}
	return ""
}

// fmtReflectCallees are the fmt SCAN functions: they PARSE a format string and
// REFLECT over their pointer varargs on every call, allocating heavily. Inside a
// loop over elements that is a per-element reflect+parse+alloc cost; a hand-parse or
// a cheap guard that skips the call for the common case is far cheaper (§T931:
// SPM.Decode ran fmt.Sscanf("<0x%02X>") PER TOKEN = 1.55M allocs/decode; gating it
// behind a len==6 && "<0x" prefix check → 20×). The FORMAT family (Sprintf/Sprint/…)
// is DELIBERATELY excluded: it is common and usually benign in loops (139 vs 2
// module-wide), so flagging it would drown the signal — scan-in-loop is the sharp,
// almost-always-a-bug shape.
var fmtReflectCallees = map[string]bool{
	"Sscanf": true, "Sscan": true, "Sscanln": true, // parse a string via reflection
	"Fscanf": true, "Fscan": true, "Fscanln": true, // parse a reader via reflection
}

// pkgFuncCall reports whether fun is `pkg.Name(...)` for a Name in set, returning the
// name. The pkg check keeps a same-named method on some other type from false-firing.
func pkgFuncCall(fun ast.Expr, pkg string, set map[string]bool) (string, bool) {
	sel, ok := fun.(*ast.SelectorExpr)
	if !ok {
		return "", false
	}
	id, ok := sel.X.(*ast.Ident)
	if !ok || id.Name != pkg || !set[sel.Sel.Name] {
		return "", false
	}
	return sel.Sel.Name, true
}

// fmtReflectCall reports a call to a fmtReflectCallees function on the fmt package.
func fmtReflectCall(fun ast.Expr) (string, bool) { return pkgFuncCall(fun, "fmt", fmtReflectCallees) }

// stringsAllocCallees are strings-package transforms that ALLOCATE a fresh string on
// every call. In a per-element loop that is a per-iteration allocation; the alloc-free
// idiom writes the transform straight into the caller's builder or slices in place
// (T934: SPM/Unigram Decode did strings.ReplaceAll(p,▁," ") per token → the inline
// writeUnescapedMeta, 52k→1 alloc, 1.65×).
var stringsAllocCallees = map[string]bool{
	"ReplaceAll": true, "Replace": true, "Map": true, "Repeat": true,
}

// regexpCompileCallees compile a pattern; INSIDE a loop they recompile the SAME
// pattern every iteration (a classic O(n)-compiles bug) — hoist the compile out.
var regexpCompileCallees = map[string]bool{
	"Compile": true, "MustCompile": true, "CompilePOSIX": true, "MustCompilePOSIX": true,
}

// regexpHoistFix is the -fix rewrite for PS2005: hoist a loop-invariant
// regexp.Compile of a STRING-LITERAL pattern into a fresh variable declared just
// above the loop, and replace the in-loop call with it. A compiled *regexp.Regexp
// is immutable and safe to reuse, so this is bit-identical; declaring it before the
// loop compiles once per call instead of once per iteration.
//
// It returns nil (advisory, no auto-fix) unless the sole argument is a plain string
// literal — a computed pattern may reference the loop variable, where hoisting would
// change behaviour. gofmt-style tab indentation is assumed for the inserted line.
func regexpHoistFix(fset *token.FileSet, loop ast.Node, call *ast.CallExpr) *suggestedFix {
	if len(call.Args) != 1 {
		return nil
	}
	if lit, ok := call.Args[0].(*ast.BasicLit); !ok || lit.Kind != token.STRING {
		return nil
	}
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, fset, call); err != nil {
		return nil
	}
	lp := fset.Position(loop.Pos())
	if lp.Column < 1 {
		return nil
	}
	name := fmt.Sprintf("perfscanRe%d", lp.Line) // unique per loop (loops start on distinct lines)
	indent := strings.Repeat("\t", lp.Column-1)
	return &suggestedFix{
		title: "hoist the compile above the loop (compile once, reuse)",
		edits: []textEdit{
			// insert "<name> := <call>\n<indent>" at the loop's first token (after its
			// existing indent), so the decl and the loop line up.
			{start: loop.Pos(), end: loop.Pos(), newText: name + " := " + buf.String() + "\n" + indent},
			// replace the in-loop compile call with the hoisted variable.
			{start: call.Pos(), end: call.End(), newText: name},
		},
	}
}

// leUintCallees / mathBitsCallees are per-element little-endian binary readers. On a
// LE host (every GoAI target) the on-disk LE bytes ARE the in-memory layout of a
// verbatim-bit tensor, so a per-element decode loop is a memcpy in disguise — one
// rawCopyLE replaces it (T720/T721/T907 format readers: safetensors 2×, npy 3–5×,
// gguf 3.4×).
var leUintCallees = map[string]bool{"Uint16": true, "Uint32": true, "Uint64": true}
var mathBitsCallees = map[string]bool{"Float32frombits": true, "Float64frombits": true}

// binaryDecodeCall matches the per-element bit decoders math.FloatNfrombits(...) and
// binary.{Little,Big}Endian.UintN(...), returning a printable name.
func binaryDecodeCall(fun ast.Expr) (string, bool) {
	sel, ok := fun.(*ast.SelectorExpr)
	if !ok {
		return "", false
	}
	if id, ok := sel.X.(*ast.Ident); ok && id.Name == "math" && mathBitsCallees[sel.Sel.Name] {
		return "math." + sel.Sel.Name, true
	}
	if inner, ok := sel.X.(*ast.SelectorExpr); ok && leUintCallees[sel.Sel.Name] {
		if pkg, ok := inner.X.(*ast.Ident); ok && pkg.Name == "binary" &&
			(inner.Sel.Name == "LittleEndian" || inner.Sel.Name == "BigEndian") {
			return "binary." + inner.Sel.Name + "." + sel.Sel.Name, true
		}
	}
	return "", false
}

// builderWriteCallees are the strings.Builder / bytes.Buffer append methods. A loop of
// these with no preceding .Grow doubles the backing buffer log(n) times, leaving
// throw-away buffers (T929: BPE/SPM/Unigram/WordPiece Decode, pre-size → −9%/allocs).
var builderWriteCallees = map[string]bool{
	"WriteString": true, "WriteByte": true, "WriteRune": true, "Write": true,
}

// isBuilderType reports whether e names strings.Builder or bytes.Buffer.
func isBuilderType(e ast.Expr) bool {
	sel, ok := e.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	pkg, ok := sel.X.(*ast.Ident)
	if !ok {
		return false
	}
	return (pkg.Name == "strings" && sel.Sel.Name == "Builder") ||
		(pkg.Name == "bytes" && sel.Sel.Name == "Buffer")
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
// (detector PS1003, the forest-predict case). The element type must itself be a
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

// sortClosureCallees / slicesClosureCallees are sort entry points whose comparator
// is a function value, PACKAGE-QUALIFIED (sort.* / slices.*) so the many non-sort
// homonyms — ops.Slice, tensor.Slice, a local Sort method — do NOT false-match. On
// a large slice keyed by a monotonic float/int the per-comparison indirect call
// dominates — an LSD radix on the key bits made top-p / typical sampling 1.9–2.25×
// faster (detector PS3002; formerly tools/perfscan P2).
var sortClosureCallees = map[string]bool{"Slice": true, "SliceStable": true, "Sort": true}
var slicesClosureCallees = map[string]bool{"SortFunc": true, "SortStableFunc": true}

// scalarTranscendentals are libm calls whose per-element cost dominates an
// elementwise kernel. When one dtype branch runs them scalar in a loop while the
// same kernel calls a hand-vectorized v*F32/F64 sibling for another dtype, the
// scalar branch is a candidate for the same SIMD treatment (detector PS4002). The F64
// SwiGLU SiLU was exactly this — scalar math.Exp on the f64 branch, vsiluF32 on
// the f32 branch — and an AVX2 f64 exp gave 1.52× Llama prefill.
var scalarTranscendentals = map[string]bool{
	"Exp": true, "Expm1": true, "Log": true, "Log1p": true, "Log2": true, "Log10": true,
	"Tanh": true, "Sinh": true, "Cosh": true, "Erf": true, "Erfc": true, "Pow": true,
}

// hasFuncLitArg reports whether any argument is a function literal (a closure).
func hasFuncLitArg(call *ast.CallExpr) bool {
	for _, a := range call.Args {
		if _, ok := a.(*ast.FuncLit); ok {
			return true
		}
	}
	return false
}

// hasFuncArg reports whether any argument is a function value — a literal, or an
// ident/selector that plausibly names a comparator (catches SortFunc(s, cmp)).
func hasFuncArg(call *ast.CallExpr) bool {
	for _, a := range call.Args {
		switch a.(type) {
		case *ast.FuncLit, *ast.Ident, *ast.SelectorExpr:
			return true
		}
	}
	return false
}

// ignoreDirectives parses `//perfscan:ignore [class[,class…]] [reason]` comments
// (staticcheck-style). It returns a line→categories map where a directive on line
// L suppresses matching findings on L AND L+1 (so the marker may sit ON or just
// ABOVE the reported line). The special category "*" means "all classes" — the
// bare `//perfscan:ignore` (reason-only counts as bare too, so a comment whose
// first token names no class silences everything at that site, matching the
// pre-consolidation tool). Naming ≥1 class silences ONLY those classes, leaving
// every other detector live at that site — the whole point of class-granular
// ignores. Requires the file to be parsed with parser.ParseComments.
func ignoreDirectives(fset *token.FileSet, f *ast.File) map[int]map[string]bool {
	out := map[int]map[string]bool{}
	for _, cg := range f.Comments {
		for _, c := range cg.List {
			i := strings.Index(c.Text, "perfscan:ignore")
			if i < 0 {
				continue
			}
			rest := strings.TrimSpace(c.Text[i+len("perfscan:ignore"):])
			cats := map[string]bool{}
			if fields := strings.Fields(rest); len(fields) > 0 {
				for _, tok := range strings.Split(fields[0], ",") {
					if cat := resolveClass(tok); cat != "" {
						cats[cat] = true
					}
				}
			}
			if len(cats) == 0 { // bare, or a reason-only comment with no class token
				cats["*"] = true
			}
			ln := fset.Position(c.Pos()).Line
			for _, l := range []int{ln, ln + 1} {
				if out[l] == nil {
					out[l] = map[string]bool{}
				}
				for cat := range cats {
					out[l][cat] = true
				}
			}
		}
	}
	return out
}

// scanFile runs every detector over one parsed file, drops sites suppressed by
// inline `//perfscan:ignore` directives, and returns the deduplicated findings
// (one per enclosing loop per category).
func scanFile(fset *token.FileSet, f *ast.File, ns nameSets) []finding {
	var out []finding
	if f.Name != nil {
		curPkg = f.Name.Name
	}
	// File-scoped fact: the package-local elementwise funcs that WRAP a libm
	// transcendental (softplus, mish, swish, …). Class PS4002 only sees a DIRECT math.X
	// in the loop; these hide it one call deep, so a hot per-element loop over them
	// reads as scalar-clean. Class PS4003 flags calls to them inside loops.
	wrappers := transcendentalWrappers(f)
	// File-scoped fact: names declared as integer-keyed maps (detector PS3003) — indexing
	// them in a loop is a dense-slice (map→slice) candidate.
	intKeyMaps := intKeyMapNames(f)
	for name := range intMapReg[curPkg] { // cross-file dispatch registries
		intKeyMaps[name] = true
	}
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		out = append(out, scanFunc(fset, fn, wrappers, intKeyMaps, ns)...)
	}
	// drop sites silenced by an inline //perfscan:ignore directive (per class).
	ign := ignoreDirectives(fset, f)
	var kept []finding
	for _, fd := range dedup(out) {
		if supp := ign[fd.pos.Line]; supp != nil && (supp["*"] || supp[fd.category]) {
			continue
		}
		kept = append(kept, fd)
	}
	return kept
}

// scanFunc analyzes a single function body. Fast-path presence (flatF64/flatF32)
// and Numel-derived loop bounds are function-scoped facts, so they are gathered up
// front; then each trigger call is attributed to its nearest enclosing loop.
// transcendentalWrappers collects package-local funcs that are scalar float→float
// AND call a libm transcendental in their body — the elementwise activations
// (softplus, mish, swish, erf-based …) a hot loop may invoke per element. Because
// the math.X hides one call deep, class PS4002's direct-call detector never sees it.
func transcendentalWrappers(f *ast.File) map[string]bool {
	w := map[string]bool{}
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil || fn.Recv != nil || !isScalarFloatFunc(fn.Type) {
			continue
		}
		calls := false
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			if _, ok := pkgFuncCall(call.Fun, "math", scalarTranscendentals); ok {
				calls = true
				return false
			}
			return true
		})
		if calls {
			w[fn.Name.Name] = true
		}
	}
	return w
}

// isScalarFloatFunc reports whether every parameter and the single result are a
// bare float32/float64 — an elementwise-kernel signature, safe to vectorize over a
// slice. Rejects funcs taking slices/structs (already batched or not elementwise).
func isScalarFloatFunc(t *ast.FuncType) bool {
	isFloat := func(e ast.Expr) bool {
		id, ok := e.(*ast.Ident)
		return ok && (id.Name == "float32" || id.Name == "float64")
	}
	if t.Params == nil || len(t.Params.List) == 0 {
		return false
	}
	for _, fld := range t.Params.List {
		if !isFloat(fld.Type) {
			return false
		}
	}
	return t.Results != nil && len(t.Results.List) == 1 && isFloat(t.Results.List[0].Type)
}

func scanFunc(fset *token.FileSet, fn *ast.FuncDecl, wrappers, intKeyMaps map[string]bool, ns nameSets) []finding {
	// Pass 1: function-scoped facts + a child→parent map for ancestor walks.
	hasFlat := false
	hasBuilder := false // declares a strings.Builder / bytes.Buffer
	hasGrow := false    // …and calls .Grow on it (so it is pre-sized — detector PS2002 stays silent)
	hasLEBulk := false  // calls rawCopyLE/rawStoreLE (the LE bulk-copy fast path is present, so a
	//                     per-element binary decode in the same function is the intended big-endian
	//                     fallback, not a candidate — detector PS4001 stays silent, mirroring hasFlat/A)
	hasVectorSibling := false // calls a v*F32/F64 / xN SIMD helper (another dtype branch is
	//                          hand-vectorized) → a scalar transcendental here is detector PS4002
	numelIdents := map[string]bool{}
	// escaping: locals that outlive a loop iteration — returned, or stored by reference
	// into a field/slot (recv.f = x, ring[i] = x). Such a buffer is NOT simple reusable
	// scratch (detector PS2004 stays silent), even though some, like a ring slot, are poolable
	// by a different fix (pre-allocate the slot). A make() feeding one is deliberately
	// excluded to keep N's "hoist to a reused field" advice correct.
	escaping := map[string]bool{}
	ptrMethod := isPointerMethod(fn)
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
			// A fast path is any typed BULK access to the backing store: the
			// flatF64/flatF32 helpers, or a direct Storage().F64()/.F32() slice grab
			// (the older idiom, e.g. fillSigmoidFocalConstants). Its presence means the
			// function's per-element Unravel loop is only the strided/other-dtype
			// fallback, not the hot path — so it must NOT be reported.
			cn := calleeName(x.Fun)
			if ns.fastPath[cn] { // a configured typed bulk accessor (flatF64/…)
				hasFlat = true
			}
			if cn == "Grow" { // strings.Builder/bytes.Buffer pre-size (stdlib)
				hasGrow = true
			}
			if ns.bulkCopy[cn] { // a configured LE bulk-copy helper (rawCopyLE/…)
				hasLEBulk = true
			}
			if ns.vectorized[cn] { // a configured hand-vectorized SIMD kernel
				hasVectorSibling = true
			}
		case *ast.CompositeLit:
			if isBuilderType(x.Type) { // b := strings.Builder{}
				hasBuilder = true
			}
			// A local used as a value in a struct/slice/map literal is stored by reference
			// into it and outlives the loop iteration — e.g. State{a: a} or []T{buf}. Mark it
			// escaping so detector PS2004 does not mis-flag it as reusable scratch.
			for _, elt := range x.Elts {
				switch e := elt.(type) {
				case *ast.KeyValueExpr:
					if id, ok := e.Value.(*ast.Ident); ok {
						escaping[id.Name] = true
					}
				case *ast.Ident:
					escaping[e.Name] = true
				}
			}
		case *ast.ValueSpec:
			if isBuilderType(x.Type) { // var b strings.Builder
				hasBuilder = true
			}
		case *ast.AssignStmt:
			// n := x.Numel()  /  n = x.Numel()  (Numel = a configured element-count method)
			for i, rhs := range x.Rhs {
				call, ok := rhs.(*ast.CallExpr)
				if !ok || !ns.elemCount[calleeName(call.Fun)] || i >= len(x.Lhs) {
					continue
				}
				if id, ok := x.Lhs[i].(*ast.Ident); ok {
					numelIdents[id.Name] = true
				}
			}
			for i, rhs := range x.Rhs {
				ix, ok := rhs.(*ast.IndexExpr)
				if !ok || i >= len(x.Lhs) {
					continue
				}
				call, ok := ix.X.(*ast.CallExpr)
				if !ok || !ns.shapeMethods[calleeName(call.Fun)] {
					continue
				}
				if id, ok := x.Lhs[i].(*ast.Ident); ok {
					numelIdents[id.Name] = true
				}
			}
			// A local stored by reference into a field/slot escapes the loop: recv.f = x,
			// ring[i] = x, m[p] = x. Mark the RHS ident (detector PS2004 excludes it).
			for i, lhs := range x.Lhs {
				switch lhs.(type) {
				case *ast.SelectorExpr, *ast.IndexExpr:
					if i < len(x.Rhs) {
						if id, ok := x.Rhs[i].(*ast.Ident); ok {
							escaping[id.Name] = true
						}
					}
				}
			}
		case *ast.ReturnStmt:
			for _, r := range x.Results { // a returned local escapes
				if id, ok := r.(*ast.Ident); ok {
					escaping[id.Name] = true
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
			perElemLoop[loop] = isNumelRange(loop, numelIdents, ns) || directlyHasUnravel(loop.Body, ns)
		case *ast.ForStmt:
			perElemLoop[loop] = isNumelForCond(loop, numelIdents, ns) || directlyHasUnravel(loop.Body, ns)
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
		if perElem && ns.accessors[name] && !hasFlat {
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "per-element-dispatch",
				msg: fmt.Sprintf("per-element .%s in an element-count/index loop with no configured typed bulk"+
					" accessor in %s() — walk the backing slice directly for the contiguous case, keeping the"+
					" per-element form as the strided/other-dtype fallback", name, fn.Name.Name),
			})
		}
		// B: allocation inside a per-element loop.
		if perElem && ns.allocators[name] {
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "alloc-in-loop",
				msg: fmt.Sprintf("allocation %q inside a per-element loop — turns an O(n) loop into O(n)"+
					" allocations; hoist it out of the loop or allocate once and reuse", name),
			})
		}
		// C: batch API fed a single-element nested-slice literal (a wrapped row), in any loop.
		for _, arg := range call.Args {
			if isSingleEltNestedSliceLit(arg) {
				out = append(out, finding{
					pos:      fset.Position(loop.Pos()),
					category: "batch-single-elt",
					msg: fmt.Sprintf("%q called with a single-element slice literal inside a loop"+
						" — call a single-item API instead of wrapping each element in a fresh slice", name),
				})
				break
			}
		}
		// D: a reflection-based fmt scan/format call in any loop (per-element reflection + alloc).
		if fname, ok := fmtReflectCall(call.Fun); ok {
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "reflection-in-loop",
				msg: fmt.Sprintf("fmt.%s in a loop — reflection-based and allocates on every call;"+
					" gate it behind a cheap check or hand-parse the input", fname),
			})
		}
		// E: a strings.Builder/bytes.Buffer written in a loop with NO .Grow anywhere in the
		// function — WriteX doubles the backing buffer log(n) times, leaving throw-away buffers.
		if hasBuilder && !hasGrow && builderWriteCallees[name] {
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "unsized-builder",
				msg: fmt.Sprintf("a strings.Builder/bytes.Buffer is written (.%s) in a loop with no .Grow —"+
					" pre-size it with .Grow(n) to skip the log(n) growth-buffer reallocations", name),
			})
		}
		// F: an allocating strings transform in any loop (a fresh string per iteration).
		if fname, ok := pkgFuncCall(call.Fun, "strings", stringsAllocCallees); ok {
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "strings-alloc-in-loop",
				msg: fmt.Sprintf("strings.%s in a loop — allocates a new string every call;"+
					" write the transform into a builder or byte slice in place", fname),
			})
		}
		// G: a per-element little-endian bit decode in a loop — a memcpy in disguise on LE hosts.
		// Silent when the function already has a rawCopyLE/rawStoreLE fast path (the loop is then
		// the big-endian fallback, not a candidate — the hasFlat discipline of detector PS1001).
		if bname, ok := binaryDecodeCall(call.Fun); ok && !hasLEBulk && decodesReadSlice(parent, call) {
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "le-decode-in-loop",
				msg: fmt.Sprintf("%s in a loop — on a little-endian host the on-disk bytes already match the"+
					" in-memory layout, so a single bulk copy replaces the per-element decode for verbatim-bit values;"+
					" The fix MUST be guarded by a build tag or a runtime byte-order check: an unconditional"+
					" bulk copy silently corrupts data on a big-endian host. Prove identical bytes and benchmark.", bname),
			})
		}
		// PS2005: a regexp compile inside a loop — recompiles the pattern every iteration.
		if rname, ok := pkgFuncCall(call.Fun, "regexp", regexpCompileCallees); ok {
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "regexp-compile-in-loop",
				msg: fmt.Sprintf("regexp.%s in a loop — recompiles the pattern on every iteration;"+
					" hoist the compile above the loop (compile once, match many)", rname),
				fix: regexpHoistFix(fset, loop, call),
			})
		}
		// K: a scalar libm transcendental in a loop while THIS kernel already calls a
		// hand-vectorized v*F32/F64 sibling for another dtype — the scalar branch is a
		// candidate for the same SIMD treatment (the F64 SwiGLU SiLU: scalar math.Exp
		// beside vsiluF32 → 1.52× Llama prefill once the f64 branch ran an AVX2 f64 exp).
		if tname, ok := pkgFuncCall(call.Fun, "math", scalarTranscendentals); ok && hasVectorSibling {
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "scalar-transcendental-vectorizable",
				msg: fmt.Sprintf("math.%s runs scalar in a loop while a configured vectorized sibling is"+
					" called in the same file — this scalar branch is a candidate for the same SIMD"+
					" treatment (keep a bit-identical scalar tail). First verify the op is not required to"+
					" be bit-for-bit identical to a scalar reference.", tname),
			})
		}
		// L: a loop calls a package-local elementwise helper that WRAPS a libm
		// transcendental (softplus = x+log1p(exp(-x)), mish, swish, …). K sees only a
		// DIRECT math.X, so these hide the scalar cost one call deep and a per-element
		// loop over them reads as clean. Candidate for a SIMD kernel / batched op.
		if wname := calleeName(call.Fun); wrappers[wname] {
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "transcendental-wrapper-in-loop",
				msg: fmt.Sprintf("%s(…) wraps a scalar libm transcendental and runs per element in a loop —"+
					" the same class as a raw math.Exp/Log (PS4002), just one call deep. Candidate for a"+
					" vectorized SIMD kernel or a batched op (compute the whole slice 4/8-wide, keep a"+
					" bit-identical scalar tail). Verify hotness and that the op need not match a scalar"+
					" reference bit-for-bit.", wname),
			})
		}
		return true
	})

	// N: poolable per-call scratch. A slice make() inside a per-item loop of a
	// pointer-receiver method, bound to a LOCAL that does not escape (not returned, not
	// stored into a field/slot), is scratch reallocated on every call. On a reusable
	// stateful object (an optimizer stepping over its params) that is pure GC churn —
	// hoist it to a reused receiver field (grow-on-demand). Ring slots / returned buffers
	// escape and are deliberately excluded (they need a different fix).
	if ptrMethod {
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok || !isSliceMake(call) {
				return true
			}
			loop := nearestLoop(parent, call)
			if loop == nil {
				return true
			}
			as, ok := parent[call].(*ast.AssignStmt)
			if !ok {
				return true
			}
			for i, rhs := range as.Rhs {
				if rhs != call || i >= len(as.Lhs) {
					continue
				}
				id, ok := as.Lhs[i].(*ast.Ident)
				if !ok || id.Name == "_" || escaping[id.Name] {
					continue
				}
				out = append(out, finding{
					pos:      fset.Position(loop.Pos()),
					category: "poolable-loop-scratch",
					msg: fmt.Sprintf("%q: make() per iteration of a pointer-method loop, bound to a"+
						" non-escaping local — per-call scratch reallocated every call. Hoist it to a"+
						" reused receiver field (grow-on-demand, zero only if read-before-write) so the"+
						" per-call allocations drop to zero. Verify the buffer is fully overwritten before use.", id.Name),
				})
			}
			return true
		})
	}

	// O: a divide by a loop-invariant scalar on every iteration of an element-wise loop.
	// Hoisting inv := 1/D once and multiplying is ~1.2-1.5x when the divide is standalone
	// (SoftCap VJP 1.28x, optimizer moments 1.1-1.3x). SAFE ONLY for a CONTINUOUS output
	// (gradient, moment, probability) whose ½-ulp shift rides a tolerance — NEVER when the
	// result feeds a discrete step (round/quantize/argmax). Advisory: verify float + intent.
	{
		// Idents accumulated via +=/-=/*= are REDUCTIONS (a softmax Σ, an attention denom):
		// dividing by one is usually a normalization whose divide is minor or parity-locked,
		// not the standalone config-scalar divide (cap, temp, eps, 1/√d) the recip-multiply
		// pays for. Exclude them to keep O's signal on the SoftCap-shaped wins.
		accumulated := map[string]bool{}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			a, ok := n.(*ast.AssignStmt)
			if !ok {
				return true
			}
			if a.Tok == token.ADD_ASSIGN || a.Tok == token.SUB_ASSIGN || a.Tok == token.MUL_ASSIGN {
				for _, lhs := range a.Lhs {
					if id, ok := lhs.(*ast.Ident); ok {
						accumulated[id.Name] = true
					}
				}
			}
			return true
		})
		loopAssigned := map[ast.Node]map[string]bool{} // per-loop cache
		invariantDivisor := func(loop ast.Node, div ast.Expr) (string, bool) {
			name, ok := invariantDivisorName(div)
			if !ok || accumulated[name] { // a reduction, not a config scalar
				return "", false
			}
			set := loopAssigned[loop]
			if set == nil {
				set = loopAssignedIdents(loop)
				loopAssigned[loop] = set
			}
			if set[name] { // the divisor is mutated in the loop → not invariant
				return "", false
			}
			return name, true
		}
		reported := map[ast.Node]bool{} // one finding per loop
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			var div, num ast.Expr
			switch e := n.(type) {
			case *ast.BinaryExpr:
				if e.Op == token.QUO {
					div, num = e.Y, e.X
				}
			case *ast.AssignStmt:
				if e.Tok == token.QUO_ASSIGN && len(e.Rhs) == 1 {
					div, num = e.Rhs[0], e.Lhs[0]
				}
			}
			if div == nil {
				return true
			}
			// Skip an integer index computation (a[i/stride]) — reciprocal-multiply is a
			// float transform, wrong for the discrete index it feeds.
			if ix, ok := parent[n].(*ast.IndexExpr); ok && ix.Index == n {
				return true
			}
			loop := nearestLoop(parent, n)
			if loop == nil || reported[loop] {
				return true
			}
			// Require an element-wise loop (indexes some slice) to keep scalar/integer
			// bookkeeping divides out of the signal, and skip loops already dominated by a
			// transcendental (there the divide is in the noise — and it is K/L territory).
			if !loopHasIndexAccess(loop) || loopHasExpensiveTranscendental(loop, wrappers) {
				return true
			}
			// Go forbids % on floating-point operands, so a modulo over the SAME
			// operand pair PROVES both are integers — and an integer divide is index
			// decomposition (`ni, rem := r/hw, r%hw`), where the recommended
			// `inv := 1/hw` evaluates to integer zero and following the advice would
			// silently zero the result. This is a proof, not a heuristic: a genuine
			// float divide cannot have a modulo sibling, so no true positive is lost.
			// The existing a[i/stride] guard misses these because the quotient is
			// assigned to a variable rather than used directly as an index.
			if num != nil && hasModuloSibling(fn.Body, num, div) {
				return true
			}
			name, ok := invariantDivisor(loop, div)
			if !ok {
				return true
			}
			reported[loop] = true
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "loop-invariant-divide",
				msg: fmt.Sprintf("divide by %q, loop-invariant, on every element — hoist inv := 1/%s"+
					" once and multiply. ~1.2-1.5x when the divide is standalone (SoftCap VJP 1.28x,"+
					" optimizer moments). SAFE ONLY for a CONTINUOUS output (gradient/moment/probability,"+
					" ½ulp rides tolerance) — NEVER feeding round/quantize/argmax. Verify float + intent.", name, name),
			})
			return true
		})
	}

	// PS4004: a loop whose only data statement copies one element from one slice to
	// another — dst[i] = src[j], no arithmetic on the value. That is a memmove written
	// out longhand: the run can be moved with a single copy() once the index pattern is
	// hoisted out of the loop. Found by the ref broadcast work, where the innermost axis
	// of a broadcast is either a verbatim contiguous row (source stride 1) or one value
	// repeated (stride 0); hoisting it out of the per-element odometer measured 4.49x on
	// BenchmarkBroadcastF64_256to256x256, bit-identically — a same-dtype copy performs no
	// arithmetic, so there is no accumulation order to disturb.
	//
	// Silent when the loop body contains ANY call: a body already reaching for copy(),
	// a helper, or a conversion is either fixed or is doing real per-element work, and
	// flagging it would be noise. Requiring distinct source and destination idents keeps
	// in-place shuffles (dst[i] = dst[j]) out, since those are permutations, not moves.
	{
		reportedCopy := map[ast.Node]bool{} // one finding per loop
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			as, ok := n.(*ast.AssignStmt)
			if !ok || as.Tok != token.ASSIGN || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
				return true
			}
			lhs, ok := as.Lhs[0].(*ast.IndexExpr)
			if !ok {
				return true
			}
			rhs, ok := as.Rhs[0].(*ast.IndexExpr) // a BARE index: no BinaryExpr, no call
			if !ok {
				return true
			}
			dstID, ok1 := lhs.X.(*ast.Ident)
			srcID, ok2 := rhs.X.(*ast.Ident)
			if !ok1 || !ok2 || dstID.Name == srcID.Name {
				return true
			}
			loop := nearestLoop(parent, n)
			if loop == nil || reportedCopy[loop] || loopBodyHasCall(loop) {
				return true
			}
			// Counted-loop gate. Without it the detector also flags rank-sized setup
			// loops (`for a := range shape { eff[a] = strides[a] }`), structurally
			// identical but running 2-4 times, where a bulk copy is noise rather than a
			// win. A copy worth hoisting is driven by an element COUNT — a three-clause
			// `for i := 0; i < n; i++`, or a range over a call such as `range t.Numel()`
			// — whereas ranging over a named container is the shape/rank idiom. This
			// keeps PS4004 a language-shape check that needs no repo configuration.
			if !isCountedLoop(loop) {
				return true
			}
			// The copy must be UNCONDITIONAL. A guarded store (`if ok { ds[i] = gs[of] }`)
			// is a filtered scatter, not a run that can be moved with one copy(), and
			// flagging it is noise. Reject if any conditional sits between the
			// assignment and its loop.
			if conditionalBefore(parent, n, loop) {
				return true
			}
			// A guarded FALLBACK is not a missed bulk copy. Where an if/else already
			// moves this same dst/src pair with copy() on the contiguous side, the
			// flagged loop IS the strided alternative that has to stay — reporting it
			// asks for a change that would be wrong. loopBodyHasCall cannot see this,
			// because the copy() sits in the sibling branch, outside the loop body.
			if siblingBranchBulkCopies(parent, loop, dstID.Name, srcID.Name) {
				return true
			}
			reportedCopy[loop] = true
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "scalar-copy-loop",
				msg: fmt.Sprintf("%s[...] = %s[...] in a loop with no arithmetic on the value — an"+
					" element-at-a-time memmove. Where the index pattern has a contiguous run, hoist the"+
					" run out of the loop and move it with one copy() (a constant source index becomes a"+
					" fill). Bit-identical by construction: a same-dtype copy does no arithmetic, so there"+
					" is no accumulation order to change. Verify the run length before acting — a wrong"+
					" run is a wrong-value bug, not a rounding one.", dstID.Name, srcID.Name),
			})
			return true
		})
	}

	// PS4005: a loop that ticks an N-D coordinate ODOMETER once per element —
	//
	//     for pos := range xs { <one element of work>
	//         for d := nd - 1; d >= 0; d-- { idx[d]++; off += stride[d]; ... } }
	//
	// The innermost axis has a CONSTANT effective stride across a full run of
	// shape[nd-1] elements, so that run is one straight walk (or a copy/fill when the
	// stride is 1 or 0) and the odometer only has to tick once per run. Hoisting it
	// leaves traversal order and every per-element operation untouched, and over a
	// full run the innermost axis contributes inner*stride - stride*inner = 0 to the
	// offset, so the outer tick is unchanged — the transform is bit-identical by
	// construction rather than by rounding argument.
	//
	// Measured three times in this tree: backend/ref/broadcast.go 4.49x,
	// backend/cpu/elementwise.go 5.29x, tensor/gatherCast 3.14x (interleaved A/B).
	//
	// Matched on the odometer's SHAPE, not on what the loop body does, so it fires
	// regardless of whether the per-element work is a copy, a cast, or an accumulate —
	// the three sites above differ in exactly that respect. The descending `d--` walk
	// with `idx[d]++` is specific enough that a plain reverse loop does not match.
	{
		reportedOdo := map[ast.Node]bool{}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			odo, ok := n.(*ast.ForStmt)
			if !ok || !isDescendingAxisWalk(odo) {
				return true
			}
			// An ALREADY-HOISTED odometer keeps this exact shape — it just runs once
			// per run instead of once per element — so shape alone would flag the fix
			// as the defect. The discriminator is where the walk starts: a per-element
			// odometer ticks every axis (`d := nd - 1`), while a hoisted one skips the
			// innermost (`d := nd - 2`) because that axis was lifted into the run.
			if startsBelowInnermost(odo) {
				return true
			}
			outer := nearestLoop(parent, n)
			if outer == nil || reportedOdo[outer] {
				return true
			}
			reportedOdo[outer] = true
			out = append(out, finding{
				pos:      fset.Position(outer.Pos()),
				category: "per-element-odometer",
				msg: "an N-D coordinate odometer ticked once per ELEMENT — the innermost axis has a" +
					" constant stride across a run, so hoist it out and tick the odometer once per run" +
					" instead (stride 1 becomes a copy, stride 0 a fill, anything else a straight strided" +
					" walk). Bit-identical by construction: traversal order and the per-element work are" +
					" unchanged, and the innermost axis contributes zero to the offset over a full run." +
					" Verify the run length and that the enclosing loop is not already a specialized" +
					" fast path before acting (PS4005)",
			})
			return true
		})
	}

	// PS4006: a matrix held as [][]T — one heap allocation per ROW — and then indexed
	// two-deep (m[i][j]) inside a nested loop. Every step of the inner loop
	// dereferences a row pointer to memory that the allocator may have placed
	// anywhere, so a column walk (m[k][p] with k varying) touches n unrelated cache
	// lines. One flat [rows*cols] buffer indexed m[i*cols+j] makes the same walk a
	// constant stride through a single allocation and collapses rows+1 allocations
	// to 1. The transform is index arithmetic only — same operands, same order — so
	// it is bit-identical.
	//
	// Measured twice in this tree: backend/ref/cholesky.go 1.5x with allocations
	// 137 -> 10, and internal/linalg SymEig's Jacobi sweep 1.2x with 277 -> 149.
	//
	// Requires BOTH the per-row allocation loop and a two-deep index inside a nested
	// loop. A [][]T that is merely passed around, or indexed once outside a loop, is
	// a legitimate ragged structure and not a candidate.
	{
		for name := range rowAllocMatrices(fn) {
			if pos, ok := nestedDoubleIndex(fn, name); ok {
				out = append(out, finding{
					pos:      fset.Position(pos),
					category: "row-slice-matrix",
					msg: fmt.Sprintf("%s is a [][]T built one row per allocation and then indexed"+
						" two-deep inside a nested loop — every inner step dereferences a separate row"+
						" pointer, and a column walk touches one cache line per row. Flatten to a single"+
						" [rows*cols] buffer indexed %s[i*cols+j]; index arithmetic only, so it is"+
						" bit-identical. Measured 1.5x (cholesky) and 1.2x (SymEig). Check first that the"+
						" rows are uniform length — a genuinely ragged matrix cannot flatten.", name, name),
				})
			}
		}
	}

	// Pass 4: call-level detectors that are NOT loop-nested — the visitor/sort call
	// IS itself the per-element / per-comparison hot loop.
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		name := calleeName(call.Fun)
		// I: a per-element visitor (readGen/fillGen/…) fed a closure → an indirect call
		// per element even on its contiguous branch.
		if ns.visitors[name] && hasFuncLitArg(call) {
			out = append(out, finding{
				pos:      fset.Position(call.Pos()),
				category: "per-element-closure",
				msg: fmt.Sprintf("%s(…) invokes a closure per element — an indirect call per element."+
					" Add a raw-slice tight loop for the contiguous case, keeping the closure as the"+
					" fallback. Verify bit-identical + benchmark.", name),
			})
		}
		// J: a closure-comparator sort → a per-comparison indirect call; radix candidate
		// when the key is a monotonic float/int over a large slice. Package-qualified to
		// sort.*/slices.* so ops.Slice / tensor.Slice do not false-match.
		sname, sortOK := pkgFuncCall(call.Fun, "sort", sortClosureCallees)
		if !sortOK {
			sname, sortOK = pkgFuncCall(call.Fun, "slices", slicesClosureCallees)
		}
		if sortOK && hasFuncArg(call) {
			out = append(out, finding{
				pos:      fset.Position(call.Pos()),
				category: "closure-comparator-sort",
				msg: fmt.Sprintf("%s uses an indirect comparator. An LSD radix on the key bits can"+
					" replace it (math.Float64bits is monotonic for non-negative f64) — measured 1.9–2.25×"+
					" on top-p / typical sampling. BOTH preconditions must hold, and this check can verify"+
					" NEITHER: (1) the sort key is a numeric float/int, not a string or a composite —"+
					" radix-on-float-bits does not apply to a string key at all; (2) the slice is long"+
					" (vocab-sized), not rank- or dimension-sized — on a short slice the radix loses and"+
					" the measurement is noise. Confirm both by reading the site before acting, then prove"+
					" identical output order and benchmark.", sname),
			})
		}
		return true
	})

	// Pass 5 (detector PS3003): a READ of an integer-keyed map inside a loop. When the keys
	// are dense over [0,N), a []T indexed by the key replaces the per-lookup hash. One
	// finding per loop; pure map-build writes (m[k] = v) are skipped.
	if len(intKeyMaps) > 0 {
		seen := map[token.Pos]bool{}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			idx, ok := n.(*ast.IndexExpr)
			if !ok || !intKeyMaps[indexedMapName(idx.X)] {
				return true
			}
			if as, ok := parent[idx].(*ast.AssignStmt); ok { // skip m[k] = v (build), keep reads incl comma-ok
				for _, l := range as.Lhs {
					if l == idx {
						return true
					}
				}
			}
			if _, ok := parent[idx].(*ast.IncDecStmt); ok {
				return true
			}
			loop := nearestLoop(parent, idx)
			if loop == nil || seen[loop.Pos()] {
				return true
			}
			seen[loop.Pos()] = true
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "int-key-map-in-loop",
				msg: fmt.Sprintf("%q (an integer-keyed map) is read in a loop — if the keys are dense over"+
					" [0,N) a []T indexed by the key replaces the per-lookup hash. Verify key density; gaps or"+
					" out-of-range keys keep the map's zero-value via a bounds check.", indexedMapName(idx.X)),
			})
			return true
		})
	}
	return out
}

// isNumelRange reports whether a range loop iterates a tensor's element count:
// `for … range x.Numel()` or `for … range n` with n bound from a .Numel() call.
func isNumelRange(r *ast.RangeStmt, numelIdents map[string]bool, ns nameSets) bool {
	switch x := r.X.(type) {
	case *ast.CallExpr:
		return ns.elemCount[calleeName(x.Fun)]
	case *ast.Ident:
		return numelIdents[x.Name]
	}
	return false
}

// isNumelForCond reports whether a 3-clause for loop is bounded by an
// element-count-derived count: `for i := 0; i < n; i++` with n from a configured
// element-count method (either operand).
func isNumelForCond(f *ast.ForStmt, numelIdents map[string]bool, ns nameSets) bool {
	bin, ok := f.Cond.(*ast.BinaryExpr)
	if !ok {
		return false
	}
	for _, side := range []ast.Expr{bin.X, bin.Y} {
		if id, ok := side.(*ast.Ident); ok && numelIdents[id.Name] {
			return true
		}
		if call, ok := side.(*ast.CallExpr); ok && ns.elemCount[calleeName(call.Fun)] {
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
func directlyHasUnravel(body *ast.BlockStmt, ns nameSets) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}
		switch n.(type) {
		case *ast.RangeStmt, *ast.ForStmt, *ast.FuncLit:
			return false // nested scope: its Unravel belongs to that loop, not this one
		}
		if call, ok := n.(*ast.CallExpr); ok && ns.indexDecompose[calleeName(call.Fun)] {
			found = true
			return false
		}
		return true
	})
	return found
}

// nearestLoop walks parent links up from n to the innermost enclosing Range/For
// (nil if n is not inside a loop).
// conditionalBefore reports whether an if/switch/select sits between node n and
// its enclosing loop. Detector PS4004 uses it to reject guarded stores: a copy that
// only happens on some iterations is a filtered scatter, not a contiguous run, so
// no bulk copy can replace it.
func conditionalBefore(parent map[ast.Node]ast.Node, n, loop ast.Node) bool {
	for p := parent[n]; p != nil && p != loop; p = parent[p] {
		switch p.(type) {
		case *ast.IfStmt, *ast.SwitchStmt, *ast.TypeSwitchStmt, *ast.SelectStmt, *ast.CaseClause:
			return true
		}
	}
	return false
}

// isCountedLoop reports whether a loop is driven by an element count rather than
// by a named container. A three-clause `for i := 0; i < n; i++` and a range over a
// call (`for i := range t.Numel()`) both iterate a count; `for a := range shape`
// walks a rank-sized container. Detector PS4004 uses the distinction to separate a
// per-element copy worth hoisting from a 2-4 iteration setup loop.
func isCountedLoop(loop ast.Node) bool {
	switch l := loop.(type) {
	case *ast.ForStmt:
		return l.Cond != nil
	case *ast.RangeStmt:
		switch l.X.(type) {
		case *ast.CallExpr, *ast.BasicLit:
			return true
		}
	}
	return false
}

// loopBodyHasCall reports whether the loop body contains any call expression.
// Detector PS4004 uses it as its silence condition: a copy loop that already
// reaches for copy(), a helper, or a per-element conversion is either fixed
// already or is doing genuine work, and flagging it would be noise.
func loopBodyHasCall(loop ast.Node) bool {
	var body *ast.BlockStmt
	switch l := loop.(type) {
	case *ast.RangeStmt:
		body = l.Body
	case *ast.ForStmt:
		body = l.Body
	}
	if body == nil {
		return false
	}
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return !found
		}
		// len/cap are free index bookkeeping, not work — suppressing on them would
		// hide real copy loops whose bound is computed from a slice length.
		if id, ok := call.Fun.(*ast.Ident); ok && (id.Name == "len" || id.Name == "cap") {
			return true
		}
		found = true
		return false
	})
	return found
}

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
	strict := flag.Bool("strict", false, "exit 1 if any finding is reported (optional CI gating)")
	tests := flag.Bool("tests", false, "include _test.go files")
	checksFlag := flag.String("checks", "", "comma-separated allow-list of checks to run, by ID (PS1001) or category; empty = all")
	exclude := flag.String("exclude", "", "comma-separated checks to silence, by ID (PS1001) or category")
	list := flag.Bool("list", false, "list the checks (ID, category, title, fixable) and exit")
	doFix := flag.Bool("fix", false, "apply the safe mechanical fixes in place (fixable checks only; see -list)")
	jsonOut := flag.Bool("json", false, "emit findings and fixes as JSON (for editor / tool integration)")
	configPath := flag.String("config", "", "path to a JSON config naming a project's element accessors/allocators/etc. that drive the domain checks (PS1001/PS1002/PS2001/PS4002); empty = discover perfscan.json/.perfscan.json upward, else stdlib-only")
	flag.Parse()

	// Load the project vocabulary that activates the domain checks. An explicit
	// -config wins; otherwise discover perfscan.json/.perfscan.json upward from the
	// working directory; otherwise stay in the fully generic, stdlib-only mode.
	var cfg Config
	if *configPath != "" {
		c, err := loadConfig(*configPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "perfscan: -config:", err)
			os.Exit(2)
		}
		cfg = c
	} else {
		cfg, _ = discoverConfig()
	}
	ns := cfg.compile()

	if *list {
		fmt.Println("perfscan checks — name any in -checks, -exclude, or a //perfscan:ignore directive (by ID or category):")
		fmt.Printf("  %-8s %-4s %-38s %s\n", "ID", "FIX", "CATEGORY", "TITLE")
		for _, c := range checks {
			fix := "—"
			if c.fixable {
				fix = "yes"
			}
			fmt.Printf("  %-8s %-4s %-38s %s\n", c.id, fix, c.category, c.title)
		}
		return
	}

	// Build the enabled set: -checks is an allow-list (empty ⇒ all checks), then
	// -exclude subtracts. Both accept a check ID (PS1001) or its category alias, so a
	// single explicit detection can be silenced precisely without touching the others.
	enabled := map[string]bool{}
	if toks := splitTokens(*checksFlag); len(toks) > 0 {
		for _, tok := range toks {
			cat := resolveClass(tok)
			if cat == "" {
				fmt.Fprintf(os.Stderr, "perfscan: -checks: unknown check %q (see -list)\n", tok)
				os.Exit(2)
			}
			enabled[cat] = true
		}
	} else {
		for _, c := range checks {
			enabled[c.category] = true
		}
	}
	for _, tok := range splitTokens(*exclude) {
		cat := resolveClass(tok)
		if cat == "" {
			fmt.Fprintf(os.Stderr, "perfscan: -exclude: unknown check %q (see -list)\n", tok)
			os.Exit(2)
		}
		delete(enabled, cat)
	}

	roots := flag.Args()
	if len(roots) == 0 {
		roots = []string{"./..."}
	}
	// A DOMAIN check is inert without its vocabulary: with no elementAccessors
	// configured, PS1001/PS1002 can never report, and the scan prints a clean
	// "no candidates" that reads as "no instances". That false assurance cost a
	// multi-round investigation into a rule that was not broken — it was simply
	// running with an empty accessor set because perfscan.json lives inside
	// internal/perfscan/ and config discovery walks UPWARD from the working
	// directory, so an invocation from the repo root never finds it. Warn loudly
	// rather than report a silent zero.
	domainVocab := map[string]map[string]bool{
		"per-element-dispatch":               ns.accessors,
		"per-element-closure":                ns.visitors,
		"alloc-in-loop":                      ns.allocators,
		"scalar-transcendental-vectorizable": ns.vectorized,
	}
	var starved []string
	for cat, vocab := range domainVocab {
		if enabled[cat] && len(vocab) == 0 {
			starved = append(starved, catToID[cat])
		}
	}
	if len(starved) > 0 {
		sort.Strings(starved)
		fmt.Fprintf(os.Stderr, "perfscan: WARNING: %s %s enabled but its vocabulary is empty — "+
			"these checks CANNOT report and a zero result here means nothing. Pass "+
			"-config <perfscan.json> (this repo: -config internal/perfscan/perfscan.json, "+
			"or run `make perfscan`).\n",
			strings.Join(starved, ", "), map[bool]string{true: "are", false: "is"}[len(starved) > 1])
	}

	files, err := goFiles(roots, *tests)
	if err != nil {
		fmt.Fprintln(os.Stderr, "perfscan:", err)
		os.Exit(2)
	}

	fset := token.NewFileSet()
	var all []finding
	// Parse everything first: the named-integer-type registry PS3003 needs is a
	// REPO-scoped fact (a map key can name a type declared in another package),
	// so it must be complete before any file is judged.
	parsed := make([]*ast.File, 0, len(files))
	for _, path := range files {
		f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if err != nil {
			fmt.Fprintf(os.Stderr, "perfscan: parse %s: %v\n", path, err)
			continue
		}
		parsed = append(parsed, f)
	}
	collectIntTypes(parsed)
	collectIntKeyMaps(parsed)
	for _, f := range parsed {
		for _, fd := range scanFile(fset, f, ns) {
			if enabled[fd.category] {
				fd.id = catToID[fd.category]
				all = append(all, fd)
			}
		}
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].pos.Filename != all[j].pos.Filename {
			return all[i].pos.Filename < all[j].pos.Filename
		}
		if all[i].pos.Line != all[j].pos.Line {
			return all[i].pos.Line < all[j].pos.Line
		}
		return all[i].id < all[j].id
	})

	if *jsonOut {
		emitJSON(fset, all)
		if *strict && len(all) > 0 {
			os.Exit(1)
		}
		return
	}

	if *doFix {
		nFix, nFile := applyFixes(fset, all)
		nSkip := 0
		for _, f := range all {
			if f.fix == nil {
				nSkip++
			}
		}
		fmt.Printf("perfscan: applied %d fix(es) across %d file(s); %d finding(s) have no safe mechanical fix (advisory — see -list)\n", nFix, nFile, nSkip)
		fmt.Println("NOTE: even applied fixes need an A/B + bit-identity check before shipping (§C3/§V22); review the diff.")
		return
	}

	// Text report — staticcheck-style "pos: message (PS1001)"; a fixable finding is
	// tagged so -fix / an editor quick-fix is discoverable.
	byID := map[string]int{}
	nFixable := 0
	for _, f := range all {
		tag := f.id
		if f.fix != nil {
			tag = f.id + ", fixable"
			nFixable++
		}
		fmt.Printf("%s: %s (%s)\n", f.pos, f.msg, tag)
		byID[f.id]++
	}
	if len(all) == 0 {
		fmt.Println("perfscan: no candidate anti-patterns found")
		return
	}
	var counts []string
	for _, c := range checks {
		if n := byID[c.id]; n > 0 {
			counts = append(counts, fmt.Sprintf("%s=%d", c.id, n))
		}
	}
	fmt.Printf("\nperfscan: %d candidate(s) — %s\n", len(all), strings.Join(counts, " "))
	if nFixable > 0 {
		fmt.Printf("perfscan: %d have a safe mechanical fix — run `perfscan -fix` (or `-json` for an editor)\n", nFixable)
	}
	fmt.Println("NOTE: candidates, not confirmed wins — measure hotness (§C3) + prove bit-identity (§V22) before shipping.")
	if *strict {
		os.Exit(1)
	}
}

// splitTokens splits a comma list, trimming blanks.
func splitTokens(s string) []string {
	var out []string
	for _, tok := range strings.Split(s, ",") {
		if tok = strings.TrimSpace(tok); tok != "" {
			out = append(out, tok)
		}
	}
	return out
}

// applyFixes rewrites every finding that carries a suggestedFix, in place. Edits
// are grouped per file and applied HIGH-offset-first so earlier offsets stay valid.
// It returns (edits applied, files touched). Overlapping edits within a file are
// dropped defensively (first-wins) so a fix can never corrupt the source.
func applyFixes(fset *token.FileSet, all []finding) (int, int) {
	type off struct {
		start, end int
		text       string
	}
	perFile := map[string][]off{}
	for _, f := range all {
		if f.fix == nil {
			continue
		}
		for _, e := range f.fix.edits {
			ps, pe := fset.Position(e.start), fset.Position(e.end)
			perFile[ps.Filename] = append(perFile[ps.Filename], off{ps.Offset, pe.Offset, e.newText})
		}
	}
	nEdit, nFile := 0, 0
	for name, edits := range perFile {
		src, err := os.ReadFile(name)
		if err != nil {
			fmt.Fprintf(os.Stderr, "perfscan -fix: read %s: %v\n", name, err)
			continue
		}
		sort.Slice(edits, func(i, j int) bool { return edits[i].start > edits[j].start })
		out := append([]byte(nil), src...)
		lastStart := len(src) + 1
		applied := 0
		for _, e := range edits {
			if e.start < 0 || e.end > len(out) || e.start > e.end || e.end > lastStart {
				continue // out of range or overlaps a later (already-applied) edit
			}
			out = append(out[:e.start], append([]byte(e.text), out[e.end:]...)...)
			lastStart = e.start
			applied++
		}
		if applied == 0 {
			continue
		}
		if err := os.WriteFile(name, out, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "perfscan -fix: write %s: %v\n", name, err)
			continue
		}
		nEdit += applied
		nFile++
	}
	return nEdit, nFile
}

// jsonEdit / jsonFix / jsonFinding are the -json wire shapes. Positions are 1-based
// line and 1-based column (byte); ranges are half-open. An editor can build a
// quick-fix straight from fix.edits, and a range from {line,col}→{endLine,endCol}.
type jsonEdit struct {
	Line    int    `json:"line"`
	Col     int    `json:"col"`
	EndLine int    `json:"endLine"`
	EndCol  int    `json:"endCol"`
	Offset  int    `json:"offset"`
	EndOff  int    `json:"endOffset"`
	NewText string `json:"newText"`
}

type jsonFix struct {
	Title string     `json:"title"`
	Edits []jsonEdit `json:"edits"`
}

type jsonFinding struct {
	ID       string   `json:"id"`
	Category string   `json:"category"`
	File     string   `json:"file"`
	Line     int      `json:"line"`
	Col      int      `json:"col"`
	EndLine  int      `json:"endLine,omitempty"`
	EndCol   int      `json:"endCol,omitempty"`
	Message  string   `json:"message"`
	Fixable  bool     `json:"fixable"`
	Fix      *jsonFix `json:"fix,omitempty"`
}

// emitJSON writes the findings as a JSON array to stdout.
func emitJSON(fset *token.FileSet, all []finding) {
	out := make([]jsonFinding, 0, len(all))
	for _, f := range all {
		jf := jsonFinding{
			ID: f.id, Category: f.category, File: f.pos.Filename,
			Line: f.pos.Line, Col: f.pos.Column, Message: f.msg,
			Fixable: f.fix != nil,
		}
		if f.end.IsValid() {
			jf.EndLine, jf.EndCol = f.end.Line, f.end.Column
		}
		if f.fix != nil {
			jx := &jsonFix{Title: f.fix.title}
			for _, e := range f.fix.edits {
				ps, pe := fset.Position(e.start), fset.Position(e.end)
				jx.Edits = append(jx.Edits, jsonEdit{
					Line: ps.Line, Col: ps.Column, EndLine: pe.Line, EndCol: pe.Column,
					Offset: ps.Offset, EndOff: pe.Offset, NewText: e.newText,
				})
			}
			jf.Fix = jx
		}
		out = append(out, jf)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(out); err != nil {
		fmt.Fprintln(os.Stderr, "perfscan -json:", err)
		os.Exit(2)
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

// siblingBranchBulkCopies reports whether an if/else alternative to the branch
// holding this loop already moves the same dst/src pair with copy(). That shape —
// `if stride == 1 { copy(dst, src[...]) } else { for ... { dst[i] = src[...] } }` —
// is a deliberate contiguous/strided split, and its else arm must stay exactly as
// written. Matching on the dst/src pair rather than on any copy() keeps an unrelated
// copy elsewhere in the sibling branch from silencing a genuine finding.
func siblingBranchBulkCopies(parent map[ast.Node]ast.Node, loop ast.Node, dst, src string) bool {
	child := loop
	for p := parent[child]; p != nil; child, p = p, parent[p] {
		ifs, ok := p.(*ast.IfStmt)
		if !ok {
			continue
		}
		var other ast.Node
		switch child {
		case ast.Node(ifs.Body):
			other = ifs.Else
		case ast.Node(ifs.Else):
			other = ifs.Body
		default:
			continue
		}
		if other != nil && blockHasBulkCopy(other, dst, src) {
			return true
		}
	}
	return false
}

// blockHasBulkCopy reports whether n contains copy(dst..., src...) over the given
// base identifiers, ignoring any indexing or slicing applied to them.
func blockHasBulkCopy(n ast.Node, dst, src string) bool {
	found := false
	ast.Inspect(n, func(x ast.Node) bool {
		if found {
			return false
		}
		call, ok := x.(*ast.CallExpr)
		if !ok {
			return true
		}
		id, ok := call.Fun.(*ast.Ident)
		if !ok || id.Name != "copy" || len(call.Args) != 2 {
			return true
		}
		if baseIdentName(call.Args[0]) == dst && baseIdentName(call.Args[1]) == src {
			found = true
		}
		return !found
	})
	return found
}

// baseIdentName peels index and slice expressions down to the underlying
// identifier: x, x[i] and x[a:b] all yield "x". Empty for anything else.
func baseIdentName(e ast.Expr) string {
	for {
		switch v := e.(type) {
		case *ast.Ident:
			return v.Name
		case *ast.IndexExpr:
			e = v.X
		case *ast.SliceExpr:
			e = v.X
		case *ast.ParenExpr:
			e = v.X
		default:
			return ""
		}
	}
}

// isDescendingAxisWalk reports whether a for-statement is the axis-ticking half of an
// N-D odometer: `for d := <expr>; d >= 0; d--` whose body increments a container at
// index d (`idx[d]++`). Detector PS4005 uses it to find odometers ticked per element.
// Requiring all three of the descending post, the `>= 0` bound and the indexed
// increment keeps ordinary reverse loops out.
func isDescendingAxisWalk(f *ast.ForStmt) bool {
	if f.Cond == nil || f.Post == nil || f.Body == nil {
		return false
	}
	cond, ok := f.Cond.(*ast.BinaryExpr)
	if !ok || cond.Op != token.GEQ {
		return false
	}
	axis, ok := cond.X.(*ast.Ident)
	if !ok {
		return false
	}
	if lit, ok := cond.Y.(*ast.BasicLit); !ok || lit.Value != "0" {
		return false
	}
	post, ok := f.Post.(*ast.IncDecStmt)
	if !ok || post.Tok != token.DEC {
		return false
	}
	if id, ok := post.X.(*ast.Ident); !ok || id.Name != axis.Name {
		return false
	}
	ticks := false
	ast.Inspect(f.Body, func(n ast.Node) bool {
		inc, ok := n.(*ast.IncDecStmt)
		if !ok || inc.Tok != token.INC || ticks {
			return !ticks
		}
		ix, ok := inc.X.(*ast.IndexExpr)
		if !ok {
			return true
		}
		if id, ok := ix.Index.(*ast.Ident); ok && id.Name == axis.Name {
			ticks = true
		}
		return !ticks
	})
	return ticks
}

// startsBelowInnermost reports whether an odometer's axis walk begins at `<expr> - 2`
// rather than `<expr> - 1`, which is the signature of an odometer whose innermost
// axis has already been hoisted into a run. Detector PS4005 uses it so that applying
// its own recommendation silences it, instead of the fixed code reporting forever.
func startsBelowInnermost(f *ast.ForStmt) bool {
	as, ok := f.Init.(*ast.AssignStmt)
	if !ok || len(as.Rhs) != 1 {
		return false
	}
	be, ok := as.Rhs[0].(*ast.BinaryExpr)
	if !ok || be.Op != token.SUB {
		return false
	}
	lit, ok := be.Y.(*ast.BasicLit)
	return ok && lit.Value == "2"
}

// hasModuloSibling reports whether fn contains `num % div` over the same operand
// expressions as a divide. Detector PS5001 uses it as an integer PROOF: Go rejects
// % on floats at compile time, so the pair cannot be floating-point.
func hasModuloSibling(body *ast.BlockStmt, num, div ast.Expr) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}
		be, ok := n.(*ast.BinaryExpr)
		if !ok || be.Op != token.REM {
			return true
		}
		if exprEqual(be.X, num) && exprEqual(be.Y, div) {
			found = true
		}
		return !found
	})
	return found
}

// exprEqual compares two expressions structurally over the shapes a divisor can
// take, so `r % (ho * wo)` matches `r / (ho * wo)` regardless of parenthesization.
// Conservative: anything it does not model compares unequal, which can only leave a
// finding reported.
func exprEqual(a, b ast.Expr) bool {
	for {
		if p, ok := a.(*ast.ParenExpr); ok {
			a = p.X
			continue
		}
		if p, ok := b.(*ast.ParenExpr); ok {
			b = p.X
			continue
		}
		break
	}
	switch x := a.(type) {
	case *ast.Ident:
		y, ok := b.(*ast.Ident)
		return ok && x.Name == y.Name
	case *ast.BasicLit:
		y, ok := b.(*ast.BasicLit)
		return ok && x.Kind == y.Kind && x.Value == y.Value
	case *ast.BinaryExpr:
		y, ok := b.(*ast.BinaryExpr)
		return ok && x.Op == y.Op && exprEqual(x.X, y.X) && exprEqual(x.Y, y.Y)
	case *ast.SelectorExpr:
		y, ok := b.(*ast.SelectorExpr)
		return ok && x.Sel.Name == y.Sel.Name && exprEqual(x.X, y.X)
	}
	return false
}

// decodesReadSlice reports whether a decode call both STORES its result verbatim
// into an element (`dst[i] = decode(...)`) and READS its bits straight out of a
// slice (`decode(src[i])`, `decode(raw[o:o+2])`). A bulk copy can only replace the
// loop when both hold: anything applied to the value on the way out, and anything
// computed on the way in, is work no memmove reproduces.
//
// Both halves are load-bearing against real code in this tree. Without the store
// half, the quant block-scale reads qualify (`d := f16ToF32(Uint16(blk))` — one
// scale per 32 elements, feeding a conversion). Without the read half, two more
// classes qualify: the IQ sign trick
// (`y[k] = Float32frombits(Float32bits(db*grid[k]) ^ sbit)`, arithmetic wearing a
// bit-cast) and the radix key inversion in classic/gbm_hist.go
// (`col[i] = Float64frombits(u)` after u was conditionally complemented).
func decodesReadSlice(parent map[ast.Node]ast.Node, call *ast.CallExpr) bool {
	if !storesVerbatim(parent, call) || len(call.Args) != 1 {
		return false
	}
	switch call.Args[0].(type) {
	case *ast.IndexExpr, *ast.SliceExpr:
		return true
	}
	return false
}

// storesVerbatim reports whether a call is the direct right-hand side of an
// assignment into an index expression — `dst[i] = decode(...)`.
func storesVerbatim(parent map[ast.Node]ast.Node, call ast.Node) bool {
	as, ok := parent[call].(*ast.AssignStmt)
	if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 || as.Rhs[0] != call {
		return false
	}
	_, ok = as.Lhs[0].(*ast.IndexExpr)
	return ok
}

// rowAllocMatrices returns the names of locals assigned from make([][]T, …) that are
// later filled one row at a time (`m[i] = make([]T, …)` inside a loop). Detector
// PS4006 needs both halves: the outer make alone could be a ragged or borrowed
// structure, while the per-row make is what turns a matrix into n allocations.
func rowAllocMatrices(fn *ast.FuncDecl) map[string]bool {
	outer := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		id, ok := as.Lhs[0].(*ast.Ident)
		if !ok {
			return true
		}
		call, ok := as.Rhs[0].(*ast.CallExpr)
		if !ok || len(call.Args) == 0 {
			return true
		}
		if fn, ok := call.Fun.(*ast.Ident); !ok || fn.Name != "make" {
			return true
		}
		if at, ok := call.Args[0].(*ast.ArrayType); ok {
			if _, inner := at.Elt.(*ast.ArrayType); inner {
				outer[id.Name] = true
			}
		}
		return true
	})
	filled := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		ix, ok := as.Lhs[0].(*ast.IndexExpr)
		if !ok {
			return true
		}
		id, ok := ix.X.(*ast.Ident)
		if !ok || !outer[id.Name] {
			return true
		}
		call, ok := as.Rhs[0].(*ast.CallExpr)
		if !ok {
			return true
		}
		if f, ok := call.Fun.(*ast.Ident); ok && f.Name == "make" {
			filled[id.Name] = true
		}
		return true
	})
	return filled
}

// nestedDoubleIndex reports the position of a two-deep index on name (m[i][j])
// occurring inside at least two nested loops — the region where the row-pointer
// dereference is paid repeatedly rather than once.
func nestedDoubleIndex(fn *ast.FuncDecl, name string) (token.Pos, bool) {
	var found token.Pos
	var ok bool
	var walk func(n ast.Node, depth int)
	walk = func(n ast.Node, depth int) {
		if n == nil || ok {
			return
		}
		switch v := n.(type) {
		case *ast.ForStmt:
			ast.Inspect(v.Body, func(c ast.Node) bool {
				if c == v.Body {
					return true
				}
				walk(c, depth+1)
				return false
			})
			return
		case *ast.RangeStmt:
			ast.Inspect(v.Body, func(c ast.Node) bool {
				if c == v.Body {
					return true
				}
				walk(c, depth+1)
				return false
			})
			return
		case *ast.IndexExpr:
			if depth >= 2 {
				if inner, isIdx := v.X.(*ast.IndexExpr); isIdx {
					if id, isID := inner.X.(*ast.Ident); isID && id.Name == name {
						found, ok = v.Pos(), true
						return
					}
				}
			}
		}
		ast.Inspect(n, func(c ast.Node) bool {
			if c == n {
				return true
			}
			walk(c, depth)
			return false
		})
	}
	walk(fn.Body, 0)
	return found, ok
}
