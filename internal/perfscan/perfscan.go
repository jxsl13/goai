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
	{"PS1004", "spread-accessor-in-loop", "a variadic AtF64/SetF64(idx...) spread call in a loop outside PS1001's Numel/Unravel domain (rebuilds the flat offset + bounds-checks each call)", false},
	// PS2xxx — allocation inside loops
	{"PS2001", "alloc-in-loop", "a tensor allocation inside a per-element loop", false},
	{"PS2002", "unsized-builder", "a strings.Builder/bytes.Buffer written in a loop with no .Grow", false},
	{"PS2003", "strings-alloc-in-loop", "an allocating strings transform (Replace/Map/Repeat) in a loop", false},
	{"PS2004", "poolable-loop-scratch", "per-call scratch make() bound to a non-escaping local in a pointer-method loop", false},
	{"PS2005", "regexp-compile-in-loop", "a regexp.Compile/MustCompile inside a loop", true},
	{"PS2006", "quadratic-cache-append", "a per-token cache slot reassigned to a concat of ITSELF and a new row — O(T\u00b2) copy traffic where an amortized row buffer is O(T)", false},
	{"PS2007", "build-nxn-use-one-row", "a call given the same size argument twice, whose square result is then read at exactly that position — an N×N object materialized to consume one row", false},
	// PS3xxx — indirection / reflection overhead
	{"PS3001", "reflection-in-loop", "a reflection-based fmt scan (Sscanf/Sscan/Fscanf) in a loop", false},
	{"PS3002", "closure-comparator-sort", "a package sort (sort.Slice/SliceStable) with a comparator closure", false},
	{"PS3003", "int-key-map-in-loop", "a read of an integer-keyed map inside a loop", false},
	{"PS3005", "indirect-key-comparator", "a sort of an index slice whose comparator dereferences the sorted element into a 2-D structure — hoist the key into a flat column first", false},
	// PS4xxx — vectorization candidates
	{"PS4001", "le-decode-in-loop", "a per-element little-endian bit decode in a loop with no bulk-copy fast path", false},
	{"PS4002", "scalar-transcendental-vectorizable", "a scalar libm transcendental in a loop while a vectorized sibling is called", false},
	{"PS4003", "transcendental-wrapper-in-loop", "a loop calls a helper that wraps a libm transcendental", false},
	{"PS4004", "scalar-copy-loop", "an element-by-element slice copy in a loop where a bulk copy would do", false},
	{"PS4005", "per-element-odometer", "an N-D coordinate odometer ticked once per element instead of once per run", false},
	{"PS4006", "row-slice-matrix", "a [][]T matrix built row-by-row and then indexed inside a nested loop", false},
	{"PS4007", "vjp-scalar-elementwise-binop", "a *VJP with a scalar single-op elementwise loop (dst[i]=a[i]∘b[i]) that a SIMD backend op would vectorize+parallelize, and no backend.Execute dispatch", false},
	{"PS4008", "serial-dot-matmul", "a matmul whose innermost loop is a serial scalar dot accumulator — latency-bound where an ikj/axpy form has independent accumulators", false},
	// PS5xxx — arithmetic
	{"PS5001", "loop-invariant-divide", "a divide by a loop-invariant scalar on every element", false},
	{"PS5004", "multi-sweep-fusable", "three or more consecutive loops over the same range each indexing a shared slice (fuse the passes into one sweep)", false},
	{"PS5002", "symmetric-accumulation", "a nested loop accumulating a full symmetric matrix (m[i][j] += x[i]*x[j]) where one triangle + mirror would halve the work", false},
	{"PS5003", "inner-invariant-recompute", "an inner-loop expression that varies with the INNER index but not the outer one — recomputed once per outer iteration where a precomputed row would do", false},
	{"PS5005", "loop-invariant-transcendental", "a pure libm transcendental (math.Pow/Exp/Log/Sin) whose args vary with the INNER index but not the outer one, recomputed every outer iteration", false},
	// PS6xxx — verification gaps
	{"PS6001", "full-sort-bounded-prefix", "a full-vocabulary descending index sort whose result feeds an early-breaking (threshold-bounded) prefix consumer, with no quickselect/pre-filter guard — an O(n log n) sort for an O(prefix) need", false},
	{"PS6002", "spatial-bounds-branch", "an innermost window/kernel loop re-testing a compound spatial bounds guard (iy>=0 && iy<h && ix>=0 && ix<wd) per tap, where the in-bounds taps form one contiguous run the guard can be hoisted around", false},
	{"PS6003", "partial-fast-path-coverage", "a fast path that bypasses the general path for only SOME members of a variant family a switch in the same function enumerates — the uncovered variants silently pay the slow path", false},
	{"PS6004", "unverified-dual-path", "a devirtualized fast path with a generic fallback — a bit-identity claim needing a bit-exact test", true},
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
			// Apply to every line of the ENCLOSING COMMENT BLOCK and to the line after
			// it. A comment block attaches to the statement below it, so a directive
			// anywhere in the block must suppress that statement — anchoring only to the
			// directive's own line + 1 makes a wrapped explanation silently INERT, which
			// is worse than no suppression: the comment reads as if it took effect. Two
			// directives in this repo were dead that way before this covered the block.
			first := fset.Position(cg.Pos()).Line
			last := fset.Position(cg.End()).Line
			ln := fset.Position(c.Pos()).Line
			if first > ln {
				first = ln
			}
			if last < ln {
				last = ln
			}
			for l := first; l <= last+1; l++ {
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
	// PS5002 is a whole-file structural check (consecutive sibling loops), not a
	// per-function trigger attribution, so it runs once over the file's blocks.
	out = append(out, scanFusableLoops(fset, f)...)
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
		// A2: a variadic accessor called with a SPREAD index slice — .AtF64(coords...) —
		// inside a loop that is NOT a Numel/Unravel per-element loop, so PS1001's domain
		// check (A) misses it. The dynamic-rank spread form recomputes the flat offset and
		// bounds-checks on every call; take the contiguous backing slice + precomputed
		// row-major strides once and index it directly. (einsum EinsumContract: -71%.)
		if !perElem && ns.accessors[name] && call.Ellipsis.IsValid() {
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "spread-accessor-in-loop",
				msg: fmt.Sprintf("variadic .%s(idx...) with a spread index slice inside a loop — each call"+
					" rebuilds the flat offset and bounds-checks; take the contiguous backing slice and"+
					" precomputed row-major strides once, then index it directly (typed fast path)", name),
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
			// Skip pointer-element slices (make([]*T, …)): a handful of pointers used as
			// orchestration scaffolding (fully overwritten before a concat/reduce reads
			// them), not the numeric value scratch the growF64 pool win targets. Pooling a
			// pointer-slice header is negligible churn dwarfed by the elements' own allocs.
			if at, ok := call.Args[0].(*ast.ArrayType); ok {
				if _, isPtr := at.Elt.(*ast.StarExpr); isPtr {
					return true
				}
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
					" walk). Measured 3.66x on autograd's broadcast VJP (543 -> 148 us). Bit-identical" +
					" for a STORING loop (dst[i] = f(src)): each destination is written once, so only" +
					" the order in which DISTINCT destinations are touched changes. NOT automatically" +
					" bit-identical for an ACCUMULATING loop (dst[expr] += v): keep every destination's" +
					" summation order, and if the hoisted run accumulates into a scalar that scalar MUST" +
					" keep the element type — widening it to float64 and narrowing once is MORE accurate" +
					" and therefore a different answer. Verify the run length and that the enclosing loop" +
					" is not already a specialized fast path before acting (PS4005)",
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
						" bit-identical. Measured 2.15x (solvespd), 1.5x (cholesky), 1.35x (qr), 1.2x"+
						" (SymEig, SVD) — but 0.93x on classic cholSolve, where the factorization is a"+
						" small part of an OLS fit dominated by building the Gram matrix. The flatten"+
						" pays when the flagged loop IS the enclosing operation's work, so measure the"+
						" OPERATION end to end, not the loop. Check too that the rows are uniform"+
						" length — a genuinely ragged matrix cannot flatten.", name, name),
				})
			}
		}
	}

	// PS6004: a function carrying BOTH a devirtualized fast path (guarded by a
	// configured flat-view helper such as f64Data/outF64) and a generic fallback.
	// That structure is a bit-identity CLAIM: the two arms must agree exactly, and
	// the claim is usually written in a comment rather than in a test.
	//
	// Four such kernels were probed in one session and ALL FOUR were blind — a
	// one-ulp change in the fast path passed every test (backend/ref distill, blas1,
	// zloss, retention). Nothing here is a performance defect; the finding is an
	// unverified invariant, which is why this rule reports a RISK rather than a fix.
	//
	// It cannot tell whether a bit-exact test exists — test sensitivity is not an AST
	// property — so it lists the population that needs one. Probe with a deliberate
	// one-ulp mutation (PROC-009) to decide, and write the oracle from the kernel's
	// own algorithm (PROC-011).
	if ns.fastPath != nil {
		if name, ok := dualPathFunc(fn, ns); ok {
			out = append(out, finding{
				pos:      fset.Position(fn.Pos()),
				category: "unverified-dual-path",
				msg: fmt.Sprintf("%s has a devirtualized fast path guarded by %s plus a generic"+
					" fallback — the two arms claim to agree bit-for-bit. Verify that claim with a"+
					" bit-exact oracle: probe first with a one-ulp mutation, and if the suite stays"+
					" green the claim is untested. Four kernels with this exact shape were found"+
					" blind in one session. Not a performance finding — an unverified invariant.",
					fn.Name.Name, name),
			})
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
		isSortPkg := sortOK // sort.Slice/SliceStable pay reflect.Swapper per swap; slices.* do not
		if !sortOK {
			sname, sortOK = pkgFuncCall(call.Fun, "slices", slicesClosureCallees)
		}
		if sortOK && hasFuncArg(call) {
			msg := fmt.Sprintf("%s uses an indirect comparator. An LSD radix on the key bits can"+
				" replace it (math.Float64bits is monotonic for non-negative f64) — measured 1.9–2.25×"+
				" on top-p / typical sampling. BOTH preconditions must hold, and this check can verify"+
				" NEITHER: (1) the sort key is a numeric float/int, not a string or a composite —"+
				" radix-on-float-bits does not apply to a string key at all; (2) the slice is long"+
				" (vocab-sized), not rank- or dimension-sized — on a short slice the radix loses and"+
				" the measurement is noise. Confirm both by reading the site before acting, then prove"+
				" identical output order and benchmark.", sname)
			if isSortPkg {
				// sort.Slice/SliceStable dispatch every element swap through reflect.Swapper.
				// For a multi-key total order over a struct slice (radix infeasible), switching to
				// slices.SortFunc/SortStableFunc keeps the comparator + permutation but monomorphizes
				// the swap - bit-exact, -30..-45% on struct slices (JLens, CosineRerank, beam).
				msg += " Or, for a multi-key total order over a struct slice, switch sort.Slice/SliceStable -> slices.SortFunc/SortStableFunc: same comparator + permutation, monomorphized swap (no reflect.Swapper) - bit-exact, -30..-45% on struct slices."
			}
			out = append(out, finding{
				pos:      fset.Position(call.Pos()),
				category: "closure-comparator-sort",
				msg:      msg,
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
	out = append(out, symmetricAccumFindings(fset, fn)...)
	out = append(out, serialDotMatmulFindings(fset, fn)...)
	out = append(out, quadraticCacheAppendFindings(fset, fn)...)
	out = append(out, buildNxNUseOneRowFindings(fset, fn)...)
	out = append(out, indirectKeyComparatorFindings(fset, fn)...)
	out = append(out, innerInvariantRecomputeFindings(fset, fn)...)
	out = append(out, invariantTranscendentalRecomputeFindings(fset, fn)...)
	out = append(out, partialFastPathFindings(fset, fn)...)
	out = append(out, vjpScalarBinopFindings(fset, fn)...)
	out = append(out, fullSortBoundedPrefixFindings(fset, fn)...)
	out = append(out, spatialBoundsBranchFindings(fset, fn)...)
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
// scanFusableLoops reports PS5004 (multi-sweep-fusable): three or more CONSECUTIVE
// sibling loops in one block that iterate over the SAME range and each index a shared
// slice — the pass-fusion / scratch-elimination pattern behind the Adafactor (-3.4%),
// CautiousAdamW (-8%) and GrokfastMA (-54%) wins. Sweeping one buffer N times moves it
// through memory N times; folding the passes into a single loop cuts that traffic ~Nx.
// Advisory only: fusion is bit-identical only when no earlier pass writes a slot a
// later pass reads at a DIFFERENT index (a genuine loop-carried dependency), which a
// static tool cannot rule out — the finding asks a human to confirm before fusing.
func scanFusableLoops(fset *token.FileSet, f *ast.File) []finding {
	render := func(e ast.Expr) string {
		var buf bytes.Buffer
		if err := printer.Fprint(&buf, fset, e); err != nil {
			return ""
		}
		return buf.String()
	}
	var out []finding
	ast.Inspect(f, func(n ast.Node) bool {
		blk, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		stmts := blk.List
		for i := 0; i < len(stmts); {
			sig, slices, ok := fusableLoop(stmts[i], render)
			if !ok {
				i++
				continue
			}
			run := 1
			shared := map[string]int{}
			for name := range slices {
				shared[name]++
			}
			j := i + 1
			for j < len(stmts) {
				sig2, sl2, ok2 := fusableLoop(stmts[j], render)
				if !ok2 || sig2 != sig {
					break
				}
				for name := range sl2 {
					shared[name]++
				}
				run++
				j++
			}
			if run >= 3 {
				// Require a slice indexed by >=2 of the passes: that proves they stream the
				// SAME buffer (the traffic fusion removes), not three unrelated arrays. Pick
				// the shared name deterministically for a stable message.
				keys := make([]string, 0, len(shared))
				for name := range shared {
					keys = append(keys, name)
				}
				sort.Strings(keys)
				common := ""
				for _, name := range keys {
					if shared[name] >= 2 {
						common = name
						break
					}
				}
				if common != "" {
					out = append(out, finding{
						pos:      fset.Position(stmts[i].Pos()),
						end:      fset.Position(stmts[j-1].End()),
						category: "multi-sweep-fusable",
						msg:      fmt.Sprintf("%d consecutive loops over the same range each index %q; fuse the passes into one loop to cut memory traffic ~%dx (verify no cross-pass dependency blocks fusion)", run, common, run),
					})
				}
			}
			i = j
		}
		return true
	})
	return out
}

// fusableLoop classifies s as a simple, fusion-eligible loop — a `for k := range X`
// or classic `for i := 0; i < E; i++` whose body only reads/writes indexed slices with
// no nested loop, closure, branch, return, defer, go or switch/select that would make
// fusion unsafe or change the trip count. It returns a signature identifying the range
// (two loops fuse only if their signatures match) and the set of slice names the body
// indexes by the loop variable.
func fusableLoop(s ast.Stmt, render func(ast.Expr) string) (sig string, slices map[string]bool, ok bool) {
	var body *ast.BlockStmt
	var idxVar string
	switch l := s.(type) {
	case *ast.RangeStmt:
		key, kok := l.Key.(*ast.Ident)
		if !kok || l.Value != nil || key.Name == "_" {
			return "", nil, false
		}
		idxVar = key.Name
		sig = "range " + render(l.X)
		body = l.Body
	case *ast.ForStmt:
		v, b, fok := classicForBound(l, render)
		if !fok {
			return "", nil, false
		}
		idxVar = v
		sig = "for " + b
		body = l.Body
	default:
		return "", nil, false
	}
	if !simpleLoopBody(body) {
		return "", nil, false
	}
	slices = indexedByVar(body, idxVar, render)
	if len(slices) == 0 {
		return "", nil, false
	}
	return sig, slices, true
}

// classicForBound matches `i := 0; i < E; i++` (also `i <= E`), returning the index
// variable and a rendered bound key. A custom step, downward count, multi-clause init
// or non-zero start is rejected — those do not fuse by simple alignment.
func classicForBound(l *ast.ForStmt, render func(ast.Expr) string) (idxVar, bound string, ok bool) {
	init, iok := l.Init.(*ast.AssignStmt)
	if !iok || len(init.Lhs) != 1 || len(init.Rhs) != 1 {
		return "", "", false
	}
	name, nok := init.Lhs[0].(*ast.Ident)
	lit, lok := init.Rhs[0].(*ast.BasicLit)
	if !nok || !lok || lit.Value != "0" {
		return "", "", false
	}
	inc, pok := l.Post.(*ast.IncDecStmt)
	if !pok || inc.Tok != token.INC {
		return "", "", false
	}
	if id, ok := inc.X.(*ast.Ident); !ok || id.Name != name.Name {
		return "", "", false
	}
	bin, bok := l.Cond.(*ast.BinaryExpr)
	if !bok || (bin.Op != token.LSS && bin.Op != token.LEQ) {
		return "", "", false
	}
	if id, ok := bin.X.(*ast.Ident); !ok || id.Name != name.Name {
		return "", "", false
	}
	return name.Name, bin.Op.String() + render(bin.Y), true
}

// simpleLoopBody reports whether a loop body is safe to consider for fusion: no nested
// loop, closure, branch/return/defer/go/labeled/switch/select. Straight-line
// assignments, increments, if-blocks and expression calls are allowed (Cautious's sign
// test is an if; a math.Sqrt in a fused body is fine).
func simpleLoopBody(b *ast.BlockStmt) bool {
	simple := true
	ast.Inspect(b, func(n ast.Node) bool {
		if !simple {
			return false
		}
		switch n.(type) {
		case *ast.ForStmt, *ast.RangeStmt, *ast.FuncLit, *ast.BranchStmt,
			*ast.ReturnStmt, *ast.DeferStmt, *ast.GoStmt, *ast.LabeledStmt,
			*ast.SwitchStmt, *ast.TypeSwitchStmt, *ast.SelectStmt:
			simple = false
			return false
		}
		return true
	})
	return simple
}

// indexedByVar collects the names of slices indexed by the loop variable in the body —
// s[i], s[i+1], s[i-1], g.sum[i] — i.e. the buffers this pass streams. A slice touched
// by >=2 fused passes is the shared traffic fusion removes. The name is the rendered
// indexed expression's base, so a local slice and a field slice both key stably.
func indexedByVar(b *ast.BlockStmt, idxVar string, render func(ast.Expr) string) map[string]bool {
	names := map[string]bool{}
	ast.Inspect(b, func(n ast.Node) bool {
		ix, ok := n.(*ast.IndexExpr)
		if !ok {
			return true
		}
		if indexUsesVar(ix.Index, idxVar) {
			if key := render(ix.X); key != "" {
				names[key] = true
			}
		}
		return true
	})
	return names
}

// indexUsesVar reports whether the index expression references idxVar (directly as i,
// or in an offset like i+1 / i-1).
func indexUsesVar(e ast.Expr, idxVar string) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && id.Name == idxVar {
			found = true
			return false
		}
		return true
	})
	return found
}

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
		// PS6004 needs the accessor set to see a generic fallback arm: with
		// elementAccessors empty, hasAccessorFallback is never set and the check
		// reports zero on any input. Same false assurance, same warning.
		"unverified-dual-path": ns.accessors,
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
	case *ast.IndexExpr:
		// c.K[l] — needed by PS2006 to prove a concat accumulates into its own slot.
		y, ok := b.(*ast.IndexExpr)
		return ok && exprEqual(x.X, y.X) && exprEqual(x.Index, y.Index)
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
// It reports the position of the INNERMOST ENCLOSING LOOP, not of the index expression
// itself. That distinction is not cosmetic: `//perfscan:ignore` applies to a comment
// block and the statement below it, so a finding anchored to an expression one line
// INSIDE the loop cannot be suppressed by a directive written above the loop — which is
// where any reader puts it. PS4006 was unsuppressable that way until this changed, and
// only a bare directive failing to silence it revealed the cause. PS4005 and PS4008
// already anchor to the loop; this makes the family consistent.
func nestedDoubleIndex(fn *ast.FuncDecl, name string) (token.Pos, bool) {
	var found token.Pos
	var ok bool
	var walk func(n ast.Node, depth int, loop token.Pos)
	walk = func(n ast.Node, depth int, loop token.Pos) {
		if n == nil || ok {
			return
		}
		switch v := n.(type) {
		case *ast.ForStmt:
			ast.Inspect(v.Body, func(c ast.Node) bool {
				if c == v.Body {
					return true
				}
				walk(c, depth+1, v.Pos())
				return false
			})
			return
		case *ast.RangeStmt:
			ast.Inspect(v.Body, func(c ast.Node) bool {
				if c == v.Body {
					return true
				}
				walk(c, depth+1, v.Pos())
				return false
			})
			return
		case *ast.IndexExpr:
			if depth >= 2 {
				if inner, isIdx := v.X.(*ast.IndexExpr); isIdx {
					if id, isID := inner.X.(*ast.Ident); isID && id.Name == name {
						found, ok = loop, true
						return
					}
				}
			}
		}
		ast.Inspect(n, func(c ast.Node) bool {
			if c == n {
				return true
			}
			walk(c, depth, loop)
			return false
		})
	}
	walk(fn.Body, 0, token.NoPos)
	return found, ok
}

// dualPathFunc reports whether fn guards a fast path with a configured flat-view
// helper (fastPathHelpers) AND retains a generic fallback, returning the helper
// name. Both halves are required: a function with only the fast path has no second
// arm to disagree with, and one with only accessors makes no bit-identity claim.
func dualPathFunc(fn *ast.FuncDecl, ns nameSets) (string, bool) {
	var helper string
	var hasAccessorFallback bool
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch v := n.(type) {
		case *ast.AssignStmt:
			// COMMA-OK is the discriminator: `xs, ok := f64Data(x)` returns a success
			// flag, which is what makes the function dual-armed. Bare storage
			// accessors such as .Storage().F64() are also in fastPathHelpers and are
			// single-valued, so requiring two results excludes them — without this the
			// rule matched 193 functions. Matching the ASSIGNMENT rather than an
			// if-init is what catches the common `xs, ok := ...` followed by
			// `if ok && ok2 {` form, which an if-init test misses.
			if len(v.Lhs) < 2 || len(v.Rhs) != 1 {
				return true
			}
			call, ok := v.Rhs[0].(*ast.CallExpr)
			if !ok {
				return true
			}
			if name := calleeName(call.Fun); helper == "" && ns.fastPath[name] {
				helper = name
			}
		case *ast.SwitchStmt:
			// The other dual-arm form: `switch x.Dtype() { case F64: <typed>; default:
			// <accessor loop> }`. There is no comma-ok here, so the assignment test
			// above misses it — blas1 is the known example, and leaving it out made the
			// floor 6 of 7. A default clause is required: without one the switch is
			// exhaustive over dtypes and has no fallback arm to disagree with.
			if helper != "" || v.Tag == nil {
				return true
			}
			call, ok := v.Tag.(*ast.CallExpr)
			if !ok || calleeName(call.Fun) != "Dtype" {
				return true
			}
			for _, st := range v.Body.List {
				if cc, ok := st.(*ast.CaseClause); ok && cc.List == nil {
					helper = "Dtype switch"
				}
			}
		case *ast.CallExpr:
			if ns.accessors[calleeName(v.Fun)] {
				hasAccessorFallback = true
			}
		}
		return true
	})
	if helper != "" && hasAccessorFallback {
		return helper, true
	}
	// THIRD FORM: a typed fast path that DECLINES to its caller. `v, ok := x.data.([]T)`
	// discriminates on concrete storage rather than through a configured comma-ok
	// helper, and the generic arm lives in the CALLER, reached by returning false — so
	// neither the helper test nor the same-function accessor test above can see it.
	// This shape was found by shipping one: tensor.gatherHalfTyped devirtualizes four
	// half-cast arms and returns false for everything else, a 3.19x dual path that
	// PS6004 was blind to. Blindness in a rule whose entire job is to demand a
	// bit-identity proof is the worst kind — it reads as "nothing to verify here".
	if name, ok := decliningTypedFastPath(fn); ok {
		return name, true
	}
	return "", false
}

// decliningTypedFastPath recognizes a function that selects a typed fast path by
// comma-ok TYPE ASSERTION and signals "not my shape" with a bool return, leaving the
// generic path to its caller. Requires all four of: a bool result, at least two
// distinct comma-ok assertions to SLICES OF NUMERIC TYPES, and a `return false` — the
// decline that hands work back.
//
// The numeric-slice requirement is what makes this usable. Without it the check fired
// on 13 functions inside perfscan itself: an AST visitor is wall-to-wall
// `x, ok := n.(*ast.Foo)` followed by `return false`, which is structurally identical
// and semantically nothing like a devirtualized kernel. Asserting []float32 or []uint16
// is the signal; asserting *ast.Ident is not.
func decliningTypedFastPath(fn *ast.FuncDecl) (string, bool) {
	if fn.Type.Results == nil || len(fn.Type.Results.List) != 1 {
		return "", false
	}
	if id, ok := fn.Type.Results.List[0].Type.(*ast.Ident); !ok || id.Name != "bool" {
		return "", false
	}
	asserted := map[string]bool{}
	declines := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch v := n.(type) {
		case *ast.AssignStmt:
			if len(v.Lhs) != 2 || len(v.Rhs) != 1 {
				return true
			}
			if ta, ok := v.Rhs[0].(*ast.TypeAssertExpr); ok && isNumericSliceType(ta.Type) {
				asserted[typeExprText(ta.Type)] = true
			}
		case *ast.ReturnStmt:
			if len(v.Results) == 1 {
				if id, ok := v.Results[0].(*ast.Ident); ok && id.Name == "false" {
					declines = true
				}
			}
		}
		return true
	})
	if len(asserted) < 2 || !declines {
		return "", false
	}
	return "typed storage assertion", true
}

// typeExprText renders the asserted type compactly, enough to tell []float32 from
// []uint16 when counting how many concrete arms a dispatch has.
func typeExprText(e ast.Expr) string {
	switch x := e.(type) {
	case *ast.Ident:
		return x.Name
	case *ast.ArrayType:
		return "[]" + typeExprText(x.Elt)
	case *ast.StarExpr:
		return "*" + typeExprText(x.X)
	case *ast.SelectorExpr:
		return typeExprText(x.X) + "." + x.Sel.Name
	}
	return "?"
}

// loopVarBody returns the index/key variable name and body block of a for/range loop.
func loopVarBody(n ast.Node) (string, *ast.BlockStmt, bool) {
	switch l := n.(type) {
	case *ast.RangeStmt:
		if id, ok := l.Key.(*ast.Ident); ok && id.Name != "_" {
			return id.Name, l.Body, true
		}
	case *ast.ForStmt:
		if as, ok := l.Init.(*ast.AssignStmt); ok && len(as.Lhs) == 1 {
			if id, ok := as.Lhs[0].(*ast.Ident); ok && id.Name != "_" {
				return id.Name, l.Body, true
			}
		}
	}
	return "", nil, false
}

// symBase returns the base identifier of an index expr whose ROW index mentions v (and
// not other) — matching both a 1-D operand B[v] and a 2-D gram operand B[v][k] (v in the
// row index, the shared k in the column index). Returns "" for anything else.
func symBase(e ast.Expr, v, other string) (string, bool) {
	ix, ok := e.(*ast.IndexExpr)
	if !ok {
		return "", false
	}
	if id, ok := ix.X.(*ast.Ident); ok { // 1-D: B[v]
		if exprMentions(ix.Index, v) && !exprMentions(ix.Index, other) {
			return id.Name, true
		}
		return "", false
	}
	if inner, ok := ix.X.(*ast.IndexExpr); ok { // 2-D gram: B[v][k]
		if id, ok := inner.X.(*ast.Ident); ok &&
			exprMentions(inner.Index, v) && !exprMentions(inner.Index, other) &&
			!exprMentions(ix.Index, v) && !exprMentions(ix.Index, other) {
			return id.Name, true
		}
	}
	return "", false
}

// symmetricAccumFindings flags PS5002 — nested i/j loops that accumulate a FULL symmetric
// matrix (m[i][j] += x[i]·x[j], or a gram M[i][k]·M[j][k] reduced into m[i][j]). Every
// off-diagonal is computed twice; if the consumer reads only one triangle (Cholesky /
// eigendecomposition) the upper triangle + a mirror pass halves the accumulation. Requires
// BOTH a same-base symmetric product AND an m[i][j] write, which excludes matmul (whose
// factors are different bases). Shipped: GMM, PCA.
func symmetricAccumFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		iName, obody, ok := loopVarBody(n)
		if !ok {
			return true
		}
		for _, stmt := range obody.List {
			jName, _, ok2 := loopVarBody(stmt)
			if !ok2 || jName == iName || innerLoopIsTriangular(stmt, iName) {
				continue
			}
			base, hasProd := symmetricProductBase(stmt, iName, jName)
			if hasProd && hasMatrixWrite(stmt, iName, jName) {
				out = append(out, finding{
					pos:      fset.Position(n.Pos()),
					category: "symmetric-accumulation",
					msg: fmt.Sprintf("nested %s/%s loop accumulates a FULL symmetric matrix"+
						" (m[%s][%s] built from a %s[%s]·%s[%s] product) — every off-diagonal is"+
						" computed twice. If the consumer reads only one triangle (Cholesky /"+
						" eigendecomposition), accumulate the upper triangle + diagonal and mirror"+
						" once (m[%s][%s]=m[%s][%s]) — ~2x the accumulation, bit-identical when the"+
						" product is commutative. Shipped: GMM, PCA. Verify the consumer + benchmark.",
						iName, jName, iName, jName, base, iName, base, jName, jName, iName, iName, jName),
				})
				return false
			}
		}
		return true
	})
	return out
}

// exprMentions reports whether e references the identifier name.
func exprMentions(e ast.Expr, name string) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && id.Name == name {
			found = true
			return false
		}
		return true
	})
	return found
}

// innerLoopIsTriangular reports whether the inner loop's bounds reference the outer index
// (for j := i; …  or  for j := 0; j <= i; …) — it already covers only one triangle, so the
// symmetric-accumulation optimization does not apply (e.g. cholSolve's j<=i Cholesky loop).
func innerLoopIsTriangular(loop ast.Node, iName string) bool {
	fl, ok := loop.(*ast.ForStmt)
	if !ok {
		return false
	}
	return (fl.Init != nil && stmtMentions(fl.Init, iName)) ||
		(fl.Cond != nil && exprMentions(fl.Cond, iName))
}

// hasMatrixWrite reports whether the subtree writes (= or +=) into a target indexed by
// BOTH i and j — nested m[i][j] or flat m[i*n+j].
func hasMatrixWrite(root ast.Node, iName, jName string) bool {
	found := false
	ast.Inspect(root, func(n ast.Node) bool {
		if found {
			return false
		}
		as, ok := n.(*ast.AssignStmt)
		if !ok || (as.Tok != token.ADD_ASSIGN && as.Tok != token.ASSIGN) {
			return true
		}
		for _, lhs := range as.Lhs {
			ix, ok := lhs.(*ast.IndexExpr)
			if !ok {
				continue
			}
			if exprMentions(ix.Index, iName) && exprMentions(ix.Index, jName) { // flat
				found = true
			}
			if inner, ok := ix.X.(*ast.IndexExpr); ok && // nested m[i][j]
				exprMentions(inner.Index, iName) && exprMentions(ix.Index, jName) {
				found = true
			}
		}
		return !found
	})
	return found
}

// stmtMentions reports whether the statement references the identifier name.
func stmtMentions(s ast.Stmt, name string) bool {
	found := false
	ast.Inspect(s, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && id.Name == name {
			found = true
			return false
		}
		return true
	})
	return found
}

// symmetricProductBase searches a subtree for a multiply B[..i..]·B[..j..] of the SAME base
// (the symmetry signal that separates a covariance/gram accumulation from a matmul, whose
// factors have different bases) and returns that base.
func symmetricProductBase(root ast.Node, iName, jName string) (string, bool) {
	var base string
	var found bool
	ast.Inspect(root, func(n ast.Node) bool {
		if found {
			return false
		}
		be, ok := n.(*ast.BinaryExpr)
		if !ok || be.Op != token.MUL {
			return true
		}
		if bi, ok := symBase(be.X, iName, jName); ok {
			if bj, ok := symBase(be.Y, jName, iName); ok && bi == bj {
				base, found = bi, true
			}
		}
		if !found {
			if bi, ok := symBase(be.X, jName, iName); ok {
				if bj, ok := symBase(be.Y, iName, jName); ok && bi == bj {
					base, found = bi, true
				}
			}
		}
		return !found
	})
	return base, found
}

// serialDotMatmulFindings flags PS4008 — a triple-nested loop whose innermost body is
// a single scalar dot accumulation, `s += A[…] * B[…]`, with the result stored to an
// indexed destination after the loop. That accumulator is a SERIAL dependency chain:
// every FMADD waits on the previous one's latency, so the loop runs at the FMA's
// latency rather than its throughput. Transposing the k-dim operand once and rewriting
// as ikj/axpy (`c[j] += av * bt[j]`) gives independent accumulators across j, which
// measured 0.92 → 0.32 ns/MAC on nn.matmulABt (Muon Step 418 → 200 ms, 2.09×).
//
// Requires the accumulator to be declared in the middle loop and stored to an IndexExpr
// after the inner loop — that store is what distinguishes a matmul from a plain
// reduction (a norm or a dot product returning a scalar has nowhere to hoist to and is
// not a candidate). Both operands must be IndexExpr over DISTINCT base identifiers, so
// an in-place accumulation into one of its own operands is not flagged.
//
// The rewrite is BIT-IDENTICAL when the ikj form keeps the same ascending accumulation
// order — the same argument backend/cpu/gemm.go makes for its tolerance-0 gate — but
// that must be PROVEN by a cross-reference test against the pre-rewrite form, not
// assumed, and a zero-skip (`if av == 0 { continue }`) must NOT be carried along: it
// drops 0·±Inf NaNs and is not order-preserving.
func serialDotMatmulFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		// outer i loop
		_, obody, ok := loopVarBody(n)
		if !ok {
			return true
		}
		for _, mid := range obody.List {
			_, mbody, ok := loopVarBody(mid)
			if !ok {
				continue
			}
			acc, inner, store := dotAccumShape(mbody)
			if acc == "" {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(inner.Pos()),
				category: "serial-dot-matmul",
				msg: fmt.Sprintf("%q accumulates a scalar dot in the innermost loop and is then stored to %s"+
					" — a serial FMADD chain running at latency, not throughput. Transpose the k-dim operand"+
					" once and rewrite as ikj/axpy so the accumulators are independent across the output index"+
					" (measured 2.9x per-MAC on nn.matmulABt). Prove bit-identity against the current form with"+
					" a tolerance-0 cross-reference test, and do NOT carry over a zero-skip.", acc, store),
			})
		}
		return true
	})
	return out
}

// dotAccumShape recognizes the matmul dot body inside a middle loop: a scalar
// declaration, an inner loop whose ONLY statement accumulates a product of two
// distinctly-based index expressions into it, and a store of it into an IndexExpr.
// Returns the accumulator name, the inner loop node and the store target's text.
func dotAccumShape(mbody *ast.BlockStmt) (string, ast.Node, string) {
	var acc string
	var inner ast.Node
	var store string
	for _, st := range mbody.List {
		switch v := st.(type) {
		case *ast.DeclStmt:
			if gd, ok := v.Decl.(*ast.GenDecl); ok && gd.Tok == token.VAR {
				for _, sp := range gd.Specs {
					if vs, ok := sp.(*ast.ValueSpec); ok && len(vs.Names) == 1 && len(vs.Values) == 0 {
						if isFloatIdent(vs.Type) {
							acc = vs.Names[0].Name
						}
					}
				}
			}
		case *ast.AssignStmt:
			// `s := 0.0` also declares an accumulator; and `ci[j] = s` is the store.
			if v.Tok == token.DEFINE && len(v.Lhs) == 1 && len(v.Rhs) == 1 {
				if id, ok := v.Lhs[0].(*ast.Ident); ok && isZeroLit(v.Rhs[0]) {
					acc = id.Name
				}
				continue
			}
			if v.Tok == token.ASSIGN && len(v.Lhs) == 1 && len(v.Rhs) == 1 && acc != "" {
				if id, ok := v.Rhs[0].(*ast.Ident); ok && id.Name == acc {
					if ix, ok := v.Lhs[0].(*ast.IndexExpr); ok {
						store = baseIdentName(ix.X)
					}
				}
			}
		default:
			if acc == "" || inner != nil {
				continue
			}
			if _, ibody, ok := loopVarBody(st); ok && accumulatesProduct(ibody, acc) {
				inner = st
			}
		}
	}
	if acc == "" || inner == nil || store == "" {
		return "", nil, ""
	}
	return acc, inner, store
}

// accumulatesProduct reports whether body is exactly `acc += X[…] * Y[…]` with X and
// Y distinct base identifiers. A single statement is required: anything else in the
// inner loop means the dot is not the whole cost and the rewrite is not the fix.
func accumulatesProduct(body *ast.BlockStmt, acc string) bool {
	if len(body.List) != 1 {
		return false
	}
	as, ok := body.List[0].(*ast.AssignStmt)
	if !ok || as.Tok != token.ADD_ASSIGN || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
		return false
	}
	if id, ok := as.Lhs[0].(*ast.Ident); !ok || id.Name != acc {
		return false
	}
	be, ok := as.Rhs[0].(*ast.BinaryExpr)
	if !ok || be.Op != token.MUL {
		return false
	}
	lx, lok := be.X.(*ast.IndexExpr)
	rx, rok := be.Y.(*ast.IndexExpr)
	if !lok || !rok {
		return false
	}
	l, r := baseIdentName(lx.X), baseIdentName(rx.X)
	return l != "" && r != "" && l != r
}

// isFloatIdent reports whether a type expression names a Go float type.
func isFloatIdent(e ast.Expr) bool {
	id, ok := e.(*ast.Ident)
	return ok && (id.Name == "float64" || id.Name == "float32")
}

// isZeroLit reports whether e is a literal zero (`0` or `0.0`), the shape a
// short-variable-declared accumulator takes.
func isZeroLit(e ast.Expr) bool {
	bl, ok := e.(*ast.BasicLit)
	if !ok || (bl.Kind != token.INT && bl.Kind != token.FLOAT) {
		return false
	}
	for _, r := range bl.Value {
		if r != '0' && r != '.' {
			return false
		}
	}
	return true
}

// quadraticCacheAppendFindings flags PS2006 — a cache slot reassigned to a concat of
// ITSELF and a new row, inside a loop, in a per-token step function:
//
//	c.K[l] = concatRows(c.K[l], k)
//
// The concat allocates a fresh [t+1, width] buffer and recopies all t existing rows
// EVERY token, so a T-token decode moves O(T²) bytes and allocates O(T²). An amortized
// row buffer — write row t in place into a doubling backing store, hand back a
// zero-copy prefix view — makes the same sequence O(T) total. Measured in this repo at
// width 2048, T=512: 101x time and 104x bytes on the append microbenchmark; end to end
// on a 500-token GPT decode, 2.21 GB -> 159 MB (13.9x) and 1.32x wall clock.
//
// Requires (a) the assignment target to be an IndexExpr whose base also appears as the
// FIRST argument of the call, (b) an enclosing loop, and (c) an enclosing function
// whose name marks a per-token step. Without (c) this shape is an ordinary one-shot
// concatenation and pooling it would be premature.
//
// The risk the fix carries is ALIASING, not numerics: the concat hands back a fresh
// buffer each step while a row buffer hands back a VIEW into a growing one. Audit for
// callers that retain an earlier view AND mutate it in place before switching.
func quadraticCacheAppendFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Name == nil || !perTokenStepName(fn.Name.Name) {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		_, body, ok := loopVarBody(n)
		if !ok {
			return true
		}
		ast.Inspect(body, func(m ast.Node) bool {
			as, ok := m.(*ast.AssignStmt)
			if !ok || as.Tok != token.ASSIGN {
				return true
			}
			for i, lhs := range as.Lhs {
				if i >= len(as.Rhs) {
					break
				}
				target, ok := lhs.(*ast.IndexExpr)
				if !ok {
					continue
				}
				call, ok := as.Rhs[i].(*ast.CallExpr)
				if !ok || len(call.Args) < 2 {
					continue
				}
				if !concatLikeName(calleeName(call.Fun)) || !exprEqual(call.Args[0], target) {
					continue
				}
				out = append(out, finding{
					pos:      fset.Position(as.Pos()),
					category: "quadratic-cache-append",
					msg: fmt.Sprintf("%q is reassigned to %s of ITSELF plus a new row, inside a loop in"+
						" %s — the call reallocates and recopies every existing row per token, so a"+
						" T-token run moves O(T²) bytes. Replace with an amortized row buffer (write in"+
						" place into a doubling backing store, return a zero-copy prefix view): measured"+
						" 2.21 GB -> 159 MB on a 500-token decode. AUDIT FOR ALIASING FIRST — the buffer"+
						" hands back a view, not a fresh tensor.",
						exprText(target), calleeName(call.Fun), fn.Name.Name),
				})
			}
			return true
		})
		return true
	})
	return out
}

// perTokenStepName reports whether a function name marks a per-token decode step,
// which is what makes a repeated concat quadratic rather than incidental.
func perTokenStepName(name string) bool {
	for _, suffix := range []string{"DecodeStep", "StepKV", "StreamStep", "Step"} {
		if strings.HasSuffix(name, suffix) {
			return true
		}
	}
	return false
}

// concatLikeName reports whether a callee builds a new buffer sized from both of its
// operands — the allocating concatenation a row buffer replaces.
func concatLikeName(name string) bool {
	switch name {
	case "concatRows", "Concat", "concat", "ConcatRows", "appendRows", "stackRows":
		return true
	}
	return false
}

// exprText renders the shapes a cache slot takes — x, x.F, x.F[i] — for a finding
// message. baseIdentName alone renders a selector as "", which turned the PS2006
// message into a bare "[g]".
func exprText(e ast.Expr) string {
	switch x := e.(type) {
	case *ast.Ident:
		return x.Name
	case *ast.SelectorExpr:
		if base := exprText(x.X); base != "" {
			return base + "." + x.Sel.Name
		}
		return x.Sel.Name
	case *ast.IndexExpr:
		return exprText(x.X) + "[" + exprText(x.Index) + "]"
	case *ast.ParenExpr:
		return exprText(x.X)
	}
	return baseIdentName(e)
}

// buildNxNUseOneRowFindings flags PS2007 — a call given the SAME size-like argument
// twice, whose result is then indexed at exactly that position:
//
//	full, _ := d.relBias.Bias(ctx, pos+1, pos+1)   // builds [pos+1, pos+1, heads]
//	… fs[(pos*kk+k)*heads+h] …                     // reads row pos, discards the rest
//
// The callee's cost grows with the argument in BOTH dimensions while the consumer
// needs one row, so the work is an order higher than required — and when the call sits
// on a per-token path, one order higher again over the whole run. Measured in this
// repo on the T5 decoder's relative-position bias: replacing the build-then-slice with
// a direct gather was 130x at pos=32, 261x at pos=128 and 713x at pos=512, with
// allocation falling 86.8 MB -> 19.7 KB per call, and the end-to-end decode going
// 2,679 -> 556 ms (4.82x) and 13.87 GB -> 125 MB (111x).
//
// The give-away is the REPEATED argument: a genuinely square result whose row index is
// the same expression fed to both size parameters. That is what distinguishes it from
// an ordinary 2-D build that is legitimately consumed in full.
//
// The fix is to compute the needed row from whatever the callee derives it from,
// which usually means exposing the per-element rule (here a bucket lookup) rather than
// its materialized matrix. Check bit-identity: a value delivered through a one-hot
// matmul equals the gathered entry for finite tables, but NOT for +-Inf/NaN (0*Inf is
// NaN) nor for a stored -0 (0 + -0 = +0).
func buildNxNUseOneRowFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Rhs) != 1 {
			return true
		}
		call, ok := as.Rhs[0].(*ast.CallExpr)
		if !ok || len(call.Args) < 2 {
			return true
		}
		// The last two arguments must be the SAME non-trivial expression — the square
		// build. A bare literal (foo(x, 2, 2)) is a fixed small shape, not this.
		a, b := call.Args[len(call.Args)-2], call.Args[len(call.Args)-1]
		if !exprEqual(a, b) || isConstSizeExpr(a) {
			return true
		}
		base := sizeDrivingIdent(a)
		if base == "" {
			return true
		}
		// The result — or a value derived from it — must be INDEXED by the driving
		// position. Asking only "is `base` used in some index anywhere in this
		// function" was far too weak: it flagged a.AtF64(j, j) (a diagonal element
		// read, not a builder) and the full Decode path, where the square bias
		// genuinely IS the attention mask and is consumed whole.
		// The driving position must be FIXED, not a loop bound. When it bounds a loop
		// the square result is being walked in full — the T5 Decode path writes -Inf
		// across every row of its [dseq, dseq] mask, and `dseq` appears in those
		// indices only as a STRIDE. In the real pattern the position is a parameter
		// that selects one row and is never itself iterated.
		if identIsLoopBound(fn.Body, base) {
			return true
		}
		lhs, ok := assignedIdent(as)
		if !ok || !derivedValueIndexedBy(fn.Body, lhs, base) {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(as.Pos()),
			category: "build-nxn-use-one-row",
			msg: fmt.Sprintf("%s is called with %q twice, building a square result, and %q is then used"+
				" as an index into it — an N×N object materialized to consume one row. Compute the row"+
				" directly from whatever the callee derives it from (measured 713x at N=512, 86.8 MB ->"+
				" 19.7 KB per call, on the T5 relative-position bias). Verify bit-identity: a one-hot"+
				" matmul equals the gathered entry for finite tables, but not for ±Inf/NaN or a stored -0.",
				calleeName(call.Fun), exprText(a), base),
		})
		return true
	})
	return out
}

// sizeDrivingIdent returns the identifier a size expression is built from — `pos` for
// `pos`, `pos+1`, `pos*2`. Empty when the expression is not driven by a single ident.
func sizeDrivingIdent(e ast.Expr) string {
	switch x := e.(type) {
	case *ast.Ident:
		return x.Name
	case *ast.ParenExpr:
		return sizeDrivingIdent(x.X)
	case *ast.BinaryExpr:
		if l := sizeDrivingIdent(x.X); l != "" {
			return l
		}
		return sizeDrivingIdent(x.Y)
	}
	return ""
}

// isConstSizeExpr reports whether an expression is built purely from literals, which
// makes a square call an ordinary fixed shape rather than a growing one.
func isConstSizeExpr(e ast.Expr) bool {
	return sizeDrivingIdent(e) == ""
}

// assignedIdent returns the first non-blank identifier an assignment binds, which for
// `full, err := f(…)` is "full" — the handle on the square result.
func assignedIdent(as *ast.AssignStmt) (string, bool) {
	for _, lhs := range as.Lhs {
		if id, ok := lhs.(*ast.Ident); ok && id.Name != "_" && id.Name != "err" {
			return id.Name, true
		}
	}
	return "", false
}

// derivedValueIndexedBy reports whether root — or any value transitively derived from
// it by assignment — is indexed by an expression mentioning pos. Following the
// derivation chain is what makes this precise: the T5 site reads the matrix as
// `fs := full.Contiguous().Storage().F64()` and then indexes fs, never full.
func derivedValueIndexedBy(body *ast.BlockStmt, root, pos string) bool {
	derived := map[string]bool{root: true}
	// Two passes so a chain assigned in order (a := f(root); b := g(a)) is followed.
	for range 2 {
		ast.Inspect(body, func(n ast.Node) bool {
			as, ok := n.(*ast.AssignStmt)
			if !ok {
				return true
			}
			for _, rhs := range as.Rhs {
				if !mentionsAnyOf(rhs, derived) {
					continue
				}
				if name, ok := assignedIdent(as); ok {
					derived[name] = true
				}
			}
			return true
		})
	}
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}
		ix, ok := n.(*ast.IndexExpr)
		if !ok {
			return true
		}
		if derived[baseIdentName(ix.X)] && exprMentions(ix.Index, pos) {
			found = true
		}
		return !found
	})
	return found
}

// mentionsAnyOf reports whether e references any of the given identifiers.
func mentionsAnyOf(e ast.Expr, names map[string]bool) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && names[id.Name] {
			found = true
			return false
		}
		return !found
	})
	return found
}

// identIsLoopBound reports whether name appears in a loop's bound — a for-condition or
// a range expression. That marks the value as something the function walks over, which
// is the opposite of selecting a single row by it.
func identIsLoopBound(body *ast.BlockStmt, name string) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}
		switch l := n.(type) {
		case *ast.ForStmt:
			if l.Cond != nil && exprMentions(l.Cond, name) {
				found = true
			}
		case *ast.RangeStmt:
			if l.X != nil && exprMentions(l.X, name) {
				found = true
			}
		}
		return !found
	})
	return found
}

// isNumericSliceType reports whether an asserted type is a slice of a numeric builtin —
// the shape a devirtualized storage fast path asserts, as opposed to the pointer-to-
// struct assertions that fill any AST or reflection walker.
func isNumericSliceType(e ast.Expr) bool {
	at, ok := e.(*ast.ArrayType)
	if !ok || at.Len != nil {
		return false
	}
	id, ok := at.Elt.(*ast.Ident)
	if !ok {
		return false
	}
	switch id.Name {
	case "float32", "float64", "uint16", "uint8", "int8", "int16", "int32", "int64", "byte":
		return true
	}
	return false
}

// indirectKeyComparatorFindings flags PS3005 — sorting an INDEX slice with a comparator
// that dereferences the sorted element into a 2-D structure:
//
//	sort.Slice(idx, func(a, b int) bool { return m[idx[a]][f] < m[idx[b]][f] })
//
// Every comparison pays a row-pointer load plus an index, O(n log n) times, to read a
// value that depends only on the element — so filling a flat id-indexed key column once
// (O(n)) and comparing THAT removes the indirection from the hot loop entirely. The
// rewrite keeps the SAME PREDICATE, so the sort returns the same permutation, ties
// included; it is not an argument about acceptable tie orders.
//
// Measured three times in this repo, all the same shape: the GBM presort (1.05x, and
// 1.10x cumulative once the flat key made a radix pass practical) and the ball-tree
// median split (1.088x on KNN fit, the half that loses to sklearn). The CART builder had
// already solved it locally, which is what made the siblings findable.
//
// Requires the comparator to reach TWO levels deep THROUGH the sorted slice —
// m[idx[a]][f]. A comparator reading a flat key (key[idx[a]]) is the FIXED form and is
// deliberately silent, so applying the advice clears the finding.
func indirectKeyComparatorFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || len(call.Args) != 2 {
			return true
		}
		switch calleeName(call.Fun) {
		case "Slice", "SliceStable", "SortFunc", "SortStableFunc":
		default:
			return true
		}
		sorted := baseIdentName(call.Args[0])
		if sorted == "" {
			return true
		}
		lit, ok := call.Args[1].(*ast.FuncLit)
		if !ok || lit.Type.Params == nil {
			return true
		}
		var params []string
		for _, f := range lit.Type.Params.List {
			for _, nm := range f.Names {
				params = append(params, nm.Name)
			}
		}
		if len(params) != 2 {
			return true
		}
		if base, ok := doubleIndexThrough(lit.Body, sorted, params); ok {
			out = append(out, finding{
				pos:      fset.Position(call.Pos()),
				category: "indirect-key-comparator",
				msg: fmt.Sprintf("the comparator sorting %q dereferences %s[%s[...]][...] on every"+
					" comparison — a row-pointer load plus an index, O(n log n) times, for a value that"+
					" depends only on the element. Fill a flat id-indexed key column once (O(n)) and"+
					" compare that; the predicate is unchanged so the permutation is identical, ties"+
					" included. Measured 1.05x on the GBM presort and 1.088x on the ball-tree split,"+
					" and the flat key is what makes a radix pass practical afterwards.",
					sorted, base, sorted),
			})
		}
		return true
	})
	return out
}

// doubleIndexThrough reports whether the body contains m[sorted[p]][...] for one of the
// comparator's parameters p — the two-level dereference through the sorted slice that
// makes the comparator expensive. Returns the outer base name for the message.
func doubleIndexThrough(body *ast.BlockStmt, sorted string, params []string) (string, bool) {
	var base string
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}
		outer, ok := n.(*ast.IndexExpr)
		if !ok {
			return true
		}
		// Shape: m[idx[a]][f]. The OUTER IndexExpr is m[idx[a]] indexed by f, so the
		// sorted slice sits in the outer's own Index (idx[a]), not in its X — getting
		// that nesting backwards made the first cut of this rule silent on all three
		// sites it was written from.
		lookup, ok := outer.X.(*ast.IndexExpr)
		if !ok {
			return true
		}
		sel, ok := lookup.Index.(*ast.IndexExpr)
		if !ok || baseIdentName(sel.X) != sorted {
			return true
		}
		for _, p := range params {
			if exprMentions(sel.Index, p) {
				base, found = baseIdentName(lookup.X), true
				return false
			}
		}
		return true
	})
	return base, found
}

// innerInvariantRecomputeFindings flags PS5003 — an expression in the innermost loop
// that depends on the INNER index but NOT the outer one:
//
//	for i := range n {          // outer
//	    for j := range m {      // inner
//	        h[i][j] = a*h[i][j] + b*(x[off+j] * delta)   // x[off+j]*delta has no i
//	    }
//	}
//
// The parenthesized product is recomputed n times for every j, though it is the same
// value each time. Precomputing it once into an m-sized scratch before the outer loop
// leaves an indexed read. The rewrite is BIT-IDENTICAL when the expression is pure: it
// is the SAME product, so it rounds the same way — this is not a reassociation.
//
// Go will not do it: the operands are index expressions the compiler cannot prove
// unaliased across the outer iteration, so the load and the multiply stay in the loop.
//
// Measured on the Mamba2 SSM scan, where x[t][hOff+j]*delta was recomputed N times per
// j: 1.08x-1.10x on prefill end to end, the largest single win in that function and
// bigger than fixing its pathological access pattern was.
//
// Requires the expression to be NON-TRIVIAL — a multiply or divide, at least one index
// expression, and no call (a call may not be pure, and hoisting it changes evaluation
// count observably). A bare index read is excluded: hoisting `x[j]` buys nothing.
func innerInvariantRecomputeFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		outerVar, outerBody, ok := loopVarBody(n)
		if !ok {
			return true
		}
		for _, st := range outerBody.List {
			innerVar, innerBody, ok := loopVarBody(st)
			if !ok || innerVar == outerVar {
				continue
			}
			// A tight kernel only: one statement in the inner loop. Anything longer and
			// the recompute is not plausibly the dominant cost, which is what turned an
			// untightened version of this check into 445 findings.
			if len(innerBody.List) != 1 {
				continue
			}
			ast.Inspect(innerBody, func(m ast.Node) bool {
				be, isBin := m.(*ast.BinaryExpr)
				if !isBin || (be.Op != token.MUL && be.Op != token.QUO) {
					return true
				}
				// Must be a PARENTHESIZED or nested subexpression of a larger one — the
				// shape a compiler cannot hoist and a human does not notice. A whole
				// right-hand side is already as hoisted as it is going to get.
				if !nestedInArithmetic(innerBody, be) {
					return true
				}
				if !exprMentions(be, innerVar) || exprMentions(be, outerVar) {
					return true
				}
				if !hasIndexOperand(be) || containsCall(be) {
					return true
				}
				// AND none of its operands may be WRITTEN inside the outer loop. Textual
				// independence from the outer index is not semantic invariance: a
				// per-row softmax `p` mentions no outer variable yet is rebuilt every
				// outer iteration, so hoisting it would be wrong, not merely useless.
				// Without this the check was unsound — every sampled finding was of
				// exactly that kind.
				if mentionsAnyOf(be, assignedIn(outerBody)) {
					return true
				}
				out = append(out, finding{
					pos:      fset.Position(be.Pos()),
					category: "inner-invariant-recompute",
					msg: fmt.Sprintf("this product varies with %q but not with the enclosing %q, so it is"+
						" recomputed on every iteration of the outer loop for the same result. Precompute"+
						" it once into a scratch indexed by %q. Bit-identical — it is the same product,"+
						" not a reassociation. Measured 1.10x on the Mamba2 SSM scan, where the value was"+
						" rebuilt N times per inner index.", innerVar, outerVar, innerVar),
				})
				return true
			})
		}
		return true
	})
	return out
}

// hasIndexOperand reports whether the expression reads through an index — the marker
// that recomputing it costs a load and not just an arithmetic op.
// invariantTranscendentalRecomputeFindings flags PS5005: a PURE libm transcendental
// (the oExpensiveOps set: math.Pow/Exp/Log/Sin/Cos/...) called in an inner loop whose
// arguments vary with the INNER index but NOT the outer one, e.g. a per-frequency
// 1/Pow(base, 2i/d) recomputed on every outer position. Precompute the per-inner-index
// values into a scratch hoisted above the outer loop, then index it. Measured 3.6x
// (-72%) on sinusoidal positional encoding at seqLen=dModel=512.
//
// This is the transcendental sibling of PS5003 (inner-invariant-recompute), which
// deliberately EXCLUDES calls because a general call may not be pure. The oExpensiveOps
// functions are the exception: deterministic, side-effect-free libm math, so hoisting
// is bit-identical and sound. Unlike PS5003 it needs no index operand or one-statement
// inner body: the transcendental itself is the cost. Conservative like PS5003: the call
// must mention the inner index, must NOT mention the outer index, and none of its
// operands may be written by the outer loop outside the inner loop.
// outerCallArgTaint collects identifiers passed as a BARE argument to a call in body,
// skipping the `skip` subtree (the inner loop) and any math.* call (pure, never mutates
// an argument). A variable filled through such a call — softmaxRowFlat(p, row) — carries
// whatever the call wrote, so a transcendental that reads it is not outer-invariant even
// though the variable never appears on an assignment LHS. Bare identifier only: an
// indexed or expression argument (x[i], 2*i) is a value, not a mutable output slot.
func outerCallArgTaint(body *ast.BlockStmt, skip ast.Node) map[string]bool {
	taint := map[string]bool{}
	ast.Inspect(body, func(n ast.Node) bool {
		if n == skip {
			return false
		}
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
			if id, ok := sel.X.(*ast.Ident); ok && id.Name == "math" {
				return true
			}
		}
		for _, a := range call.Args {
			if id, ok := a.(*ast.Ident); ok {
				taint[id.Name] = true
			}
		}
		return true
	})
	return taint
}

func invariantTranscendentalRecomputeFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		outerVar, outerBody, ok := loopVarBody(n)
		if !ok {
			return true
		}
		for _, st := range outerBody.List {
			innerVar, innerBody, ok := loopVarBody(st)
			if !ok || innerVar == outerVar {
				continue
			}
			// Everything the outer loop body ASSIGNS — including locals set inside the
			// inner loop — taints the transcendental: a local like dt := delta[t][d]
			// carries the outer index t even though math.Exp(dt*A[d][n]) never names t
			// textually. Only the loop INDEX variables are safe to exclude (indexing the
			// precomputed scratch by the inner index is the whole point). Subtracting the
			// full inner assignment set instead of just innerVar was unsound — it dropped
			// exactly those tainted locals (the SSM scan's dt), so mirror PS5003 and keep
			// the whole outer-body write set, minus the two index vars.
			outerWrites := assignedIn(outerBody)
			delete(outerWrites, innerVar)
			delete(outerWrites, outerVar)
			// A slice/pointer filled by a CALL is mutated invisibly to assignedIn (which
			// only sees =/:=/++): softmaxRowFlat(p, teacherRow) rebuilds p every outer
			// iteration, so math.Log(p[j]) is NOT outer-invariant though p is never on an
			// assignment LHS. Taint any bare-identifier passed to a non-math call in the
			// outer body OUTSIDE the inner loop (math.* is pure and never mutates its args;
			// restricting to outside-the-inner-loop keeps a helper(i) call in the inner body
			// from spuriously tainting the index we precompute on).
			callTaint := outerCallArgTaint(outerBody, st)
			seen := map[token.Pos]bool{}
			ast.Inspect(innerBody, func(m ast.Node) bool {
				call, ok := m.(*ast.CallExpr)
				if !ok {
					return true
				}
				tname, ok := pkgFuncCall(call.Fun, "math", oExpensiveOps)
				if !ok {
					return true
				}
				if seen[call.Pos()] {
					return true
				}
				if !exprMentions(call, innerVar) || exprMentions(call, outerVar) {
					return true
				}
				if mentionsAnyOf(call, outerWrites) {
					return true
				}
				if mentionsAnyOf(call, callTaint) {
					return true
				}
				seen[call.Pos()] = true
				out = append(out, finding{
					pos:      fset.Position(call.Pos()),
					category: "loop-invariant-transcendental",
					msg: fmt.Sprintf("math.%s varies with %q but not with the enclosing %q, so this pure"+
						" transcendental is re-evaluated for the same value on every outer iteration."+
						" Precompute it once into a scratch indexed by %q, hoisted above the outer loop."+
						" Bit-identical: math.%s is pure, so the hoist is the same value, not a"+
						" reassociation. Measured 3.6x on sinusoidal positional encoding.",
						tname, innerVar, outerVar, innerVar, tname),
				})
				return true
			})
		}
		return true
	})
	return out
}

func hasIndexOperand(e ast.Expr) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if _, ok := n.(*ast.IndexExpr); ok {
			found = true
			return false
		}
		return !found
	})
	return found
}

// containsCall reports whether the expression calls anything. A call may not be pure and
// hoisting it changes how often it runs, which is observable; those are left alone.
func containsCall(e ast.Expr) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if _, ok := n.(*ast.CallExpr); ok {
			found = true
			return false
		}
		return !found
	})
	return found
}

// nestedInArithmetic reports whether e appears as an OPERAND of another arithmetic
// expression rather than standing alone as a statement's value — `b*(x[j]*d)` counts,
// `v := x[j]*d` does not, because the latter is already a single hoistable assignment a
// reader would see.
func nestedInArithmetic(root ast.Node, e ast.Expr) bool {
	found := false
	ast.Inspect(root, func(n ast.Node) bool {
		if found {
			return false
		}
		be, ok := n.(*ast.BinaryExpr)
		if !ok {
			return true
		}
		for _, side := range []ast.Expr{be.X, be.Y} {
			if inner, ok := side.(*ast.ParenExpr); ok && inner.X == e {
				found = true
			}
			if side == e {
				found = true
			}
		}
		return !found
	})
	return found
}

// assignedIn collects the base identifiers written anywhere in a loop body — by
// assignment, compound assignment, or increment. An expression that reads any of them
// is not invariant across that loop no matter what its indices say.
func assignedIn(body *ast.BlockStmt) map[string]bool {
	w := map[string]bool{}
	ast.Inspect(body, func(n ast.Node) bool {
		switch v := n.(type) {
		case *ast.AssignStmt:
			for _, lhs := range v.Lhs {
				if name := baseIdentName(lhs); name != "" {
					w[name] = true
				}
			}
		case *ast.IncDecStmt:
			if name := baseIdentName(v.X); name != "" {
				w[name] = true
			}
		}
		return true
	})
	return w
}

// partialFastPathFindings flags PS6003 — a function that short-circuits the general
// path for SOME members of a variant family, where a switch in the same function shows
// the family is larger. The uncovered variants keep paying the general path, and nothing
// about the code says so: the fast path reads as "this case is handled", not "only this
// case is handled".
//
// Found in the wild by its symptom rather than its shape, which is the argument for the
// rule. gguf.QMatMul had a fused single-token path for Q8_0 and none for Q4_0, so Q4_0
// decode ran SLOWER than Q8_0 despite half the memory traffic — backwards for the
// smaller format. Fusing it was 1.40x on the enclosing decode step. Nobody reading
// QMatMul would have suspected a gap; the switch below the fast path listed seven types.
//
// Deliberately NOT a defect report. A fast path may legitimately cover one variant —
// the others may be rare, or unfusable, or already fast. What the rule asserts is only
// that the asymmetry is INTENTIONAL-OR-NOT and nothing in the code distinguishes those,
// so it is worth one benchmark per uncovered variant.
func partialFastPathFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	// Fast paths first: an `if` whose condition tests <subject> == <named constant> and
	// whose body RETURNS. The early return is what makes it a bypass rather than a
	// branch — without it the general path still runs and there is no coverage gap.
	//
	// The guard must CLOSE BEFORE the switch it is judged against opens. Without that,
	// an `if vt == vtI16 { … return }` sitting inside one case clause of `switch vt`
	// reads as a fast path for the whole switch, when it bypasses nothing — it is a
	// sub-case of the dispatch. That was the rule's only false positive on this tree.
	type fastPath struct {
		eqCond
		end token.Pos
	}
	var fast []fastPath
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		ifs, ok := n.(*ast.IfStmt)
		if !ok || !endsInReturn(ifs.Body) {
			return true
		}
		for _, c := range eqConds(ifs.Cond) {
			fast = append(fast, fastPath{c, ifs.End()})
		}
		return true
	})
	if len(fast) == 0 {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		sw, ok := n.(*ast.SwitchStmt)
		if !ok || sw.Tag == nil {
			return true
		}
		sub := identName(sw.Tag)
		if sub == "" {
			return true
		}
		covered := map[string]bool{}
		for _, f := range fast {
			if f.subject == sub && f.end <= sw.Pos() {
				covered[f.constant] = true
			}
		}
		if len(covered) == 0 {
			return true
		}
		// Only NAMED constants count as a variant family. Allowing literals would fire
		// on every `switch n { case 1: case 2: ... }`, which is not a family of formats.
		var members []string
		for _, cl := range sw.Body.List {
			cc, ok := cl.(*ast.CaseClause)
			if !ok {
				continue
			}
			for _, e := range cc.List {
				if c := constName(e); c != "" {
					members = append(members, c)
				}
			}
		}
		// Three is the floor for a "family": a two-way switch is a branch, and a fast
		// path for one of two arms is just an if/else written twice.
		if len(members) < 3 {
			return true
		}
		var missing []string
		hits := 0
		for _, m := range members {
			if covered[m] {
				hits++
			} else {
				missing = append(missing, m)
			}
		}
		// Both halves must be non-empty. All covered means there is no gap; none covered
		// means the fast path is keyed on something unrelated to this switch.
		if hits == 0 || len(missing) == 0 {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(sw.Pos()),
			end:      fset.Position(sw.End()),
			category: "partial-fast-path-coverage",
			msg: fmt.Sprintf("fast path short-circuits %d of the %d %q variants this switch handles; %s still take the general path — benchmark whether they should",
				hits, len(members), sub, strings.Join(missing, ", ")),
		})
		return true
	})
	return out
}

// endsInReturn reports whether a block's LAST statement returns. A return buried in a
// nested conditional is not a bypass — the block can fall through to the general path.
func endsInReturn(b *ast.BlockStmt) bool {
	if b == nil || len(b.List) == 0 {
		return false
	}
	_, ok := b.List[len(b.List)-1].(*ast.ReturnStmt)
	return ok
}

// eqCond is one `<subject> == <constant>` test recovered from a guard condition.
type eqCond struct{ subject, constant string }

// eqConds collects every `<ident> == <named constant>` reachable through && and || in a
// guard condition. BOTH operators establish coverage, and restricting this to && was a
// real defect — the rule reported Q4_K and Q6_K as uncovered immediately after they were
// fused, because their guard spells the pair as `(qt == Q4_K || qt == Q6_K) && m == 1`.
//
// Either operator is sound here. Under && the branch is taken when the constant holds
// and the rest of the guard passes; under || it is taken whenever the constant holds,
// regardless of the rest. In both cases a path exists that short-circuits that variant,
// which is exactly what coverage means. A negated test (!=) is not coverage, and a
// unary ! is not descended into, so neither is collected.
func eqConds(e ast.Expr) []eqCond {
	var out []eqCond
	var walk func(ast.Expr)
	walk = func(e ast.Expr) {
		be, ok := ast.Unparen(e).(*ast.BinaryExpr)
		if !ok {
			return
		}
		switch be.Op {
		case token.LAND, token.LOR:
			walk(be.X)
			walk(be.Y)
		case token.EQL:
			sub, con := identName(be.X), constName(be.Y)
			if sub == "" || con == "" { // also accept the reversed spelling
				sub, con = identName(be.Y), constName(be.X)
			}
			if sub != "" && con != "" {
				out = append(out, eqCond{sub, con})
			}
		}
	}
	walk(e)
	return out
}

// identName renders a plain identifier, the shape a switch tag and a guard subject
// share. Anything more complex is not matched across the two by name.
func identName(e ast.Expr) string {
	if id, ok := ast.Unparen(e).(*ast.Ident); ok {
		return id.Name
	}
	return ""
}

// constName renders an identifier or a qualified pkg.Const, the two spellings a
// variant-family member takes. Literals are excluded on purpose (see the caller).
func constName(e ast.Expr) string {
	switch v := ast.Unparen(e).(type) {
	case *ast.Ident:
		return v.Name
	case *ast.SelectorExpr:
		if p, ok := v.X.(*ast.Ident); ok {
			return p.Name + "." + v.Sel.Name
		}
	}
	return ""
}

// unwrapNumConv strips enclosing numeric conversions (float32(x), float64(x), etc.) so a
// conversion-wrapped index expression like float64(gs[i]) is seen as the underlying gs[i].
func unwrapNumConv(e ast.Expr) ast.Expr {
	for {
		call, ok := e.(*ast.CallExpr)
		if !ok || len(call.Args) != 1 {
			return e
		}
		id, ok := call.Fun.(*ast.Ident)
		if !ok {
			return e
		}
		switch id.Name {
		case "float32", "float64", "int", "int32", "int64", "uint32", "uint64", "byte":
			e = call.Args[0]
		default:
			return e
		}
	}
}

// indexBase returns the slice identifier of an s[i]-style index expression whose index
// mentions loop var iName (after stripping numeric conversions): float64(gs[i]) → ("gs", true).
func indexBase(e ast.Expr, iName string) (string, bool) {
	ix, ok := unwrapNumConv(e).(*ast.IndexExpr)
	if !ok || !exprMentions(ix.Index, iName) {
		return "", false
	}
	if id, ok := ix.X.(*ast.Ident); ok {
		return id.Name, true
	}
	return "?", true
}

// vjpScalarBinopFindings flags PS4007 — a reverse-mode *VJP whose hot loop is a SINGLE
// elementwise binary op dst[i] = a[i] ∘ b[i] (∘ ∈ *,+,-,/) written as a scalar Go loop
// instead of dispatching the matching backend op. gc does not autovectorize, so on the
// GOEXPERIMENT=simd build the loop stays scalar+single-core while backend.Execute(ctx,
// OpMul/OpAdd/…) routes to the 8-wide AVX kernel + parallel(). A LONE IEEE binop is
// correctly-rounded regardless of vectorization/chunking, so the dispatch is bit-exact —
// this is exactly the expVJP → OpMul win. Scoped to *VJP names (the autograd dispatch
// layer); backend kernels are excluded (their scalar loop IS the implementation). The
// single-statement guard is essential: multi-op VJP bodies (tanh g·(1−y²), sigmoid
// g·y·(1−y)) keep f64 intermediates and narrow once — composing f32 backend ops would
// diverge, so they need a fused backward kernel and are NOT flagged.
func vjpScalarBinopFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Name == nil || !strings.HasSuffix(fn.Name.Name, "VJP") || fn.Body == nil {
		return nil
	}
	// A VJP that already dispatches to the backend is fine.
	dispatches := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if sel, ok := n.(*ast.SelectorExpr); ok && sel.Sel.Name == "Execute" {
			if id, ok := sel.X.(*ast.Ident); ok && id.Name == "backend" {
				dispatches = true
				return false
			}
		}
		return true
	})
	if dispatches {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		iName, body, ok := loopVarBody(n)
		if !ok || body == nil || len(body.List) != 1 {
			return true
		}
		as, ok := body.List[0].(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 || as.Tok != token.ASSIGN {
			return true
		}
		if _, ok := indexBase(as.Lhs[0], iName); !ok { // LHS must be dst[i]
			return true
		}
		bin, ok := unwrapNumConv(as.Rhs[0]).(*ast.BinaryExpr)
		if !ok {
			return true
		}
		op, ok := binopBackendOp[bin.Op]
		if !ok {
			return true
		}
		aBase, aok := indexBase(bin.X, iName)
		bBase, bok := indexBase(bin.Y, iName)
		if !aok || !bok {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(n.Pos()),
			category: "vjp-scalar-elementwise-binop",
			msg: fmt.Sprintf("VJP %s has a scalar elementwise loop dst[%s] = %s[%s] %s %s[%s] — gc does"+
				" not autovectorize, so on the simd build it stays scalar+single-core. A lone IEEE %s is"+
				" correctly-rounded regardless of vectorization, so dispatch backend.Execute(ctx, backend.Op%s,"+
				" …) to reach the 8-wide SIMD + parallel kernel (bit-exact; see expVJP → OpMul). Applies ONLY"+
				" to single-op bodies — multi-op VJPs (tanh, sigmoid) need a fused kernel and are not flagged.",
				fn.Name.Name, iName, aBase, iName, bin.Op.String(), bBase, iName, op[1], op[0]),
		})
		return false
	})
	return out
}

// fullSortBoundedPrefixFindings flags PS6001 — a full-vocabulary descending index sort whose
// result feeds an early-breaking (threshold-bounded) prefix, e.g. sorting all V tokens to take
// the top-p / surprise-≤-μ / top-k prefix. When only a small prefix is consumed, an O(V) filter
// (or quickselect) into the small candidate set + sorting just those is O(V + k log k) instead
// of O(V log V). Shipped: Mirostat prob pre-filter, nucleusTopP quickselect (§T627).
//
// Fires only when ALL hold, keeping already-optimized code silent:
//   - a full-index fill (S[i] = i) initializes the sorted slice → it is the WHOLE vocabulary;
//   - the function calls sortIdxDescByProb / sortIdxDescByKey on that slice;
//   - a bounded consumer follows — a for-loop with an early break (the prefix cutoff);
//   - NO quickselectIdxDesc guard — its presence marks the optimized top-K-then-fallback form
//     (nucleusTopP), whose retained full-sort fallback must NOT be flagged.
//
// A pure full-sort HELPER that just returns the sorted slice (no in-function break consumer,
// e.g. Mirostat's sortedKeep fallback) is likewise silent.
func fullSortBoundedPrefixFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	guarded, hasBreak := false, false
	var sortCall *ast.CallExpr
	var sortSlice string
	fullFill := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.CallExpr:
			if id, ok := x.Fun.(*ast.Ident); ok {
				switch id.Name {
				case "quickselectIdxDesc":
					guarded = true
				case "sortIdxDescByProb", "sortIdxDescByKey":
					if len(x.Args) >= 1 {
						if s, ok := x.Args[0].(*ast.Ident); ok {
							sortCall, sortSlice = x, s.Name
						}
					}
				}
			}
		case *ast.BranchStmt:
			if x.Tok == token.BREAK {
				hasBreak = true
			}
		case *ast.AssignStmt:
			// full-index fill: S[i] = i
			if len(x.Lhs) == 1 && len(x.Rhs) == 1 {
				ix, ok1 := x.Lhs[0].(*ast.IndexExpr)
				rid, ok2 := x.Rhs[0].(*ast.Ident)
				if ok1 && ok2 {
					s, sok := ix.X.(*ast.Ident)
					iid, iok := ix.Index.(*ast.Ident)
					if sok && iok && iid.Name == rid.Name {
						fullFill[s.Name] = true
					}
				}
			}
		}
		return true
	})
	if sortCall == nil || guarded || !hasBreak || !fullFill[sortSlice] {
		return nil
	}
	return []finding{{
		pos:      fset.Position(sortCall.Pos()),
		category: "full-sort-bounded-prefix",
		msg: fmt.Sprintf("full-vocabulary descending sort of %q (initialized S[i]=i) feeds an"+
			" early-breaking prefix — an O(n log n) sort for an O(prefix) need. If only a"+
			" threshold-bounded prefix is consumed (top-p mass, surprise ≤ μ, top-k), pre-filter"+
			" to the small candidate set in O(n) (or quickselect) and sort only those; guard the"+
			" full sort as a fallback. Bit-exact for n ≥ radixSortCutoff (stable radix; sort the"+
			" candidates with SortStableFunc to match the tie order). Shipped: Mirostat, nucleusTopP"+
			" §T627. Verify the consumer is genuinely bounded, then benchmark.", sortSlice),
	}}
}

// andedComparisons counts the relational comparisons (<, >, <=, >=) joined by && in a boolean
// expression tree — the shape of a compound spatial bounds guard (iy>=0 && iy<h && ix>=0 && ix<wd).
func andedComparisons(e ast.Expr) int {
	switch x := e.(type) {
	case *ast.ParenExpr:
		return andedComparisons(x.X)
	case *ast.BinaryExpr:
		switch x.Op {
		case token.LAND:
			return andedComparisons(x.X) + andedComparisons(x.Y)
		case token.LSS, token.GTR, token.LEQ, token.GEQ:
			return 1
		}
	}
	return 0
}

// loopHasNested reports whether a loop body contains another for/range loop (so the outer is
// not the innermost).
func loopHasNested(body *ast.BlockStmt) bool {
	nested := false
	for _, s := range body.List {
		ast.Inspect(s, func(n ast.Node) bool {
			switch n.(type) {
			case *ast.ForStmt, *ast.RangeStmt:
				nested = true
				return false
			}
			return true
		})
	}
	return nested
}

// stmtHasIndexedTarget reports whether the statement is a single assignment/compound-assignment
// whose LHS is an indexed expression (a[i] = … or a[i] += …) — one scatter/gather move.
func stmtHasIndexedTarget(s ast.Stmt) bool {
	as, ok := s.(*ast.AssignStmt)
	if !ok || len(as.Lhs) != 1 {
		return false
	}
	_, ok = as.Lhs[0].(*ast.IndexExpr)
	return ok
}

// spatialBoundsBranchFindings flags PS6002 — an INNERMOST window/kernel loop whose body is a
// single compound-bounds `if` (≥3 anded relational comparisons: the iy>=0 && iy<h && ix>=0 &&
// ix<wd shape) guarding one indexed load/store. In a convolution / pooling im2col-style loop the
// index steps by 1 per tap, so the in-bounds taps form ONE contiguous run and the guard hoists
// out of the inner loop into a [lo,hi) bulk copy / scatter-add — branch-free and bit-identical
// (padding taps contribute nothing). Shipped: conv2d forward im2colFillBand + backward col2im.
func spatialBoundsBranchFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		_, body, ok := loopVarBody(n)
		if !ok || body == nil || loopHasNested(body) {
			return true
		}
		for _, stmt := range body.List {
			ifs, ok := stmt.(*ast.IfStmt)
			if !ok || ifs.Else != nil || ifs.Init != nil {
				continue
			}
			if andedComparisons(ifs.Cond) < 3 || len(ifs.Body.List) != 1 {
				continue
			}
			if !stmtHasIndexedTarget(ifs.Body.List[0]) {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(ifs.Pos()),
				category: "spatial-bounds-branch",
				msg: "innermost window/kernel loop re-tests a compound spatial bounds guard per tap" +
					" (≥3 anded comparisons) around a single indexed move. If the index steps by 1" +
					" per tap the in-bounds taps form one contiguous run — hoist the guard out into a" +
					" [lo,hi) bulk copy / scatter-add (branch-free, bit-identical: padding taps add" +
					" nothing). See conv2d im2colFillBand / col2im. Verify the stride-1 assumption.",
			})
			return false
		}
		return true
	})
	return out
}

var binopBackendOp = map[token.Token][2]string{
	token.MUL: {"Mul", "multiply"}, token.ADD: {"Add", "add"},
	token.SUB: {"Sub", "subtract"}, token.QUO: {"Div", "divide"},
}
