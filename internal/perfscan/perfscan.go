// Command perfscan is a static finder for the per-element hot-loop anti-patterns
// this codebase repeatedly optimizes away (§base-perf-sweep). It parses Go source
// with go/ast — it never builds or imports the packages, so cgo/build-tagged
// backends are scanned too — and reports the STRUCTURAL SHAPE of fifteen patterns
// (this is the SINGLE perfscan for the repo — the earlier tools/perfscan P1/P2
// prototype was folded in here as detectors I/J):
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
// constructor is fine (§C3, measure don't assume). Each hit still needs an A/B
// measurement and a bit-identity proof before it ships (§V22). perfscan replaces
// the ad-hoc awk scans in docs/perf-notes-training.md with an AST-accurate,
// comment/string-safe, CI-wirable pass.
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
func integerKeyType(e ast.Expr) bool {
	id, ok := e.(*ast.Ident)
	if !ok {
		return false
	}
	switch id.Name {
	case "int", "int8", "int16", "int32", "int64",
		"uint", "uint8", "uint16", "uint32", "uint64",
		"byte", "rune", "uintptr":
		return true
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
		case *ast.ValueSpec: // var m map[int]V
			for _, nm := range x.Names {
				add(x.Type, nm.Name)
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

// allocCallees are constructors/converters that allocate a fresh tensor; called
// per element they turn an O(n) loop into O(n) allocations (§base-perf §2).
var allocCallees = map[string]bool{
	"New": true, "NewOn": true, "FromFloat64": true, "FromFloat32": true,
	"Zeros": true, "Ones": true, "Full": true, "Scalar": true,
	"scalarTensor": true, "Cast": true,
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

// perElemVisitors are helpers that invoke their function-literal argument ONCE
// PER ELEMENT. Passing a closure to them means an indirect call per element even
// on their contiguous fast branch — the cost that made SWA/EMA/GradAccum/GradFn
// 1.6–4.2× slower than a raw-slice loop (detector PS1002; formerly tools/perfscan P1).
var perElemVisitors = map[string]bool{
	"readGen": true, "fillGen": true, "visitGen": true, "forEach": true, "eachElem": true,
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

// vectorizedSibling reports whether name looks like a hand-vectorized numeric
// kernel helper (vsiluF32, vexpF32, vgeluF32, expF64x4, …): a 'v'-prefixed
// F32/F64 helper, or an xN-lane-suffixed SIMD primitive.
func vectorizedSibling(name string) bool {
	if strings.HasPrefix(name, "v") && (strings.Contains(name, "F32") || strings.Contains(name, "F64")) {
		return true
	}
	return strings.Contains(name, "x8") || strings.Contains(name, "x4") || strings.Contains(name, "x16")
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
func scanFile(fset *token.FileSet, f *ast.File) []finding {
	var out []finding
	// File-scoped fact: the package-local elementwise funcs that WRAP a libm
	// transcendental (softplus, mish, swish, …). Class PS4002 only sees a DIRECT math.X
	// in the loop; these hide it one call deep, so a hot per-element loop over them
	// reads as scalar-clean. Class PS4003 flags calls to them inside loops.
	wrappers := transcendentalWrappers(f)
	// File-scoped fact: names declared as integer-keyed maps (detector PS3003) — indexing
	// them in a loop is a dense-slice (map→slice) candidate.
	intKeyMaps := intKeyMapNames(f)
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		out = append(out, scanFunc(fset, fn, wrappers, intKeyMaps)...)
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

func scanFunc(fset *token.FileSet, fn *ast.FuncDecl, wrappers, intKeyMaps map[string]bool) []finding {
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
			switch calleeName(x.Fun) {
			case "flatF64", "flatF32", "F64", "F32":
				hasFlat = true
			case "Grow":
				hasGrow = true
			case "rawCopyLE", "rawStoreLE":
				hasLEBulk = true
			}
			if vectorizedSibling(calleeName(x.Fun)) {
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
				msg: fmt.Sprintf("per-element .%s in a Numel/Unravel loop, no typed fast path (flatF64/flatF32/Storage().F64()) in %s()"+
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
		// D: a reflection-based fmt scan/format call in any loop (per-element reflection + alloc).
		if fname, ok := fmtReflectCall(call.Fun); ok {
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "reflection-in-loop",
				msg: fmt.Sprintf("fmt.%s in a loop — reflection-based + allocates on every call;"+
					" gate it behind a cheap check or hand-parse (T931: SPM.Decode fmt.Sscanf per token = 1.55M allocs → 20×)", fname),
			})
		}
		// E: a strings.Builder/bytes.Buffer written in a loop with NO .Grow anywhere in the
		// function — WriteX doubles the backing buffer log(n) times, leaving throw-away buffers.
		if hasBuilder && !hasGrow && builderWriteCallees[name] {
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "unsized-builder",
				msg: fmt.Sprintf("a strings.Builder/bytes.Buffer is written (.%s) in a loop with no .Grow —"+
					" pre-size it to skip the log(n) growth-buffer churn (T929: BPE/SPM/Unigram Decode pre-size, allocs↓, up to 1.65×)", name),
			})
		}
		// F: an allocating strings transform in any loop (a fresh string per iteration).
		if fname, ok := pkgFuncCall(call.Fun, "strings", stringsAllocCallees); ok {
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "strings-alloc-in-loop",
				msg: fmt.Sprintf("strings.%s in a loop — allocates a new string every call;"+
					" write the transform into the builder or slice in place (T934: Decode ReplaceAll per token → inline, 52k→1 alloc, 1.65×)", fname),
			})
		}
		// G: a per-element little-endian bit decode in a loop — a memcpy in disguise on LE hosts.
		// Silent when the function already has a rawCopyLE/rawStoreLE fast path (the loop is then
		// the big-endian fallback, not a candidate — the hasFlat discipline of detector PS1001).
		if bname, ok := binaryDecodeCall(call.Fun); ok && !hasLEBulk {
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "le-decode-in-loop",
				msg: fmt.Sprintf("%s in a loop — on a little-endian host the on-disk bytes ARE the in-memory layout;"+
					" a single rawCopyLE replaces the per-element decode for VERBATIM-bit dtypes (F32/F64/F16), T720/T907"+
					" (2–5×). A quant-widen path genuinely decodes per-element — triage before acting.", bname),
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
				msg: fmt.Sprintf("math.%s runs scalar in a loop while a vectorized v*F32/F64 sibling is"+
					" called in the same kernel — vectorize this dtype branch on a SIMD transcendental"+
					" (keep a bit-identical scalar tail). FIRST verify the op is NOT under a bit-exact"+
					" CPU==Ref invariant (some f64 ops are locked to the scalar reference). T667: F64 SiLU"+
					" vexp = 1.52× Llama prefill.", tname),
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
					" the same class as a raw math.Exp/Log (K), just one call deep. Candidate for a"+
					" vectorized SIMD kernel or a batched tensor op (compute the whole slice 4/8-wide,"+
					" keep a bit-identical scalar tail). Verify hotness + that the op is not under a"+
					" bit-exact CPU==Ref invariant. F64 OpSoftplus vsoftplus = 1.62× Mamba prefill.", wname),
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
						" non-escaping local - per-call scratch reallocated every call. Hoist it to a"+
						" reused receiver field (grow-on-demand, zero only if read-before-write) so the"+
						" per-Step allocations drop to zero (Adafactor/Cautious/LAMB StepOnly 1.2-1.5x,"+
						" 12 allocs to 0). Verify the buffer is fully overwritten before use.", id.Name),
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
			var div ast.Expr
			switch e := n.(type) {
			case *ast.BinaryExpr:
				if e.Op == token.QUO {
					div = e.Y
				}
			case *ast.AssignStmt:
				if e.Tok == token.QUO_ASSIGN && len(e.Rhs) == 1 {
					div = e.Rhs[0]
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
		if perElemVisitors[name] && hasFuncLitArg(call) {
			out = append(out, finding{
				pos:      fset.Position(call.Pos()),
				category: "per-element-closure",
				msg: fmt.Sprintf("%s(…) invokes a closure per element — add a contiguous raw-slice fast"+
					" path (IsContiguous && Offset==0) for F32/F64, keep the closure as the strided"+
					" fallback (SWA/EMA/GradAccum 1.6–4.2×). Verify bit-identical + benchmark.", name),
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
				msg: fmt.Sprintf("%s uses an indirect comparator — if the key is a monotonic float/int"+
					" over a large slice, replace with an LSD radix on the key bits (math.Float64bits is"+
					" monotonic for non-negative f64). Top-p / typical sampling 1.9–2.25×. Verify identical"+
					" order + benchmark.", sname),
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
					" [0,N) a []T indexed by the key replaces the per-lookup hash (map→slice: BPE Decode 2.85×,"+
					" GGUF Decode 3.67×, forest votes T312). Verify key density; gaps/out-of-range keep the"+
					" map's zero-value via a bounds check.", indexedMapName(idx.X)),
			})
			return true
		})
	}
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
	strict := flag.Bool("strict", false, "exit 1 if any finding is reported (optional CI gating)")
	tests := flag.Bool("tests", false, "include _test.go files")
	checksFlag := flag.String("checks", "", "comma-separated allow-list of checks to run, by ID (PS1001) or category; empty = all")
	exclude := flag.String("exclude", "", "comma-separated checks to silence, by ID (PS1001) or category")
	list := flag.Bool("list", false, "list the checks (ID, category, title, fixable) and exit")
	doFix := flag.Bool("fix", false, "apply the safe mechanical fixes in place (fixable checks only; see -list)")
	jsonOut := flag.Bool("json", false, "emit findings and fixes as JSON (for editor / tool integration)")
	flag.Parse()

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
	files, err := goFiles(roots, *tests)
	if err != nil {
		fmt.Fprintln(os.Stderr, "perfscan:", err)
		os.Exit(2)
	}

	fset := token.NewFileSet()
	var all []finding
	for _, path := range files {
		f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if err != nil {
			fmt.Fprintf(os.Stderr, "perfscan: parse %s: %v\n", path, err)
			continue
		}
		for _, fd := range scanFile(fset, f) {
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
