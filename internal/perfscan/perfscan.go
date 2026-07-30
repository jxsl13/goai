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
	"strconv"
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
	{"PS1005", "manual-walk-dispatch", "a per-element AtF64/SetF64 whose 2+ index args are enclosing-loop variables — a manual multi-dim tensor walk via dispatch that PS1001 Numel-loop check misses", false},
	{"PS1006", "strided-inner-reduction", "a reduction whose INNER loop var is the high-stride (multiplied) part of a flat index ARR[inner*stride + outer] while the OUTER loop var is the contiguous (additive) part — the inner loop strides ARR by `stride` every step (cache-thrashing). Interchange to inner-outer/outer-inner so ARR is walked contiguously; per output element the reduction stays in the same order, so it is bit-identical. Shipped: MLA value-mix (cpu 1.13x / ref 1.27x)", false},
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
	{"PS4009", "transposed-gram-colstride", "a symmetric-gram reduction M[k][i]·M[k][j] whose reduction index k is the OUTER (row) index of a row-major/jagged matrix — the innermost loop strides down a column across rows; reblock to k-outer rank-1 (load M[k] once, walk contiguously)", false},
	// PS5xxx — arithmetic
	{"PS5001", "loop-invariant-divide", "a divide by a loop-invariant scalar on every element", false},
	{"PS5004", "multi-sweep-fusable", "three or more consecutive loops over the same range each indexing a shared slice (fuse the passes into one sweep)", false},
	{"PS5002", "symmetric-accumulation", "a nested loop accumulating a full symmetric matrix (m[i][j] += x[i]*x[j]) where one triangle + mirror would halve the work", false},
	{"PS5003", "inner-invariant-recompute", "an inner-loop expression that varies with the INNER index but not the outer one — recomputed once per outer iteration where a precomputed row would do", false},
	{"PS5005", "loop-invariant-transcendental", "a pure libm transcendental (math.Pow/Exp/Log/Sin) whose args vary with the INNER index but not the outer one, recomputed every outer iteration", false},
	// PS6xxx — verification gaps
	{"PS6001", "full-sort-bounded-prefix", "a full-vocabulary descending index sort whose result feeds an early-breaking (threshold-bounded) prefix consumer, with no quickselect/pre-filter guard — an O(n log n) sort for an O(prefix) need", false},
	{"PS6002", "spatial-bounds-branch", "an innermost window/kernel loop re-testing a compound spatial bounds guard (iy>=0 && iy<h && ix>=0 && ix<wd) per tap, where the in-bounds taps form one contiguous run the guard can be hoisted around", false},
	{"PS6005", "monotone-index-bound", "an innermost loop guarding its tap work with a single relational bound on an index affine in the loop var (j:=t-(K-1)+k; if j>=0) — the in-bounds iterations are one contiguous run, clamp the loop bound instead of branching per tap", false},
	{"PS6021", "fanout-without-worker-seam", "a fan-out helper whose callback receives ONLY a single work index and no scratch, with no sibling scratch-constructor parameter — callers have nowhere to hoist a per-item buffer to, so every caller that needs a working buffer must allocate one PER ITEM; give the callback a scratch parameter or take a func() T constructor the helper calls once per worker", false},
	{"PS6019", "jam-tail-delegates", "an unroll-and-jammed loop (i+N <= bound) whose scalar remainder loop DELEGATES to a method on the receiver while the wide body is inlined — the two are separate code paths, so a fix applied to the wide one (per-worker scratch, a pinned product, a bounds hoist) silently misses the tail, and a test at a trip count divisible by N never executes it", false},
	{"PS6018", "layout-op-cluster-unfused", "three or more calls dispatching a pure DATA-MOVEMENT op (layoutOpConstants: slice, reshape, transpose, concat) in one function with no fused raw-storage path — movement cannot change a value, so gathering and scattering directly is bit-identical and removes every one of those dispatches", false},
	{"PS6017", "unpooled-variadic-sibling", "a VARIADIC helper called inside a loop at a fixed argument count, when the same package declares a non-variadic sibling with identical leading parameters and exactly that many trailing ones — the variadic form allocates a slice per call and the sibling exists to avoid it", false},
	{"PS6016", "loop-invariant-literal-arg", "a struct composite literal built INSIDE a loop and passed straight to a call, whose every field initializer is loop-invariant — the same value is rebuilt every iteration, and when the parameter is an interface it is heap-boxed every iteration; construct it once above the loop", false},
	{"PS6015", "batch1-call-feeds-only-postloop-slice", "a PURE (pureComputeFuncs) call inside a loop, given a single-element batch, whose result is used ONLY to append to a slice declared outside the loop — nothing in the loop reads it, so N batch-1 calls can become one batched call after the loop", false},
	{"PS6014", "redundant-pure-recompute", "two syntactically identical calls to a function declared PURE (pureComputeFuncs) in one block, with nothing between them assigning any name the call reads — the second recomputes what the first already holds; keep one and read its result twice", false},
	{"PS6013", "sort-feeds-counted-prefix", "a full sort whose result is then read only by a COUNTED prefix loop (for r := 0; r < k; r++ over the sorted slice) — the order past k is computed and discarded; a selection answers the same question in O(n) instead of O(n log n)", false},
	{"PS6012", "inconsistent-fma-pinning", "a function that rounds SOME products explicitly (float64(a*b)) to stop FMA contraction, but leaves a sibling product feeding an add or subtract unpinned — including one assigned to a named local, which the compiler still inlines and contracts", false},
	{"PS6011", "strided-inner-walk", "an inner loop whose index into a flat buffer MULTIPLIES the inner loop variable by a stride — consecutive iterations jump a whole row, so each touches its own cache line to use one element; interchange the loops, or block four adjacent outer indices so one fetched line serves four accumulators", false},
	{"PS6010", "output-invariant-operand-reload", "an output loop whose accumulator re-reads an operand that does NOT vary with the output index — unrolling the output loop by 4 amortizes that load across 4 accumulators (register blocking / unroll-and-jam)", false},
	{"PS6006", "receiver-scratch-buffer", "a method using a receiver SLICE FIELD as a per-call temporary (indexed write, then indexed read, same call) — unsafe to call concurrently, and every caller contends on one cache line", false},
	{"PS6007", "search-feeds-reduction", "a loop whose expensive per-item CALL produces an index into an accumulation — split it into a parallel search pass and a sequential fold, since chunked partials would reassociate the sums", false},
	{"PS6008", "alloc-in-parallel-body", "a buffer allocated INSIDE a parallel dispatch body — free when the dispatch is infrequent, ruinous when it sits in a hot loop; hoist to per-chunk buffers indexed by the chunk", false},
	{"PS6009", "reflect-swapper-sort", "sort.Slice/SliceStable reaches its swap through reflectlite.Swapper, which ALLOCATES on every call — slices.SortFunc/SortStableFunc is the same comparator with a monomorphized swap and no allocation", false},
	{"PS6003", "partial-fast-path-coverage", "a fast path that bypasses the general path for only SOME members of a variant family a switch in the same function enumerates — the uncovered variants silently pay the slow path", false},
	{"PS6004", "unverified-dual-path", "a devirtualized fast path with a generic fallback — a bit-identity claim needing a bit-exact test", true},
	{"PS4010", "vectorizable-butterfly", "an in-loop butterfly p,q = x+y,x-y (add and subtract of the SAME operand pair written to two indexed slots) — a scalar FWHT/FFT/Hadamard stage a SIMD Add/Sub over the contiguous stride-separated runs would vectorize", false},
	{"PS4011", "op-dispatch-recurrence", "a sequential loop dispatching 2+ backend ops (calls passing a backend.Op* constant) per iteration in a function with NO fused typed fast path (no flatF64 guard) — O(seq) dispatch+alloc overhead on tiny per-step tensors; add a raw-slice fused path", false},
	{"PS4012", "scaled-serial-dot", "a serial scalar dot accumulator whose result is SCALED/dequantized (acc*scale…) before being stored — a quantized/dequant GEMM inner loop; latency-bound like PS4008 but missed by it (acc isn't stored raw). Break the chain with independent accumulators; bit-identical when the products are integer-valued (int8·int8 partials < 2^53 reassociate exactly), else tolerance-gated", false},
	{"PS5006", "nested-subrange-rescan", "an innermost loop recomputing a running reduction (acc *= / += arr[k]) over a [j..i] sub-range whose bounds are the two enclosing loop vars — an O(T\u00b3) triangular rescan replaceable by a prefix/suffix scan precomputed once per outer index (O(T\u00b2))", false},
	{"PS5007", "f32-abs-via-f64", "a float32(math.Abs(float64(x))) round-trip on an f32 value — replace with a direct sign-bit clear math.Float32frombits(math.Float32bits(x) &^ (1<<31)); bit-identical |x|, no f64 conversion or call", false},
	{"PS5008", "sincos-fusable", "a function calling BOTH math.Sin(x) and math.Cos(x) on the SAME argument expression — each does the full argument reduction of x independently; fuse to `sin, cos := math.Sincos(x)` (one reduction, both polynomials). Go's math.Sincos shares Sin/Cos's exact reduction+polynomials so it is bit-identical. Verified: sinusoidal PE builder, RoPE trig fill (attn_extra.go already uses Sincos)", false},
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
	// PS6018 — backend op constants that only MOVE data (slice, reshape, transpose,
	// concat). A cluster of these around a little arithmetic is fusable bit-identically,
	// because a gather and a scatter cannot change a value. Empty by default.
	LayoutOpConstants []string `json:"layoutOpConstants,omitempty"`
	// PS6014 — entry points that are PURE with respect to their arguments: same
	// arguments, same result, no observable side effect (e.g. a network forward pass).
	// The purity judgment is a project's to make and cannot be derived from syntax, so
	// naming a function here is what licenses the check to call a second identical call
	// redundant. Empty by default: without it PS6014 cannot report.
	PureComputeFuncs []string `json:"pureComputeFuncs,omitempty"`
}

// nameSets is Config compiled to maps for O(1) lookup during a scan.
type nameSets struct {
	accessors, fastPath, elemCount, indexDecompose map[string]bool
	shapeMethods                                   map[string]bool
	allocators, visitors, bulkCopy, vectorized     map[string]bool
	pureCompute                                    map[string]bool
	layoutOps                                      map[string]bool
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
		pureCompute:    toSet(c.PureComputeFuncs),
		layoutOps:      toSet(c.LayoutOpConstants),
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
	// File-scoped fact for PS6006: receiver fields that are INDEXED in exactly one
	// function of this file. A per-call temporary is used element-wise in the one method
	// that needs it and merely allocated elsewhere; persistent state is indexed by
	// several. This replaces a name heuristic that missed two of the three real instances
	// found in this repo (b.vals, b.part) because they are not spelled like buffers.
	curSoleIndexed = soleIndexedFields(f)
	// …and the subset of those declared as a SLICE. A per-call buffer is a slice; the
	// false positives the structural test produced were overwhelmingly MAPS used as
	// persistent registries — optimizer per-parameter state (a.st, g.st, s.st), intern
	// tables, memo caches. Indexing a map is not buffer reuse, and a map field is never
	// the thing standing between a loop and its parallel form.
	curSliceFields = sliceTypedFields(f)
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
		// A3: a per-element accessor whose index arguments are 2+ ENCLOSING-LOOP variables —
		// a hand-written multi-dimensional tensor walk (t.AtF64(i, j) inside for i { for j }).
		// PS1001 (A) only fires inside a Numel/Unravel loop, so it misses these explicit
		// [rows,cols] walks; each element still pays an interface dispatch + flat-offset
		// recompute. Grab the contiguous typed storage once (Storage().F64()/F32(), a fast
		// path per dtype) and index it, keeping the accessor as the exotic-dtype fallback.
		// (VQ-VAE codebook assignment: -94%.)
		if !perElem && ns.accessors[name] && !call.Ellipsis.IsValid() {
			lv := enclosingLoopVars(parent, call)
			loopArgs := 0
			for _, a := range call.Args {
				if id, ok := a.(*ast.Ident); ok && lv[id.Name] {
					loopArgs++
				}
			}
			if loopArgs >= 2 {
				out = append(out, finding{
					pos:      fset.Position(loop.Pos()),
					category: "manual-walk-dispatch",
					msg: fmt.Sprintf(".%s walks a tensor by explicit loop indices in %s() — an interface"+
						" dispatch + flat-offset recompute per element that PS1001's Numel-loop check"+
						" misses. Take the contiguous typed backing slice once (Storage().F64()/F32())"+
						" and index it directly, keeping the accessor as the exotic-dtype fallback.",
						name, fn.Name.Name),
				})
			}
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
	out = append(out, f32AbsViaF64Findings(fset, fn)...)
	out = append(out, opDispatchRecurrenceFindings(fset, fn)...)
	out = append(out, nestedSubrangeRescanFindings(fset, fn)...)
	out = append(out, butterflyFindings(fset, fn)...)
	out = append(out, transposedGramFindings(fset, fn)...)
	out = append(out, stridedInnerReductionFindings(fset, fn)...)
	out = append(out, serialDotMatmulFindings(fset, fn)...)
	out = append(out, scaledSerialDotFindings(fset, fn)...)
	out = append(out, quadraticCacheAppendFindings(fset, fn)...)
	out = append(out, buildNxNUseOneRowFindings(fset, fn)...)
	out = append(out, indirectKeyComparatorFindings(fset, fn)...)
	out = append(out, innerInvariantRecomputeFindings(fset, fn)...)
	out = append(out, stridedInnerWalkFindings(fset, fn)...)
	out = append(out, inconsistentFMAPinningFindings(fset, fn)...)
	out = append(out, sortFeedsCountedPrefixFindings(fset, fn)...)
	out = append(out, redundantPureRecomputeFindings(fset, fn, ns)...)
	out = append(out, batch1FeedsPostloopSliceFindings(fset, fn, ns)...)
	out = append(out, loopInvariantLiteralArgFindings(fset, fn)...)
	out = append(out, unpooledVariadicSiblingFindings(fset, fn)...)
	out = append(out, layoutOpClusterFindings(fset, fn, ns)...)
	out = append(out, jamTailDelegatesFindings(fset, fn)...)
	out = append(out, fanoutWithoutWorkerSeamFindings(fset, fn)...)
	out = append(out, invariantTranscendentalRecomputeFindings(fset, fn)...)
	out = append(out, partialFastPathFindings(fset, fn)...)
	out = append(out, outputInvariantReloadFindings(fset, fn)...)
	out = append(out, receiverScratchFindings(fset, fn)...)
	out = append(out, searchFeedsReductionFindings(fset, fn)...)
	out = append(out, allocInParallelBodyFindings(fset, fn)...)
	out = append(out, reflectSwapperSortFindings(fset, fn)...)
	out = append(out, vjpScalarBinopFindings(fset, fn)...)
	out = append(out, fullSortBoundedPrefixFindings(fset, fn)...)
	out = append(out, spatialBoundsBranchFindings(fset, fn)...)
	out = append(out, monotoneIndexBoundFindings(fset, fn)...)
	out = append(out, sincosFusableFindings(fset, fn)...)
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

// enclosingLoopVars returns the iteration-variable names of every Range/For loop enclosing
// n. A dispatch call whose index arguments are drawn from this set (t.AtF64(i, j) inside
// for i { for j { … } }) is a manual per-dimension tensor walk — the shape PS1005 flags.
func enclosingLoopVars(parent map[ast.Node]ast.Node, n ast.Node) map[string]bool {
	vars := map[string]bool{}
	for p := parent[n]; p != nil; p = parent[p] {
		switch l := p.(type) {
		case *ast.RangeStmt:
			if id, ok := l.Key.(*ast.Ident); ok && id.Name != "_" {
				vars[id.Name] = true
			}
			if id, ok := l.Value.(*ast.Ident); ok && id.Name != "_" {
				vars[id.Name] = true
			}
		case *ast.ForStmt:
			if as, ok := l.Init.(*ast.AssignStmt); ok {
				for _, lhs := range as.Lhs {
					if id, ok := lhs.(*ast.Ident); ok && id.Name != "_" {
						vars[id.Name] = true
					}
				}
			}
		}
	}
	return vars
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
		// PS6014 cannot judge purity from syntax; with pureComputeFuncs empty it reports
		// zero on any input, which is the same false assurance.
		"redundant-pure-recompute": ns.pureCompute,
		// PS6015 hoists a call across a loop, which is only legal if the call is pure;
		// same config, same reason, same zero-means-nothing warning.
		"batch1-call-feeds-only-postloop-slice": ns.pureCompute,
		// PS6018 keys on the project's own movement-op constants.
		"layout-op-cluster-unfused": ns.layoutOps,
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
	collectVariadicSiblings(fset, parsed)
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

// butterflyFindings flags PS4010 — an in-loop butterfly assignment p,q = x+y, x-y (the same
// two operands added AND subtracted, written to two indexed slots of one base). This is the
// core step of a Fast Walsh-Hadamard / FFT / Hadamard-rotation kernel: an overhead/L1-bound
// scalar add/sub loop that a SIMD Add/Sub over the contiguous stride-separated operand runs
// vectorizes (bit-identical — each output is exactly x±y of the same operands, no cross-lane
// reduction). Shipped: nlp FWHT butterfly (kernel 1.5x, RotationHadamard -25%).
func butterflyFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	seen := map[token.Pos]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		_, body, ok := loopVarBody(n)
		if !ok {
			return true
		}
		ast.Inspect(body, func(m ast.Node) bool {
			as, ok := m.(*ast.AssignStmt)
			if !ok || len(as.Lhs) != 2 || len(as.Rhs) != 2 || seen[as.Pos()] {
				return true
			}
			l0, ok0 := as.Lhs[0].(*ast.IndexExpr)
			l1, ok1 := as.Lhs[1].(*ast.IndexExpr)
			if !ok0 || !ok1 {
				return true
			}
			b0, b1 := baseIdentName(l0), baseIdentName(l1)
			if b0 == "" || b0 != b1 { // both indexed writes into the same base slice
				return true
			}
			add, aok := as.Rhs[0].(*ast.BinaryExpr)
			sub, sok := as.Rhs[1].(*ast.BinaryExpr)
			if !aok || !sok || add.Op != token.ADD || sub.Op != token.SUB {
				return true
			}
			if exprEqual(add.X, sub.X) && exprEqual(add.Y, sub.Y) { // x+y and x-y, same x,y
				seen[as.Pos()] = true
				out = append(out, finding{
					pos:      fset.Position(as.Pos()),
					category: "vectorizable-butterfly",
					msg: fmt.Sprintf("in-loop butterfly %s[..],%s[..] = x+y, x-y — a scalar"+
						" FWHT/FFT/Hadamard stage. When the two operand runs are contiguous and"+
						" stride-separated by >= the SIMD lane count, replace the inner block with"+
						" one Float64x4/Float32x8 Add (-> first slot) and one Sub (-> second slot)"+
						" of the same loaded x,y — bit-identical (no cross-lane reduction), keep the"+
						" stage order. Shipped: nlp FWHT (kernel 1.5x). Gate behind the simd build"+
						" tag with a scalar fallback + a Float64bits parity test.", b0, b0),
				})
			}
			return true
		})
		return true
	})
	return out
}

// transposedGramFindings flags PS4009 — a symmetric-gram reduction accumulated as
// M[k][i]·M[k][j] where the innermost (reduction) loop var k is the OUTER/row index of a
// row-major or jagged 2-D matrix. Walking k then strides DOWN a column across separately
// stored rows: L2-bound (one cache line touched per element) plus a serial dependent-FMA
// chain. Reblock to k-outer rank-1 accumulation — load M[k] once, walk it contiguously
// across the (i,j) triangle into a scratch, then blend once. Bit-identical (same ascending-k
// order into a +0 accumulator). Shipped: SOAP/Shampoo R-gram (-6.5%/-9.3%); same class as
// dino.go (-88%) and Muon matmulABt (2.09x). Contrast the cache-friendly gram M[i][k]·M[j][k]
// (k the inner/contiguous index) which is NOT flagged.
func transposedGramFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		iName, ibody, ok := loopVarBody(n)
		if !ok {
			return true
		}
		for _, s := range ibody.List {
			jName, jbody, ok := loopVarBody(s)
			if !ok || jName == iName {
				continue
			}
			for _, s2 := range jbody.List {
				kName, kbody, ok := loopVarBody(s2)
				if !ok || kName == iName || kName == jName {
					continue
				}
				if base, ok := transposedGramProduct(kbody, kName, iName, jName); ok {
					out = append(out, finding{
						pos:      fset.Position(n.Pos()),
						category: "transposed-gram-colstride",
						msg: fmt.Sprintf("nested %s/%s/%s loop reduces a gram %s[%s][%s]·%s[%s][%s]"+
							" with the reduction index %s as the OUTER (row) index — the innermost"+
							" loop strides DOWN a column across %s's rows (L2-bound + serial FMA"+
							" chain). Reblock to %s-outer rank-1: for %s { load %s[%s] once, walk it"+
							" contiguously across the (%s,%s) triangle into a scratch }, then blend"+
							" once — bit-identical (same ascending-%s order). Shipped: SOAP/Shampoo,"+
							" dino.go, Muon. Benchmark the step.",
							iName, jName, kName, base, kName, iName, base, kName, jName,
							kName, base, kName, kName, base, kName, iName, jName, kName),
					})
					return false
				}
			}
		}
		return true
	})
	return out
}

// transposedGramProduct searches root for a multiply M[k][i]·M[k][j] of the SAME base M
// where k is the row (first) index and the two column (second) indices are exactly the
// enclosing loop vars i and j — the column-stride signature. Returns the base M.
func transposedGramProduct(root ast.Node, kName, iName, jName string) (string, bool) {
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
		lb, lc, lok := rowStridedBase(be.X, kName)
		rb, rc, rok := rowStridedBase(be.Y, kName)
		if lok && rok && lb == rb &&
			((lc == iName && rc == jName) || (lc == jName && rc == iName)) {
			base, found = lb, true
		}
		return !found
	})
	return base, found
}

// stridedInnerReductionFindings flags PS1006 — a nested loop (outer o, inner j) whose inner
// loop reduces over a flat 1-D access ARR[j*stride + o]: the INNER var j is the high-stride
// (multiplied) part while the OUTER var o is the contiguous (additive) part. Walking j then
// strides ARR by `stride` every inner step (cache-thrashing). Interchange the loops (j outer,
// o inner) so ARR[j*stride+o] is walked contiguously in o. Per output element the reduction
// stays in the same j-ascending order, so it is bit-identical. Requires a += reduction in the
// inner body so pure strided reads (which may be intentional gathers) are not flagged. Shipped:
// MLA value-mix (backend/{cpu,ref}/mla.go, cpu 1.13x / ref 1.27x).
func stridedInnerReductionFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		oName, obody, ok := loopVarBody(n)
		if !ok {
			return true
		}
		for _, s := range obody.List {
			jName, jbody, ok := loopVarBody(s)
			if !ok || jName == oName {
				continue
			}
			if base, stride, ok := stridedInnerAccess(jbody, jName, oName); ok {
				out = append(out, finding{
					pos:      fset.Position(n.Pos()),
					category: "strided-inner-reduction",
					msg: fmt.Sprintf("nested %s/%s loop reduces over %s[%s*%s + %s] — the INNER var %s is the"+
						" high-stride (×%s) part while the OUTER var %s is contiguous, so the inner loop strides"+
						" %s by %s every step (cache-thrashing). Interchange to %s-outer/%s-inner so %s is walked"+
						" contiguously in %s — bit-identical (same ascending-%s reduction order per element)."+
						" Shipped: MLA value-mix. Benchmark the kernel.",
						oName, jName, base, jName, stride, oName, jName, stride, oName, base, stride,
						jName, oName, base, oName, jName),
				})
				return false
			}
		}
		return true
	})
	return out
}

// stridedInnerAccess searches root (an inner loop body over jName, nested in an outer loop over
// oName) for a flat access ARR[jName*stride + oName] where jName is the multiplied (strided) part
// and oName the additive (contiguous) part, gated on a += reduction being present. Returns the
// array base and the textual stride operand.
func stridedInnerAccess(root ast.Node, jName, oName string) (string, string, bool) {
	hasReduce := false
	ast.Inspect(root, func(n ast.Node) bool {
		if as, ok := n.(*ast.AssignStmt); ok && as.Tok == token.ADD_ASSIGN {
			hasReduce = true
		}
		return true
	})
	if !hasReduce {
		return "", "", false
	}
	var base, stride string
	var found bool
	ast.Inspect(root, func(n ast.Node) bool {
		if found {
			return false
		}
		ix, ok := n.(*ast.IndexExpr)
		if !ok {
			return true
		}
		baseID, ok := ix.X.(*ast.Ident) // flat 1-D base
		if !ok {
			return true
		}
		add, ok := ix.Index.(*ast.BinaryExpr)
		if !ok || add.Op != token.ADD {
			return true
		}
		if st, ok := strideOperand(add.X, jName); ok && isPlainIdent(add.Y, oName) {
			base, stride, found = baseID.Name, st, true
		} else if st, ok := strideOperand(add.Y, jName); ok && isPlainIdent(add.X, oName) {
			base, stride, found = baseID.Name, st, true
		}
		return !found
	})
	return base, stride, found
}

// strideOperand reports whether e is a multiply `v * K` (or `K * v`) and returns the textual
// stride K. v must appear as one whole factor (the loop var), K as the other.
func strideOperand(e ast.Expr, v string) (string, bool) {
	be, ok := e.(*ast.BinaryExpr)
	if !ok || be.Op != token.MUL {
		return "", false
	}
	if isPlainIdent(be.X, v) {
		return exprText(be.Y), true
	}
	if isPlainIdent(be.Y, v) {
		return exprText(be.X), true
	}
	return "", false
}

func isPlainIdent(e ast.Expr, name string) bool {
	id, ok := e.(*ast.Ident)
	return ok && id.Name == name
}

// rowStridedBase matches an access base[k][col] where k is the reduction (row) index and
// col is a plain identifier. Returns (base, col). Rejects base[col][k] (the contiguous form).
func rowStridedBase(e ast.Expr, kName string) (string, string, bool) {
	ix, ok := e.(*ast.IndexExpr)
	if !ok {
		return "", "", false
	}
	col, ok := ix.Index.(*ast.Ident) // second index = column
	if !ok {
		return "", "", false
	}
	inner, ok := ix.X.(*ast.IndexExpr) // base[k]
	if !ok {
		return "", "", false
	}
	row, ok := inner.Index.(*ast.Ident)
	if !ok || row.Name != kName { // first index must be the reduction var
		return "", "", false
	}
	baseID, ok := inner.X.(*ast.Ident)
	if !ok {
		return "", "", false
	}
	return baseID.Name, col.Name, true
}

// nestedSubrangeRescanFindings flags PS5006 — a triple-nested loop where the innermost
// C-style for-loop iterates k over a [j..i] sub-range (its bounds reference BOTH enclosing
// loop vars) and accumulates a running reduction acc *= / += arr[k]. Because the sub-range
// shrinks/grows by one element between adjacent (i,j), the whole product/sum is recomputed
// from scratch T²/2 times — an O(T³) serial-dependent chain. Precompute a prefix/suffix scan
// once per outer index i (O(i)), then read the reduction in O(1). Shipped: Mamba-2
// SSDQuadratic decay (1.92x). Tolerance-gated when the scan reassociates the fold.
func nestedSubrangeRescanFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		iName, ibody, ok := loopVarBody(n)
		if !ok {
			return true
		}
		for _, s := range ibody.List {
			jName, jbody, ok := loopVarBody(s)
			if !ok || jName == iName {
				continue
			}
			for _, s2 := range jbody.List {
				fl, ok := s2.(*ast.ForStmt)
				if !ok || fl.Init == nil || fl.Cond == nil {
					continue
				}
				kName, kbody, ok := loopVarBody(fl)
				if !ok || kName == iName || kName == jName {
					continue
				}
				// innermost bounds span [j..i]: one end references j, the other i.
				spans := (stmtMentions(fl.Init, jName) && exprMentions(fl.Cond, iName)) ||
					(stmtMentions(fl.Init, iName) && exprMentions(fl.Cond, jName))
				if !spans {
					continue
				}
				if op, arr, ok := subrangeReduce(kbody, kName); ok {
					out = append(out, finding{
						pos:      fset.Position(fl.Pos()),
						category: "nested-subrange-rescan",
						msg: fmt.Sprintf("innermost %s-loop recomputes a running reduction (%s %s[%s])"+
							" over the [%s..%s] sub-range for every (%s,%s) — an O(T\u00b3) serial"+
							" recompute. Precompute a prefix/suffix scan of %s once per %s (O(%s)),"+
							" then read it in O(1) in the inner loop. Shipped: Mamba-2 SSDQuadratic"+
							" decay (1.92x). Tolerance-gated if the scan reassociates the fold;"+
							" benchmark + gate on the existing tolerance test.",
							kName, op, arr, kName, jName, iName, iName, jName, arr, iName, iName),
					})
					return false
				}
			}
		}
		return true
	})
	return out
}

// subrangeReduce reports whether body accumulates acc *= arr[k] or acc += arr[k] (a running
// product/sum over the k-indexed array) and returns the operator text and array base.
func subrangeReduce(body *ast.BlockStmt, kName string) (string, string, bool) {
	for _, st := range body.List {
		as, ok := st.(*ast.AssignStmt)
		if !ok || (as.Tok != token.MUL_ASSIGN && as.Tok != token.ADD_ASSIGN) || len(as.Rhs) != 1 {
			continue
		}
		ix, ok := as.Rhs[0].(*ast.IndexExpr)
		if !ok || !exprMentions(ix.Index, kName) {
			continue
		}
		if base, ok := ix.X.(*ast.Ident); ok {
			op := "*="
			if as.Tok == token.ADD_ASSIGN {
				op = "+="
			}
			return op, base.Name, true
		}
	}
	return "", "", false
}

// opDispatchRecurrenceFindings flags PS4011 — a sequential recurrence loop that dispatches
// 2+ backend ops per iteration (each call passing a backend.Op* constant) while the enclosing
// function has NO fused typed fast path (no flatF64 guard). Each dispatch is a map lookup + a
// fresh tensor.New on a microscopic per-step tensor, so the loop is O(seq) dispatch+alloc
// overhead, not arithmetic. The fix is a raw-[]float64 fused path (grab storage once, run the
// recurrence in plain Go, reuse the state slice) as in kda.go / retention.go / ssd.go — usually
// bit-exact (the backend ops are plain ascending-order loops), gated on ctx.Recorder==nil so
// autograd training keeps the dispatch path. Shipped: DeltaNet/GatedDeltaNet (2.0-3.6x).
func opDispatchRecurrenceFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || funcCallsIdent(fn.Body, "flatF64") {
		return nil // already has (or is) a fused typed fast path
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		_, body, ok := loopVarBody(n)
		if !ok {
			return true
		}
		count := 0
		ast.Inspect(body, func(m ast.Node) bool {
			if call, ok := m.(*ast.CallExpr); ok && callHasBackendOpArg(call) {
				count++
			}
			return true
		})
		if count >= 2 {
			out = append(out, finding{
				pos:      fset.Position(n.Pos()),
				category: "op-dispatch-recurrence",
				msg: fmt.Sprintf("sequential loop dispatches %d backend ops per iteration (calls"+
					" passing a backend.Op* constant) and the function has no flatF64 fused path —"+
					" O(seq) dispatch + tiny-tensor alloc overhead. Add a raw-[]float64 fused path"+
					" (storage grabbed once, state slice reused, plain-Go recurrence) gated on"+
					" ctx.Recorder==nil so training keeps the dispatch path; usually bit-exact."+
					" Shipped: DeltaNet/GatedDeltaNet 2.0-3.6x. Benchmark it.", count),
			})
			return false
		}
		return true
	})
	return out
}

// callHasBackendOpArg reports whether any argument of call is a backend.Op* constant
// (a selector X.OpName), the marker of a backend op dispatch.
func callHasBackendOpArg(call *ast.CallExpr) bool {
	for _, a := range call.Args {
		if sel, ok := a.(*ast.SelectorExpr); ok {
			if _, ok := sel.X.(*ast.Ident); ok && strings.HasPrefix(sel.Sel.Name, "Op") && len(sel.Sel.Name) > 2 {
				return true
			}
		}
	}
	return false
}

// funcCallsIdent reports whether the subtree contains a call to the named function.
func funcCallsIdent(root ast.Node, name string) bool {
	found := false
	ast.Inspect(root, func(n ast.Node) bool {
		if found {
			return false
		}
		if call, ok := n.(*ast.CallExpr); ok {
			if id, ok := call.Fun.(*ast.Ident); ok && id.Name == name {
				found = true
				return false
			}
		}
		return true
	})
	return found
}

// f32AbsViaF64Findings flags PS5007 — float32(math.Abs(float64(x))): an f32 abs computed by
// converting to f64, calling math.Abs, and converting back. The conversions are exact, so a
// direct f32 sign-bit clear math.Float32frombits(math.Float32bits(x) &^ (1<<31)) is BIT-IDENTICAL
// and skips two conversions + a call. Hot in quantizer absmax scans. Shipped: quantizeQ8_0/Q6_K.
func f32AbsViaF64Findings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		outer, ok := n.(*ast.CallExpr)
		if !ok || !isConvTo(outer, "float32") || len(outer.Args) != 1 {
			return true
		}
		abs, ok := outer.Args[0].(*ast.CallExpr)
		if !ok || !isSelCall(abs, "math", "Abs") || len(abs.Args) != 1 {
			return true
		}
		if inner, ok := abs.Args[0].(*ast.CallExpr); ok && isConvTo(inner, "float64") {
			out = append(out, finding{
				pos:      fset.Position(outer.Pos()),
				category: "f32-abs-via-f64",
				msg: "float32(math.Abs(float64(x))) round-trips an f32 through f64 for abs —" +
					" replace with a direct sign-bit clear math.Float32frombits(math.Float32bits(x)" +
					" &^ (1<<31)): bit-identical |x|, no conversions or call. Shipped: quantizeQ8_0/Q6_K.",
			})
			return false
		}
		return true
	})
	return out
}

// isConvTo reports whether call is a conversion T(x) for the builtin numeric type name.
func isConvTo(call *ast.CallExpr, name string) bool {
	id, ok := call.Fun.(*ast.Ident)
	return ok && id.Name == name
}

// isSelCall reports whether call is pkg.Fn(...).
func isSelCall(call *ast.CallExpr, pkg, fn string) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != fn {
		return false
	}
	id, ok := sel.X.(*ast.Ident)
	return ok && id.Name == pkg
}

// scaledSerialDotFindings flags PS4012 — a serial scalar-dot accumulator whose result is
// SCALED/dequantized (used in a * or / expression) before being stored, the quantized-GEMM
// inner loop that PS4008 misses because the accumulator is not written out raw. Same fix as
// PS4008 (independent accumulators break the latency chain); bit-identical when the products
// are integer-valued (int8·int8 partials stay < 2^53 so any grouping sums the same exact
// integer), else tolerance-gated. Shipped: nn.LLMInt8MatMul (~2x).
func scaledSerialDotFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		_, obody, ok := loopVarBody(n)
		if !ok {
			return true
		}
		for _, mid := range obody.List {
			_, mbody, ok := loopVarBody(mid)
			if !ok {
				continue
			}
			acc, inner := scaledDotAcc(mbody)
			// require the accumulator to be consumed by a scale (* or /) — the dequant
			// step — which distinguishes this from PS4008's raw `out[i] = acc` store.
			if acc == "" || !identInScaleExpr(mbody, acc) {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(inner.Pos()),
				category: "scaled-serial-dot",
				msg: fmt.Sprintf("%q accumulates a serial scalar dot that is then SCALED/dequantized"+
					" (acc*scale…) before being stored — a quantized-GEMM inner loop, latency-bound"+
					" like PS4008 but not stored raw. Break the dependency chain with independent"+
					" accumulators (e.g. 4-way unroll). Bit-identical when the products are"+
					" integer-valued (int8·int8 partials < 2^53 reassociate exactly); else"+
					" tolerance-gate against the sequential form. Shipped: nn.LLMInt8MatMul (~2x).", acc),
			})
		}
		return true
	})
	return out
}

// scaledDotAcc finds a float accumulator declared in mbody (`var acc float64` or `acc := 0.0`)
// whose value is built by an innermost loop that accumulates a product into it. Returns the
// accumulator name and the inner-loop node, or ("", nil) if the shape is absent.
func scaledDotAcc(mbody *ast.BlockStmt) (string, ast.Node) {
	var acc string
	var inner ast.Node
	for _, st := range mbody.List {
		switch v := st.(type) {
		case *ast.DeclStmt:
			if gd, ok := v.Decl.(*ast.GenDecl); ok && gd.Tok == token.VAR {
				for _, sp := range gd.Specs {
					if vs, ok := sp.(*ast.ValueSpec); ok && len(vs.Names) == 1 && len(vs.Values) == 0 && isFloatIdent(vs.Type) {
						acc = vs.Names[0].Name
					}
				}
			}
		case *ast.AssignStmt:
			if v.Tok == token.DEFINE && len(v.Lhs) == 1 && len(v.Rhs) == 1 {
				if id, ok := v.Lhs[0].(*ast.Ident); ok && isZeroLit(v.Rhs[0]) {
					acc = id.Name
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
	if acc == "" || inner == nil {
		return "", nil
	}
	return acc, inner
}

// identInScaleExpr reports whether name appears inside a multiply or divide expression in root
// — i.e. the accumulator is scaled (dequantized) rather than stored verbatim.
func identInScaleExpr(root ast.Node, name string) bool {
	found := false
	ast.Inspect(root, func(n ast.Node) bool {
		if found {
			return false
		}
		if be, ok := n.(*ast.BinaryExpr); ok && (be.Op == token.MUL || be.Op == token.QUO) {
			if exprMentions(be.X, name) || exprMentions(be.Y, name) {
				found = true
				return false
			}
		}
		return true
	})
	return found
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

// sincosNames is the PS5008 trigger set: the two libm calls that share an argument reduction.
var sincosNames = map[string]bool{"Sin": true, "Cos": true}

// sincosFusableFindings flags PS5008 — a function that calls BOTH math.Sin(E) and
// math.Cos(E) on the SAME argument expression E. Each call independently performs the
// full argument reduction of E and then evaluates its own polynomial; math.Sincos(E)
// reduces once and evaluates both. Go's math.Sincos shares Sin/Cos's exact reduction
// and _sin/_cos polynomials, so `sinE, cosE := math.Sincos(E)` is bit-identical to the
// separate calls (the codebase already treats it so — backend/cpu/attn_extra.go RoPE).
//
// Advisory (no auto-fix): the enclosing fill/scan loop still needs an A/B bench to
// confirm the trig is a real fraction of it, and the rewrite must bind the (sin, cos)
// return order at a single site the tool cannot always place safely. Verified win:
// nn/sinusoidal.go PE builder (~15-22% on the trig-fill loop). Argument matching is
// structural via exprEqual, so it is conservative — args differing only by a value
// conversion (float64(i)…) are not matched, avoiding false positives.
func sincosFusableFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var sins, coss []*ast.CallExpr
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || len(call.Args) != 1 {
			return true
		}
		switch name, ok := pkgFuncCall(call.Fun, "math", sincosNames); {
		case ok && name == "Sin":
			sins = append(sins, call)
		case ok && name == "Cos":
			coss = append(coss, call)
		}
		return true
	})
	if len(sins) == 0 || len(coss) == 0 {
		return nil
	}
	var out []finding
	seen := map[token.Pos]bool{}
	for _, s := range sins {
		for _, c := range coss {
			if !exprEqual(s.Args[0], c.Args[0]) {
				continue
			}
			rep := s // report at the earlier of the two call sites
			if c.Pos() < s.Pos() {
				rep = c
			}
			if seen[rep.Pos()] {
				continue
			}
			seen[rep.Pos()] = true
			out = append(out, finding{
				pos:      fset.Position(rep.Pos()),
				end:      fset.Position(rep.End()),
				category: "sincos-fusable",
				msg:      "math.Sin and math.Cos on the same argument — fuse to `sin, cos := math.Sincos(x)` (one argument reduction, bit-identical)",
			})
			break
		}
	}
	return out
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
	// slicePrefix[name] — the sorted slice is consumed via a fixed prefix `name[:k]`
	// (High bound present). Like an early break, this is a bounded-prefix consumer: only
	// the top-k SET is used, so a quickselect replaces the full sort (SnapKV keep-mask).
	slicePrefix := map[string]bool{}
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
		case *ast.RangeStmt:
			// range over a fixed prefix `for _, i := range cand[:k]` — an order-INDEPENDENT
			// consumption of the top-k as a set (contrast `return idx[:keep]`, which hands
			// the caller the sorted ORDER and so genuinely needs the sort).
			if se, ok := x.X.(*ast.SliceExpr); ok && se.High != nil && se.Low == nil {
				if id, ok := se.X.(*ast.Ident); ok {
					slicePrefix[id.Name] = true
				}
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
	if sortCall == nil || guarded || !(hasBreak || slicePrefix[sortSlice]) || !fullFill[sortSlice] {
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

// usedAsIndex reports whether name appears inside an array/slice INDEX expression within e —
// the discriminator that the guarded index is a real address offset (xs[j*D+c]) and not a
// data-dependent value merely fed to a threshold compare (a quantized sample qlo>=16).
func usedAsIndex(e ast.Node, name string) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if ix, ok := n.(*ast.IndexExpr); ok && exprMentions(ix.Index, name) {
			found = true
			return false
		}
		return !found
	})
	return found
}

// isTapWork reports whether s is the single "tap work" statement a bounds guard wraps:
// an accumulate (acc += … / acc -= …) or an indexed scatter/gather move (a[i] = … / a[i] += …).
func isTapWork(s ast.Stmt) bool {
	as, ok := s.(*ast.AssignStmt)
	if !ok || len(as.Lhs) != 1 {
		return false
	}
	switch as.Lhs[0].(type) {
	case *ast.Ident:
		return as.Tok == token.ADD_ASSIGN || as.Tok == token.SUB_ASSIGN
	case *ast.IndexExpr:
		return true
	}
	return false
}

// monotoneIndexBoundFindings flags PS6003 — an INNERMOST loop that derives an index affine in
// the loop variable (j := t-(K-1)+k) and then guards its single tap-work statement with ONE
// relational comparison on that index (if j >= 0). Because the index is monotone in the loop
// variable, the in-bounds iterations form one contiguous run: hoist a clamped loop bound
// (kStart := max(0,(K-1)-t)) and drop the per-iteration branch — bit-identical, the skipped
// iterations are exactly the guard's empty false branch. This is the 1-D causal / left-pad-conv
// sibling of PS6002 (which handles the compound ≥3-comparison spatial guard around an indexed
// move). Shipped: conv1d. Conservative: needs the affine index, a SINGLE relational comparison
// on it, and a one-statement accumulate/indexed-move body — verify the index is monotone (±1
// coeff in the loop var) before acting.
func monotoneIndexBoundFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		kvar, body, ok := loopVarBody(n)
		if !ok || body == nil || loopHasNested(body) {
			return true
		}
		// indices assigned in the loop body from an expression mentioning the loop var.
		affine := map[string]bool{}
		for _, st := range body.List {
			if as, ok := st.(*ast.AssignStmt); ok && len(as.Lhs) == 1 && len(as.Rhs) == 1 {
				if id, ok := as.Lhs[0].(*ast.Ident); ok && exprMentions(as.Rhs[0], kvar) {
					affine[id.Name] = true
				}
			}
		}
		if len(affine) == 0 {
			return true
		}
		for _, st := range body.List {
			ifs, ok := st.(*ast.IfStmt)
			if !ok || ifs.Init != nil || ifs.Else != nil {
				continue
			}
			be, ok := ifs.Cond.(*ast.BinaryExpr)
			if !ok || andedComparisons(be) != 1 {
				continue
			}
			// the single comparison must bound an affine index, the guarded body must be
			// exactly one tap-work statement, AND that index must be used as an array index
			// in it (so a data-dependent value fed to a threshold — a quantized sample — is
			// not mistaken for a monotone address offset).
			if len(ifs.Body.List) != 1 || !isTapWork(ifs.Body.List[0]) {
				continue
			}
			idxName := ""
			for name := range affine {
				if exprMentions(be, name) {
					idxName = name
					break
				}
			}
			if idxName == "" || !usedAsIndex(ifs.Body.List[0], idxName) {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(ifs.Pos()),
				category: "monotone-index-bound",
				msg: "innermost loop guards its tap work with a single relational bound on an index" +
					" affine in the loop variable (the j:=t-(K-1)+k; if j>=0 causal-conv shape). If the" +
					" index is monotone in the loop var the in-bounds iterations are one contiguous run" +
					" — hoist a clamped loop start/end (kStart := max(0,(K-1)-t)) and drop the per-tap" +
					" branch (branch-free, bit-identical: the skipped taps hit the guard's empty else)." +
					" Sibling of PS6002. Verify the ±1 monotonicity before acting.",
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

// outputInvariantReloadFindings flags PS6010 — an output loop whose accumulator reads an
// operand that does NOT vary with the output index. That operand is re-loaded and
// re-converted once per output, when unrolling the output loop by 4 would amortize one
// load across 4 accumulators. This is register blocking, the m==1 dual of unroll-and-jam.
//
// Found because main had already applied it to exactly one of seven sibling kernels.
// gguf.QMatMul's Q8_0 single-token path is blocked by 4 and measured 526us -> 233us per
// decode step (2.26x) — a larger factor than either fusing the dequant into the dot or
// parallelizing the row loop delivered. Q4_0, structurally the closest sibling, was still
// one row at a time; blocking it measured 1.55x. NOTHING IN PERFSCAN FLAGGED EITHER. The
// existing serial-dot rule (PS4008) requires a plain `acc += A[i]*B[i]`, and these loops
// unpack nibbles and convert between float widths on the way, so it stays silent.
//
// The remedy is NOT the ikj rewrite PS4008 asks for. Here the dependency chain is already
// broken by having one accumulator per output; what is wasted is the repeated load of the
// SHARED operand. Different defect, different fix, so it is a separate check.
func outputInvariantReloadFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		outVar, outBody, ok := loopVarBody(n)
		if !ok || outVar == "" {
			return true
		}
		// Already blocked: a stride > 1 means someone has done this. `range n` and `i++`
		// are the unblocked forms.
		if f, isFor := n.(*ast.ForStmt); isFor && stridesByMoreThanOne(f.Post) {
			return true
		}
		// Everything reachable from the output index, transitively: `rowBits :=
		// weight[ni*rowBytes:]` then `q := rowBits[o:]` makes q output-dependent even
		// though it never mentions ni. Without the closure the shared operand cannot be
		// told apart from the per-output one.
		derived := derivedFrom(outVar, outBody)
		for _, acc := range scalarAccumulators(outBody) {
			shared, perOutput := false, false
			ast.Inspect(outBody, func(m ast.Node) bool {
				as, ok := m.(*ast.AssignStmt)
				if !ok || as.Tok != token.ADD_ASSIGN || len(as.Lhs) != 1 {
					return true
				}
				if identName(as.Lhs[0]) != acc {
					return true
				}
				for _, ix := range indexExprs(as.Rhs[0]) {
					base := identName(ix.X)
					if base == "" {
						continue
					}
					if derived[base] || mentions(ix.Index, outVar) {
						perOutput = true
					} else {
						shared = true
					}
				}
				// A bare identifier counts too: an unpacked byte carried in as a range
				// value is per-output without ever appearing as an index.
				if !perOutput && mentionsAnyOf(as.Rhs[0], derived) {
					perOutput = true
				}
				return true
			})
			// BOTH are required. Without a shared operand there is nothing to amortize;
			// without a per-output one the whole accumulation is loop-invariant, which is
			// PS5003's finding and a different fix (hoist it out entirely, do not unroll).
			// And the accumulator must be STORED to an index that varies with the output
			// variable. That is what makes this an output loop producing one value per
			// iteration — the thing unrolling replicates. Without it the check fires on
			// every scalar reduction that happens to sit inside some loop, which is what
			// took it to 145 findings tree-wide.
			if shared && perOutput && storedToIndexOf(outBody, acc, outVar) {
				out = append(out, finding{
					pos:      fset.Position(n.Pos()),
					end:      fset.Position(n.End()),
					category: "output-invariant-operand-reload",
					msg:      fmt.Sprintf("accumulator %q re-reads an operand that does not vary with output index %q — unroll this loop by 4 so one load feeds 4 accumulators (register blocking)", acc, outVar),
				})
				break // one finding per output loop, not one per accumulator
			}
		}
		return true
	})
	return out
}

// stridesByMoreThanOne reports whether a for-post is `i += c` with c > 1 — the signature
// of a loop that is already register-blocked.
func stridesByMoreThanOne(post ast.Stmt) bool {
	as, ok := post.(*ast.AssignStmt)
	if !ok || as.Tok != token.ADD_ASSIGN || len(as.Rhs) != 1 {
		return false
	}
	lit, ok := as.Rhs[0].(*ast.BasicLit)
	return ok && lit.Kind == token.INT && lit.Value != "1"
}

// derivedFrom returns every identifier reachable from name by assignment inside body,
// transitively. Iterated to a fixed point because the chain can be several links long.
func derivedFrom(name string, body *ast.BlockStmt) map[string]bool {
	set := map[string]bool{name: true}
	for changed := true; changed; {
		changed = false
		ast.Inspect(body, func(n ast.Node) bool {
			switch v := n.(type) {
			case *ast.AssignStmt:
				for i, lhs := range v.Lhs {
					if i >= len(v.Rhs) {
						break
					}
					if l := identName(lhs); l != "" && !set[l] && mentionsAnyOf(v.Rhs[i], set) {
						set[l] = true
						changed = true
					}
				}
			case *ast.RangeStmt:
				// `for i, q := range qs` makes q output-dependent when qs is. Omitting
				// this is what made the check miss the very loop it was written from:
				// the per-output operand arrived as a range VALUE, never as an index.
				if !mentionsAnyOf(v.X, set) {
					return true
				}
				for _, e := range []ast.Expr{v.Key, v.Value} {
					if l := identName(e); l != "" && l != "_" && !set[l] {
						set[l] = true
						changed = true
					}
				}
			}
			return true
		})
	}
	return set
}

// scalarAccumulators returns names declared in body as a float scalar — `var acc float64`
// or `acc := 0.0` — the shape an output accumulator takes.
func scalarAccumulators(body *ast.BlockStmt) []string {
	var out []string
	for _, st := range body.List {
		switch v := st.(type) {
		case *ast.DeclStmt:
			gd, ok := v.Decl.(*ast.GenDecl)
			if !ok || gd.Tok != token.VAR {
				continue
			}
			for _, sp := range gd.Specs {
				vs, ok := sp.(*ast.ValueSpec)
				if !ok {
					continue
				}
				if t := identName(vs.Type); t == "float64" || t == "float32" {
					for _, nm := range vs.Names {
						out = append(out, nm.Name)
					}
				}
			}
		case *ast.AssignStmt:
			if v.Tok != token.DEFINE || len(v.Lhs) != 1 || len(v.Rhs) != 1 {
				continue
			}
			if lit, ok := v.Rhs[0].(*ast.BasicLit); ok && lit.Kind == token.FLOAT {
				if nm := identName(v.Lhs[0]); nm != "" {
					out = append(out, nm)
				}
			}
		}
	}
	return out
}

// indexExprs collects every IndexExpr in e.
func indexExprs(e ast.Expr) []*ast.IndexExpr {
	var out []*ast.IndexExpr
	ast.Inspect(e, func(n ast.Node) bool {
		if ix, ok := n.(*ast.IndexExpr); ok {
			out = append(out, ix)
		}
		return true
	})
	return out
}

// mentions reports whether e references name.
func mentions(e ast.Expr, name string) bool {
	return mentionsAnyOf(e, map[string]bool{name: true})
}

// storedToIndexOf reports whether acc is assigned into an index expression that varies
// with outVar — `outf[ni] = float32(acc)`, the signature of a per-output accumulator.
func storedToIndexOf(body *ast.BlockStmt, acc, outVar string) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || found {
			return !found
		}
		for i, lhs := range as.Lhs {
			ix, ok := ast.Unparen(lhs).(*ast.IndexExpr)
			if !ok || !mentions(ix.Index, outVar) {
				continue
			}
			if i < len(as.Rhs) && mentions(as.Rhs[i], acc) {
				found = true
			}
		}
		return !found
	})
	return found
}

// receiverScratchFindings flags PS6006 — a method that uses a SLICE FIELD ON ITS RECEIVER
// as a per-call temporary: written element-wise and read back element-wise within the
// same call, carrying nothing between calls.
//
// It is two defects wearing one shape. The method cannot be called concurrently, which
// silently blocks parallelizing any loop over it; and when someone parallelizes anyway,
// every worker writes the same cache line.
//
// Both were measured on classic.GaussianMixture.logGaussian, whose triangular-solve
// buffer was such a field. It carried a comment saying the method "runs serially" — the
// precondition was known, written down, and still violated the moment the E-step was
// parallelized. -race caught the correctness half. The performance half is the striking
// one: the racy version measured 1.16x, and moving the buffer to a parameter took the
// same parallelization to 1.93x. The contention cost more than the allocation saved.
//
// The fix is always the same: make it a parameter. Then the requirement lives in the
// signature instead of a comment, and the next caller cannot miss it.
func receiverScratchFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || fn.Recv == nil || len(fn.Recv.List) == 0 {
		return nil
	}
	recv := ""
	if len(fn.Recv.List[0].Names) > 0 {
		recv = fn.Recv.List[0].Names[0].Name
	}
	if recv == "" || recv == "_" {
		return nil
	}
	// ALIASES FIRST, and they are the whole reason a naive version of this check found
	// nothing. The shape in the wild is `y := m.yScratch` followed by `y[i] = …`, never
	// `m.yScratch[i] = …`. A detector that insists on the literal selector cannot see the
	// case it was written from — this one could not, until aliases were tracked.
	alias := map[string]string{} // local ident -> receiver field it aliases
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != len(as.Rhs) {
			return true
		}
		for i, rhs := range as.Rhs {
			sel, ok := ast.Unparen(rhs).(*ast.SelectorExpr)
			if !ok {
				continue
			}
			if id, ok := sel.X.(*ast.Ident); !ok || id.Name != recv {
				continue
			}
			if l := identName(as.Lhs[i]); l != "" && l != "_" {
				alias[l] = sel.Sel.Name
			}
		}
		return true
	})
	// Fields written through an INDEX (m.buf[i] = … or alias[i] = …) and fields read
	// through one. A temporary is both; persistent state is written here, read elsewhere.
	written, read := map[string]token.Pos{}, map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			// A slice of the field passed to a call is a read: copy(cf[w:end], b.part[:r])
			// is how the gbm partition consumed its scratch, and requiring an INDEXED read
			// missed it entirely.
			for _, f := range slicedFields(call.Args, recv, alias) {
				read[f] = true
			}
			for _, a := range call.Args {
				for _, ix := range indexExprs(a) {
					if f, okf := indexedField(ix, recv, alias); okf {
						read[f] = true
					}
				}
			}
			return true
		}
		if as, ok := n.(*ast.AssignStmt); ok {
			for _, lhs := range as.Lhs {
				if f, okf := indexedField(lhs, recv, alias); okf {
					if _, seen := written[f]; !seen {
						written[f] = as.Pos()
					}
				}
			}
			for _, rhs := range as.Rhs {
				for _, ix := range indexExprs(rhs) {
					if f, okf := indexedField(ix, recv, alias); okf {
						read[f] = true
					}
				}
			}
			for _, f := range slicedFields(as.Rhs, recv, alias) {
				read[f] = true
			}
			return true
		}
		return true
	})
	var out []finding
	for f, pos := range written {
		if !read[f] {
			continue // written but never read back here: an output, not a temporary
		}
		// Written-and-read-in-one-method is NOT enough on its own: the first cut of this
		// check flagged m.Means, persistent model state that the M-step legitimately
		// fills and reads back.
		//
		// THE DISCRIMINATOR IS STRUCTURAL, NOT THE NAME. An earlier version keyed on
		// scratch/buf/tmp-style names and missed two of the three real instances in this
		// repo — gbmBuilder.vals and gbmBuilder.part — because they are not spelled like
		// buffers. What actually separates a temporary from state is that a temporary is
		// INDEXED IN EXACTLY ONE FUNCTION: the method that needs it uses it element-wise
		// and everything else at most allocates it. State (m.Means, b.cols) is indexed by
		// several. The name is kept only as a fallback for a field whose sole indexed use
		// happens to be split across a file boundary this per-file scanner cannot see.
		// An EXPORTED field is part of the type's API, so callers outside this file can
		// read it and it cannot be a private temporary — whatever it is indexed by here.
		// Without this the structural test flags m.Means in a file where the M-step is
		// the only method that indexes it.
		if ast.IsExported(f) {
			continue
		}
		if !curSliceFields[f] {
			continue
		}
		if !curSoleIndexed[f] && !looksLikeScratchName(f) {
			continue
		}
		out = append(out, finding{
			pos:      fset.Position(pos),
			category: "receiver-scratch-buffer",
			msg:      fmt.Sprintf("%q is a receiver slice field used as a per-call temporary — pass it as a parameter instead: as a field it makes this method unsafe to call concurrently, and concurrent callers contend on one cache line", recv+"."+f),
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].pos.Offset < out[j].pos.Offset })
	return out
}

// recvIndexField matches `recv.field[...]` and returns "field".
func recvIndexField(e ast.Expr, recv string) (string, bool) {
	ix, ok := ast.Unparen(e).(*ast.IndexExpr)
	if !ok {
		return "", false
	}
	sel, ok := ast.Unparen(ix.X).(*ast.SelectorExpr)
	if !ok {
		return "", false
	}
	if id, ok := sel.X.(*ast.Ident); !ok || id.Name != recv {
		return "", false
	}
	return sel.Sel.Name, true
}

// scratchNameParts are the substrings that mark a field as a per-call temporary by
// convention. Advisory by nature: a name is evidence of intent, not proof of it.
var scratchNameParts = []string{"scratch", "scr", "buf", "tmp", "temp", "work"}

func looksLikeScratchName(f string) bool {
	l := strings.ToLower(f)
	for _, p := range scratchNameParts {
		if strings.Contains(l, p) {
			return true
		}
	}
	return false
}

// indexedField matches an indexed use of a receiver slice field, whether written
// directly (recv.field[i]) or through a local alias (y := recv.field; y[i]).
func indexedField(e ast.Expr, recv string, alias map[string]string) (string, bool) {
	if f, ok := recvIndexField(e, recv); ok {
		return f, true
	}
	ix, ok := ast.Unparen(e).(*ast.IndexExpr)
	if !ok {
		return "", false
	}
	if id, ok := ast.Unparen(ix.X).(*ast.Ident); ok {
		if f, isAlias := alias[id.Name]; isAlias {
			return f, true
		}
	}
	return "", false
}

// curSoleIndexed holds, per file, the receiver fields indexed in exactly one function —
// see the note at its assignment in scanFile. File-scoped like curPkg.
var curSoleIndexed map[string]bool

// soleIndexedFields returns the field names that appear as recv.field[...] (or through a
// local alias of recv.field) inside exactly ONE function of f. Wholesale assignment
// (b.vals = make(...)) does not count as an indexed use, so an allocation site elsewhere
// leaves a temporary sole-indexed.
func soleIndexedFields(f *ast.File) map[string]bool {
	users := map[string]map[string]bool{} // field -> set of function names indexing it
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil || fn.Recv == nil || len(fn.Recv.List) == 0 {
			continue
		}
		recv := ""
		if len(fn.Recv.List[0].Names) > 0 {
			recv = fn.Recv.List[0].Names[0].Name
		}
		if recv == "" || recv == "_" {
			continue
		}
		alias := receiverAliases(fn.Body, recv)
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			ix, ok := n.(*ast.IndexExpr)
			if !ok {
				return true
			}
			if fld, ok := indexedField(ix, recv, alias); ok {
				if users[fld] == nil {
					users[fld] = map[string]bool{}
				}
				users[fld][fn.Name.Name] = true
			}
			return true
		})
	}
	out := map[string]bool{}
	for fld, fns := range users {
		if len(fns) == 1 {
			out[fld] = true
		}
	}
	return out
}

// receiverAliases maps locals assigned directly from a receiver field to that field.
func receiverAliases(body *ast.BlockStmt, recv string) map[string]string {
	alias := map[string]string{}
	ast.Inspect(body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != len(as.Rhs) {
			return true
		}
		for i, rhs := range as.Rhs {
			sel, ok := ast.Unparen(rhs).(*ast.SelectorExpr)
			if !ok {
				continue
			}
			if id, ok := sel.X.(*ast.Ident); !ok || id.Name != recv {
				continue
			}
			if l := identName(as.Lhs[i]); l != "" && l != "_" {
				alias[l] = sel.Sel.Name
			}
		}
		return true
	})
	return alias
}

// slicedFields returns the receiver fields appearing as recv.field[a:b] (or through a
// local alias) anywhere in exprs. A slice is a READ of the buffer just as an index is;
// treating only IndexExpr as a read missed gbmBuilder.part, whose whole consumption is
// copy(dst, b.part[:r]).
func slicedFields(exprs []ast.Expr, recv string, alias map[string]string) []string {
	var out []string
	for _, e := range exprs {
		ast.Inspect(e, func(n ast.Node) bool {
			se, ok := n.(*ast.SliceExpr)
			if !ok {
				return true
			}
			if f, ok := recvField(se.X, recv); ok {
				out = append(out, f)
			} else if id, ok := ast.Unparen(se.X).(*ast.Ident); ok {
				if f, isAlias := alias[id.Name]; isAlias {
					out = append(out, f)
				}
			}
			return true
		})
	}
	return out
}

// recvField matches a bare recv.field selector (no index or slice).
func recvField(e ast.Expr, recv string) (string, bool) {
	sel, ok := ast.Unparen(e).(*ast.SelectorExpr)
	if !ok {
		return "", false
	}
	if id, ok := sel.X.(*ast.Ident); !ok || id.Name != recv {
		return "", false
	}
	return sel.Sel.Name, true
}

// curSliceFields holds, per file, the struct fields declared with a slice type.
var curSliceFields map[string]bool

// sliceTypedFields returns every struct field in f declared as a slice ([]T). Scratch is
// a buffer; the persistent registries this check kept mistaking for one are maps.
func sliceTypedFields(f *ast.File) map[string]bool {
	out := map[string]bool{}
	ast.Inspect(f, func(n ast.Node) bool {
		st, ok := n.(*ast.StructType)
		if !ok || st.Fields == nil {
			return true
		}
		for _, fld := range st.Fields.List {
			at, ok := fld.Type.(*ast.ArrayType)
			if !ok || at.Len != nil { // Len != nil is a fixed-size array, not a buffer
				continue
			}
			// PRIMITIVE ELEMENTS ONLY. Every genuine scratch buffer found in this repo is
			// []float64, []int, []bool or []uint16. The persistent registries that kept
			// being mistaken for one hold structs or pointers — optimizer per-parameter
			// state ([]soapState), fitted models ([]*tree). A buffer of records is a
			// collection someone keeps; a buffer of numbers is working space.
			if !isBasicIdent(at.Elt) {
				continue
			}
			for _, nm := range fld.Names {
				out[nm.Name] = true
			}
		}
		return true
	})
	return out
}

// isBasicIdent reports whether e is a predeclared numeric/bool element type.
func isBasicIdent(e ast.Expr) bool {
	id, ok := ast.Unparen(e).(*ast.Ident)
	if !ok {
		return false
	}
	switch id.Name {
	case "float64", "float32", "int", "int8", "int16", "int32", "int64",
		"uint", "uint8", "uint16", "uint32", "uint64", "byte", "rune", "bool":
		return true
	}
	return false
}

// searchFeedsReductionFindings flags PS6007 — a loop that computes an expensive per-item
// value with a CALL and then uses that value as the INDEX of an accumulation:
//
//	for _, x := range data {
//	    b := nearest(x, cent)          // expensive, independent per item
//	    cnt[b]++                       // …but the accumulation is order-dependent
//	    for t := range dim { sums[b][t] += x[t] }
//	}
//
// The loop cannot be partitioned as written: the accumulation is a reduction over items
// in order, and per-chunk partial sums reassociate it. The fix is to SPLIT it — run the
// search in parallel into an assignment array, then fold sequentially in the original
// order. The expensive half parallelizes and the order-dependent half does not move.
//
// Both halves matter to the diagnosis. Parallelizing the whole loop with per-chunk
// partials is faster to write, silently NOT bit-identical, and passes any test that only
// checks reproducibility rather than preservation — which is what the determinism tests
// in this repo do.
//
// Shipped twice: AQLM's k-means assignment (part of 990ms -> 278ms end to end) and the
// GMM E-step's log-likelihood total (part of 76.5ms -> 18.7ms). In both the reduction was
// a small fraction of the work, so leaving it serial cost nothing measurable.
func searchFeedsReductionFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		// NOT loopVarBody: it requires a NAMED loop variable, and the k-means assignment
		// this check was written from is `for _, x := range data`. Insisting on a named
		// key made the rule miss its own motivating case — found by replaying it against
		// the pre-fix revision, which fixtures written from the same mental model could
		// not have shown.
		body, ok := anyLoopBody(n)
		if !ok {
			return true
		}
		// Locals in THIS loop body assigned straight from a call — the per-item search.
		searched := map[string]bool{}
		for _, st := range body.List {
			as, ok := st.(*ast.AssignStmt)
			if !ok || as.Tok != token.DEFINE || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
				continue
			}
			if _, isCall := ast.Unparen(as.Rhs[0]).(*ast.CallExpr); !isCall {
				continue
			}
			if l := identName(as.Lhs[0]); l != "" && l != "_" {
				searched[l] = true
			}
		}
		if len(searched) == 0 {
			return true
		}
		// …used as the INDEX of an accumulation anywhere below.
		for name := range searched {
			if pos, okAcc := accumulationIndexedBy(body, name); okAcc {
				out = append(out, finding{
					pos:      fset.Position(pos),
					category: "search-feeds-reduction",
					msg:      fmt.Sprintf("%q comes from a per-item call and then indexes an accumulation — split the loop into a parallel search pass writing an assignment array and a sequential fold over it; partitioning as written would reassociate the sums", name),
				})
				break // one finding per loop
			}
		}
		return true
	})
	return out
}

// accumulationIndexedBy reports an `acc[name]++` or `acc[name]... += …` inside body — an
// accumulation whose destination is chosen by the searched value.
func accumulationIndexedBy(body *ast.BlockStmt, name string) (token.Pos, bool) {
	var pos token.Pos
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}
		var lhs ast.Expr
		switch v := n.(type) {
		case *ast.IncDecStmt:
			lhs = v.X
		case *ast.AssignStmt:
			if v.Tok != token.ADD_ASSIGN && v.Tok != token.SUB_ASSIGN {
				return true
			}
			if len(v.Lhs) != 1 {
				return true
			}
			lhs = v.Lhs[0]
		default:
			return true
		}
		// The destination must be indexed BY the searched value somewhere in its chain:
		// cnt[b] and sums[b][t] both qualify, x[t] does not.
		for _, ix := range indexExprs(lhs) {
			if mentions(ix.Index, name) {
				pos, found = n.Pos(), true
				return false
			}
		}
		return true
	})
	return pos, found
}

// anyLoopBody returns the body of any for/range loop, whatever its variables are named.
func anyLoopBody(n ast.Node) (*ast.BlockStmt, bool) {
	switch v := n.(type) {
	case *ast.RangeStmt:
		return v.Body, v.Body != nil
	case *ast.ForStmt:
		return v.Body, v.Body != nil
	}
	return nil, false
}

// allocInParallelBodyFindings flags PS6008 — a buffer allocated inside the body handed to
// a parallel dispatch:
//
//	parallelFeatures(d, n, func(lo, hi int) {
//	    vals := make([]float64, n)   // once per DISPATCH, not once per program
//	    …
//	})
//
// WHETHER THIS IS FREE OR RUINOUS DEPENDS ENTIRELY ON HOW OFTEN THE DISPATCH RUNS, which
// is why the check reports rather than condemns. Measured both sides in this repo: AQLM
// (49 -> 51MB) and GMM (4 -> 4MB) dispatch once per encode pass or EM iteration and paid
// nothing. The GBM exact grower dispatches once per TREE NODE — thousands of times per
// fit — and the identical code shape took it from 64MB/883 allocs to 2007MB/8965, a 31x
// memory regression that shipped hidden behind a 2.80x speedup because the commit
// reported only ns/op.
//
// The fix is not to avoid the allocation but to move it: one buffer per CHUNK on the
// caller's struct, allocated once, selected by the chunk index (parallel.RowsIdx).
//
// Deliberately NOT restricted to dispatches inside a visible loop: the GBM case is not
// one. bestSplit contains no loop around its dispatch — it is bestSplit ITSELF that runs
// per node, one call frame up. A check that demanded a local loop would have missed the
// only case that mattered.
func allocInParallelBodyFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || !isParallelDispatch(call.Fun) {
			return true
		}
		for _, arg := range call.Args {
			lit, ok := arg.(*ast.FuncLit)
			if !ok || lit.Body == nil {
				continue
			}
			for _, st := range lit.Body.List {
				as, ok := st.(*ast.AssignStmt)
				if !ok || as.Tok != token.DEFINE || len(as.Rhs) != 1 {
					continue
				}
				mk, ok := ast.Unparen(as.Rhs[0]).(*ast.CallExpr)
				if !ok || identName(mk.Fun) != "make" {
					continue
				}
				out = append(out, finding{
					pos:      fset.Position(as.Pos()),
					category: "alloc-in-parallel-body",
					msg: fmt.Sprintf("%q is allocated inside a parallel body — once per DISPATCH. Check how often this dispatch runs: if the enclosing function is itself called in a loop, hoist it to one buffer per chunk on the receiver and select it with the chunk index",
						identName(as.Lhs[0])),
				})
			}
		}
		return true
	})
	return out
}

// isParallelDispatch matches the naming this project gives its fan-out helpers —
// parallel.Rows/RowsIdx and the parallelX wrappers each package defines over them.
func isParallelDispatch(fun ast.Expr) bool {
	switch v := ast.Unparen(fun).(type) {
	case *ast.Ident:
		return strings.HasPrefix(v.Name, "parallel") || strings.Contains(v.Name, "Parallel")
	case *ast.SelectorExpr:
		if id, ok := v.X.(*ast.Ident); ok && id.Name == "parallel" {
			return strings.HasPrefix(v.Sel.Name, "Rows")
		}
	}
	return false
}

// reflectSwapperSortFindings flags PS6009 — a call to sort.Slice or sort.SliceStable.
//
// Both reach the swap through reflectlite.Swapper, which ALLOCATES on every call
// regardless of how short the slice is. slices.SortFunc/SortStableFunc take the same
// comparator, produce the same permutation for a total order, and monomorphize the swap.
//
// SPLIT OUT OF PS3002 DELIBERATELY. That check reports the same call sites but bundles two
// unrelated remedies — an LSD radix on the key bits, and this swap fix — and it states it
// cannot verify whether the radix precondition holds. The consequence was concrete: after
// converting classic/tree.go to slices.SortFunc, PS3002 went on flagging the REPLACEMENT,
// its own recommendation, and the site needed a suppression to go quiet. A check that
// cannot recognize its own fix cannot be cleared, only silenced. This one clears.
//
// TRIAGE BY CALL FREQUENCY, NOT SLICE LENGTH, which is the counter-intuitive part and the
// reason the message says so. The allocation is per CALL:
//
//	classic/tree.go   radixByFeature, once per node per feature   1,095,700 -> 352,027 allocs (3.11x)
//	classic/knn.go    three per-node/per-query sites                 36,004 ->  24,003 allocs (1.50x)
//	                  sorts of k results — SHORT slices, still 1.50x
//
// A long sort called once allocates one swapper and is not worth touching; a short sort
// called a million times is. Five sites were declined on exactly that basis.
//
// Both forms are unstable, so ties may land differently between them — check that the
// comparator is a total order, or gate the output, before converting. And note the
// signature difference: sort.Slice passes INDICES while slices.SortFunc passes VALUES, so
// `key[order[a]] < key[order[c]]` becomes `key[a] < key[c]` — silently wrong if
// transcribed rather than re-derived.
func reflectSwapperSortFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := ast.Unparen(call.Fun).(*ast.SelectorExpr)
		if !ok {
			return true
		}
		pkg, okp := sel.X.(*ast.Ident)
		if !okp || pkg.Name != "sort" {
			return true
		}
		if sel.Sel.Name != "Slice" && sel.Sel.Name != "SliceStable" {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(call.Pos()),
			category: "reflect-swapper-sort",
			msg: fmt.Sprintf("sort.%s allocates a reflect swapper on EVERY call, whatever the slice length — switch to slices.%sFunc. Triage by how often this line runs, not by how long the slice is: a short sort called per node or per query is worth converting, a long one called once per Fit is not. slices.SortFunc passes VALUES where sort.%s passes INDICES, so re-derive the comparator rather than transcribing it",
				sel.Sel.Name, map[string]string{"Slice": "Sort", "SliceStable": "SortStable"}[sel.Sel.Name], sel.Sel.Name),
		})
		return true
	})
	return out
}

// stridedInnerWalkFindings flags PS6011 — a nested loop that walks a flat row-major buffer
// along the WRONG axis. The signature is that the INNER loop variable appears multiplied by
// a stride inside the index while the outer one appears additively: S[r*dk+c] iterated over
// r, or vs[j*dm+off+d] iterated over j. Consecutive inner iterations then jump a whole row,
// so each one touches a separate cache line to consume a single element of it, and the
// whole traversal is repeated once per outer index.
//
// The correct spelling has the INNER variable additive (S[r*dk+c] iterated over c), which is
// why the two are distinguishable without types: this check asks only which loop variable is
// the one being scaled.
//
// Two fixes, and which one wins is a measurement, not a rule. Interchanging the loops makes
// the access sequential and is bit-neutral when the body is a pure elementwise update.
// Blocking four adjacent OUTER indices instead keeps register accumulators and reuses the
// line that was fetched anyway — the right choice while the buffer is cache-resident, per
// PERF-ACCUM-RESIDENCY-001. Measured this campaign: 2.65x on the Sinkhorn transposed half,
// 2.40x on the NSA P-times-V, and the larger share of KDA's 1.75x on its decay loop.
func stridedInnerWalkFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		outerVar, outerBody, ok := loopVarBody(n)
		if !ok {
			return true
		}
		// Inner loops are searched ANYWHERE beneath the outer body, not just among its
		// direct statements. The first draft of this check walked outerBody.List and
		// therefore missed its own motivating case — attendMask's P*V loop sits inside an
		// `if sum > 0`, one block down, and a guard like that is the norm rather than the
		// exception in this codebase.
		ast.Inspect(outerBody, func(inner ast.Node) bool {
			innerVar, innerBody, ok := loopVarBody(inner)
			if !ok || innerVar == outerVar {
				return true
			}
			// A TRANSPOSE cannot be fixed by interchange: out[j*r+i] = x[i*c+j] strides on
			// one side whichever way it is iterated, and swapping the loops just moves the
			// stride to the other operand. Detect it by the mirrored shape — some index in
			// this body scales the OUTER variable — and stay quiet. (Tiling helps a
			// transpose, but that is a different rewrite than the one this check advises.)
			if scalesVar(innerBody, outerVar) {
				return true
			}
			seen := map[string]bool{}
			ast.Inspect(innerBody, func(m ast.Node) bool {
				ix, isIx := m.(*ast.IndexExpr)
				if !isIx {
					return true
				}
				// A flat buffer named by a plain identifier. A [][]T two-deep index is
				// PS4006's shape, not this one, and double-reporting the same line helps
				// nobody.
				buf, isID := ix.X.(*ast.Ident)
				if !isID {
					return true
				}
				// Both loop variables must reach the SAME index expression, or the two
				// axes are not interchangeable and there is nothing to advise.
				if !exprMentions(ix.Index, innerVar) || !exprMentions(ix.Index, outerVar) {
					return true
				}
				// The discriminator: the INNER variable is scaled and the OUTER one is
				// not. Reversed, this is ordinary row-major iteration and correct.
				if !multipliedBy(ix.Index, innerVar) || multipliedBy(ix.Index, outerVar) {
					return true
				}
				// A call in the index means the stride is not a plain scalar and the
				// rewrite is not mechanical.
				if containsCall(ix.Index) {
					return true
				}
				// A pure PERMUTATION COPY — dst[j*a+i] = src[row+j], with or without a
				// conversion — has no reduction to interchange: one side must stride
				// whichever way it runs, and the real fix is tiling. This is the same
				// exclusion as the transpose check above, but it survives the stride
				// being hoisted (row := i*b), which makes the mirrored multiplication
				// invisible to a syntactic test. nlp's gguf transposes are already
				// tiled and were the false positives that motivated it.
				if permutationCopy(innerBody, ix) {
					return true
				}
				if seen[buf.Name] {
					return true
				}
				seen[buf.Name] = true
				out = append(out, finding{
					pos:      fset.Position(ix.Pos()),
					category: "strided-inner-walk",
					msg: fmt.Sprintf("the inner loop over %q indexes %q at a stride, so each iteration jumps a"+
						" whole row and touches its own cache line to use one element — and the walk repeats"+
						" once per %q. Interchange the loops so %q runs contiguously (bit-neutral when the body"+
						" is a pure elementwise update), or block four adjacent %q values so one fetched line"+
						" feeds four accumulators. Measured 2.40x (NSA P*V) and the larger share of KDA's 1.75x.",
						innerVar, buf.Name, outerVar, innerVar, outerVar),
				})
				return true
			})
			return true
		})
		return true
	})
	return dedupeByPos(out)
}

// dedupeByPos keeps one finding per source position. Searching inner loops at any depth
// means a triple-nested loop yields the same index expression once per enclosing loop.
func dedupeByPos(in []finding) []finding {
	seen := make(map[token.Position]bool, len(in))
	out := in[:0:0]
	for _, f := range in {
		if seen[f.pos] {
			continue
		}
		seen[f.pos] = true
		out = append(out, f)
	}
	return out
}

// multipliedBy reports whether name appears as an operand of a multiplication inside e.
func multipliedBy(e ast.Expr, name string) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		be, ok := n.(*ast.BinaryExpr)
		if !ok || be.Op != token.MUL {
			return true
		}
		if exprMentions(be.X, name) || exprMentions(be.Y, name) {
			found = true
		}
		return true
	})
	return found
}

// scalesVar reports whether any index expression in body multiplies name by something.
func scalesVar(body *ast.BlockStmt, name string) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if ix, ok := n.(*ast.IndexExpr); ok && multipliedBy(ix.Index, name) {
			found = true
		}
		return true
	})
	return found
}

// permutationCopy reports whether ix is the target (or source) of a plain assignment whose
// other side is a single indexed read, i.e. a copy that permutes indices rather than
// reducing. Conversions are unwrapped so float64(src[k]) counts.
func permutationCopy(body *ast.BlockStmt, ix *ast.IndexExpr) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.ASSIGN || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		// Only when the flagged index is the WRITE. A transpose is flagged on its
		// destination, because that is where the inner loop variable is scaled
		// (dst[j*a+i]). A GATHER is flagged on its source — col[j] = ss[j*cout+o] writes
		// contiguously and only reads strided — and that is fixable by transposing the
		// source once, so it must not be suppressed. Suppressing both was a false
		// negative that hid a 4.2M-strided-read loop in Wanda.
		if as.Lhs[0] != ast.Expr(ix) {
			return true
		}
		// The destination is exactly one indexed write; the source is a single read —
		// an index, or an accessor call in the generic fallback arms. What must NOT
		// appear is arithmetic combining several reads, which is a reduction rather
		// than a permutation.
		if countIndexReads(as.Lhs[0]) == 1 && countIndexReads(as.Rhs[0]) <= 1 {
			found = true
		}
		return true
	})
	return found
}

// countIndexReads counts IndexExprs in e, skipping the index expressions themselves so
// that the arithmetic inside brackets (j*a+i) is not mistaken for a second read.
func countIndexReads(e ast.Expr) int {
	n := 0
	var walk func(ast.Node)
	walk = func(node ast.Node) {
		ast.Inspect(node, func(m ast.Node) bool {
			ix, ok := m.(*ast.IndexExpr)
			if !ok {
				return true
			}
			n++
			walk(ix.X) // descend the operand, never the subscript
			return false
		})
	}
	walk(e)
	return n
}

// inconsistentFMAPinningFindings flags PS6012 — a product that can be contracted into an
// FMA inside a function whose author has ALREADY pinned other products against exactly that.
//
// Go contracts a*b + c into a single FMADD on arm64 and generally does not on amd64. A fused
// fast path that must reproduce a chain of separately-rounded backend ops therefore has to
// round every product explicitly, and float64(a*b) is how that is spelled. The failure this
// catches is not forgetting the technique — it is applying it incompletely, which looks
// correct and passes on amd64 CI.
//
// The discriminator is INTERNAL CONSISTENCY, which is what keeps it quiet: only functions
// that already contain at least one float64(a*b) are considered, because those have declared
// that contraction matters here. A function with no pinning at all is not making the claim
// and is none of this check's business.
//
// The case that motivated it is the one that is hardest to see: naming a subexpression does
// NOT pin it. `inc := gs[i] * th` used in `s = float64(s*et) - inc` is still inlined and
// contracted to fma(-gs[i], th, ...) — one rounding where the dispatch path does two. That
// asymmetry cost three attempts on the Titans fused path, because the branch that computes
// `s = -inc` (a negation, nothing to fuse into) always matched while every other step was off
// by one ulp.
func inconsistentFMAPinningFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || !hasPinnedProduct(fn.Body) {
		return nil
	}
	// Products pinned by an explicit float64(...) are the baseline, not findings.
	pinned := map[ast.Expr]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if p, ok := pinnedProduct(n); ok {
			pinned[p] = true
		}
		return true
	})
	// A local whose value is a bare product is a candidate operand: the compiler inlines it.
	unpinnedLocal := map[string]ast.Expr{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		id, isID := as.Lhs[0].(*ast.Ident)
		if !isID {
			return true
		}
		if be, ok := as.Rhs[0].(*ast.BinaryExpr); ok && be.Op == token.MUL && !pinned[be] {
			unpinnedLocal[id.Name] = be
		}
		return true
	})

	// Index and slice subscripts are INTEGER arithmetic — `ks[t*d : t*d+d]` is a product
	// feeding an add, and FMA has nothing to do with it. Without types the only way to tell
	// is structural, so every product appearing inside a subscript is excluded.
	// Requiring an INDEXED operand is the second half of separating float math from integer
	// offset math without types: `oy*wo + ox` is all plain identifiers, while real value
	// arithmetic reads memory (gs[i]*th). It also covers index expressions computed into a
	// helper call argument, which a subscript-only exclusion cannot see.
	inSubscript := subscriptProducts(fn.Body)

	var out []finding
	seen := map[token.Pos]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		add, ok := n.(*ast.BinaryExpr)
		if !ok || (add.Op != token.ADD && add.Op != token.SUB) {
			return true
		}
		if inSubscript[add.Pos()] {
			return true
		}
		for _, side := range []ast.Expr{add.X, add.Y} {
			var culprit ast.Expr
			var why string
			switch t := side.(type) {
			case *ast.BinaryExpr:
				if t.Op == token.MUL && !pinned[t] && hasIndexOperand(t) {
					culprit, why = t, "this product"
				}
			case *ast.Ident:
				if be, ok := unpinnedLocal[t.Name]; ok && hasIndexOperand(be) {
					culprit = be
					why = fmt.Sprintf("the product assigned to %q", t.Name)
				}
			}
			if culprit == nil || seen[culprit.Pos()] {
				continue
			}
			seen[culprit.Pos()] = true
			out = append(out, finding{
				pos:      fset.Position(culprit.Pos()),
				category: "inconsistent-fma-pinning",
				msg: why + " feeds an add or subtract and is NOT wrapped in float64(...), while this" +
					" function pins other products against FMA contraction. arm64 will fuse it into one" +
					" rounding where the path it must match does two, so the bit-exactness claim holds on" +
					" amd64 CI and fails on Apple silicon. Naming a subexpression does not pin it — the" +
					" compiler inlines the local. Wrap it: float64(a*b).",
			})
		}
		return true
	})
	return out
}

// hasPinnedProduct reports whether body contains at least one float64(a*b) — the signal that
// this function is deliberately defending against FMA contraction.
func hasPinnedProduct(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if _, ok := pinnedProduct(n); ok {
			found = true
		}
		return true
	})
	return found
}

// pinnedProduct matches float64(a*b) and returns the inner product.
func pinnedProduct(n ast.Node) (ast.Expr, bool) {
	call, ok := n.(*ast.CallExpr)
	if !ok || len(call.Args) != 1 {
		return nil, false
	}
	// float64 ONLY. float32(a*b) is overwhelmingly a store rounding on an F32 path, not a
	// declaration that contraction matters, and treating it as one made this check fire in
	// every typed F32 branch in the tree.
	id, ok := call.Fun.(*ast.Ident)
	if !ok || id.Name != "float64" {
		return nil, false
	}
	be, ok := call.Args[0].(*ast.BinaryExpr)
	if !ok || be.Op != token.MUL {
		return nil, false
	}
	return be, true
}

// subscriptProducts collects the positions of every expression that sits inside an index or
// slice subscript, where arithmetic is integer offsets rather than floating-point values.
func subscriptProducts(body *ast.BlockStmt) map[token.Pos]bool {
	out := map[token.Pos]bool{}
	mark := func(e ast.Expr) {
		if e == nil {
			return
		}
		ast.Inspect(e, func(n ast.Node) bool {
			if n != nil {
				out[n.Pos()] = true
			}
			return true
		})
	}
	ast.Inspect(body, func(n ast.Node) bool {
		switch t := n.(type) {
		case *ast.IndexExpr:
			mark(t.Index)
		case *ast.SliceExpr:
			mark(t.Low)
			mark(t.High)
			mark(t.Max)
		}
		return true
	})
	return out
}

// sortFeedsCountedPrefixFindings flags PS6013 — a slice sorted in full whose only subsequent
// reader is a COUNTED prefix loop.
//
// `sort(idx); for r := 0; r < k; r++ { use(idx[r]) }` computes a total order and then throws
// away everything past position k. When the consumer only asks WHICH elements are the k
// smallest — membership, not order — a selection answers it in O(n) against the sort's
// O(n log n).
//
// PS6001 covers a narrower relative: a descending vocabulary sort feeding a consumer that
// breaks early on a threshold. This one is the counted form, and it is the one that occurs:
// PS6001 matched nothing anywhere in this tree, while this shape was worth 5.1x in
// WandaPrune (282ms to 55ms) — 2048 sorts of 2048 elements to decide which half of each
// column to drop.
//
// SOUNDNESS rests on the prefix loop being the ONLY reader. If anything else reads the
// sorted slice afterwards, the full order is load-bearing and the sort must stay. Writes do
// not count — re-initializing the index slice for the next column is a write.
//
// Replacing a sort with a selection is bit-safe only when the comparator is a TOTAL order
// and the consumer reads membership rather than position; with ties the selected set is not
// unique and the two disagree. The message says so, because that is the precondition a
// reviewer has to check rather than assume.
func sortFeedsCountedPrefixFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	// Local closures that sort a CAPTURED slice count as sorters. The motivating case is
	// written that way — `sortCol := func(col []float64) { slices.SortFunc(idx, …) }` — and
	// a detector that only matched direct calls missed it, which is the third time a rule in
	// this file has failed to find the case it was built from.
	wrapped := closureSorters(fn.Body)

	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		blk, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, st := range blk.List {
			target, sorter, ok := sortedSliceName(st)
			if !ok {
				target, sorter, ok = wrappedSortCall(st, wrapped)
			}
			if !ok {
				continue
			}
			// The next statements must contain a counted loop reading target[loopvar]…
			loop, bound := countedPrefixReader(blk.List[i+1:], target)
			if loop == nil {
				continue
			}
			// …and nothing else after the sort may READ it.
			if otherReaderAfter(blk.List[i+1:], target, loop) {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(st.Pos()),
				category: "sort-feeds-counted-prefix",
				msg: fmt.Sprintf("%s orders %q in full, but the only later reader is a counted loop over"+
					" its first %s elements — the order past that point is computed and discarded. A"+
					" selection (quickselect / nth_element) answers the same question in O(n) instead of"+
					" O(n log n); measured 5.1x on WandaPrune, 282ms to 55ms. Bit-safe ONLY when the"+
					" comparator is a TOTAL order and the consumer reads membership rather than"+
					" position — with ties the selected set is not unique.", sorter, target, bound),
			})
		}
		return true
	})
	return out
}

// sortedSliceName reports the slice identifier a statement sorts, and the sorter used.
func sortedSliceName(st ast.Stmt) (target, sorter string, ok bool) {
	es, isExpr := st.(*ast.ExprStmt)
	if !isExpr {
		return "", "", false
	}
	call, isCall := es.X.(*ast.CallExpr)
	if !isCall || len(call.Args) == 0 {
		return "", "", false
	}
	// calleeName collapses a qualified call to its selector, so match the package
	// explicitly — a bare "Sort" would catch any method of that name.
	sel, isSel := call.Fun.(*ast.SelectorExpr)
	if !isSel {
		return "", "", false
	}
	pkg, isPkg := sel.X.(*ast.Ident)
	if !isPkg {
		return "", "", false
	}
	name := pkg.Name + "." + sel.Sel.Name
	switch name {
	case "sort.Slice", "sort.SliceStable", "slices.SortFunc", "slices.SortStableFunc", "slices.Sort":
	default:
		return "", "", false
	}
	id, isID := call.Args[0].(*ast.Ident)
	if !isID {
		return "", "", false
	}
	return id.Name, name, true
}

// countedPrefixReader finds a `for r := 0; r < k; r++` loop among stmts whose body indexes
// target with the loop variable, returning the loop and the bound's source text.
func countedPrefixReader(stmts []ast.Stmt, target string) (*ast.ForStmt, string) {
	var found *ast.ForStmt
	var bound string
	for _, st := range stmts {
		ast.Inspect(st, func(n ast.Node) bool {
			if found != nil {
				return false
			}
			f, ok := n.(*ast.ForStmt)
			if !ok || f.Cond == nil {
				return true
			}
			v, _, ok := loopVarBody(f)
			if !ok {
				return true
			}
			cmpb, isBin := f.Cond.(*ast.BinaryExpr)
			if !isBin || cmpb.Op != token.LSS {
				return true
			}
			// The bound must not be the slice's own length — that reads everything.
			if c, isCall := cmpb.Y.(*ast.CallExpr); isCall && calleeName(c.Fun) == "len" {
				return true
			}
			if !indexesWithVar(f.Body, target, v) {
				return true
			}
			found = f
			if id, isID := cmpb.Y.(*ast.Ident); isID {
				bound = id.Name
			} else {
				bound = "k"
			}
			return false
		})
		if found != nil {
			break
		}
	}
	return found, bound
}

// indexesWithVar reports whether body contains target[v].
func indexesWithVar(body *ast.BlockStmt, target, v string) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		ix, ok := n.(*ast.IndexExpr)
		if !ok {
			return true
		}
		id, isID := ix.X.(*ast.Ident)
		if !isID || id.Name != target {
			return true
		}
		if iv, isIV := ix.Index.(*ast.Ident); isIV && iv.Name == v {
			found = true
		}
		return true
	})
	return found
}

// otherReaderAfter reports whether target is READ anywhere in stmts outside the given loop.
// An assignment into target[i] is a write and does not keep the sort alive.
func otherReaderAfter(stmts []ast.Stmt, target string, skip *ast.ForStmt) bool {
	found := false
	for _, st := range stmts {
		ast.Inspect(st, func(n ast.Node) bool {
			if n == skip {
				return false
			}
			if as, ok := n.(*ast.AssignStmt); ok {
				for _, r := range as.Rhs {
					if exprMentions(r, target) {
						found = true
					}
				}
				for _, l := range as.Lhs {
					if ix, isIx := l.(*ast.IndexExpr); isIx {
						if exprMentions(ix.Index, target) {
							found = true
						}
						continue
					}
					if exprMentions(l, target) {
						found = true
					}
				}
				return false
			}
			if call, ok := n.(*ast.CallExpr); ok {
				for _, a := range call.Args {
					if exprMentions(a, target) {
						found = true
					}
				}
			}
			return true
		})
	}
	return found
}

// closureSorters maps a local closure's name to the captured slice it sorts.
func closureSorters(body *ast.BlockStmt) map[string]string {
	out := map[string]string{}
	ast.Inspect(body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		name, isID := as.Lhs[0].(*ast.Ident)
		if !isID {
			return true
		}
		lit, isLit := as.Rhs[0].(*ast.FuncLit)
		if !isLit || lit.Body == nil {
			return true
		}
		params := map[string]bool{}
		for _, f := range lit.Type.Params.List {
			for _, pn := range f.Names {
				params[pn.Name] = true
			}
		}
		ast.Inspect(lit.Body, func(m ast.Node) bool {
			if st, ok := m.(ast.Stmt); ok {
				if t, _, ok := sortedSliceName(st); ok && !params[t] {
					out[name.Name] = t // sorts a captured slice, not one handed in
				}
			}
			return true
		})
		return true
	})
	return out
}

// wrappedSortCall reports the slice sorted by a call to one of the closures above.
func wrappedSortCall(st ast.Stmt, wrapped map[string]string) (target, sorter string, ok bool) {
	es, isExpr := st.(*ast.ExprStmt)
	if !isExpr {
		return "", "", false
	}
	call, isCall := es.X.(*ast.CallExpr)
	if !isCall {
		return "", "", false
	}
	id, isID := call.Fun.(*ast.Ident)
	if !isID {
		return "", "", false
	}
	if t, found := wrapped[id.Name]; found {
		return t, id.Name, true
	}
	return "", "", false
}

// redundantPureRecomputeFindings flags PS6014 — the same pure call made twice in one block
// with nothing between them that could change its answer:
//
//	qPred, _ := forward(backend.NewContext(), d.Net, states) // untaped preview
//	target := buildTargets(qPred, ...)                       // reads qPred only
//	q, _ := forward(tape.Context(), d.Net, states)           // SAME network, SAME input
//
// The second call recomputes what the first already holds. In the motivating case that was
// one Context, five backend dispatches and their intermediates on a path that runs once per
// environment step — worth 1.30-1.35x once removed (rl.DQN.learn), because at these widths
// the cost is dispatch rather than arithmetic. The shape recurs wherever an untaped preview
// pass precedes the real taped pass, which is a natural way to write a TD target and a
// natural way to write it twice.
//
// PURITY IS NOT DERIVABLE FROM SYNTAX, so it is not guessed: only callees named in
// pureComputeFuncs qualify. Without that list the check reports nothing (and says so via
// the starved-vocabulary warning). Flagging any repeated call would fire on every
// rng.IntN(n) and every env.Step(a) — calls whose whole purpose is to differ.
//
// THE LEADING CONTEXT ARGUMENT IS IGNORED when comparing, which is the point: the two calls
// in the motivating case differ in exactly that argument (a fresh Context versus the tape's)
// and are otherwise identical. A comparison that included it would miss every instance.
// Soundness therefore rests on the remaining arguments, and the check requires all of them
// to be plain names or selector chains, so "identical text" means "identical value" unless
// something in between reassigns a name — which is what the invalidation scan looks for.
func redundantPureRecomputeFindings(fset *token.FileSet, fn *ast.FuncDecl, ns nameSets) []finding {
	if fn.Body == nil || len(ns.pureCompute) == 0 {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		blk, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		type site struct {
			key  string
			call *ast.CallExpr
			idx  int
		}
		var sites []site
		for i, st := range blk.List {
			call, key, ok := pureCallSite(st, ns)
			if !ok {
				continue
			}
			sites = append(sites, site{key, call, i})
		}
		for a := 0; a < len(sites); a++ {
			for b := a + 1; b < len(sites); b++ {
				if sites[a].key != sites[b].key {
					continue
				}
				if pureCallInvalidated(blk.List[sites[a].idx+1:sites[b].idx+1], sites[b].call, ns) {
					continue
				}
				out = append(out, finding{
					pos:      fset.Position(sites[b].call.Pos()),
					end:      fset.Position(sites[b].call.End()),
					category: "redundant-pure-recompute",
					msg: fmt.Sprintf("this call to %q recomputes what the identical call on line %d already produced "+
						"(same arguments apart from the leading context, and nothing between them assigns any name it reads) — "+
						"keep one result and read it twice. Measured 1.30-1.35x on rl.DQN.learn, where the duplicate cost "+
						"one context plus five backend dispatches per step.",
						calleeName(sites[b].call.Fun), fset.Position(sites[a].call.Pos()).Line),
				})
				break // one report per redundant site
			}
		}
		return true
	})
	return out
}

// pureCallSite recognizes `x, err := PURE(ctx, args…)` and returns a comparison key built
// from the callee and every argument AFTER the first. Only plain names and selector chains
// qualify as arguments: a composite literal or a nested call would make textual equality a
// weaker claim than value equality.
func pureCallSite(st ast.Stmt, ns nameSets) (*ast.CallExpr, string, bool) {
	as, ok := st.(*ast.AssignStmt)
	if !ok || len(as.Rhs) != 1 {
		return nil, "", false
	}
	call, ok := as.Rhs[0].(*ast.CallExpr)
	if !ok || len(call.Args) < 2 || call.Ellipsis.IsValid() {
		return nil, "", false
	}
	if !ns.pureCompute[calleeName(call.Fun)] {
		return nil, "", false
	}
	// The key carries the FULL callee expression, not just the trailing name. calleeName
	// collapses a qualified call to its last segment, so `b.Wq.Forward(ctx, xn)` and
	// `b.Wk.Forward(ctx, xn)` — two different projections of the same input, the most common
	// three lines in an attention block — keyed identically and reported as a recompute.
	// Every hit outside the package this rule was built from was that false positive. The
	// vocabulary is still matched on the trailing name, since that is how a method is named
	// in config; only the identity comparison needs the receiver.
	key := exprText(call.Fun)
	if key == "" {
		return nil, "", false
	}
	for _, arg := range call.Args[1:] {
		t := exprText(arg)
		if t == "" {
			return nil, "", false
		}
		switch arg.(type) {
		case *ast.Ident, *ast.SelectorExpr:
		default:
			return nil, "", false
		}
		key += "\x00" + t
	}
	return call, key, true
}

// pureCallInvalidated reports whether anything in stmts could change what call returns:
// an assignment to a name the call reads, or a call to a NON-pure function handed one of
// those names (which might mutate what it points at). Both scans descend into nested nodes,
// so a loop or closure in between is covered rather than skipped.
func pureCallInvalidated(stmts []ast.Stmt, call *ast.CallExpr, ns nameSets) bool {
	reads := map[string]bool{}
	for _, arg := range call.Args[1:] {
		for _, nm := range identNamesIn(arg) {
			reads[nm] = true
		}
	}
	for _, nm := range identNamesIn(call.Fun) {
		reads[nm] = true
	}
	bad := false
	for _, st := range stmts {
		if bad {
			break
		}
		ast.Inspect(st, func(n ast.Node) bool {
			if bad {
				return false
			}
			switch x := n.(type) {
			case *ast.AssignStmt:
				for _, lhs := range x.Lhs {
					for _, nm := range identNamesIn(lhs) {
						if reads[nm] {
							bad = true
						}
					}
				}
			case *ast.IncDecStmt:
				for _, nm := range identNamesIn(x.X) {
					if reads[nm] {
						bad = true
					}
				}
			case *ast.UnaryExpr:
				if x.Op == token.AND { // &x — the address escapes, assume a write
					for _, nm := range identNamesIn(x.X) {
						if reads[nm] {
							bad = true
						}
					}
				}
			case *ast.CallExpr:
				if ns.pureCompute[calleeName(x.Fun)] {
					return true
				}
				// len and cap are builtins that provably cannot mutate their argument, and
				// they are how a size gets read between the two calls (Shape{len(batch), k}).
				// Treating them as potential writers suppressed the very case this rule was
				// built from when that case was written with the same slice name.
				if id, ok := x.Fun.(*ast.Ident); ok && (id.Name == "len" || id.Name == "cap") {
					return true
				}
				// A method on something the call reads, or a call handed one of those
				// names, may mutate it.
				for _, nm := range identNamesIn(x.Fun) {
					if reads[nm] {
						bad = true
					}
				}
				for _, arg := range x.Args {
					for _, nm := range mutableIdentNamesIn(arg) {
						if reads[nm] {
							bad = true
						}
					}
				}
			}
			return true
		})
	}
	return bad
}

// mutableIdentNamesIn is identNamesIn minus anything reachable only through a len or cap
// call. Those builtins provably cannot mutate their argument, and reading a size is the
// normal reason a name appears between two otherwise-identical calls —
// `New(F64, Shape{len(states), k})`. Counting the `states` inside that `len` as a possible
// write suppressed the exact case this rule was built from.
//
// Worth recording HOW that was caught, because the usual order failed here: replaying the
// detector against the real pre-fix source PASSED, since that source happened to size its
// tensor from a different slice than it fed the forward. The synthetic positive test is what
// exposed it. Replay proves a rule finds the case it was built from; it does not prove the
// rule finds the SHAPE, and one incidental difference in naming is enough to hide the gap.
func mutableIdentNamesIn(e ast.Expr) []string {
	var out []string
	ast.Inspect(e, func(n ast.Node) bool {
		if c, ok := n.(*ast.CallExpr); ok {
			if id, ok := c.Fun.(*ast.Ident); ok && (id.Name == "len" || id.Name == "cap") {
				return false
			}
		}
		if id, ok := n.(*ast.Ident); ok {
			out = append(out, id.Name)
		}
		return true
	})
	return out
}

// identNamesIn collects every identifier name appearing in e, including the base of a
// selector chain (so `d.Net` contributes both "d" and "Net").
func identNamesIn(e ast.Expr) []string {
	var out []string
	ast.Inspect(e, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok {
			out = append(out, id.Name)
		}
		return true
	})
	return out
}

// batch1FeedsPostloopSliceFindings flags PS6015 — a pure call made once per iteration on a
// single-element batch, whose result NOTHING IN THE LOOP READS:
//
//	for { // per environment step
//		v, _ := forward(NewContext(), critic, [][]float64{obs}) // batch of one
//		ro.values = append(ro.values, v.AtF64(0, 0))            // the only use
//	}
//	// ro.values consumed here, after the loop
//
// Because the result only accumulates into a slice that outlives the loop, the N batch-1
// calls answer a question one batch-N call answers — the loop does not depend on the answer
// while it runs. Hoisting the critic out of `rl.rlRollout` this way was **1.59x** on
// collection (29239 allocations down to 15471) and **1.19x** end to end, since each batch-1
// forward was five backend dispatches on a one-row tensor.
//
// THIS IS DIFFERENT ADVICE FROM PS1003, which matches the same call shape and says "call a
// single-item API instead" — avoid the wrapper allocation, keep N calls. That is the right
// fix when the loop READS the result (the actor forward in the same loop feeds a softmax
// that feeds the action that feeds the environment, so it cannot move). PS1003 also reports
// once per loop, so where both a hoistable and a non-hoistable call share a loop it flags
// only the first and the hoistable one can go unmentioned entirely. This check reports per
// call site and only for the hoistable case, so the two are complementary rather than
// redundant.
//
// SOUNDNESS. Hoisting across a loop is legal only if the call is pure, so the callee must be
// named in pureComputeFuncs — the same licensing PS6014 uses, and for a stronger reason
// here, since a call that consumed RNG would move draws out of the stream and change every
// subsequent iteration. Beyond that, EVERY use of the result inside the loop must be an
// append to a slice declared outside it. One use in a branch condition, one use handed to
// another call, one use feeding the iteration state, and the result is loop-carried — the
// check stays silent rather than proposing a hoist that changes behavior.
func batch1FeedsPostloopSliceFindings(fset *token.FileSet, fn *ast.FuncDecl, ns nameSets) []finding {
	if fn.Body == nil || len(ns.pureCompute) == 0 {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		body := loopBody(n)
		if body == nil {
			return true
		}
		declared := declaredInBlock(body)
		for _, st := range body.List {
			as, ok := st.(*ast.AssignStmt)
			if !ok || len(as.Rhs) != 1 || len(as.Lhs) == 0 {
				continue
			}
			call, ok := as.Rhs[0].(*ast.CallExpr)
			if !ok || !ns.pureCompute[calleeName(call.Fun)] || !hasSingleElemBatchArg(call) {
				continue
			}
			res, ok := as.Lhs[0].(*ast.Ident)
			if !ok || res.Name == "_" {
				continue
			}
			if !onlyAppendedOutside(body, res.Name, declared) {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(call.Pos()),
				end:      fset.Position(call.End()),
				category: "batch1-call-feeds-only-postloop-slice",
				msg: fmt.Sprintf("%q runs once per iteration on a batch of one, and %q is only appended to a slice "+
					"that outlives the loop — nothing in the loop reads it, so these calls can become ONE batched "+
					"call after the loop. Measured 1.59x on rl.rlRollout (29239 allocations to 15471). Unlike PS1003, "+
					"which keeps N calls and drops the wrapper, this removes N-1 calls; it applies only because no "+
					"loop-carried use was found.",
					calleeName(call.Fun), res.Name),
			})
		}
		return true
	})
	return out
}

// loopBody returns the body of a for/range statement, or nil for any other node.
func loopBody(n ast.Node) *ast.BlockStmt {
	switch x := n.(type) {
	case *ast.ForStmt:
		return x.Body
	case *ast.RangeStmt:
		return x.Body
	}
	return nil
}

// declaredInBlock collects names introduced anywhere inside body (:= or var), so an append
// target can be told from a per-iteration local.
func declaredInBlock(body *ast.BlockStmt) map[string]bool {
	out := map[string]bool{}
	ast.Inspect(body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.AssignStmt:
			if x.Tok == token.DEFINE {
				for _, lhs := range x.Lhs {
					if id, ok := lhs.(*ast.Ident); ok {
						out[id.Name] = true
					}
				}
			}
		case *ast.ValueSpec:
			for _, nm := range x.Names {
				out[nm.Name] = true
			}
		case *ast.RangeStmt:
			for _, e := range []ast.Expr{x.Key, x.Value} {
				if id, ok := e.(*ast.Ident); ok {
					out[id.Name] = true
				}
			}
		}
		return true
	})
	return out
}

// hasSingleElemBatchArg reports whether any argument is a one-element slice literal — the
// batch-of-one wrapper.
func hasSingleElemBatchArg(call *ast.CallExpr) bool {
	for _, arg := range call.Args {
		cl, ok := arg.(*ast.CompositeLit)
		if !ok || len(cl.Elts) != 1 {
			continue
		}
		if _, ok := cl.Type.(*ast.ArrayType); ok {
			return true
		}
	}
	return false
}

// onlyAppendedOutside reports whether EVERY use of name inside body sits in an
// `outer = append(outer, …)` whose target is declared outside body. A single use anywhere
// else means the result is loop-carried and the hoist is not available.
func onlyAppendedOutside(body *ast.BlockStmt, name string, declared map[string]bool) bool {
	appendUses, otherUses := 0, 0
	var walk func(n ast.Node, inAppend bool)
	walk = func(n ast.Node, inAppend bool) {
		if n == nil {
			return
		}
		if as, ok := n.(*ast.AssignStmt); ok && len(as.Lhs) == 1 && len(as.Rhs) == 1 {
			if c, ok := as.Rhs[0].(*ast.CallExpr); ok && calleeName(c.Fun) == "append" && len(c.Args) >= 2 {
				target := exprText(as.Lhs[0])
				if target != "" && target == exprText(c.Args[0]) && !declared[baseIdentName(as.Lhs[0])] {
					// The accumulating append: uses of name in the appended values are fine.
					for _, arg := range c.Args[1:] {
						for _, nm := range identNamesIn(arg) {
							if nm == name {
								appendUses++
							}
						}
					}
					// Anything else in this statement still counts as another use.
					for _, nm := range identNamesIn(as.Lhs[0]) {
						if nm == name {
							otherUses++
						}
					}
					return
				}
			}
		}
		ast.Inspect(n, func(m ast.Node) bool {
			if m == n {
				return true
			}
			if st, ok := m.(ast.Stmt); ok {
				walk(st, inAppend)
				return false
			}
			if id, ok := m.(*ast.Ident); ok && id.Name == name {
				otherUses++
			}
			return true
		})
	}
	for _, st := range body.List {
		walk(st, false)
	}
	// The defining assignment itself contributes one non-append use of name (its LHS), so
	// require exactly that one and at least one append use.
	return appendUses >= 1 && otherUses <= 1
}

// loopInvariantLiteralArgFindings flags PS6016 — a struct literal rebuilt every iteration
// from values that do not change:
//
//	for l, b := range m.Blocks {
//		q, _ = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: pos}, q)
//	}
//
// Nothing in the literal depends on l or b. Rebuilding it is cheap on its own; what is not
// cheap is that `exec1` takes an INTERFACE, so each construction is also a heap box — once
// per layer per decoded token. Hoisting these above the loop across six nlp decode paths
// removed 8490 allocations from a 500-token generate (-2.9%) and 391KB of garbage.
//
// SOUNDNESS. The literal must be passed DIRECTLY as a call argument and nowhere else: not
// appended, not assigned, not address-taken. A literal that escapes into a slice needs its
// per-iteration identity, and hoisting it would make every element alias one value. Field
// initializers must reference nothing the loop assigns and no loop variable, so the hoisted
// value is the value every iteration built.
//
// WHAT THIS CANNOT SEE, and it is half the defect. The same waste occurs when the struct is
// ALREADY hoisted but the interface conversion still happens at the call site — the form an
// earlier pass in this repo produced and then missed, because hoisting the struct looks like
// the fix. Recognizing it requires knowing the parameter is an interface type, which needs
// go/types; this scanner is deliberately go/ast-only. The tool that does see it is the
// compiler: `go build -gcflags='<pkg>=-m'` names every escaping literal, and that is how
// both forms were actually found here. This check covers the form a parser can prove and
// points at escape analysis for the rest rather than guessing.
func loopInvariantLiteralArgFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		body := loopBody(n)
		if body == nil {
			return true
		}
		moving := loopMovingNames(n, body)
		// Every use of each candidate literal has to be the one call argument, so collect
		// the literals reached as a direct argument and reject any seen elsewhere.
		reported := map[string]bool{}
		ast.Inspect(body, func(m ast.Node) bool {
			call, ok := m.(*ast.CallExpr)
			if !ok || calleeName(call.Fun) == "append" {
				return true
			}
			for _, arg := range call.Args {
				cl, ok := arg.(*ast.CompositeLit)
				if !ok || len(cl.Elts) == 0 || cl.Type == nil {
					continue
				}
				if _, ok := cl.Type.(*ast.ArrayType); ok {
					continue // a slice literal is the pooling question (PS2001), not this one
				}
				if _, ok := cl.Type.(*ast.MapType); ok {
					continue
				}
				if !allElemsInvariant(cl, moving) {
					continue
				}
				tname := exprText(cl.Type)
				if tname == "" {
					continue
				}
				// Dedup per SITE, not per type: the q and k RoPE attrs in a decode loop are
				// two distinct literals of the same type, and both need hoisting. Keying on
				// the type reported one and hid the other.
				key := fmt.Sprintf("%s@%d", tname, fset.Position(cl.Pos()).Line)
				if reported[key] {
					continue
				}
				reported[key] = true
				out = append(out, finding{
					pos:      fset.Position(cl.Pos()),
					end:      fset.Position(cl.End()),
					category: "loop-invariant-literal-arg",
					msg: fmt.Sprintf("this %s literal is rebuilt every iteration from values the loop never changes, and it is "+
						"passed straight to a call — if that parameter is an interface, each construction is also a heap box. "+
						"Construct it once above the loop. Doing this across six nlp decode paths removed 8490 allocations from a "+
						"500-token generate (-2.9%%). Confirm with go build -gcflags='<pkg>=-m', which is also the only way to see "+
						"the variant where the struct is already hoisted but still boxed at the call site.",
						tname),
				})
			}
			return true
		})
		return true
	})
	return out
}

// loopMovingNames is every name that changes across iterations of the loop at n: its
// induction variables plus anything assigned anywhere in its body.
func loopMovingNames(n ast.Node, body *ast.BlockStmt) map[string]bool {
	moving := map[string]bool{}
	switch x := n.(type) {
	case *ast.RangeStmt:
		for _, e := range []ast.Expr{x.Key, x.Value} {
			if id, ok := e.(*ast.Ident); ok {
				moving[id.Name] = true
			}
		}
	case *ast.ForStmt:
		for _, st := range []ast.Stmt{x.Init, x.Post} {
			if st == nil {
				continue
			}
			ast.Inspect(st, func(m ast.Node) bool {
				switch y := m.(type) {
				case *ast.AssignStmt:
					for _, lhs := range y.Lhs {
						if id, ok := lhs.(*ast.Ident); ok {
							moving[id.Name] = true
						}
					}
				case *ast.IncDecStmt:
					if id, ok := y.X.(*ast.Ident); ok {
						moving[id.Name] = true
					}
				}
				return true
			})
		}
	}
	ast.Inspect(body, func(m ast.Node) bool {
		switch y := m.(type) {
		case *ast.AssignStmt:
			for _, lhs := range y.Lhs {
				for _, nm := range identNamesIn(lhs) {
					moving[nm] = true
				}
			}
		case *ast.IncDecStmt:
			for _, nm := range identNamesIn(y.X) {
				moving[nm] = true
			}
		case *ast.RangeStmt:
			for _, e := range []ast.Expr{y.Key, y.Value} {
				if id, ok := e.(*ast.Ident); ok {
					moving[id.Name] = true
				}
			}
		}
		return true
	})
	return moving
}

// allElemsInvariant reports whether every field initializer of cl references only names the
// loop leaves alone. A call in an initializer is allowed only if its own arguments are
// invariant — cfg.attnScale() is fine, next() is not, and neither is distinguishable by name,
// so any zero-argument call is treated as invariant only when its receiver chain is.
func allElemsInvariant(cl *ast.CompositeLit, moving map[string]bool) bool {
	ok := true
	for _, el := range cl.Elts {
		val := el
		if kv, isKV := el.(*ast.KeyValueExpr); isKV {
			val = kv.Value
		}
		ast.Inspect(val, func(n ast.Node) bool {
			if !ok {
				return false
			}
			if id, isID := n.(*ast.Ident); isID && moving[id.Name] {
				ok = false
			}
			return true
		})
	}
	return ok
}

// variadicSiblings maps a package name to each variadic function it declares that HAS
// fixed-arity siblings, and from that function to arity -> sibling name.
//
// A sibling is a function with IDENTICAL leading parameter types followed by exactly n
// parameters of the variadic element type. In this repository that is exec1(ctx, op, attrs,
// ins ...*tensor.Tensor) against exec1a/exec2/exec3, which exist precisely so a call with a
// known argument count does not allocate a `[]*tensor.Tensor` to pass three pointers.
//
// Package-level, because the variadic form and its siblings are often declared in one file
// and called from twenty others — the same reason intMapReg exists. Nothing here needs
// go/types: two parameter types are the same when their source text is, which within one
// package is what "same type" means for this purpose.
type variadicFamily struct {
	fixed   int            // number of non-variadic leading parameters
	byArity map[int]string // trailing-argument count -> sibling name
}

var variadicSiblings = map[string]map[string]variadicFamily{}

// collectVariadicSiblings pre-scans every package for variadic functions and pairs each with
// the fixed-arity siblings that could replace a call of known arity.
// typeText renders a parameter type to source form. exprText is NOT usable here: it has no
// StarExpr case and returns empty for every pointer, which breaks this comparison in both
// directions depending on how the empty is handled. Treated as a placeholder, all pointer
// types collapse to one value and *backend.Context compares equal to *tensor.Tensor — that is
// how concat1D(parts ...*tensor.Tensor) acquired project(ctx *backend.Context, …) as a
// "sibling", the one hit in the tree that was obviously wrong. Treated as unrenderable and
// skipped, every candidate with a pointer parameter drops out and the rule reports nothing at
// all — which is the worse failure, because a silent check reads as a clean codebase.
// Rendering the type properly is what makes the comparison mean anything.
func typeText(fset *token.FileSet, e ast.Expr) string {
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, fset, e); err != nil {
		return ""
	}
	return buf.String()
}

func collectVariadicSiblings(fset *token.FileSet, files []*ast.File) {
	type sig struct {
		name     string
		lead     []string // rendered leading parameter types
		elem     string   // rendered variadic element type, "" if not variadic
		trailing []string // rendered trailing parameter types (non-variadic funcs)
	}
	byPkg := map[string][]sig{}
	for _, f := range files {
		if f.Name == nil {
			continue
		}
		pkg := f.Name.Name
		for _, decl := range f.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok || fd.Recv != nil || fd.Type.Params == nil {
				continue
			}
			var flat []string
			variadic := ""
			skip := false
			for _, fld := range fd.Type.Params.List {
				n := len(fld.Names)
				if n == 0 {
					n = 1
				}
				if ell, isEll := fld.Type.(*ast.Ellipsis); isEll {
					variadic = typeText(fset, ell.Elt)
					if variadic == "" {
						skip = true // an unrenderable type cannot be compared
					}
					break // the variadic parameter is always last
				}
				t := typeText(fset, fld.Type)
				if t == "" {
					skip = true
					break
				}
				for range n {
					flat = append(flat, t)
				}
			}
			if skip {
				continue
			}
			if variadic != "" {
				byPkg[pkg] = append(byPkg[pkg], sig{name: fd.Name.Name, lead: flat, elem: variadic})
				continue
			}
			byPkg[pkg] = append(byPkg[pkg], sig{name: fd.Name.Name, trailing: flat})
		}
	}
	for pkg, sigs := range byPkg {
		for _, v := range sigs {
			// At least one FIXED leading parameter is required. With none, "same trailing
			// types" is far too weak a relation: concat1D(parts ...*tensor.Tensor) matched
			// every two-tensor function in the package, none of which was a sibling of it in
			// any useful sense. The shared prefix is what makes a family — for exec1 it is
			// (ctx, op, attrs), which identifies the operation the siblings all perform.
			if v.elem == "" || len(v.lead) == 0 {
				continue
			}
			for _, c := range sigs {
				if c.elem != "" || c.name == v.name || len(c.trailing) <= len(v.lead) {
					continue
				}
				// leading types must match exactly...
				same := true
				for i, t := range v.lead {
					if c.trailing[i] != t {
						same = false
						break
					}
				}
				if !same {
					continue
				}
				// ...and every remaining parameter must be the variadic element type.
				rest := c.trailing[len(v.lead):]
				for _, t := range rest {
					if t != v.elem {
						same = false
						break
					}
				}
				if !same {
					continue
				}
				if variadicSiblings[pkg] == nil {
					variadicSiblings[pkg] = map[string]variadicFamily{}
				}
				fam, ok := variadicSiblings[pkg][v.name]
				if !ok {
					fam = variadicFamily{fixed: len(v.lead), byArity: map[int]string{}}
				}
				// A stable pick when two siblings share an arity: shortest name, then
				// lexicographic, so the report does not change between runs.
				if prev, seen := fam.byArity[len(rest)]; !seen ||
					len(c.name) < len(prev) || (len(c.name) == len(prev) && c.name < prev) {
					fam.byArity[len(rest)] = c.name
				}
				variadicSiblings[pkg][v.name] = fam
			}
		}
	}
}

// unpooledVariadicSiblingFindings flags PS6017 — a variadic helper called in a loop at an
// arity a fixed-arity sibling already covers:
//
//	for l, b := range m.Blocks {
//		a, _ := exec1(ctx, backend.OpMHA, attn, q, kNew, vNew) // allocates a 3-elem slice
//	}
//
// `exec3` takes the same three tensors as named parameters and pools the slice it builds, so
// the variadic call is a per-iteration allocation with a ready-made replacement. Switching
// these by hand across the nlp decode paths was part of a change measured at -3.1% allocs;
// the point of the rule is that 140 such call sites remained package-wide and no one was
// going to find them by reading.
//
// SOUNDNESS. The sibling must have identical leading parameter types and exactly n trailing
// ones of the variadic element type, so the call transfers argument for argument. Calls that
// SPREAD (f(xs...)) are skipped — their arity is not known here. Restricted to loop bodies:
// the allocation is per call either way, but once per invocation is rarely worth a diff, and
// keeping the report to per-iteration costs is what makes the output actionable.
//
// WHAT IT CANNOT CHECK is whether the sibling is semantically equivalent rather than merely
// type-compatible — that is a judgment about the two bodies. In this repository the siblings
// pool only when ctx.Recorder == nil and delegate to the variadic form otherwise, which is
// exactly the equivalence wanted; elsewhere it must be read before the swap. Advisory, like
// the rest.
func unpooledVariadicSiblingFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	sibs := variadicSiblings[curPkg]
	if fn.Body == nil || len(sibs) == 0 {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		body := loopBody(n)
		if body == nil {
			return true
		}
		ast.Inspect(body, func(m ast.Node) bool {
			call, ok := m.(*ast.CallExpr)
			if !ok || call.Ellipsis.IsValid() {
				return true
			}
			id, ok := call.Fun.(*ast.Ident)
			if !ok {
				return true
			}
			fam, ok := sibs[id.Name]
			if !ok {
				return true
			}
			// Trailing count = total arguments minus the variadic function's fixed leading
			// parameters. The sibling table is keyed by that count, so one lookup answers
			// both whether a replacement exists and which one it is.
			arity := len(call.Args) - fam.fixed
			sib, ok := fam.byArity[arity]
			if !ok {
				return true
			}
			out = append(out, finding{
				pos:      fset.Position(call.Pos()),
				end:      fset.Position(call.End()),
				category: "unpooled-variadic-sibling",
				msg: fmt.Sprintf("%q is variadic and is called here with %d trailing argument(s) inside a loop, so it "+
					"allocates a slice every iteration — %q takes the same arguments as named parameters and exists to "+
					"avoid that. Switching these across the nlp decode paths was part of a change measured at -3.1%% "+
					"allocations. Read the sibling first: the rule proves the signatures transfer, not that the bodies agree.",
					id.Name, arity, sib),
			})
			return true
		})
		return true
	})
	return out
}

// layoutOpClusterFindings flags PS6018 — a function that spends most of its dispatches moving
// data rather than computing:
//
//	flat, _ := exec1(ctx, backend.OpReshape, …, x)      // 7 layout dispatches
//	rot, _  := exec1(ctx, backend.OpSlice, …, flat)     // around
//	pass, _ := exec1(ctx, backend.OpSlice, …, flat)     // exactly
//	rotWide, _ := exec1(ctx, backend.OpReshape, …, rot) // one
//	rotWide, _ = exec1a(ctx, backend.OpRoPE, r, rotWide)// arithmetic op
//	merged, _ := exec1(ctx, backend.OpConcat, …, rot, pass)
//
// Movement CANNOT change a value, so gathering the operands out of raw storage and scattering
// the result back is bit-identical BY CONSTRUCTION — no reassociation, no FMA question, no
// tolerance argument. That is what makes this class worth flagging on sight where a general
// "too many dispatches" report would not be: the fix needs no numerical judgment, only index
// arithmetic. Shipped three times, all in one session: partialRoPE 1.25-1.33x with 38-43%
// fewer allocations across three architectures, Gemma2 capped attention 1.21x / -27.6%, and
// DeepSeekV2 absorbed attention 1.12x / -9.3%.
//
// PS4011 IS THE LOOP-SHAPED RELATIVE and does not subsume this. It requires the dispatches to
// sit in a sequential loop; partialRoPE is straight-line code, so PS4011 could not see the
// largest of the three wins. Together they cover the two shapes.
//
// SUPPRESSED once the function already has a fused path — a ctx.Recorder == nil guard, a
// configured fast-path helper, or a direct Storage() grab. That is what the fix looks like, so
// a fixed function must stop reporting; without it the rule would keep flagging its own
// successes.
//
// The threshold is three because two movement ops around one arithmetic op is often the
// irreducible shape of an operation (transpose-then-matmul), while three or more means the
// layout algebra has outgrown the computation.
func layoutOpClusterFindings(fset *token.FileSet, fn *ast.FuncDecl, ns nameSets) []finding {
	if fn.Body == nil || len(ns.layoutOps) == 0 {
		return nil
	}
	if hasFusedStoragePath(fn.Body, ns) {
		return nil
	}
	var sites []token.Pos
	names := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		for _, arg := range call.Args {
			sel, ok := arg.(*ast.SelectorExpr)
			if !ok || !ns.layoutOps[sel.Sel.Name] {
				continue
			}
			sites = append(sites, call.Pos())
			names[sel.Sel.Name] = true
			break
		}
		return true
	})
	if len(sites) < 3 {
		return nil
	}
	sorted := make([]string, 0, len(names))
	for nm := range names {
		sorted = append(sorted, nm)
	}
	sort.Strings(sorted)
	return []finding{{
		pos:      fset.Position(sites[0]),
		end:      fset.Position(sites[0]),
		category: "layout-op-cluster-unfused",
		msg: fmt.Sprintf("this function dispatches %d pure data-movement ops (%s) and has no fused raw-storage path. "+
			"Movement cannot change a value, so gathering the operands out of storage and scattering the result back is "+
			"bit-identical BY CONSTRUCTION — index arithmetic only, no numerical judgment. Gate the fused arm on "+
			"ctx.Recorder == nil so a taped context keeps the dispatch nodes as gradient edges. Shipped: partialRoPE "+
			"1.25-1.33x with 38-43%% fewer allocations, Gemma2 capped attention 1.21x, DeepSeekV2 absorbed attention 1.12x.",
			len(sites), strings.Join(sorted, ", ")),
	}}
}

// hasFusedStoragePath reports whether fn already reaches raw storage behind a tape guard —
// the shape the PS6018 fix takes, and therefore the shape that must silence it.
func hasFusedStoragePath(body *ast.BlockStmt, ns nameSets) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}
		switch x := n.(type) {
		case *ast.BinaryExpr:
			// ctx.Recorder == nil (either operand order)
			if x.Op == token.EQL || x.Op == token.NEQ {
				for _, side := range []ast.Expr{x.X, x.Y} {
					if sel, ok := side.(*ast.SelectorExpr); ok && sel.Sel.Name == "Recorder" {
						found = true
					}
				}
			}
		case *ast.CallExpr:
			nm := calleeName(x.Fun)
			if nm == "Storage" || ns.fastPath[nm] {
				found = true
			}
		}
		return true
	})
	return found
}

// fanoutWithoutWorkerSeamFindings flags PS6021 — a fan-out helper whose callback signature
// gives its callers no per-worker seam.
//
// Three separate wins in this repository were blocked by the identical shape, and in all
// three the arithmetic was already parallel; what was missing was a place to put a buffer:
//
//	func knnParallelRows(n int, body func(i int))      // KNN: 3 allocations per QUERY
//	func nbPredictParallel(n, feat int, body func(i int)) // GaussianNB: 1 per ROW
//
// The callback is invoked once per ITEM, so a buffer the caller allocates inside it is
// allocated per item. Hoisting it above the helper makes it shared mutable state that every
// worker races on — and a receiver field is the same bug with a longer fuse (PS6006). With
// no per-worker seam in the signature, the per-item allocation is the only CORRECT option
// available, which is why these sites survive review: the code is not careless, the
// interface is short a parameter. Measured after adding one: GaussianNB predict 1.28x with
// 99.2% fewer allocations, KNN predict 99.4% fewer allocations, DBSCAN fit 78.8% fewer.
//
// THE FIX IS THE SIGNATURE. Either give the callback a scratch parameter the helper supplies
// per worker (gmmParallelRows, moeParallelTokens, wkvParallelChannels all do this), or take
// a func() T constructor the helper calls once per worker and passes down.
//
// A callback taking a RANGE rather than a single index is NOT reported: with (lo, hi) the
// caller can allocate inside the chunk closure, which is per-chunk and therefore already
// per-worker. That is why parallel.Rows and the many (lo, hi) helpers here are silent, and
// it is the distinction that makes this check quiet enough to act on.
//
// A helper that takes a func() T scratch constructor needs no separate exception: it must
// pass what the constructor returns down to the callback, so that callback already carries a
// scratch parameter and fails the index-only test. A clause for the constructor was written
// first and removed — mutation testing showed nothing depended on it, and a predicate clause
// no test can reach implies coverage it does not have.
//
// A helper that creates a CHANNEL in its own body is also skipped: that is a work-queue
// primitive (parallelBuild), where the callback is the job and the queue is what other
// helpers build their own seam on top of by passing a worker count as the job count.
// Reporting the primitive would be reporting the mechanism every fix uses.
func fanoutWithoutWorkerSeamFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || fn.Type.Params == nil {
		return nil
	}
	if !fanoutDispatches(fn.Body) || fanoutMakesChan(fn.Body) {
		return nil
	}
	var (
		cbName string
		cbPos  token.Pos
	)
	for _, p := range fn.Type.Params.List {
		ft, ok := p.Type.(*ast.FuncType)
		if !ok || ft.Params == nil || !fanoutIndexOnlyParams(ft.Params) {
			continue
		}
		for _, nm := range p.Names {
			cbName, cbPos = nm.Name, nm.Pos()
		}
	}
	if cbName == "" {
		return nil
	}
	return []finding{{
		pos:      fset.Position(cbPos),
		category: "fanout-without-worker-seam",
		msg: fmt.Sprintf("callback %q takes only a work index, so a caller needing a per-item buffer has nowhere to hoist it — every call site must allocate per item, and hoisting above %q would be a data race. Give the callback a scratch parameter, or take a func() T constructor this helper calls once per worker",
			cbName, fn.Name.Name),
	}}
}

// fanoutIndexOnlyParams reports whether a callback takes exactly one parameter and that
// parameter is a bare integer index. Exactly one is the load-bearing part: a (lo, hi) range
// hands the caller a per-chunk closure, which is already a per-worker seam.
func fanoutIndexOnlyParams(ps *ast.FieldList) bool {
	n := 0
	for _, f := range ps.List {
		id, ok := f.Type.(*ast.Ident)
		if !ok {
			return false
		}
		switch id.Name {
		case "int", "int32", "int64":
		default:
			return false
		}
		if len(f.Names) == 0 {
			n++
			continue
		}
		n += len(f.Names)
	}
	return n == 1
}

// fanoutDispatches reports whether a body fans work out: a go statement, or a call to one of
// the project's parallel helpers (the same naming isParallelDispatch matches).
func fanoutDispatches(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}
		switch v := n.(type) {
		case *ast.GoStmt:
			found = true
		case *ast.CallExpr:
			if isParallelDispatch(v.Fun) {
				found = true
			}
		}
		return !found
	})
	return found
}

// fanoutMakesChan reports whether a body constructs a channel, marking it a work-queue
// primitive rather than an index fan-out.
func fanoutMakesChan(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}
		call, ok := n.(*ast.CallExpr)
		if !ok || identName(call.Fun) != "make" || len(call.Args) == 0 {
			return true
		}
		if _, isChan := call.Args[0].(*ast.ChanType); isChan {
			found = true
		}
		return !found
	})
	return found
}

// jamTailDelegatesFindings flags PS6019 — an unroll-and-jammed loop whose remainder is handled
// by a DIFFERENT code path:
//
//	for ; c+4 <= k; c += 4 {
//		y0, y1, y2, y3 := y4[0], y4[1], y4[2], y4[3] // wide body: buffers passed in
//		…
//	}
//	for ; c < k; c++ {
//		ld[c], _ = m.logGaussian(x, c) // tail: reads the buffer off the RECEIVER
//	}
//
// The two are separate code paths that happen to compute the same thing, and every property
// established for the wide body has to be re-established for the tail. This shipped as a data
// race: parallelizing the caller required moving the wide body's scratch off the receiver, the
// tail still read the receiver, and the scan raced for any k not divisible by four. Nothing
// caught it because every benchmark and test used k=8 — the race detector cannot flag a line
// that never runs, and a parity test compares equal on a path that is empty.
//
// The properties that go stale in a tail are exactly the ones worth flagging for: per-worker
// scratch (a race), explicit FMA pinning (§NUM-FUSED-PATH-FMA — a one-ulp divergence on the
// remainder only), and hoisted bounds or dtype checks (a panic on the last elements).
//
// DELEGATION IS THE SIGNAL, not the presence of a tail. A tail that repeats the wide body
// inline shares its edits by construction, because they sit in the same text. A tail that calls
// a method inherits nothing. So this reports only when the remainder loop invokes a method on
// the receiver and the wide body does not — which is also what keeps it quiet on the ordinary
// scalar-arithmetic tail that most jammed kernels have.
//
// THIS RULE DOES NOT GO QUIET WHEN THE BUG IS FIXED, which is deliberate and unlike every other
// rule here. Threading the scratch through a parameter fixed the race; it did not remove the
// duplication, so the next property established for the wide body faces the same gap. Reporting
// it permanently is the point — it is a maintenance hazard attached to a shape, not a defect
// with a closing state. Suppressing once state is threaded was considered and rejected: telling
// "argument the wide body also uses as a buffer" from "argument it merely reads" needs alias
// analysis (the racy call passed x, which the wide body reads too), and an unsound suppression
// here hides a race.
func jamTailDelegatesFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || fn.Recv == nil {
		return nil
	}
	recv := ""
	if len(fn.Recv.List) > 0 && len(fn.Recv.List[0].Names) > 0 {
		recv = fn.Recv.List[0].Names[0].Name
	}
	if recv == "" {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		blk, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i := 0; i+1 < len(blk.List); i++ {
			wide, ok := blk.List[i].(*ast.ForStmt)
			if !ok {
				continue
			}
			stride, ok := jamStride(wide)
			if !ok || stride < 2 {
				continue
			}
			tail, ok := blk.List[i+1].(*ast.ForStmt)
			if !ok || tail.Body == nil {
				continue
			}
			// The tail must delegate to a receiver method while the wide body does not.
			if !callsReceiverMethod(tail.Body, recv) || callsReceiverMethod(wide.Body, recv) {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(tail.Pos()),
				end:      fset.Position(tail.Pos()),
				category: "jam-tail-delegates",
				msg: fmt.Sprintf("this remainder loop delegates to a method on %q while the loop jammed by %d above it is "+
					"inlined — two code paths for one computation. A STANDING HAZARD rather than a defect to close: "+
					"anything established for the wide body (per-worker scratch, explicit FMA pinning, a hoisted bounds "+
					"check) has to be re-established here, and a test at a trip count divisible by %d never executes it. "+
					"This exact shape shipped as a data race in GMM's full-covariance density kernel, where the wide body "+
					"took its scratch as a parameter and the tail still read the receiver. Threading the state through does "+
					"NOT silence this: the duplication remains and the next change faces it again. Whenever you touch the "+
					"wide body, test at a trip count NOT divisible by %d.",
					recv, stride, stride, stride),
			})
		}
		return true
	})
	return out
}

// jamStride returns N from a `for ; i+N <= bound; i += N` header, the unroll-and-jam idiom.
func jamStride(f *ast.ForStmt) (int, bool) {
	cond, ok := f.Cond.(*ast.BinaryExpr)
	if !ok || (cond.Op != token.LEQ && cond.Op != token.LSS) {
		return 0, false
	}
	add, ok := cond.X.(*ast.BinaryExpr)
	if !ok || add.Op != token.ADD {
		return 0, false
	}
	n, ok := intLit(add.Y)
	if !ok {
		return 0, false
	}
	// the post statement must advance by the same N
	as, ok := f.Post.(*ast.AssignStmt)
	if !ok || as.Tok != token.ADD_ASSIGN || len(as.Rhs) != 1 {
		return 0, false
	}
	if step, ok := intLit(as.Rhs[0]); !ok || step != n {
		return 0, false
	}
	return n, true
}

// intLit reads a non-negative integer literal.
func intLit(e ast.Expr) (int, bool) {
	bl, ok := e.(*ast.BasicLit)
	if !ok || bl.Kind != token.INT {
		return 0, false
	}
	v, err := strconv.Atoi(bl.Value)
	if err != nil {
		return 0, false
	}
	return v, true
}

// callsReceiverMethod reports whether body invokes a method on the named receiver.
func callsReceiverMethod(body *ast.BlockStmt, recv string) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		if id, ok := sel.X.(*ast.Ident); ok && id.Name == recv {
			found = true
		}
		return true
	})
	return found
}
