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
//	PS3009. ONE column of a slice-of-slices read through an INDIRECT row index (M[idx[k]][f]) —
//	   a gather with no nest to interchange; a feature-major copy makes the column contiguous
//	   (GBM exact split scan -7.74%/-6.55%, at the cost of one n*d*8 copy).
//	PS3008. a monotone (non-negative) accumulator tested against a threshold EVERY iteration —
//	   the test can run every 4th iteration for the same answer, dropping an unpredictable branch
//	   (DBSCAN leaf test -17.4%/-8.5%; the branch was 450ms against 30ms of arithmetic).
//	PS3007. a membership SET (map[K]bool/struct{}) built by ranging a slice, then probed in
//	   a loop — the source slice already answers the question, so a small set is faster scanned
//	   than hashed (applyDRY -18.72%, mapaccess1_fast64 gone; crossover 8–16 elements).
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
	{"PS1005", "manual-walk-dispatch", "a per-element AtF64/SetF64 whose 2+ index args are enclosing-loop variables — a manual multi-dim tensor walk via dispatch that PS1001 Numel-loop check misses. THE BIGGEST MEASURED PAYOFF IN THIS TOOL, and also one of the most share-dependent, so both numbers matter. classic SoftmaxRegression.PredictProba: an n*d SetF64 fill plus an n*k AtF64 copy-out, 94208 dispatches at n=4096, replaced by contiguous row copies — sec/op -48.67%/-45.57%/-55.53% across three shapes, geomean -50.10% (2.0x), allocs -97.95%. nlp QuantMamba2 prefill gated RMSNorm: seq*intermediate iterations each paying TWO dispatches, an AtF64 on a gain that does not vary with the outer variable plus a SetF64 recomputing a flat offset — removed entirely, and the result was geomean -0.61% with 3 of 4 cells indistinguishable, because the enclosing prefill is dominated by quantized matmul. Same transform, same class, an 80x difference in outcome. RANK BY (a) the dispatch count, the product of the loop bounds times dispatches per iteration, and (b) the loop SHARE of its enclosing function — a walk bolted to a matmul-dominated path will not move a wall clock however many dispatches it removes (PERF-HOTNESS-IS-NOT-SYNTAX-001, PERF-ISOLATED-RATIO-IS-NOT-A-SITE-001). THE COST OF THE FIX IS DECIDED BY ONE THING: whether the dtype is statically fixed. Where a constructor pins it (tensor.New(tensor.F64, ...) or tensor.New(tensor.F32, ...)) the fix is three lines — take Storage().F64()/F32() once and index it — with no fallback needed and none wanted (PERF-NO-FALLBACK-FOR-A-FIXED-DTYPE-001). Where the dtype comes from an input, tensor.New(xin.Dtype(), ...), a direct storage write needs a dual path, which is a bit-identity claim between two arms that PS6004 then reports forever unless a test reaches BOTH. That is the case at the remaining quant_mamba, quant_jamba and quant_deepseekv2 sites, and it is why they were left alone rather than converted. Read the construction before estimating the work. Bit-identity is otherwise free: AtF64 on an f32 tensor widens exactly and SetF64 on one stores float32(v), so the same expression in the same order over the typed slice gives the same result — a converted site can even be made TEXTUALLY identical to a sibling that already hoists, which is worth doing where a bit-match invariant between two paths is relied on", false},
	{"PS1006", "strided-inner-reduction", "a reduction whose INNER loop var is the high-stride (multiplied) part of a flat index ARR[inner*stride + outer] while the OUTER loop var is the contiguous (additive) part — the inner loop strides ARR by `stride` every step (cache-thrashing). Interchange to inner-outer/outer-inner so ARR is walked contiguously; per output element the reduction stays in the same order, so it is bit-identical. Shipped: MLA value-mix (cpu 1.13x / ref 1.27x), spectral-norm power-iter 2.57x (#592). WIN SCALES WITH `stride`×working-set: big when it EXCEEDS L2 (spectral-norm 512²), ~noise when it stays L1-resident — rank candidates by the strided dim's size. When the strided access sits inside a FUSED per-column O(seq²) scan that can't be interchanged (the outer var `c` is fixed per whole scan, e.g. an RWKV/attention recurrence), the remedy is GATHER/SCATTER: copy column c into contiguous scratch once (O(seq) strided), scan the scratch, scatter grads back once — same bit-exact remedy, shipped WKV backward 2.2-2.9x", false},
	{"PS1007", "output-row-restreamed", "an inner loop that accumulates into a loop-INVARIANT output row — OUT[d] += f(outer)*IN[outer*stride+d] — so all of OUT is loaded and stored once per OUTER step instead of once in total. TWO remedies, and WHICH ONE depends on whether IN is already contiguous in the inner var; both are bit-identical when OUT enters zeroed, since each element still sums the outer var ascending over the same terms with the same association. (a) IN is NOT contiguous in d (the d-strided gather, e.g. a sparse P·V reading IN[key*stride+d]): strip-mine the d loop by 4 and hold four partial sums in registers with the outer loop INNERMOST. Measured on the sparse P·V against the axpy form: MoBAAttention 57.20ms->46.07ms (1.24x), DSAAttention_seq1024 53.14ms->48.89ms (1.09x). (b) IN IS already contiguous in d (a row-major rank-1 update, IN[i*n+d] read as a row slice): do NOT strip-mine — that trades one contiguous pass over the whole submatrix for n/4 strided passes, and the measurement shows its gain DECAYING with the outer trip count (LstsqMat -2.56% at n=64, -1.92% at 256, -0.52% at 512, indistinguishable from base at 768). Instead unroll the OUTER loop by 2 and emit SEPARATE accumulating adds (`OUT[d] += v0*r0[d]` then `OUT[d] += v1*r1[d]`), so OUT[d] stays in a register across the pair while both input rows stay contiguous: -2.31% to -3.16% at every size, geomean -2.69% (linalg QR, shipped). Case (b) was found by measuring case (a)'s advice on a contiguous site and watching it decay. Also the MIRROR of PS1006, and interchange is NEVER the remedy here — the inner var is already the contiguous part. RANKING for case (b) is NOT about cache residency, and an earlier version of this text got that wrong: the loop visits outer*|OUT| elements and does 3 memory ops per visit (load IN, load OUT, store OUT), which an unroll by U cuts to 1+2/U — 2.0 at U=2, 1.5 at U=4 — so the saving is a fixed fraction of the loop's own traffic whether OUT is L1-resident or not. Both shipped sites confirm it: linalg QR unrolled by 2 with a 6 KB L1-resident accumulator, and nlp SnapKV unrolled by 4 with a 57 KB one. What actually decides the payoff is the loop's SHARE of runtime and the unroll factor the outer trip count allows: QR's accumulate is one of several phases in Lstsq and gave geomean -2.69%, while SnapKV's aggregate dominates its function and gave -21.08% at U=4 (-17.6% at U=2, so the third and fourth rows earned their registers). Rank by share-of-runtime x achievable U, not by whether OUT fits in cache. (c) SEVERAL output rows are ALREADY accumulated together (a GEMM band kernel holding c0..c3): case (b)'s refusal does NOT transfer, and this was measured after (b) predicted it would. Strip-mining the inner var by 4 there is not n/4 strided passes over one row — it is a Ux4 REGISTER TILE, and the output traffic it removes (one whole pass over U rows per OUTER step) dominates the input locality it costs. Measured on the portable GEMM band kernels, where IN is contiguous and (b) would have said no: f32 -31.06%%/-38.86%%/-34.17%% and f64 -41.51%%/-50.50%%/-37.84%% at 256/512/1024, all p<=0.002, with the gain GROWING with size instead of decaying as (b)'s LstsqMat curve did. The quantity that separates (b) from (c) is arithmetic intensity: one output row gives 4 FMAs per 5 loads, four rows give 16 per 8", false},
	{"PS1009", "unconverted-dtype-arm", "a per-element accessor walk sitting in one clause of a Dtype() switch whose SIBLING clause already takes typed storage — a fast path that covers some dtypes and leaves the rest on interface dispatch. PS1001 CANNOT REPORT THIS, structurally: it suppresses on hasFlat, which is satisfied by the sibling arm, so a helper that is fast for f64 and slow for everything else reads as already optimized. Which arm is hot is a workload fact, not a syntax one — nlp rows2D had an f64 copy arm and an accessor default, and every QUANTIZED model went through the default because activations are f32. Adding an f32 arm measured geomean -6.46% on the whole QuantMamba2 prefill (Q4_K_128 -6.34%, Q8_0_128 -6.56%, seq256 -4.79%, seq512 -8.09%, all p=0.000, 18 samples per arm) — a whole-benchmark win from one helper, since rows2D has 20+ call sites. Bit-identical where the missing arm is a WIDENING: AtF64 on an f32 tensor is exactly float64(v). NOT every arm is worth adding — f16/bf16 store as u16 and need a real conversion rather than a widening, so those clauses are legitimately left on the accessor; this check reports the population and the reader picks. Confirm which dtype the workload actually takes (a panic in the clause under the relevant benchmark settles it in one run) before converting. NARROWED AFTER SHIPPING, and the reason is the useful part: the first version reported 23 sites and an audit of every one found that ALL 23 already had both an f64 and an f32 arm, so once rows2D was fixed the check had zero actionable findings and 23 pieces of permanent noise. A switch that already covers F32 and leaves only its DEFAULT on the accessor is the FINISHED state - what remains there is f16/bf16 and the quantized dtypes, which live in u16 or packed storage and need a real conversion rather than an exact widening - so that shape is now suppressed. A NAMED case left on the accessor still fires, since listing a dtype and not converting it is a choice rather than an accepted tail. rows2D BEFORE its fix is the canonical hit: typed arms for f64 only, f32 falling through, every quantized model on that path", false},
	{"PS1010", "column-walk-slice-of-slices", "a nested loop over a SLICE OF SLICES whose INNER loop varies the ROW index — for j { for i { X[i][j] } } — so every step jumps a whole row, re-dereferences a row header and touches a fresh cache line to read eight bytes. The flat-array analogues (PS1006, PS6011) cannot see this: they match ARR[inner*stride + outer] index arithmetic, and X[i][j] has none. That blind spot was expensive — this check exists because the tool MISSED two measured wins in one session. classic GaussianNB.Fit computed its var-smoothing epsilon with 2*d such passes while every other loop in the same function was already row-major: -23.74% once folded into two row-major passes. classic ballTree.build chose its split dimension with one such pass per dimension, next to an enclose() that already walked rows contiguously: -18.71% on BenchmarkKNNFit once the lo/hi rode along in that pass. REPORTED ONLY WHEN INTERCHANGE IS PROFITABLE: the inner loop must assign to something that does not mention the inner variable — a scalar accumulator, or a slot indexed only by the outer one. A transpose (out[j][i] = in[i][j]) strides whichever way it is run and is excluded by that clause. Bit-identity is usually free, since interchanging preserves each accumulator's summation order when the accumulation is per-outer-index, but CONFIRM that per site rather than assuming it AMORTIZATION FILTER: an inner body that itself contains a loop is skipped, because the strided access then happens once against that loop's whole trip count. LU's elimination forced this — `mult := m[i][k]` reads a column beside a row-major `for j { ri[j] -= mult*pivRow[j] }` that does O(n) work for the same i — and without the filter the tree reported 46 sites, 19 of them amortized like that. The check finds the SHAPE, not the payoff: of its findings measured so far, classic GaussianNB.Fit paid -23.74% and linalg CholSolve's l[k][i] paid NOTHING (transposing L measured +3.74% at n=64, see PERF-TRANSPOSE-IS-NOT-FREE-001), so confirm at the site", false},
	// PS2xxx — allocation inside loops
	{"PS2001", "alloc-in-loop", "a tensor allocation inside a per-element loop", false},
	{"PS2002", "unsized-builder", "a strings.Builder/bytes.Buffer written in a loop with no .Grow", false},
	{"PS2003", "strings-alloc-in-loop", "an allocating strings transform (Replace/Map/Repeat) in a loop", false},
	{"PS2004", "poolable-loop-scratch", "per-call scratch make() bound to a non-escaping local in a pointer-method loop", false},
	{"PS2005", "regexp-compile-in-loop", "a regexp.Compile/MustCompile inside a loop", true},
	{"PS2006", "quadratic-cache-append", "a per-token cache slot reassigned to a concat of ITSELF and a new row — O(T\u00b2) copy traffic where an amortized row buffer is O(T)", false},
	{"PS2007", "build-nxn-use-one-row", "a call given the same size argument twice, whose square result is then read at exactly that position — an N×N object materialized to consume one row", false},
	{"PS2008", "per-row-make-slab", "a loop that allocates one slice per iteration into ARR[loopvar] — make([]T, len) with len LOOP-INVARIANT — where a single slab plus disjoint capped views would do. Replace with one make([]T, n*len) and ARR[i] = slab[i*len : (i+1)*len : (i+1)*len]. Bit-identical by construction: make() zeroes and so does a fresh slab, and no value, order or association changes. EXPECT NO SPEEDUP *WHEN THE LOOP BODY IS SUBSTANTIAL*, and that qualifier was added after a third site broke the unconditional form. Measured with no wall-clock movement: linalg QR's per-column reflectors -39.7% allocs/op and classic GMM's per-sample responsibility rows -62.2% (5751 to 1753 on GMMFit), both p>0.4. Measured WITH movement: classic GMM PredictProba's output rows, -11.17%% at k=4 and -7.30%% at k=8 (p=0.002) alongside allocs 605 to 94. What separates them is how much work the body does per allocation, and the cleanest evidence is INSIDE that third site rather than across sites: the same transform on the same function is -11.17%% at d=4 and INDISTINGUISHABLE at d=16, 32 and 64, where the density evaluation dominates and the allocation never was a meaningful share. Position in the loop nest matters for the same reason — per-ROW scratch can move the clock, per-WORKER scratch cannot, since it is amortized over the whole chunk (classic GMM's pooled worker buffers: allocs -63%%, time -0.61%% geomean). So: it is always a RESOURCE optimization — fewer allocations means less allocator work and less GC scan pressure on every machine, which is why it holds across systems — and it is ALSO a throughput one exactly when the allocation is a real fraction of the per-iteration work. Measure the enclosing operation rather than assuming either way. Bytes barely move: the [][]T header stays and one slab rounds to a size class (+0.1% observed). PRECONDITIONS the check cannot verify: rows must not be appended to (the 3-index cap makes that safe but it then reallocates, losing the benefit) and must not be individually replaced later, since the views are only valid while the slab is. Jagged rows are already excluded — a length that varies with the loop variable needs per-row offsets, a different transform. Cannot be ranked syntactically: the payoff scales with the ITERATION COUNT, which is a runtime fact (PERF-HOTNESS-IS-NOT-SYNTAX-001), so prefer sites whose loop bound is a data size over ones bounded by a small constant", false},
	// PS3xxx — indirection / reflection overhead
	{"PS3001", "reflection-in-loop", "a reflection-based fmt scan (Sscanf/Sscan/Fscanf) in a loop", false},
	{"PS3002", "closure-comparator-sort", "a package sort (sort.Slice/SliceStable) with a comparator closure", false},
	{"PS3003", "int-key-map-in-loop", "a read of an integer-keyed map inside a loop", false},
	{"PS3005", "indirect-key-comparator", "a sort of an index slice whose comparator dereferences the sorted element into a 2-D structure — hoist the key into a flat column first", false},
	{"PS3007", "set-map-from-slice", "a membership SET (map[K]bool / map[K]struct{}) built by ranging a slice and then probed inside a loop. PS3003 excludes set-shaped maps because a sparse set is not the dense [0,N) lookup a slice would replace — true of DENSIFICATION, but this is a different transform: when the set's contents come from a slice the caller already owns, the fix is no map at all. MEASURED: nlp applyDRY hashed its sequence-breaker set once per window position, with runtime.mapaccess1_fast64 at 1.14s of the function's 1.99s cumulative (57%% of its own time); scanning DRYBreakers directly took BenchmarkApplyDRY 19.52us to 15.87us, -18.72%% at p=0.002 (n=6, interleaved, both arm orders), allocations unchanged, and mapaccess1_fast64 left the profile. The same measurement bounds the transform: forced onto each arm the crossover is 8-16 elements on an M2 Pro, so this is a SMALL-SET fix and large sets should keep the map. Silent on a set written after its build loop (a mutable working set genuinely needs a map) and on a build already guarded by a size THRESHOLD on the source, which is code that has taken this advice already — but NOT on an emptiness guard (len(src) > 0), whose branch is the only path rather than a fallback. Hotness is not visible to the AST: confirm the source is small and the probe repeats, then benchmark", false},
	{"PS3008", "monotone-bail-per-element", "a loop accumulating a provably NON-NEGATIVE term (x*x with identical operands, math.Abs, math.Hypot, or a sum of those) into a scalar that is tested against a threshold on EVERY iteration. The accumulator never decreases, so once it passes the threshold it stays past it: testing every 4th iteration returns the SAME answer and removes a data-dependent branch the predictor cannot learn. MEASURED on classic ballTree.within, the leaf test DBSCAN runs per candidate pair, where a line-level profile put the branch at 450ms against 30ms for the subtraction and square it guarded — checking every 4th dimension gave BenchmarkDBSCANFit -17.41%% at eps=2 (p=0.000) and -8.51%% at eps=4 (p=0.010), geomean -13.07%%, allocations unchanged, exact-label goldens green. The non-negativity is the CORRECTNESS condition and is required syntactically, since a signed term can dip back under the threshold and moving its test would change the answer. Keep one accumulator in the same order so the sum stays bit-identical, and end the scalar tail with !(acc > thr) rather than acc <= thr — with a NaN term the original never bailed and returned its not-exceeded answer, and <= flips it. Silent once the loop strides by more than 1, which is the applied form. Hotness is not visible: benchmark the enclosing operation before restructuring a cold bail-out", false},
	{"PS3009", "indirect-column-gather", "a loop reading ONE column of a slice-of-slices through an INDIRECT row index — M[idx[k]][f] with f invariant — so every element lands in a different row, costing a row-header dereference and a cache line per eight bytes used. Same cache behavior PS1010 describes but NOT the same fix: PS1010 needs an interchangeable nest, and here the row order is a data-dependent permutation with no nest to swap. Keep a feature-major copy (xT[f*n+row]) instead. MEASURED on classic gbmBuilder.bestSplit, where the gather was 330ms of the function's 400ms: GBMHist_exact_80k -7.74%% (p=0.002), GBMFit -6.55%% (p=0.002), 20k -2.00%% (p=0.007), the win growing with n. TRADES MEMORY FOR TIME — the copy costs n*d*8 bytes, +26.7%% measured B/op at 80k x 20 — so weigh it against the gather's hotness rather than converting on sight. A reference implementation kept deliberately simple is the expected false positive", false},
	{"PS3006", "full-sort-take-topk", "a full sort.Slice/SliceStable (or slices.SortFunc/Sort) of a whole slice whose result is consumed ONLY through a bounded top-K prefix (s[:K] or a loop bounded by K reading s[r], K an identifier that is not len(s)) — an O(n log n) sort for an O(K) need. When K ≪ n, a bounded top-K selection (size-K min-heap or quickselect) then sorting just those is O(n log K), and it drops the O(n) sort-scratch alloc. Bit-identical when the comparator is a strict total order (a unique tiebreak → no genuine ties). Shipped: MemMemory.retrieveHead 2.98x. Silent when the slice is ALSO consumed in full (range s, s[:], s[:len(s)], a len(s) loop) or returned/passed whole. Confirm K ≪ n and the total-order tiebreak, then benchmark.", false},
	{"PS3004", "composite-key-map-probe", "a map indexed by a COMPOSITE LITERAL key — m[k{a,b}] or m[[2]T{a,b}] — which only a map can be, so the shape is unambiguous without type information. An array or struct key goes through Go's GENERIC hasher (one hash call per field, then a combine) rather than the specialized fast paths a plain string or int key gets, so it is the most expensive kind of map probe. Where the key domain is small and dense, a flat index replaces it outright. MEASURED: nlp's GGUF BPE encoder probed mergeRank[[2]string{left,right}] once per adjacent pair in the seed pass, one per input byte; a 65536-entry table indexed by the raw byte pair took BenchmarkBPEGGUFEncode 4.425ms to 2.735ms, -38.19% at p=0.000, with bytes unchanged. The sibling tiktoken path in bpe.go had already made exactly this change, its comment recording that the string hash 'dominated the profile (mapaccess2_faststr)' — and a [2]string key is WORSE than that, since it pays the generic struct hasher instead. HOTNESS IS NOT VISIBLE HERE and the check does not pretend otherwise: the site that mattered was not syntactically inside a loop at all, it was a small function CALLED from one, which no AST-only predicate can see. So this reports the whole population rather than guessing, and the population is small enough to read. Before converting, establish that the key domain really is dense and bounded — two enum-like fields or two bytes qualify, an arbitrary string pair does not — and that the probe is on a repeating path", false},
	// PS4xxx — vectorization candidates
	{"PS4001", "le-decode-in-loop", "a per-element little-endian bit decode in a loop with no bulk-copy fast path", false},
	{"PS4002", "scalar-transcendental-vectorizable", "a scalar libm transcendental in a loop while a vectorized sibling is called", false},
	{"PS4003", "transcendental-wrapper-in-loop", "a loop calls a helper that wraps a libm transcendental", false},
	{"PS4004", "scalar-copy-loop", "an element-by-element slice copy in a loop where a bulk copy would do. Reported in TWO strengths. ADVISORY when the index pattern only SUGGESTS a run: the message asks for the run length to be verified, because a wrong run is a wrong-value bug rather than a rounding one. PROVEN when the loop body is exactly that one assignment and both sides advance by exactly 1 per step over a base that does not move — then the bounds are computable, the whole loop IS the run, and one copy() replaces it. Measured on this host (M2 Pro, darwin/arm64) against the elementwise form, three run shapes x three sizes, copy() wins EVERY cell: full row 4.13x at n=64, 2.67x at 256, 2.51x at 512; upper-triangular (j from i) 3.16x/2.39x/2.35x; lower-triangular (j to i) 3.46x/2.27x/2.40x. The compiler does NOT rescue these — even the full-row case stays 2.5x off copy() when the index is written ARR[i*n+j] rather than as a bare range over the destination, so the memmove idiom does not apply. RANKING, which decides whether a site is worth touching: those ratios are for the LOOP IN ISOLATION, and all 4 sites converted when the proven classification was added moved end-to-end runtime by NOTHING — linalg Cholesky and QR assemble output in an O(n^2) loop inside an O(n^3) factorization (geomean -0.48% over six cells, all within spread), and the nlp Mamba2 prefill bias seed is one D-length run against a K*D convolution per timestep (six cells indistinguishable). So this is a WORK reduction that holds on every system — fewer instructions and fewer bounds checks for the same bytes moved, which is why those sites were still converted — but it shows in a wall clock only where the run is a meaningful SHARE of the enclosing function. Rank by that share, not by run length; PERF-HOTNESS-IS-NOT-SYNTAX-001 applies. PRECONDITIONS the AST cannot check: (1) element types must be IDENTICAL, not merely assignable — copy() rejects a []any destination fed a []string source, which the loop accepts; (2) if the two slices ALIAS with the destination offset ABOVE the source, an ascending element loop propagates values forward while copy() has memmove semantics and does not, so an in-place forward shift is the one shape to leave alone. SILENCE conditions: any call in the loop body (already fixed, or doing real per-element work), a same-identifier source and destination (a permutation, not a move), a conditional store (a filtered scatter), a sibling branch that already bulk-copies the same pair (the flagged loop is the strided arm that must stay), and — for the ADVISORY strength only — a loop that is not element-counted, which keeps rank-sized setup loops out. The counted-loop gate applies at BOTH strengths: exempting proven runs would cover four range-over-identifier sites but is indistinguishable from a rank loop (`range k` vs `range shape`), so the precision contract wins and those four are a recorded recall gap. A base that is not a bare identifier (a row of a slice-of-slices, l[i][j]) is matched ONLY at the PROVEN strength: allowing it advisorily added 12 findings to the tree and all 12 were gathers with no contiguous run — column gathers, permutation gathers through an index array, and bit-shift sources — so proving the stride is what separates the real shape from them", false},
	{"PS4005", "per-element-odometer", "an N-D coordinate odometer ticked once per element instead of once per run", false},
	{"PS4006", "row-slice-matrix", "a [][]T matrix built row-by-row and then indexed inside a nested loop", false},
	{"PS4007", "vjp-scalar-elementwise-binop", "a *VJP with a scalar single-op elementwise loop (dst[i]=a[i]∘b[i]) that a SIMD backend op would vectorize+parallelize, and no backend.Execute dispatch", false},
	{"PS4008", "serial-dot-matmul", "a matmul whose innermost loop is a serial scalar dot accumulator — latency-bound where an ikj/axpy form has independent accumulators", false},
	{"PS4009", "transposed-gram-colstride", "a symmetric-gram reduction M[k][i]·M[k][j] whose reduction index k is the OUTER (row) index of a row-major/jagged matrix — the innermost loop strides down a column across rows; reblock to k-outer rank-1 (load M[k] once, walk contiguously)", false},
	// PS5xxx — arithmetic
	{"PS5001", "loop-invariant-divide", "a divide by a loop-invariant scalar on every element. MOST FINDINGS SHOULD BE DECLINED. The reciprocal-multiply rewrite is NOT bit-identical and its speedup evaporates as soon as the loop touches memory, which an elementwise tensor loop always does. Measured on this host, 18 samples per arm interleaved in both orders: pure arithmetic with 8 independent chains and no memory 1303.5ns -> 933.7ns (-28.37%), the same arithmetic reading from an L1-resident slice 454.7ns -> 452.2ns (p=0.168, indistinguishable), a load-divide-store elementwise loop noise. The 1.2-1.5x quoted from SoftCap VJP and the optimizer moments is the memory-free ceiling. Accuracy cost, over 200k values: results differ for 66.3% of inputs at d=3.7, 35.4% at d=7, 24.5% across 50 random divisors, by up to 1 ulp — exact ONLY for a power-of-two divisor (0.000% differing at d=1024). So the usual trade is 1 ulp on a quarter to two thirds of outputs for nothing. Rank by whether the divide sits on a memory-free path; if it does not, decline", false},
	{"PS5004", "multi-sweep-fusable", "three or more consecutive loops over the same range each indexing a shared slice (fuse the passes into one sweep)", false},
	{"PS5002", "symmetric-accumulation", "a nested loop accumulating a full symmetric matrix (m[i][j] += x[i]*x[j]) where one triangle + mirror would halve the work", false},
	{"PS5003", "inner-invariant-recompute", "an inner-loop expression that varies with the INNER index but not the outer one — recomputed once per outer iteration where a precomputed row would do", false},
	{"PS5005", "loop-invariant-transcendental", "a pure libm transcendental (math.Pow/Exp/Log/Sin) whose args vary with the INNER index but not the outer one, recomputed every outer iteration", false},
	// PS6xxx — verification gaps
	{"PS6001", "full-sort-bounded-prefix", "a full-vocabulary descending index sort whose result feeds an early-breaking (threshold-bounded) prefix consumer, with no quickselect/pre-filter guard — an O(n log n) sort for an O(prefix) need", false},
	{"PS6002", "spatial-bounds-branch", "an innermost window/kernel loop re-testing a compound spatial bounds guard (iy>=0 && iy<h && ix>=0 && ix<wd) per tap, where the in-bounds taps form one contiguous run the guard can be hoisted around", false},
	{"PS6005", "monotone-index-bound", "an innermost loop guarding its tap work with a single relational bound on an index affine in the loop var (j:=t-(K-1)+k; if j>=0) — the in-bounds iterations are one contiguous run, clamp the loop bound instead of branching per tap", false},
	{"PS6023", "threshold-path-uncovered", "a package-level tuning constant that gates two code paths through a relational comparison, where NO test file in the package names it — so nothing PINS which arm the tests take. Sometimes the path is entirely unexercised (all three hand-found cases below were); sometimes it is covered incidentally by a test aimed at something else, which is weaker than it looks: the diagnosis misleads, and the coverage vanishes silently the moment the constant is retuned. Measured example of the second kind: both radix sorts behind nlp's radixSortCutoff turned out to be reachable from existing tests, but an ascending-radix STABILITY break surfaced only as a failure in TestDistQuickselectParity, a test named for something else, and nothing pinned the tie-break that diffusionRefillOrder documents. Found by hand three times in three packages before this check existed: linalg goldens ran 384 elements against a 65536 bound and a one-ulp perturbation of the parallel arm left them green; every SnapKV test used 2-3 rows against a group of 4; WandaPrune's fan-out needs 2+ panels and every test capped cout below the panel width, so making the worker scratch SHARED left all eleven Wanda tests passing with the race detector firing. REMEDY: one gate whose two arms are the SAME source selected by the threshold — never a separately written reference, which contracts to FMA differently and fails by an ulp on arithmetic that never changed. If the package's tests are EXTERNAL (package X_test) and the constant is unexported they cannot name it at all, so add an internal test file that asserts the geometry clears the bound, or select the arms through an exported knob such as GOMAXPROCS. VERIFICATION-GAP check: it reports missing evidence, not a defect — a hit is a test to write, never a code change. MEASURED CONSEQUENCE, and this check called it correctly before anyone acted on it: backend/cpu's matmulInlineWork, which decides whether an F64 matmul runs serial or fans out, sat in this check's output as gemm.go:95 while it went stale. The band kernels beneath it were register-tiled and became roughly twice as fast; fork/join cost did not change, so the crossover moved up and the untouched constant kept fanning out shapes that had stopped paying for it — measured at +37.26%% slower at the gate value itself, +22.85%% and +9.41%% just above it. Re-swept, the crossover had moved from 262144 to between 592704 and 681472. The finding was sitting here the whole time and was re-derived from first principles instead of read. TREAT THIS CHECK'S OUTPUT AS A LIVE WORKLIST: an untested threshold is not merely under-covered, it is a number nobody can re-sweep when the code it balances changes, and this repo has now had four of those go stale in one session", false},
	{"PS0001", "unused-ignore-directive", "a //perfscan:ignore directive that suppressed nothing — either the code it guarded moved out from under it (a directive reaches only its own comment block and the following line) or the finding is fixed and the comment should go. An inert suppression reads as though it took effect, which is worse than none", false},
	{"PS6022", "sort-feeds-truncation", "a slice sorted in FULL and then truncated to a smaller bound — every comparison that ordered the discarded tail was wasted, and a selection answers the same question in O(n). The counted-loop (PS6013) and threshold-break (PS6001) forms miss this one, because the consumer is neither a loop nor a break: it is a reslice", false},
	{"PS6021", "fanout-without-worker-seam", "a fan-out helper whose callback receives ONLY a single work index and no scratch, with no sibling scratch-constructor parameter — callers have nowhere to hoist a per-item buffer to, so every caller that needs a working buffer must allocate one PER ITEM; give the callback a scratch parameter or take a func() T constructor the helper calls once per worker", false},
	{"PS6019", "jam-tail-delegates", "an unroll-and-jammed loop (i+N <= bound) whose scalar remainder loop DELEGATES to a method on the receiver while the wide body is inlined — the two are separate code paths, so a fix applied to the wide one (per-worker scratch, a pinned product, a bounds hoist) silently misses the tail, and a test at a trip count divisible by N never executes it", false},
	{"PS6018", "layout-op-cluster-unfused", "three or more calls dispatching a pure DATA-MOVEMENT op (layoutOpConstants: slice, reshape, transpose, concat) in one function with no fused raw-storage path — movement cannot change a value, so gathering and scattering directly is bit-identical and removes every one of those dispatches", false},
	{"PS6017", "unpooled-variadic-sibling", "a VARIADIC helper called inside a loop at a fixed argument count, when the same package declares a non-variadic sibling with identical leading parameters and exactly that many trailing ones — the variadic form allocates a slice per call and the sibling exists to avoid it. MEASURED, in isolation, on nlp exec1 against its pooled siblings exec1a and exec2 (M2 Pro, darwin/arm64, nil Recorder so the pooled arm is taken): the pooling removes EXACTLY ONE allocation per call, 8 down to 7, with 392->384 B at one input and 432->416 B at two, and 179.8->177.6 ns (1 input) and 201.5->197.1 ns (2 inputs). So the win is real, bit-identical by construction (same op, same inputs, same attrs — only the slice provenance changes) and portable, but it is ONE allocation out of eight: the other seven come from the callee itself (output tensor, storage, shape and strides), so a whole-call ratio near 1-2% is the ceiling, not a starting point. RANKING, and this is what the count hides: all 118 findings in this tree sit on code that NO benchmark reaches. exec1 was probed with a counter under BenchmarkGemmaDecode, BenchmarkFalconDecode, BenchmarkCohereDecode, BenchmarkQuantDeepSeekV2ReconstructedDecode and BenchmarkT5Decode500RowBuf — every one confirmed to have executed — and the call count was ZERO in all five. The 97 nlp sites live in bert, blt, jamba, gemma2, granitemoe, dola and the unquantized deepseekv2 decode and prefill paths, none of which is benchmarked. So an end-to-end number for this check cannot be produced on this host without first writing the missing model benchmarks. Convert these as a portable resource win on the strength of the isolated measurement, or write the instrument first if a wall-clock claim is wanted — but do not read the finding COUNT as leverage. A 3-argument call has no sibling to move to (exec1/exec2 stop at two), which is one site here; the check reports it because the sibling test only needs SOME matching sibling, so confirm the exact arity exists before converting", false},
	{"PS6016", "loop-invariant-literal-arg", "a struct composite literal built INSIDE a loop and passed straight to a call, whose every field initializer is loop-invariant — the same value is rebuilt every iteration, and when the parameter is an interface it is heap-boxed every iteration; construct it once above the loop FALLBACK ARMS ARE SKIPPED: where an if/else has a branch reaching typed storage — in its body, its init guard or its condition — the else is the correctness path for whatever that branch declines, so a literal rebuilt there never runs on the path a benchmark takes. Verified rather than assumed: a panic in nlp quant_deepseekv2 attnReconstructed's else arm never fires under BenchmarkQuantDeepSeekV2ReconstructedPrefill or its Decode twin, both of which take the typed branch. That file went from 8 findings to 2, and the 2 that remain are the ConcatAttrs literals AFTER the if/else, on the common path — which is the behavior wanted. Tree-wide 117 to 111. PS1001 suppresses its equivalent via hasFlat and PS1009 via its dtype tail; this brings PS6016 in line, and PERF-FINDING-MAY-BE-THE-FALLBACK-ARM-001 records the pattern across all three ALL-CONSTANT LITERALS ARE ALSO SKIPPED, and the reason corrects this entry's own advice: go build -gcflags=-m reports a fully constant literal as escaping to heap, but it does not allocate — the compiler emits a pointer to a static read-only copy. Hoisting two backend.ConcatAttrs{Axis: 1} sites in nlp quant_deepseekv2, first PROVED reachable with a panic under both QuantDeepSeekV2Reconstructed benchmarks, changed allocs/op by zero with every sample equal, and was reverted. An isolated benchmark separates the cases: all fields constant is 0 allocs at 1.98ns, identical to a pre-boxed package var; ANY non-constant field is 1 alloc, 24 B and 11.4ns — and a loop-INVARIANT variable field costs the same 1 alloc as a varying one, which is why loop-invariance is still the right thing to key on once the constant literals are gone. Field constness decides, not the escape verdict. 111 findings to 91. A bare identifier is not treated as constant, since the AST cannot tell a const from a var and guessing would drop real findings RANKING, from auditing what survives: about 20% of struct literals in loops sit inside a constructor-like function (New*, *FromGGUF, *FromHF) and therefore run once per MODEL rather than per token — 8 of vision's 12 remaining findings are tensor.Randn weight initialization of exactly that kind. That is NOT filtered here, deliberately: unlike the fallback-arm and constant-field suppressions, which are structural facts, a name prefix is a guess that would misclassify any constructor called per request. Read the enclosing function before acting. The measured unit cost is 1 allocation, 24 bytes and about 9.5ns saved per occurrence, so a site needs a per-token or per-element trip count to be worth touching; vision's other 2 sit in a patch-embedding prologue an allocation profile sized at roughly 1% of ViT's objects", false},
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
	{"PS5008", "sincos-fusable", "a function calling BOTH math.Sin(x) and math.Cos(x) on the SAME argument expression — each does the full argument reduction of x independently; fuse to `sin, cos := math.Sincos(x)` (one reduction, both polynomials). Go's math.Sincos shares Sin/Cos's exact reduction+polynomials so it is bit-identical. Wins ONLY where trig DOMINATES the kernel (pure positional encoding, e.g. nn/sinusoidal.go +33% #587) — NOT where the trig is an amortized seq·half PRECOMPUTE feeding a larger matmul/attention: fusing MLA/self-extend RoPE (11 sites) was bit-exact but measured flat/within-noise because the seq²·heads score+value-mix dwarfs the trig (R-01KYRJ6RW1FJ0). Bench the ENCLOSING op, skip if trig <~10% of its work", false},
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
// ignoreAnchor records where a directive was written, so an unused one can be reported at the
// line the author will look at.
type ignoreAnchor struct {
	line       int    // the directive comment's own line — where an unused one is reported
	cat        string // the class it names, or "*" for a bare directive
	first, end int    // the suppression span this directive participates in
}

func ignoreDirectives(fset *token.FileSet, f *ast.File) (map[int]map[string]bool, []ignoreAnchor) {
	out := map[int]map[string]bool{}
	var anchors []ignoreAnchor
	for _, cg := range f.Comments {
		for _, c := range cg.List {
			// A DIRECTIVE, not a mention. The token must open the comment — strip the
			// leading marker, then any indentation — so that prose describing the feature
			// and the indented examples in this file's own package doc are not parsed as
			// live directives. Matching the token anywhere in the text made four doc
			// comments register suppressions, which was harmless only because no finding
			// ever landed on a doc-comment line; it stopped being harmless the moment
			// unused directives became reportable.
			body := strings.TrimLeft(strings.TrimPrefix(c.Text, "//"), " \t")
			if !strings.HasPrefix(body, "perfscan:ignore") {
				continue
			}
			rest := strings.TrimSpace(body[len("perfscan:ignore"):])
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
			lo, hi := first, last
			if lo > ln {
				lo = ln
			}
			if hi < ln {
				hi = ln
			}
			for cat := range cats {
				anchors = append(anchors, ignoreAnchor{line: ln, cat: cat, first: lo, end: hi + 1})
			}
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
	return out, anchors
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
	// PS6023 is FILE-level (it reports at a declaration) but reads PACKAGE-level facts the
	// pre-pass gathered, since a threshold is usually declared in one file and used in another.
	out = append(out, thresholdUncoveredFindings(fset, f)...)
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
	ign, anchors := ignoreDirectives(fset, f)
	used := map[int]map[string]bool{}
	var kept []finding
	for _, fd := range dedup(out) {
		if supp := ign[fd.pos.Line]; supp != nil && (supp["*"] || supp[fd.category]) {
			// Credit the directive that did the suppressing, so an unused one can be told
			// apart from a working one below.
			for _, a := range anchors {
				if !directiveCovers(a, fd.pos.Line) || !(a.cat == "*" || a.cat == fd.category) {
					continue
				}
				if used[a.line] == nil {
					used[a.line] = map[string]bool{}
				}
				used[a.line][a.cat] = true
			}
			continue
		}
		kept = append(kept, fd)
	}
	// REPORT DIRECTIVES THAT SUPPRESSED NOTHING. An inert suppression is worse than no
	// suppression, because it reads as though it took effect — the same reasoning that made
	// this file widen a directive's reach to its whole comment block after two directives
	// here were found dead. Widening reduced the failure mode; it cannot detect it. A
	// directive goes stale two ways: the code it guarded moved away from it (an edit inserted
	// statements between the comment and its target), or the finding was genuinely fixed. The
	// first is a silent hole, the second means the comment should be deleted. Both want the
	// author's attention, which is exactly how an unused lint suppression behaves.
	for _, a := range anchors {
		if used[a.line][a.cat] {
			continue
		}
		name := a.cat
		if name == "*" {
			name = "any check"
		}
		kept = append(kept, finding{
			pos:      token.Position{Filename: fset.Position(f.Pos()).Filename, Line: a.line},
			category: "unused-ignore-directive",
			msg: fmt.Sprintf("this //perfscan:ignore names %s but suppressed nothing. Either the code it"+
				" guarded moved out from under it — a directive reaches only its own comment block and the"+
				" line after, so inserting statements between the two silently voids it — or the finding is"+
				" fixed and the comment should be deleted. An inert suppression is worse than none: it reads"+
				" as though it took effect", name),
		})
	}
	return kept
}

// directiveCovers reports whether the directive reaches the given line. It MUST mirror the
// span ignoreDirectives applies — the enclosing comment block through the line after it — and
// not a tighter approximation.
//
// A tighter span was tried first, crediting only the directive's own line and the next, on the
// reasoning that over-crediting merely hides a stale directive while under-crediting is safe.
// That was backwards. Two directives stacked above one statement form a two-line block: the
// upper one is more than one line from the statement, so the tight span refused to credit it
// and reported a working directive as unused. Under-crediting produces false reports, which is
// the failure this check exists to prevent.
func directiveCovers(a ignoreAnchor, line int) bool {
	return line >= a.first && line <= a.end
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
					" per-element form as the strided/other-dtype fallback."+
					" MEASURED, and the ceiling is close to 2x because the dispatch IS the cost at a site"+
					" like this: the four RoPE/QKV weight permutations in nlp move every element with an"+
					" AtF64 read and a SetF64 store, and removing just the STORE half — half the dispatches,"+
					" nothing else touched — took all 8 benchmark cells down 44.98%% to 49.07%%, geomean"+
					" -47.64%% (deinterleaveRoPE, permuteInterleaveToSplit, permuteSplitToInterleave,"+
					" splitNeoXQKV; 18 samples per arm, p=0.000 everywhere). Those loops run out*in times —"+
					" 4.2M at [4096,1024] — and do nothing else, which is the shape that pays."+
					" A SITE CAN PAY TWICE, and then the ratio is far larger: nlp's rwkvShiftRows fed its"+
					" per-element stores from a rows2D() call that materialized the whole input as"+
					" [][]float64 first, so it paid the dispatches AND a second T*dim buffer. Routing it"+
					" through the package's existing typed bulk copy went -89.65%% and -93.39%% (geomean"+
					" -91.73%%, 12.1x) with bytes -50.06%% and allocs -25%%. Look for a helper the package"+
					" ALREADY has before writing a fast path: reusing a tested one avoids minting a new"+
					" bit-identity claim, and the per-element loop stays as its declined-dtype fallback."+
					" ASYMMETRY IS THE TRICK: convert only the side whose dtype is statically fixed. Each of"+
					" these builds its destination with tensor.New(tensor.F64, ...) in the same function, so"+
					" the store needs no fallback and none is wanted"+
					" (PERF-NO-FALLBACK-FOR-A-FIXED-DTYPE-001), while the SOURCE dtype comes from the file"+
					" being loaded and keeps its accessor — a typed read there would be a dual path needing a"+
					" test on both arms. Half the win, none of the risk."+
					" DO NOT GENERALIZE THE RATIO: the same transform on a walk inside a"+
					" matmul-dominated prefill measured -0.61%% (see PS1005). Rank by trip count AND by the"+
					" loop's share of its enclosing function. Note also that these four run at MODEL LOAD,"+
					" which no end-to-end benchmark exercises, so the instrument had to be written before"+
					" anything could be claimed (BENCH-PROVE-THE-CODE-RAN-001)", name, fn.Name.Name),
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
			// The advice below prescribes "keeping the accessor as the exotic-dtype fallback",
			// so an already-converted site still contains, by construction, the very loop this
			// arm matches. Reporting it re-files the applied fix as the defect, forever.
			if loopArgs >= 2 && !inDeclinedTypedFallback(parent, call) {
				out = append(out, finding{
					pos:      fset.Position(loop.Pos()),
					category: "manual-walk-dispatch",
					msg: fmt.Sprintf(".%s walks a tensor by explicit loop indices in %s() — an interface"+
						" dispatch + flat-offset recompute per element that PS1001's Numel-loop check"+
						" misses. Take the contiguous typed backing slice once (Storage().F64()/F32())"+
						" and index it directly, keeping the accessor as the exotic-dtype fallback."+
						" PAYOFF IS PER-SITE AND THE SPREAD IS 80x, so rank before converting. classic"+
						" SoftmaxRegression.PredictProba was an n*d SetF64 fill plus an n*k AtF64 copy-out,"+
						" 94208 dispatches at n=4096, and went geomean -50.10%% (2.0x) with allocs -97.95%%."+
						" nlp QuantMamba2's prefill gated RMSNorm removed just as many dispatches and went"+
						" geomean -0.61%%, 3 of 4 cells indistinguishable, because that loop sits inside a"+
						" quantized-matmul-dominated prefill. Rank by (a) the dispatch count, the product of"+
						" the loop bounds times dispatches per iteration, and (b) the loop's SHARE of its"+
						" enclosing function. THE COST OF THE FIX is decided by the dtype: a constructor-pinned"+
						" tensor.New(tensor.F64/F32, ...) makes it three lines with no fallback wanted"+
						" (PERF-NO-FALLBACK-FOR-A-FIXED-DTYPE-001), while a dtype taken from an input,"+
						" tensor.New(xin.Dtype(), ...), needs a dual path — a fresh bit-identity claim PS6004"+
						" then reports forever unless a test reaches BOTH arms.",
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
				// Positioned at the CALL, not the enclosing loop. dedup collapses findings sharing a
				// (position, category), so a loop holding two different batch-1 calls used to yield one
				// finding naming only the first — the second was invisible, not merely deduplicated.
				// rl.rlRollout was exactly that shape (actor forward plus critic forward) and PS6015's
				// doc comment recorded the gap without closing it. Per-call positioning also points at
				// the line a reader has to change.
				out = append(out, finding{
					pos:      fset.Position(call.Pos()),
					end:      fset.Position(call.End()),
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
			// Two more integer proofs, in the same spirit and for the same reason: on integer
			// operands the recommended inv := 1/n evaluates to ZERO, so a false positive here
			// costs correctness rather than a missed win.
			//
			// A divide that IS a loop bound must be integer — Go's for-condition compares an
			// index against it. And a quotient later used as a slice INDEX must be integer,
			// which the existing a[i/stride] guard misses because the quotient goes through a
			// variable first.
			//
			// COUNTED against all 89 hits before shipping (PROC-CHECK-PREDICATE-FIRST-001).
			// Ten are genuine integer divides. These two proofs catch three of them —
			// autograd/vjp.go's len(src)/inner loop bound, and nn/hqq.go and nn/qgalore.go
			// where a group index i/groupSize indexes a scale table — with ZERO false
			// positives. The other seven need type information: their quotients feed
			// attribute structs or offset arithmetic rather than a bracket index, and every
			// looser signal tried for them also swept up float sites, including a first
			// attempt that wrongly flagged six.
			//
			// WHY SHIP AT THREE INSTANCES when a perf rule at that count would be declined:
			// the asymmetry. Suppressing a wrong recommendation costs nothing, because the
			// recommendation was wrong; and the thing lost is a perf suggestion at sites where
			// it would have produced zero. For a correctness hazard, high precision at low
			// recall is the right trade — the opposite of a perf check, where low precision is
			// what makes it useless.
			if divideIsLoopBound(loop, n) || quotientIndexesASlice(loop, parent, n) {
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
					" once and multiply. MOST SITES SHOULD DECLINE THIS, and the numbers say why."+
					" CHECK THE OPERAND TYPE FIRST: with no type information this check cannot tell float"+
					" from integer, and on integer operands inv := 1/n is ZERO — a wrong-value bug, not a"+
					" missed win. SPEED: the divide is genuinely ~1.4x slower than the multiply, but only"+
					" where nothing else is going on. Measured on this host at three levels of realism,"+
					" 18 samples per arm interleaved in both orders: pure arithmetic with 8 independent"+
					" chains and NO memory, 1303.5ns -> 933.7ns (-28.37%%, p=0.000); the same arithmetic"+
					" reading its values from an L1-resident slice, 454.7ns -> 452.2ns (p=0.168,"+
					" INDISTINGUISHABLE); a plain load-divide-store elementwise loop, noise. A tensor"+
					" elementwise loop is the third case, so the expected win at a typical site here is"+
					" ZERO — the divide overlaps with the loads and stores around it. The 1.2-1.5x figures"+
					" quoted from SoftCap VJP and the optimizer moments are the memory-free ceiling, not"+
					" what a strided kernel will see. ACCURACY: the rewrite is NOT bit-identical and the"+
					" cost is larger than it looks — measured over 200k values, results differ for 66.3%%"+
					" of inputs at d=3.7, 35.4%% at d=7, 35.0%% at d=0.1, 24.5%% across 50 random divisors,"+
					" by up to 1 ulp (not half). It is EXACT only when the divisor is a power of two"+
					" (0.000%% differing at d=1024), since 1/d is then representable. So: pay 1 ulp on a"+
					" quarter to two thirds of your outputs for a win that is usually zero. SAFE ONLY for a"+
					" CONTINUOUS output (gradient/moment/probability) — NEVER feeding"+
					" round/quantize/argmax. Verify float + intent, and measure the enclosing loop before"+
					" believing the ceiling applies to it.", name, name),
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
			// The BASE may be an index/selector chain, not just a bare identifier: `out[i*n+j] =
			// l[i][j]` — a flat destination fed from a row of a slice-of-slices — is a real shape in
			// this repo (linalg Cholesky assembled its output that way) and an Ident-only base
			// silently dropped it. Distinctness is compared on the rendered base TEXT so in-place
			// permutations (out[...] = out[...], a transpose) stay excluded, and the ROOT identifier
			// is still required so the sibling-branch check below has a name to look for.
			dstBase, srcBase := srcText(fset, lhs.X), srcText(fset, rhs.X)
			dstRoot, ok1 := rootIdentName(lhs.X)
			srcRoot, ok2 := rootIdentName(rhs.X)
			// Distinctness is on the ROOT identifier, not the rendered base. Comparing base text
			// looked equivalent and was not: `dst[i][j] = dst[j][i]` renders bases `dst[i]` and
			// `dst[j]`, which differ, so a TRANSPOSE would have passed as a copy — the exact class
			// the original identifier comparison existed to exclude. Caught by the tree count
			// jumping 37 to 50 on a change that was supposed to add 0 sites beyond the four known
			// shapes.
			if !ok1 || !ok2 || dstBase == "" || srcBase == "" || dstRoot == srcRoot {
				return true
			}
			// A non-identifier base is admitted ONLY on the proven path, decided below. The advisory
			// path keeps the original bare-identifier requirement, and the reason is measured: relaxing
			// it for both strengths took the tree from 33 findings to 45, and every one of the 12 added
			// was noise — column gathers (`col[i] = x[i][f]`, `key[i] = scores[i][ex]`), permutation
			// gathers through an index array (`vals[k] = b.x[order[k]][f]`) and bit-shift sources
			// (`grid[e][b*4+k] = gridMap[(packed>>(2*k))&0x3]`). None of those has a contiguous run at
			// all, so the advisory message's "where the index pattern has a contiguous run" had nothing
			// to point at. What made the relaxation worth keeping is one real shape — a flat destination
			// fed from a row of a slice-of-slices, `out[i*n+j] = l[i][j]`, which linalg Cholesky used —
			// and that shape is always PROVEN, because proving the run is what tells it apart from a
			// gather in the first place.
			_, dstIsIdent := lhs.X.(*ast.Ident)
			_, srcIsIdent := rhs.X.(*ast.Ident)
			bareBases := dstIsIdent && srcIsIdent
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
			// PROVEN-RUN CLASSIFICATION, and it also relaxes the gate above. When the loop body is
			// EXACTLY this assignment and both sides advance by exactly 1 per step with a base that
			// does not move, the run is not a guess: its bounds are computable and the whole loop is
			// one copy(). That is the difference between this check's advisory form ("verify the run
			// length") and an actionable one, so the message says which it is.
			//
			// The single-statement requirement is load-bearing. A body that also writes something
			// else still contains a movable run, but the loop cannot be REPLACED by a copy, and
			// promising that would be a wrong-value bug rather than a rounding one.
			lv, lbody, haveVar := loopVarBody(loop)
			proven := false
			var runLo, runHi, dstRun, srcRun string
			if haveVar && lbody != nil && len(lbody.List) == 1 {
				dr, dok := unitStrideOffset(fset, as.Lhs[0], lv)
				sr, sok := unitStrideOffset(fset, as.Rhs[0], lv)
				if dok && sok {
					proven, dstRun, srcRun = true, dr, sr
					if lo, hi, ok := loopExtent(fset, loop, lv); ok {
						runLo, runHi = lo, hi
					}
				}
			}
			// The counted-loop gate keeps rank-sized setup loops out — `for a := range shape { eff[a]
			// = strides[a] }` is structurally a copy but runs 2-4 times, so a bulk copy there is
			// noise — and it applies at BOTH strengths, including proven runs.
			//
			// Exempting proven runs was tried and reverted. It is tempting because the advice is
			// correct whatever the trip count, and it would cover four range-over-identifier sites
			// (`for j := range k { codebooks[m*k+j] = cb[j] }` and three like it). But that shape is
			// INDISTINGUISHABLE from the rank loop: both range over a bare identifier, and the AST
			// cannot tell an element count from a slice of dimensions. TestDetectPS4004_SilentOnRankLoop
			// pins the exclusion, and trading a standing precision contract for four findings is the
			// wrong way round — especially since every measured conversion of this class moved
			// end-to-end runtime by nothing, so the four are low-value by this check's own numbers.
			// Recorded as a recall gap rather than closed.
			if !isCountedLoop(loop) {
				return true
			}
			if !proven && !bareBases {
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
			if siblingBranchBulkCopies(parent, loop, dstRoot, srcRoot) {
				return true
			}
			reportedCopy[loop] = true
			msg := fmt.Sprintf("%s[...] = %s[...] in a loop with no arithmetic on the value — an"+
				" element-at-a-time memmove. Where the index pattern has a contiguous run, hoist the"+
				" run out of the loop and move it with one copy() (a constant source index becomes a"+
				" fill). Bit-identical by construction: a same-dtype copy does no arithmetic, so there"+
				" is no accumulation order to change. Verify the run length before acting — a wrong"+
				" run is a wrong-value bug, not a rounding one.", dstBase, srcBase)
			if proven {
				where := "the loop's trip count"
				if runLo != "" || runHi != "" {
					where = fmt.Sprintf("exactly [%s, %s)", runLo, runHi)
				}
				msg = fmt.Sprintf("this loop's whole body is one unit-stride element move, so the LOOP"+
					" ITSELF is a contiguous run over %s: the destination advances from %s and the source"+
					" from %s, both by 1, with no arithmetic between them. Replace the entire loop with a"+
					" single copy(). Measured on this host against the elementwise form, copy() wins every"+
					" shape and size tried: full row 4.13x at n=64 down to 2.51x at 512, upper-triangular"+
					" 3.16x to 2.35x, lower-triangular 3.46x to 2.40x — the compiler does not turn"+
					" ARR[i*n+j] stores into memmove, so this is not a redundant rewrite. TWO"+
					" preconditions the AST cannot check: element types must be IDENTICAL rather than"+
					" assignable (copy() refuses a []any fed from []string), and if the slices ALIAS with"+
					" the destination offset above the source then the ascending loop propagates forward"+
					" while copy() does not — leave an in-place forward shift alone. RANK BY SHARE OF"+
					" RUNTIME: those ratios are for the loop in isolation, and all 4 sites converted when"+
					" this classification was added moved end-to-end runtime by nothing — an O(n^2)"+
					" assembly loop inside an O(n^3) factorization (linalg geomean -0.48%%, every cell"+
					" within spread) and a per-timestep bias seed against a K*D convolution (nlp, all"+
					" cells indistinguishable). It is a work reduction that holds on every system, so it"+
					" is still worth doing, but expect a wall-clock change only where the run is a real"+
					" share of the enclosing function.", where, dstRun, srcRun)
			}
			out = append(out, finding{
				pos:      fset.Position(loop.Pos()),
				category: "scalar-copy-loop",
				msg:      msg,
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
					" is not already a specialized fast path before acting",
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
	// a legitimate ragged structure and not a candidate. The per-row alloc is matched
	// whether the fill is direct (m[i] = make(...)) or indirect (r := make(...); m[i] = r),
	// and the two-deep index whether literal (m[i][j]) or via a hoisted row local
	// (row := m[i]; … row[j]) — the common idiom that lifts the row pointer once per outer
	// iteration and columns-walks it in the inner loop.
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
						" length — a genuinely ragged matrix cannot flatten. TRY THE CHEAPER FIX"+
						" FIRST: where the outer index is invariant across the inner loop, HOISTING"+
						" the row (qi := %s[i], then range qi) removes the same per-step pointer load"+
						" with no type change, no API change and no call-site churn, and ranging the"+
						" row rather than a separate extent lets the compiler drop a bounds check it"+
						" could not otherwise prove. Measured alone on nlp randomOrthogonal, an"+
						" O(d^3) Gram-Schmidt: -21.3%% (d=128/256/512, all p=0.002). Reach for the"+
						" full flatten when the hoist is not available or not enough.", name, name, name),
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
				" NEITHER: (1) the sort key is numeric, not a string — radix-on-float-bits does not"+
				" apply to a string key at all. A COMPOSITE OF NUMERICS IS NOT DISQUALIFYING, and"+
				" reading it as such is the easy mistake here: an LSD radix is stable, so passing over"+
				" the TIEBREAK key first and the primary key last yields exactly the lexicographic"+
				" composite order. It costs one pass per key, so a (score, index) comparator roughly"+
				" halves the payoff instead of removing it. classic spatialindex's (dist, idx) kNN sort"+
				" and nlp embed's (Score, Index) rerank both clear this axis and were declined on the"+
				" LENGTH one; (2) the slice is long (vocab-sized), not rank- or dimension-sized — on a"+
				" short slice the radix loses and the measurement is noise, and a k-nearest result is"+
				" k-length BY CONSTRUCTION however large the dataset behind it is, which is the trap"+
				" that makes a kNN sort look like a vocab-sized one. Confirm both by reading the site"+
				" before acting, then prove identical output order and benchmark.", sname)
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
	out = append(out, outputRowRestreamedFindings(fset, fn)...)
	out = append(out, serialDotMatmulFindings(fset, fn)...)
	out = append(out, scaledSerialDotFindings(fset, fn)...)
	out = append(out, quadraticCacheAppendFindings(fset, fn)...)
	out = append(out, buildNxNUseOneRowFindings(fset, fn)...)
	out = append(out, indirectKeyComparatorFindings(fset, fn)...)
	out = append(out, setMapFromSliceFindings(fset, fn)...)
	out = append(out, monotoneBailPerElementFindings(fset, fn)...)
	out = append(out, indirectColumnGatherFindings(fset, fn)...)
	out = append(out, innerInvariantRecomputeFindings(fset, fn)...)
	out = append(out, stridedInnerWalkFindings(fset, fn)...)
	out = append(out, inconsistentFMAPinningFindings(fset, fn)...)
	out = append(out, sortFeedsCountedPrefixFindings(fset, fn)...)
	out = append(out, sortFeedsTruncationFindings(fset, fn)...)
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
	out = append(out, unconvertedDtypeArmFindings(fset, fn, ns)...)
	out = append(out, compositeKeyMapProbeFindings(fset, fn)...)
	out = append(out, columnWalkFindings(fset, fn)...)
	out = append(out, reflectSwapperSortFindings(fset, fn)...)
	out = append(out, perRowMakeSlabFindings(fset, fn)...)
	out = append(out, vjpScalarBinopFindings(fset, fn)...)
	out = append(out, fullSortBoundedPrefixFindings(fset, fn)...)
	out = append(out, sortThenTopKFindings(fset, fn)...)
	out = append(out, spatialBoundsBranchFindings(fset, fn)...)
	out = append(out, monotoneIndexBoundFindings(fset, fn)...)
	out = append(out, sincosFusableFindings(fset, fn)...)
	return out
}

// divideIsLoopBound reports whether the divide expression appears in the loop's own
// post/condition clause, which forces it to be integer: a for-condition compares an index
// against it.
func divideIsLoopBound(loop ast.Node, div ast.Node) bool {
	f, ok := loop.(*ast.ForStmt)
	if !ok || f.Cond == nil {
		return false
	}
	found := false
	ast.Inspect(f.Cond, func(n ast.Node) bool {
		if n == div {
			found = true
		}
		return !found
	})
	return found
}

// quotientIndexesASlice reports whether the divide's result is assigned to a name that is
// later used as a slice index inside the same loop. That forces the quotient to be integer,
// and so both operands of the divide. The existing direct a[i/stride] guard cannot see this
// case, because the quotient passes through a variable first.
func quotientIndexesASlice(loop ast.Node, parent map[ast.Node]ast.Node, div ast.Node) bool {
	// Walk up to the assignment that stores the quotient.
	var as *ast.AssignStmt
	for p := parent[div]; p != nil; p = parent[p] {
		if a, ok := p.(*ast.AssignStmt); ok {
			as = a
			break
		}
		if _, ok := p.(*ast.BlockStmt); ok {
			break
		}
	}
	if as == nil || len(as.Lhs) == 0 {
		return false
	}
	name := ""
	if id, ok := as.Lhs[0].(*ast.Ident); ok {
		name = id.Name
	}
	if name == "" || name == "_" {
		return false
	}
	used := false
	ast.Inspect(loop, func(n ast.Node) bool {
		if used {
			return false
		}
		ix, ok := n.(*ast.IndexExpr)
		if !ok {
			return true
		}
		if id, ok := ast.Unparen(ix.Index).(*ast.Ident); ok && id.Name == name {
			used = true
		}
		return !used
	})
	return used
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
	collectThresholdUse(fset, parsed)
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
	// Locals bound to a 1-D make (`row := make([]T, …)`) — a single row buffer that the
	// fill loop may store into arr[i] indirectly (`row := make(...); …; arr[i] = row`)
	// rather than the direct `arr[i] = make(...)`. Both fills allocate one row per i.
	makeRow := map[string]bool{}
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
		if f, ok := call.Fun.(*ast.Ident); !ok || f.Name != "make" {
			return true
		}
		if at, ok := call.Args[0].(*ast.ArrayType); ok {
			if _, nested := at.Elt.(*ast.ArrayType); !nested { // 1-D []T row, not [][]T
				makeRow[id.Name] = true
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
		switch rhs := as.Rhs[0].(type) {
		case *ast.CallExpr: // arr[i] = make(...)
			if f, ok := rhs.Fun.(*ast.Ident); ok && f.Name == "make" {
				filled[id.Name] = true
			}
		case *ast.Ident: // arr[i] = row, where row := make([]T, …)
			if makeRow[rhs.Name] {
				filled[id.Name] = true
			}
		}
		return true
	})
	return filled
}

// rowAliases collects the local variables bound to a single-index ROW of name
// (`row := name[i]`, including within a tuple `row, pr := name[i], other[i]`). It lets
// nestedDoubleIndex also recognize the HOISTED-ROW form of a two-deep index — where the
// row pointer is lifted into a local and then indexed (`row[j]`) — which the literal
// m[i][j] matcher misses. A `_` binding is ignored.
func rowAliases(fn *ast.FuncDecl, name string) map[string]bool {
	locals := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok {
			return true
		}
		for k := 0; k < len(as.Lhs) && k < len(as.Rhs); k++ {
			lid, lok := as.Lhs[k].(*ast.Ident)
			ix, iok := as.Rhs[k].(*ast.IndexExpr)
			if lok && iok && lid.Name != "_" {
				if xid, ok := ix.X.(*ast.Ident); ok && xid.Name == name {
					locals[lid.Name] = true
				}
			}
		}
		return true
	})
	return locals
}

// nestedDoubleIndex reports the position of a two-deep index on name — either the literal
// m[i][j] or the hoisted-row form `row := m[i]; … row[j]` — occurring inside at least two
// nested loops — the region where the row-pointer dereference is paid repeatedly rather
// than once.
// It reports the position of the INNERMOST ENCLOSING LOOP, not of the index expression
// itself. That distinction is not cosmetic: `//perfscan:ignore` applies to a comment
// block and the statement below it, so a finding anchored to an expression one line
// INSIDE the loop cannot be suppressed by a directive written above the loop — which is
// where any reader puts it. PS4006 was unsuppressable that way until this changed, and
// only a bare directive failing to silence it revealed the cause. PS4005 and PS4008
// already anchor to the loop; this makes the family consistent.
func nestedDoubleIndex(fn *ast.FuncDecl, name string) (token.Pos, bool) {
	rowLocals := rowAliases(fn, name) // locals bound to name[i] — the hoisted-row form
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
				// literal m[i][j]
				if inner, isIdx := v.X.(*ast.IndexExpr); isIdx {
					if id, isID := inner.X.(*ast.Ident); isID && id.Name == name {
						found, ok = loop, true
						return
					}
				}
				// hoisted row: row[j] where `row := m[i]` earlier
				if id, isID := v.X.(*ast.Ident); isID && rowLocals[id.Name] {
					found, ok = loop, true
					return
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
		for _, s := range immediateInnerLoops(obody) {
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
		// Flatten the additive index into its terms so an intervening constant offset
		// — ARR[j*stride + off + o], the attention P·V / per-head-offset shape — matches,
		// not just the exact two-term ARR[j*stride + o]. Flag when SOME term is the inner
		// var strided (j*stride) and SOME term is the plain outer var o; extra terms (off)
		// are position-invariant and don't change the stride/contiguity analysis.
		terms := flattenAdd(ix.Index)
		if len(terms) < 2 {
			return true
		}
		var st string
		strided, contig := false, false
		for _, t := range terms {
			if s, ok := strideOperand(t, jName); ok {
				st, strided = s, true
			} else if isPlainIdent(t, oName) {
				contig = true
			}
		}
		if strided && contig {
			base, stride, found = baseID.Name, st, true
		}
		return !found
	})
	return base, stride, found
}

// outputRowRestreamedFindings flags PS1007 — the MIRROR of PS1006. An inner loop over d
// accumulates into an output row that is INVARIANT across the enclosing loop:
//
//	for j := range keys {          // outer
//	    w := scores[j]
//	    for d := 0; d < dk; d++ {  // inner, d is already the contiguous part
//	        out[d] += w * vs[j*dm+d]
//	    }
//	}
//
// Every one of the dk elements of `out` is loaded and stored once per OUTER step, so out is
// re-streamed len(keys) times when it only needs writing once. PS1006 deliberately stays
// SILENT here (its own silence fixture is this exact shape) because interchanging would put
// the strided access on the inside — the wrong direction. The fix is to strip-mine the d loop
// by 4 with the outer loop innermost, keeping four partial sums in registers: the input is
// still read four-contiguous from a single cache line, and out is written once per element.
//
// Bit-identical when out enters zeroed — each element still sums the outer var ascending over
// the same terms with the same association. If out is pre-seeded the register must be seeded
// from out[d] first.
//
// Requires the accumulated base to be a plain identifier indexed by EXACTLY the inner var. A
// base whose index also mentions the outer var writes a different location each outer step —
// a scatter, not a re-stream — and is not flagged. A base ASSIGNED inside the outer body (a
// per-iteration row slice) is likewise excluded for the same reason.
func outputRowRestreamedFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		ovs, obody, ok := loopBoundVarsBody(n)
		if !ok {
			return true
		}
		for _, s := range immediateInnerLoops(obody) {
			iName, ibody, ok := loopVarBody(s)
			if !ok || iName == "" {
				continue
			}
			skip := false
			for _, ov := range ovs {
				if ov == iName {
					skip = true
				}
			}
			if skip {
				continue
			}
			base, rhs, ok := rowAccumulatedAt(ibody, iName)
			if !ok {
				continue
			}
			// The accumulated value must depend on the OUTER loop, or the inner loop is
			// simply summing something the outer loop does not vary and the whole nest is
			// redundant work of a different kind.
			if !dependsOnOuterVars(rhs, ovs, obody) {
				continue
			}
			// Loop-INVARIANT across the outer loop: a base reassigned per outer step is a
			// scatter into distinct rows, which is not re-streamed.
			if identAssignedIn(obody, base) {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(n.Pos()),
				category: "output-row-restreamed",
				msg: fmt.Sprintf("inner %s-loop accumulates into %s[%s], and %s does not vary with the"+
					" enclosing %s loop — so every element of %s is loaded and stored once per %s step"+
					" instead of once in total. CHECK THE INPUT FIRST: if it is read as a row slice"+
					" contiguous in %s (a row-major rank-1 update), do NOT strip-mine — unroll the %s"+
					" loop by 2 with SEPARATE adds (%s[%s] += v0*r0[%s] then += v1*r1[%s]) so %s[%s]"+
					" stays in a register across the pair while both rows stay contiguous (shipped"+
					" linalg QR, geomean -2.69%%; strip-mining the same site decayed from -2.56%% at"+
					" n=64 to nothing at n=768). If the input is instead %s-STRIDED, strip-mine the %s"+
					" loop by 4 with the %s loop innermost (measured 1.24x MoBA / 1.09x DSA against"+
					" the axpy form). Either way bit-identical if %s enters zeroed (same ascending-%s"+
					" terms, same association). NOT PS1006 — %s is already the contiguous part, so"+
					" interchanging is never the remedy. Rank by the %s trip count: that is how many"+
					" times %s is re-streamed. BUT IF SEVERAL OUTPUT ROWS ARE ALREADY ACCUMULATED"+
					" TOGETHER here (a band kernel holding c0..c3), the contiguous-input refusal"+
					" above does NOT apply: strip-mining by 4 then builds a Ux4 REGISTER TILE whose"+
					" removed output traffic dominates the input locality it costs. Measured on the"+
					" portable GEMM bands, where the input IS contiguous: f32 -34.8%% and f64 -50.5%%"+
					" geomean, growing with size rather than decaying.",
					iName, base, iName, base, strings.Join(ovs, "/"), base, strings.Join(ovs, "/"),
					iName, strings.Join(ovs, "/"), base, iName, iName, iName, base, iName,
					iName, iName, strings.Join(ovs, "/"),
					base, strings.Join(ovs, "/"), iName, strings.Join(ovs, "/"), base),
			})
			return false
		}
		return true
	})
	return out
}

// loopBoundVarsBody is loopVarBody widened to every non-blank name a loop binds, and to the
// range VALUE. `for _, j := range act` — iterating a list of selected indices, the shape a
// sparse mask produces — binds j as Value with a blank Key, which loopVarBody rejects
// outright; PS1007 must see it, since that is the exact loop the measurement came from.
//
// KNOWN BOUNDARY, deliberately not widened: a `for ; cond; post` loop whose index was hoisted
// above it (`i := k` then `for ; i+2 <= m; i += 2`) has no Init and is NOT recognized, so
// PS1007 does not fire on it. That form is the idiom this repo uses for a strip-mined or
// unrolled loop — i.e. for code where the fix has ALREADY been applied — so the silence is
// usually right, but it is right by luck rather than by analysis. Widening this would make
// PS1007 nag about its own shipped remedies (linalg/qr.go is exactly such a site), which would
// need two more exclusions to suppress again: an inner body with 2+ accumulating adds into one
// base, and a remainder loop trailing an unrolled pair over the same variable. Recorded here
// so the next reader knows this is a chosen boundary and not an oversight. Widen it only
// together with those exclusions, and only once a genuine un-optimized instance turns up.
func loopBoundVarsBody(n ast.Node) ([]string, *ast.BlockStmt, bool) {
	var names []string
	add := func(e ast.Expr) {
		if id, ok := e.(*ast.Ident); ok && id.Name != "_" {
			names = append(names, id.Name)
		}
	}
	switch l := n.(type) {
	case *ast.RangeStmt:
		if l.Key != nil {
			add(l.Key)
		}
		if l.Value != nil {
			add(l.Value)
		}
		if len(names) == 0 {
			return nil, nil, false
		}
		return names, l.Body, true
	case *ast.ForStmt:
		if as, ok := l.Init.(*ast.AssignStmt); ok {
			for _, lhs := range as.Lhs {
				add(lhs)
			}
		}
		if len(names) == 0 {
			return nil, nil, false
		}
		return names, l.Body, true
	}
	return nil, nil, false
}

// rowAccumulatedAt finds a `BASE[iName] += RHS` in root where BASE is a plain identifier and
// the index is EXACTLY iName. Returns the base name and the accumulated expression.
func rowAccumulatedAt(root ast.Node, iName string) (string, ast.Expr, bool) {
	var base string
	var rhs ast.Expr
	found := false
	ast.Inspect(root, func(n ast.Node) bool {
		if found {
			return false
		}
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.ADD_ASSIGN || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		ix, ok := as.Lhs[0].(*ast.IndexExpr)
		if !ok {
			return true
		}
		id, ok := ix.X.(*ast.Ident)
		if !ok {
			return true
		}
		if !isPlainIdent(ix.Index, iName) {
			return true
		}
		base, rhs, found = id.Name, as.Rhs[0], true
		return false
	})
	return base, rhs, found
}

// dependsOnOuterVars reports whether e mentions one of the outer loop's variables, either
// directly or through a local assigned from one inside the outer body (the hoisted `w :=
// scores[j]` shape, which is the common form).
func dependsOnOuterVars(e ast.Expr, ovs []string, obody *ast.BlockStmt) bool {
	for _, v := range ovs {
		if exprMentions(e, v) {
			return true
		}
	}
	for _, st := range obody.List {
		as, ok := st.(*ast.AssignStmt)
		if !ok || len(as.Rhs) != 1 {
			continue
		}
		via := false
		for _, v := range ovs {
			if exprMentions(as.Rhs[0], v) {
				via = true
			}
		}
		if !via {
			continue
		}
		for _, lhs := range as.Lhs {
			if id, ok := lhs.(*ast.Ident); ok && exprMentions(e, id.Name) {
				return true
			}
		}
	}
	return false
}

// identAssignedIn reports whether name appears on the LHS of an assignment in root.
func identAssignedIn(root ast.Node, name string) bool {
	found := false
	ast.Inspect(root, func(n ast.Node) bool {
		if found {
			return false
		}
		as, ok := n.(*ast.AssignStmt)
		if !ok {
			return true
		}
		for _, lhs := range as.Lhs {
			if id, ok := lhs.(*ast.Ident); ok && id.Name == name {
				found = true
			}
		}
		return !found
	})
	return found
}

// Package-scoped state for PS6023. Keyed by directory, because a threshold is routinely
// DECLARED in one file and used as a bound in another (linalg's factorParThreshold is declared
// in solvecols.go and gates loops in qr.go, cholesky.go and linalg.go), so neither the
// declaration nor the gate is decidable from a single file.
var (
	thrGated     = map[string]bool{} // dir/name -> used in a relational comparison
	thrInTest    = map[string]bool{} // dir/name -> named by some _test.go in that dir
	thrExtOnly   = map[string]bool{} // dir -> every test file there is an external X_test package
	thrTestSeen  = map[string]bool{} // dir -> at least one test file was read
	thrProcsKnob = map[string]bool{} // dir -> some test there selects arms via GOMAXPROCS
)

// bailOnlyBlock reports whether every statement in b is a bail-out: a return, a panic, or a
// break/continue. Such a branch rejects its input rather than choosing an alternative
// computation, which is what separates a validation limit from a two-path performance gate.
func bailOnlyBlock(b *ast.BlockStmt) bool {
	if b == nil || len(b.List) == 0 {
		return false
	}
	for _, st := range b.List {
		switch t := st.(type) {
		case *ast.ReturnStmt, *ast.BranchStmt:
		case *ast.ExprStmt:
			call, ok := t.X.(*ast.CallExpr)
			if !ok {
				return false
			}
			id, ok := call.Fun.(*ast.Ident)
			if !ok || id.Name != "panic" {
				return false
			}
		default:
			return false
		}
	}
	return true
}

// collectThresholdUse is the PS6023 pre-pass. It records, per directory, which candidate
// threshold names are used as a bound and which are named by a test.
//
// It parses the sibling _test.go files ITSELF rather than relying on the scan set, because the
// default run excludes tests — and a check about test coverage that could only see test files
// when -tests was passed would silently report every threshold as uncovered on a normal run.
// Test files are read for identifier mentions only; nothing in them is ever reported.
func collectThresholdUse(fset *token.FileSet, files []*ast.File) {
	dirs := map[string]bool{}
	for _, f := range files {
		dir := filepath.Dir(fset.Position(f.Pos()).Filename)
		dirs[dir] = true
		ast.Inspect(f, func(n ast.Node) bool {
			// ONLY an if-condition counts. A relational use in a FOR condition is a loop
			// bound, not a choice between two code paths, and reporting one would ask for a
			// two-arm test where there is only one arm.
			is, ok := n.(*ast.IfStmt)
			if !ok || is.Cond == nil {
				return true
			}
			// A branch that only BAILS OUT — return, panic, break, continue, with no else —
			// is a validation limit, not a path gate: the alternative is an error, not a
			// second implementation of the same result, so there are no two arms to compare
			// and no bit-identity to prove. format/npy's maxHeaderLen and maxElems,
			// format/npz's maxEntries and format/pytorch's maxTensors are all this shape, and
			// an earlier version of this check asked all of them for an arm-selecting test.
			if is.Else == nil && bailOnlyBlock(is.Body) {
				return true
			}
			ast.Inspect(is.Cond, func(m ast.Node) bool {
				be, ok := m.(*ast.BinaryExpr)
				if !ok {
					return true
				}
				switch be.Op {
				case token.LSS, token.GTR, token.LEQ, token.GEQ:
				default:
					return true
				}
				for _, side := range []ast.Expr{be.X, be.Y} {
					ast.Inspect(side, func(q ast.Node) bool {
						if id, ok := q.(*ast.Ident); ok {
							thrGated[dir+"/"+id.Name] = true
						}
						return true
					})
				}
				return true
			})
			return true
		})
	}
	for dir := range dirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		external, internal := false, false
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), "_test.go") {
				continue
			}
			tf, err := parser.ParseFile(fset, filepath.Join(dir, e.Name()), nil, 0)
			if err != nil {
				continue
			}
			thrTestSeen[dir] = true
			if strings.HasSuffix(tf.Name.Name, "_test") {
				external = true
			} else {
				internal = true
			}
			ast.Inspect(tf, func(n ast.Node) bool {
				switch t := n.(type) {
				case *ast.Ident:
					thrInTest[dir+"/"+t.Name] = true
				case *ast.SelectorExpr:
					if t.Sel.Name == "GOMAXPROCS" {
						thrProcsKnob[dir] = true
					}
				}
				return true
			})
		}
		thrExtOnly[dir] = external && !internal
	}
}

// thresholdUncoveredFindings flags PS6023 at the DECLARATION of a tuning constant that gates two
// paths but is named by no test in its package. See the registry entry for why that matters and
// what the remedy is.
//
// Deliberately keyed on the constant being NAMED by a test rather than on whether some test
// geometry clears it. Whether a given test input exceeds a bound is not decidable from syntax —
// it depends on values computed at runtime — and incidental coverage is exactly what this check
// exists to distrust: it evaporates when the constant is retuned, without any test turning red.
// Naming the threshold is what makes the coverage provable and durable, and it is the shape the
// tests in this repo that DO cover their fast paths already use.
func thresholdUncoveredFindings(fset *token.FileSet, f *ast.File) []finding {
	dir := filepath.Dir(fset.Position(f.Pos()).Filename)
	if !thrTestSeen[dir] {
		return nil // no tests at all in this package: PS6023 has nothing to say that is specific
	}
	var out []finding
	for _, d := range f.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok || (gd.Tok != token.CONST && gd.Tok != token.VAR) {
			continue
		}
		for _, sp := range gd.Specs {
			vs, ok := sp.(*ast.ValueSpec)
			if !ok || len(vs.Values) != len(vs.Names) {
				continue
			}
			for i, name := range vs.Names {
				if !isTuningLiteral(vs.Values[i]) {
					continue
				}
				key := dir + "/" + name.Name
				if !thrGated[key] || thrInTest[key] {
					continue
				}
				remedy := "add a gate whose two arms are the SAME source selected by " + name.Name
				if thrExtOnly[dir] && !name.IsExported() {
					remedy = "this package's tests are all external (X_test) and " + name.Name +
						" is unexported, so no test CAN name it — add an internal test file that" +
						" asserts its geometry clears the bound, or select the arms through an" +
						" exported knob such as GOMAXPROCS"
					if thrProcsKnob[dir] {
						remedy += " (some test here already uses GOMAXPROCS, but nothing ties it to" +
							" this threshold)"
					}
				}
				out = append(out, finding{
					pos:      fset.Position(name.Pos()),
					category: "threshold-path-uncovered",
					msg: fmt.Sprintf("%s gates two code paths through a relational comparison, and no"+
						" test in this package names it — so nothing PINS which arm the tests take. The"+
						" guarded path may be entirely unexercised, or covered only incidentally by a"+
						" test aimed elsewhere, which misdiagnoses failures and disappears the moment"+
						" %s is retuned, with nothing turning red. Note one threshold can gate SEVERAL"+
						" paths: naming it in a test that covers one of them silences this check for"+
						" all of them, so cover every path it gates. %s. This reports MISSING"+
						" EVIDENCE, not a defect: the fix is a test, never a code change.",
						name.Name, name.Name, remedy),
				})
			}
		}
	}
	return out
}

// isTuningLiteral reports whether e is a compile-time numeric constant expression — a literal, a
// shift or arithmetic combination of literals, or a parenthesized one. It is what separates a
// tuning knob from configuration read at runtime: only the former is a threshold a test can
// reason about, and only the former stays fixed across a package's tests.
func isTuningLiteral(e ast.Expr) bool {
	switch t := e.(type) {
	case *ast.BasicLit:
		return t.Kind == token.INT || t.Kind == token.FLOAT
	case *ast.ParenExpr:
		return isTuningLiteral(t.X)
	case *ast.UnaryExpr:
		return isTuningLiteral(t.X)
	case *ast.BinaryExpr:
		return isTuningLiteral(t.X) && isTuningLiteral(t.Y)
	}
	return false
}

// immediateInnerLoops returns the loop statements nested in body that are not themselves
// inside a deeper loop — descending through if/else/block wrappers but STOPPING at the first
// loop on each path. This finds an inner reduction loop even when it is guarded by an `if`
// (the attention P·V `if sum > 0 { for j … }` shape), without reaching into deeper unrelated
// loop bodies (those are handled when their own enclosing loop becomes the outer candidate).
func immediateInnerLoops(body *ast.BlockStmt) []ast.Stmt {
	var loops []ast.Stmt
	var walk func(stmts []ast.Stmt)
	walk = func(stmts []ast.Stmt) {
		for _, s := range stmts {
			switch t := s.(type) {
			case *ast.ForStmt, *ast.RangeStmt:
				loops = append(loops, s)
			case *ast.IfStmt:
				if t.Body != nil {
					walk(t.Body.List)
				}
				switch e := t.Else.(type) {
				case *ast.BlockStmt:
					walk(e.List)
				case *ast.IfStmt:
					walk([]ast.Stmt{e})
				}
			case *ast.BlockStmt:
				walk(t.List)
			}
		}
	}
	walk(body.List)
	return loops
}

// flattenAdd splits an additive expression tree (a + b + c …) into its addends, recursing
// only through ADD binary ops. A non-add expression returns itself. Used so a strided index
// carrying a constant offset (ARR[j*stride + off + o]) is analyzed by its individual terms.
func flattenAdd(e ast.Expr) []ast.Expr {
	be, ok := e.(*ast.BinaryExpr)
	if !ok || be.Op != token.ADD {
		return []ast.Expr{e}
	}
	return append(flattenAdd(be.X), flattenAdd(be.Y)...)
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
		if loopIteratesArchitectureCount(n) {
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
				msg: fmt.Sprintf("loop dispatches %d backend ops per iteration (calls passing a"+
					" backend.Op* constant) and the function has no flatF64 fused path. If the trip"+
					" count is a SEQUENCE LENGTH, that is per-step dispatch and tiny-tensor alloc"+
					" overhead, and a raw-[]float64 fused path (storage grabbed once, state slice"+
					" reused, plain-Go recurrence) gated on ctx.Recorder==nil is usually bit-exact"+
					" — shipped DeltaNet/GatedDeltaNet 2.0-3.6x. VERIFY THE TRIP COUNT FIRST: this"+
					" check cannot see it, and an audit of every hit found the architectural"+
					" shapes (layer stacks, per-head and per-expert fan-outs) outnumbered genuine"+
					" recurrences by two orders of magnitude before the trip-count-origin filter"+
					" was added. Benchmark it.", count),
			})
			return false
		}
		return true
	})
	return out
}

// loopIteratesArchitectureCount reports whether the loop's trip count comes from a FIELD —
// `range m.Blocks`, `range cfg.Heads`, `range m.Experts`, or a three-clause loop bounded by
// `m.MaxRecursion`. Those are architecture counts: layer, head and expert collections sized by
// the model, on the order of tens, iterated once per forward. A sequence length, by contrast,
// arrives as a local or a parameter — `range seq`, `for t := 0; t < seq; t++`.
//
// THIS IS A SUPPRESSION, AND IT WAS COUNTED BEFORE IT WAS WRITTEN
// (PROC-CHECK-PREDICATE-FIRST-001). An audit classified all 110 hits of this check: ZERO were
// the sequential recurrence its message described, 57 were transformer layer stacks, and 35
// more were per-head, per-window or per-expert fan-outs. The genuine class exists — six loops
// in this tree, all carrying explicit scalar-or-small state across a `range seq` — and every
// one of them iterates a LOCAL, so none is suppressed here. This filter prunes 84 of the 110
// and keeps 6 of 6 on the real class.
//
// Four richer predicates were counted and REJECTED rather than shipped. Loop-carried state
// fires on every layer stack too — the residual is exactly that, so it is what the two shapes
// have in common. Requiring the carried value to feed two or more dispatches loses the
// canonical single-chain recurrence, dropping recall to four of six. Filtering on elementwise
// versus matmul ops has recall ZERO: 22 layer stacks show no visible matmul because theirs sit
// behind method calls, while all six real recurrences do show one. And matching a literal
// row-slice attribute has recall zero as well.
//
// An AST walker cannot tell `range m.Blocks` (a slice field) from `range cfg.Heads` (an int
// field), and does not need to: both are architecture counts and both belong on this side.
//
// The known cost: a recurrence written `for t := range m.Config.Ctx` would be missed. There
// are none in this tree today.
func loopIteratesArchitectureCount(n ast.Node) bool {
	switch v := n.(type) {
	case *ast.RangeStmt:
		_, isSel := ast.Unparen(v.X).(*ast.SelectorExpr)
		return isSel
	case *ast.ForStmt:
		cmp, ok := v.Cond.(*ast.BinaryExpr)
		if !ok {
			return false
		}
		if _, isSel := ast.Unparen(cmp.Y).(*ast.SelectorExpr); isSel {
			return true
		}
		_, isSel := ast.Unparen(cmp.X).(*ast.SelectorExpr)
		return isSel
	}
	return false
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
					" tolerance-gate against the sequential form. ~2x was measured on nn.LLMInt8MatMul, not"+
					" here; like PS4008 the gain turns on the contraction extent and cache residency,"+
					" which this check cannot see.", acc),
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
					" (2.9x per-MAC was measured on nn.matmulABt, not here; the gain turns on the contraction"+
					" extent and the operands' cache residency, neither of which this check can see)."+
					" Prove bit-identity against the current form with"+
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
// nn/sinusoidal.go PE builder (~15-22% on the trig-fill loop, +33% at op level #587)
// — a kernel that is NOTHING BUT trig. Verified NON-win: MLA/self-extend RoPE (11
// sites, backend/{cpu,ref}/mla.go + autograd/vjp_mla.go + nlp/self_extend.go) fused
// bit-exact (12.8M-arg ulp check clean) but benched flat/within-noise, because there
// the trig is a seq·half per-position PRECOMPUTE dwarfed by the seq²·heads score +
// value-mix (R-01KYRJ6RW1FJ0). Rule of thumb: ship only if trig is >~10% of the
// enclosing op's work. Argument matching is structural via exprEqual, so it is
// conservative — args differing only by a value conversion (float64(i)…) are not
// matched, avoiding false positives.
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

// sortThenTopKFindings flags PS3006 — a raw package sort of a WHOLE slice
// (sort.Slice/SliceStable(s,…) or slices.Sort/SortFunc/SortStableFunc(s,…)) whose result is
// consumed ONLY through a bounded top-K prefix: a reslice s[:K] or a loop bounded by an
// identifier K (≠ len(s)) that reads s[r]. That is an O(n log n) sort done for an O(K) need;
// a size-K min-heap or quickselect (then sorting just those K) is O(n log K) and drops the
// O(n) sort scratch. Bit-identical when the comparator is a strict total order. Shipped:
// MemMemory.retrieveHead (2.98×, #597).
//
// Consumption is analyzed only AFTER the sort's position (so the comparator closure's own
// s[i]/s[j] and the pre-sort fill are ignored). Kept SILENT — vetoed as full-use — when the
// sorted slice is also range-d, resliced s[:] / s[:len(s)], returned, or passed whole to a
// call, since the caller may then need the entire order.
func sortThenTopKFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	sortEnd := map[string]token.Pos{} // sorted slice → End() of its (last) sort call
	sortPos := map[string]token.Pos{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		c, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := c.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		pkg, ok := sel.X.(*ast.Ident)
		if !ok || len(c.Args) < 1 {
			return true
		}
		m := sel.Sel.Name
		isSort := (pkg.Name == "sort" && (m == "Slice" || m == "SliceStable")) ||
			(pkg.Name == "slices" && (m == "Sort" || m == "SortFunc" || m == "SortStableFunc"))
		if !isSort {
			return true
		}
		if s, ok := c.Args[0].(*ast.Ident); ok && c.End() > sortEnd[s.Name] {
			sortEnd[s.Name], sortPos[s.Name] = c.End(), c.Pos()
		}
		return true
	})
	if len(sortEnd) == 0 {
		return nil
	}
	after := func(name string, pos token.Pos) bool { e, ok := sortEnd[name]; return ok && pos > e }
	// makeSize[s] = the identifiers used as the len/cap of `s := make(…, x[, y])`. A prefix
	// bound that equals one of these means the slice is ALREADY that size (K == len(s), just a
	// different name — e.g. `heap := make([]T, 0, topM)` consumed as heap[:topM]); such a sort
	// is already O(K log K), not the full sort, so it must not be flagged.
	makeSize := map[string]map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		lhs, ok := as.Lhs[0].(*ast.Ident)
		if !ok {
			return true
		}
		call, ok := as.Rhs[0].(*ast.CallExpr)
		if !ok {
			return true
		}
		if id, ok := call.Fun.(*ast.Ident); !ok || id.Name != "make" {
			return true
		}
		for _, a := range call.Args[1:] { // skip the type arg
			if id, ok := a.(*ast.Ident); ok {
				if makeSize[lhs.Name] == nil {
					makeSize[lhs.Name] = map[string]bool{}
				}
				makeSize[lhs.Name][id.Name] = true
			}
		}
		return true
	})
	isMakeSize := func(slice, bound string) bool { return makeSize[slice] != nil && makeSize[slice][bound] }
	isLenOf := func(e ast.Expr, name string) bool {
		c, ok := e.(*ast.CallExpr)
		if !ok || len(c.Args) != 1 {
			return false
		}
		id, ok := c.Fun.(*ast.Ident)
		if !ok || id.Name != "len" {
			return false
		}
		a, ok := c.Args[0].(*ast.Ident)
		return ok && a.Name == name
	}
	bounded, full := map[string]bool{}, map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.SliceExpr:
			if id, ok := x.X.(*ast.Ident); ok && after(id.Name, x.Pos()) {
				hi, _ := x.High.(*ast.Ident)
				switch {
				case x.High == nil || isLenOf(x.High, id.Name):
					full[id.Name] = true // s[:] / s[low:] / s[:len(s)] — whole tail
				case hi != nil && isMakeSize(id.Name, hi.Name):
					// s[:K] where K is s's own make size → already K-sized, not a full sort
				default:
					bounded[id.Name] = true // s[:K]
				}
			}
		case *ast.RangeStmt:
			if id, ok := x.X.(*ast.Ident); ok && after(id.Name, x.Pos()) {
				full[id.Name] = true // range s
			}
		case *ast.ReturnStmt:
			for _, r := range x.Results {
				if id, ok := r.(*ast.Ident); ok && after(id.Name, r.Pos()) {
					full[id.Name] = true // return s (whole)
				}
			}
		case *ast.CallExpr:
			for _, a := range x.Args {
				if id, ok := a.(*ast.Ident); ok && after(id.Name, a.Pos()) {
					full[id.Name] = true // s passed whole to append/copy/f(s)
				}
			}
		}
		return true
	})
	// Bounded-loop form: `for r := range K { … s[r] … }` or `for …; r < K; … { s[r] }`, K an
	// identifier that is neither the sorted slice nor len(s) — reads only the top-K prefix.
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		var loopVar, boundName string
		var body *ast.BlockStmt
		switch x := n.(type) {
		case *ast.RangeStmt:
			if k, ok := x.Key.(*ast.Ident); ok && x.Value == nil {
				if b, ok := x.X.(*ast.Ident); ok {
					loopVar, boundName, body = k.Name, b.Name, x.Body
				}
			}
		case *ast.ForStmt:
			if c, ok := x.Cond.(*ast.BinaryExpr); ok && c.Op == token.LSS {
				lv, lok := c.X.(*ast.Ident)
				b, bok := c.Y.(*ast.Ident)
				if lok && bok {
					loopVar, boundName, body = lv.Name, b.Name, x.Body
				}
			}
		}
		if body == nil {
			return true
		}
		ast.Inspect(body, func(m ast.Node) bool {
			ix, ok := m.(*ast.IndexExpr)
			if !ok {
				return true
			}
			s, sok := ix.X.(*ast.Ident)
			iv, iok := ix.Index.(*ast.Ident)
			if sok && iok && iv.Name == loopVar && boundName != s.Name && after(s.Name, ix.Pos()) && !isMakeSize(s.Name, boundName) {
				bounded[s.Name] = true
			}
			return true
		})
		return true
	})
	var out []finding
	for name := range sortEnd {
		if bounded[name] && !full[name] {
			out = append(out, finding{
				pos:      fset.Position(sortPos[name]),
				category: "full-sort-take-topk",
				msg: fmt.Sprintf("full sort of %q but only a bounded top-K prefix is consumed — an"+
					" O(n log n) sort for an O(K) need. Replace with a size-K min-heap or quickselect"+
					" (then sort just those K); bit-identical when the comparator is a strict total"+
					" order (a unique tiebreak → no genuine ties). Shipped: retrieveHead 2.98x."+
					" Confirm K ≪ len(%s), then benchmark.", name, name),
			})
		}
	}
	sort.Slice(out, func(a, b int) bool { return out[a].pos.Offset < out[b].pos.Offset })
	return out
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

// KNOWN FALSE NEGATIVES, recorded because "PS6010 reports nothing here" must not be read as
// "there is no load to amortize". linalg.Lstsq's right-hand-side column loop is the measured
// case: jamming it four wide was worth -13.9% to -29.5% across the size sweep, and this check
// was silent on it. It fails three of the predicates above at once — the body calls AtF64, it
// carries two accumulators rather than one, and neither is stored to an index varying with the
// column. Those filters are what hold precision at 95.7%, so they stay.
//
// One of the three rationales does NOT survive that site, though, and the message repeats it:
// that a body containing a call is bottlenecked on the call, so the reload is already free. In
// Lstsq the call is a per-element AtF64 that a line-level profile puts at 30ms of 13.58s — 0.2%
// — while the reloaded operands were the top three lines at 61%. The exclusion is a precision
// tradeoff that loses real sites, not a statement about where the time goes.
//
// The shape this check cannot see is the wider one: an INDEPENDENT outer dimension whose whole
// body re-reads a large shared read-only array, where the fix replicates the body rather than
// one accumulator. No detector is proposed for it — see PERF-NO-CHECK-FOR-ROW-HOIST-001 and
// PERF-NO-CHECK-FOR-JAM-REMAINDER-001 for the two nearest predicates that were counted and
// declined on population.
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
		// A body that calls something the compiler will not fold away is bottlenecked on that
		// call, not on the reloaded operand, so blocking the loop buys nothing.
		//
		// COUNTED against all 65 hits before shipping (PROC-CHECK-PREDICATE-FIRST-001).
		// Removes 19, of which 18 are false positives — 13 are sites this repository itself
		// labels a generic fallback for exotic or mixed dtypes, reaching every element through
		// AtF64. Precision 69.2% -> 95.7% at 97.8% recall. The single genuine casualty is
		// backend/ref/mha_masked_backward.go, where math.Inf and math.IsInf sit in a mask
		// bail-out branch and never in the dot product.
		//
		// CONVERSIONS AND BUILTINS ARE NOT CALLS, and this is the whole reason the obvious
		// version of this predicate is wrong. Go models float64(x) as an *ast.CallExpr, so
		// "the body contains a call" removes 28 and loses TEN genuine sites — and seven of
		// those are the f32-widening twin of an f64 hit the same predicate keeps, the same
		// kernel in the same file differing only by float64(...) around each load. math.Inf
		// likewise compiles to a constant load and no call at all.
		if loopCallsNonTrivial(outBody) {
			return true
		}
		if f, isFor := n.(*ast.ForStmt); isFor && stridesByMoreThanOne(f.Post) {
			return true
		}
		// A body declaring MORE THAN ONE scalar float accumulator is not the
		// single-accumulator reload shape this check targets. Either the operand already feeds
		// several accumulators — which IS the recommended transform, so recommending it again
		// multiplies code paths, the hazard PS6019 reports — or the accumulators consume it
		// differently and nothing invariant is left to amortize. The two surviving false
		// positives are one of each: a NEON kernel tail already feeding t0..t3 from one load,
		// and a site whose two accumulators read the same operand through different indices.
		//
		// COUNTED AGAINST THE SURVIVORS, not the original population
		// (PROC-RECOUNT-AFTER-FILTERING-001). Against the pre-filter 65 hits this shape
		// appeared 3 times; against the 46 that survive the non-trivial-call exclusion it
		// appears twice, because one of the three was already removed there. Those two are
		// EXACTLY the two false positives left in the population: a NEON kernel's own
		// commented scalar column tail already feeding t0..t3 from one load, and a site where
		// the sole operand is booked as both shared and per-output because the index goes
		// through a variable. So this takes the check from 44 genuine of 46 to 44 of 44 —
		// zero genuine sites lost.
		//
		// The existing stride exclusion cannot reach either, because it inspects only the
		// flagged loop's own post statement, and in both cases the blocking lives on an
		// ENCLOSING loop or in the body's accumulator set rather than in this loop's stride.
		if len(scalarAccumulators(outBody)) > 1 {
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
					msg: fmt.Sprintf("accumulator %q re-reads an operand that does not vary with output index %q — unrolling"+
						" the output loop by 4 would let one load feed 4 accumulators (register blocking)."+
						" NOTHING HERE IS MEASURED, and two things must be checked before acting. Confirm the"+
						" load actually survives register allocation — build with -gcflags='<pkg>=-S' and count"+
						" loads of that operand in the loop body; a scalar the compiler already keeps in a"+
						" register gives nothing to amortize. And a body containing a call or a transcendental"+
						" is bottlenecked elsewhere, so the load is already free. The remainder path an unroll"+
						" needs is itself a hazard — see PS6019.", acc, outVar),
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
// loopCallsNonTrivial reports whether body calls something the compiler will not fold away.
//
// A predeclared type conversion and a builtin are deliberately NOT such calls. Go's AST models
// float64(x) as an *ast.CallExpr, and treating that as a call was measured to lose ten genuine
// sites out of forty-five — seven of them the f32-widening twin of an f64 site the same test
// keeps. len and cap likewise compile to a field load, not a call.
func loopCallsNonTrivial(body ast.Node) bool {
	trivial := map[string]bool{
		"float64": true, "float32": true, "int": true, "int8": true, "int16": true,
		"int32": true, "int64": true, "uint": true, "uint8": true, "uint16": true,
		"uint32": true, "uint64": true, "uintptr": true, "byte": true, "rune": true,
		"bool": true, "string": true, "complex64": true, "complex128": true,
		"len": true, "cap": true, "make": true, "new": true, "append": true,
		"copy": true, "delete": true, "min": true, "max": true, "clear": true,
		"panic": true, "print": true, "println": true, "recover": true,
		"complex": true, "real": true, "imag": true,
	}
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if id, isID := ast.Unparen(call.Fun).(*ast.Ident); isID && trivial[id.Name] {
			return true // a conversion or a builtin — keep descending into its arguments
		}
		found = true
		return false
	})
	return found
}

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

// columnWalkFindings flags PS1010 — a nested loop over a slice of slices whose INNER loop varies the
// ROW index, so consecutive iterations jump a whole row.
//
// WHY THIS IS NOT PS1006 OR PS6011. Those match a flat array indexed as ARR[inner*stride + outer] and
// reason about the stride arithmetic. A slice of slices has no such arithmetic — X[i][j] is two
// index operations — so both checks are structurally blind to it. The cost is the same or worse: a
// row-header dereference per step on top of the cache miss.
//
// THE INTERCHANGE CLAUSE is what keeps this from flagging every transpose in the tree. Interchanging
// only helps when the inner loop accumulates into something that does not depend on the inner
// variable, so the whole nest can be rewritten as one row-major pass. A transpose writes
// out[j][i], which mentions the inner variable, and strides whichever way it is run.
func columnWalkFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	seen := map[int]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		outer, obody, ok := loopIndexVar(n)
		if !ok || obody == nil {
			return true
		}
		ast.Inspect(obody, func(m ast.Node) bool {
			inner, ibody, ok := loopIndexVar(m)
			if !ok || ibody == nil || inner == outer {
				return true
			}
			if !assignsOutsideVar(ibody, inner) {
				return true // a transpose or an in-place shuffle: interchange buys nothing
			}
			// AMORTIZATION: if the inner loop's body contains a loop of its own, the strided access
			// happens ONCE per inner step against that nested loop's whole trip count, so it is not
			// what the nest costs. LU's elimination is the case that forced this — `mult := m[i][k]`
			// reads a column, but it sits beside `for j { ri[j] -= mult*pivRow[j] }`, which is
			// row-major and does O(n) work per i. Reporting it invites a rewrite of the cheap part.
			if containsLoop(ibody) {
				return true
			}
			ast.Inspect(ibody, func(k ast.Node) bool {
				ix, ok := k.(*ast.IndexExpr)
				if !ok {
					return true
				}
				col, ok := ix.Index.(*ast.Ident)
				if !ok || col.Name != outer {
					return true
				}
				row, ok := ix.X.(*ast.IndexExpr)
				if !ok {
					return true
				}
				rid, ok := row.Index.(*ast.Ident)
				if !ok || rid.Name != inner {
					return true
				}
				line := fset.Position(ix.Pos()).Line
				if seen[line] {
					return true
				}
				seen[line] = true
				out = append(out, finding{
					pos:      fset.Position(ix.Pos()),
					end:      fset.Position(ix.End()),
					category: "column-walk-slice-of-slices",
					msg: fmt.Sprintf("%s is read with the INNER loop varying the ROW index (%q inner,"+
						" %q outer), so every step jumps a whole row — a row-header dereference plus a"+
						" fresh cache line to use eight bytes of it. The inner loop assigns to something"+
						" that does not mention %q, so the nest is interchangeable: run rows outer and"+
						" accumulate all the outer-indexed targets in ONE contiguous pass."+
						" PS1006 and PS6011 cannot see this — they match ARR[inner*stride+outer]"+
						" arithmetic and a slice of slices has none, which is why this check exists:"+
						" the tool MISSED two measured wins of exactly this shape. classic"+
						" GaussianNB.Fit did 2*d such passes for its epsilon prepass, -23.74%% once"+
						" folded into two row-major passes; classic ballTree.build did one per"+
						" dimension beside an enclose() that already walked rows, -18.71%% on"+
						" BenchmarkKNNFit. Interchange usually preserves each accumulator's summation"+
						" order, but CONFIRM that at the site rather than assuming it.",
						srcText(fset, ix), inner, outer, inner),
				})
				return true
			})
			return true
		})
		return true
	})
	return out
}

// containsLoop reports whether a block contains a for/range statement, which for PS1010 means the
// strided access it holds is amortized against that loop's trip count rather than dominating.
func containsLoop(b *ast.BlockStmt) bool {
	found := false
	ast.Inspect(b, func(n ast.Node) bool {
		if found {
			return false
		}
		switch n.(type) {
		case *ast.ForStmt, *ast.RangeStmt:
			found = true
		}
		return !found
	})
	return found
}

// loopIndexVar returns the index variable a loop advances and its body. A range statement may bind
// the index as the KEY (for i := range xs) or as the VALUE (for _, i := range idx, where idx holds
// point ids) — the ball-tree scan that motivated this check uses the second form.
func loopIndexVar(n ast.Node) (string, *ast.BlockStmt, bool) {
	switch x := n.(type) {
	case *ast.RangeStmt:
		if id, ok := x.Value.(*ast.Ident); ok && id.Name != "_" {
			return id.Name, x.Body, true
		}
		if id, ok := x.Key.(*ast.Ident); ok && id.Name != "_" {
			return id.Name, x.Body, true
		}
	case *ast.ForStmt:
		if as, ok := x.Init.(*ast.AssignStmt); ok && len(as.Lhs) == 1 {
			if id, ok := as.Lhs[0].(*ast.Ident); ok && id.Name != "_" {
				return id.Name, x.Body, true
			}
		}
	}
	return "", nil, false
}

// assignsOutsideVar reports whether the body writes any target that never mentions v — a scalar
// accumulator, or a slot indexed only by the outer variable. That is the signal that interchanging
// the nest is profitable rather than merely moving the stride somewhere else.
func assignsOutsideVar(body *ast.BlockStmt, v string) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}
		var lhs []ast.Expr
		switch x := n.(type) {
		case *ast.AssignStmt:
			lhs = x.Lhs
		case *ast.IncDecStmt:
			lhs = []ast.Expr{x.X}
		default:
			return true
		}
		for _, l := range lhs {
			if !mentionsVar(l, v) {
				found = true
			}
		}
		return !found
	})
	return found
}

// compositeKeyMapProbeFindings flags PS3004 — a map probed with a composite-literal key.
//
// The predicate is sound without types, which is rare here: a composite literal is not a valid index
// for a slice or an array, so `X[T{...}]` can only be a map lookup. What it costs is that an array or
// struct key misses Go's specialized hashers and takes the generic one, a hash call per field plus a
// combine, making it the priciest probe shape available.
//
// DELIBERATELY NOT GATED ON BEING INSIDE A LOOP. The site that motivated this check — the GGUF BPE
// seed pass, worth -38.19% when replaced by a dense byte-pair table — sits in a four-line helper that
// a loop calls, not in the loop itself, so a loop gate would have missed the only measured win. The
// whole population in this tree is a handful of sites, small enough that reporting all of them and
// letting a reader judge hotness beats a predicate that silently drops the interesting one.
func compositeKeyMapProbeFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		ix, ok := n.(*ast.IndexExpr)
		if !ok {
			return true
		}
		cl, ok := ix.Index.(*ast.CompositeLit)
		if !ok || cl.Type == nil {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(ix.Pos()),
			end:      fset.Position(ix.End()),
			category: "composite-key-map-probe",
			msg: fmt.Sprintf("%s is probed with a COMPOSITE-LITERAL key (%s), so it is a map, and an"+
				" array or struct key takes Go's GENERIC hasher — one hash call per field plus a"+
				" combine — rather than the specialized path a plain string or int key gets. Where the"+
				" key domain is small and dense, a flat index removes the probe entirely. Measured:"+
				" nlp's GGUF BPE seed pass probed a map[[2]string] once per input byte, and a"+
				" 65536-entry table indexed by the raw byte pair took BenchmarkBPEGGUFEncode -38.19%%"+
				" (p=0.000). The sibling tiktoken encoder had already made that change, its comment"+
				" recording that the string hash dominated its profile — and a [2]string key is worse"+
				" still. HOTNESS IS NOT VISIBLE TO THIS CHECK: the measured site was not in a loop, it"+
				" was a helper CALLED from one, which is why there is no loop gate here. Confirm the key"+
				" domain is genuinely dense and bounded (two enum-like fields or two bytes qualify; an"+
				" arbitrary string pair does not) and that the probe repeats, before converting.",
				srcText(fset, ix.X), srcText(fset, cl.Type)),
		})
		return true
	})
	return out
}

// unconvertedDtypeArmFindings flags PS1009 — an accessor walk in one clause of a Dtype() switch
// whose sibling clause already reads typed storage.
//
// THE POINT IS WHAT PS1001 CANNOT SEE. That check suppresses whenever the enclosing function calls a
// configured fast-path helper anywhere (hasFlat), which is right when the fast path covers the hot
// case and the accessor is a rare fallback. A dtype switch breaks that assumption: the fast arm
// covers ONE dtype and every other dtype still pays a dispatch per element, and which arm is hot
// depends on the workload rather than on the source. nlp's rows2D was the case that proved it — an
// f64 copy arm and an accessor default, with every quantized model taking the default because
// activations are f32. PS1001 was silent; adding the f32 arm was worth 6.46% of a whole prefill.
//
// Reports the clause, not the switch, so a switch with two unconverted arms yields two findings.
// Deliberately advisory: an unconverted arm is not automatically worth converting, since f16/bf16
// live in u16 storage and need a genuine conversion rather than the exact widening that makes the
// f32 case a one-liner.
func unconvertedDtypeArmFindings(fset *token.FileSet, fn *ast.FuncDecl, ns nameSets) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		sw, ok := n.(*ast.SwitchStmt)
		if !ok || !isDtypeSwitch(sw) {
			return true
		}
		typed, cheapCovered := false, false
		var slow []*ast.CaseClause
		for _, st := range sw.Body.List {
			cc, ok := st.(*ast.CaseClause)
			if !ok {
				continue
			}
			if clauseReadsTypedStorage(cc) {
				typed = true
				if clauseMentionsCheapDtype(cc) {
					cheapCovered = true
				}
			} else if clauseAccessorName(cc, ns) != "" {
				slow = append(slow, cc)
			}
		}
		if !typed {
			return true // no fast arm at all — that is PS1001's domain, not this one
		}
		// SUPPRESS THE ACCEPTED TAIL. A switch that already has an F32 arm and leaves only its
		// `default` on the accessor is in the FINISHED state, not an unfinished one: what remains
		// there is f16/bf16 and the quantized dtypes, which live in u16 or packed storage and need a
		// genuine conversion rather than the exact widening that makes an f32 arm a one-liner.
		//
		// This was measured, not assumed. Auditing all 23 sites this check reported found that every
		// single one already covered both f64 and f32, so after the rows2D fix the check had ZERO
		// actionable findings and would have emitted 23 pieces of noise forever. rows2D BEFORE that
		// fix is the shape worth firing on — typed arms for f64 only, f32 falling through to the
		// accessor, and every quantized model taking that path — and it still fires here.
		if cheapCovered && onlyDefaultIsSlow(slow) {
			return true
		}
		for _, cc := range slow {
			kind := "case"
			if cc.List == nil {
				kind = "default"
			}
			out = append(out, finding{
				pos:      fset.Position(cc.Pos()),
				category: "unconverted-dtype-arm",
				msg: fmt.Sprintf("this %s clause walks elements with .%s while a SIBLING clause of the same"+
					" Dtype() switch already reads typed storage — the fast path covers some dtypes and"+
					" leaves the rest on interface dispatch. PS1001 cannot report this: it suppresses"+
					" whenever the function has any fast path, and the sibling arm satisfies that, so a"+
					" helper fast for one dtype and slow for the others reads as already optimized."+
					" Which arm is HOT is a workload fact — nlp rows2D had exactly this shape and every"+
					" QUANTIZED model took the accessor default, because activations are f32; adding an"+
					" f32 arm measured geomean -6.46%% across the whole QuantMamba2 prefill (all 4 cells"+
					" p=0.000) from one helper with 20+ call sites. Bit-identical when the missing arm is"+
					" a WIDENING — AtF64 on an f32 tensor is exactly float64(v). Not every arm earns one:"+
					" f16/bf16 store as u16 and need a real conversion, so leaving those on the accessor"+
					" is correct. Establish which dtype your workload takes — a panic in this clause under"+
					" the relevant benchmark settles it in one run — before converting.",
					kind, clauseAccessorName(cc, ns)),
			})
		}
		return true
	})
	return out
}

// clauseMentionsCheapDtype reports whether a case label names a dtype whose conversion to float64 is
// an exact WIDENING, which is what makes converting that arm a one-liner. F32 is the only such dtype
// besides F64 itself: f16 and bf16 live in u16 storage and need real conversion code.
func clauseMentionsCheapDtype(cc *ast.CaseClause) bool {
	for _, e := range cc.List {
		if sel, ok := e.(*ast.SelectorExpr); ok && sel.Sel.Name == "F32" {
			return true
		}
		if id, ok := e.(*ast.Ident); ok && id.Name == "F32" {
			return true
		}
	}
	return false
}

// onlyDefaultIsSlow reports whether every unconverted clause is the `default` one.
func onlyDefaultIsSlow(slow []*ast.CaseClause) bool {
	for _, cc := range slow {
		if cc.List != nil {
			return false
		}
	}
	return true
}

// isDtypeSwitch reports whether the switch tag is an X.Dtype() call.
func isDtypeSwitch(sw *ast.SwitchStmt) bool {
	call, ok := sw.Tag.(*ast.CallExpr)
	if !ok {
		return false
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	return ok && sel.Sel.Name == "Dtype"
}

// clauseReadsTypedStorage reports whether the clause reaches the backing store through
// Storage().<Typed>() — the shape that makes it a converted arm.
func clauseReadsTypedStorage(cc *ast.CaseClause) bool { return readsTypedStorage(cc) }

// inDeclinedTypedFallback reports whether n sits in the arm a typed fast path DECLINED: the else
// of an if whose taken side reaches Storage().<Typed>(), or the default of a switch that has a
// converted case. Such an arm is the exotic-dtype correctness path — precisely what the fix these
// checks recommend LEAVES BEHIND — so reporting it flags the applied fix as the defect. PS1001,
// PS1009 and PS6016 already suppress their own equivalents; this is the shared form.
//
// THE SIBLING-ARM TEST IS THE WHOLE POINT, and a per-function one is not a weaker version of it but
// a wrong one. "The enclosing function mentions Storage() somewhere" flags 16 of the 84 PS1005
// sites, and reading them shows the coarse rule is right for the wrong reason and sometimes simply
// wrong: nlp jlens builds a typed DESTINATION slice and then walks a DIFFERENT, provably non-F64
// source by accessor inside the same function. That walk is not a fallback, and the identity fill
// a few lines below it is a genuine site the coarse rule would have swallowed with it.
func inDeclinedTypedFallback(parent map[ast.Node]ast.Node, n ast.Node) bool {
	for cur := n; cur != nil; cur = parent[cur] {
		switch s := parent[cur].(type) {
		case *ast.IfStmt:
			// Only the ELSE side is inert. The taken side is the fast path itself, and an
			// accessor loop there would be a real finding.
			if s.Else == cur && (readsTypedStorage(s.Body) ||
				(s.Init != nil && readsTypedStorage(s.Init)) ||
				(s.Cond != nil && readsTypedStorage(s.Cond))) {
				return true
			}
		case *ast.CaseClause:
			// A default clause carries no expression list; a converted sibling is any other
			// clause that reaches typed storage.
			if len(s.List) == 0 && switchHasConvertedClause(parent, s) {
				return true
			}
		}
	}
	return false
}

// switchHasConvertedClause reports whether the switch owning def has some OTHER, non-default clause
// that reaches typed storage — the sibling that makes def the declined arm.
func switchHasConvertedClause(parent map[ast.Node]ast.Node, def *ast.CaseClause) bool {
	// The parent map is built from the walk stack, so a BlockStmt sits between the switch and
	// its clauses; walk up until the switch itself appears.
	var body *ast.BlockStmt
	for p := parent[def]; p != nil; p = parent[p] {
		if sw, ok := p.(*ast.SwitchStmt); ok {
			body = sw.Body
			break
		}
		if sw, ok := p.(*ast.TypeSwitchStmt); ok {
			body = sw.Body
			break
		}
		if _, ok := p.(*ast.BlockStmt); !ok {
			return false
		}
	}
	if body == nil {
		return false
	}
	for _, st := range body.List {
		cc, ok := st.(*ast.CaseClause)
		if !ok || cc == def || len(cc.List) == 0 {
			continue
		}
		if readsTypedStorage(cc) {
			return true
		}
	}
	return false
}

// readsTypedStorage reports whether a subtree reaches the backing store through
// Storage().<Typed>(). Shared by PS1009, which uses it to recognize a converted dtype arm, and by
// PS6016, which uses it to recognize the arm a fallback sits behind.
func readsTypedStorage(root ast.Node) bool {
	found := false
	ast.Inspect(root, func(n ast.Node) bool {
		if found {
			return false
		}
		sel, ok := n.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		inner, ok := sel.X.(*ast.CallExpr)
		if !ok {
			return true
		}
		if s2, ok := inner.Fun.(*ast.SelectorExpr); ok && s2.Sel.Name == "Storage" {
			found = true
		}
		return !found
	})
	return found
}

// clauseAccessorName returns the first configured per-element accessor called in the clause.
func clauseAccessorName(cc *ast.CaseClause, ns nameSets) string {
	name := ""
	ast.Inspect(cc, func(n ast.Node) bool {
		if name != "" {
			return false
		}
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok && ns.accessors[sel.Sel.Name] {
			name = sel.Sel.Name
		}
		return name == ""
	})
	return name
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
					msg: fmt.Sprintf("%q is allocated inside a parallel body — once per DISPATCH, so whether"+
						" this is free or ruinous depends ENTIRELY on how often the dispatch runs, which the"+
						" check cannot see. Both sides were measured in this repo. Dispatch-once sites pay"+
						" nothing: AQLM went 49 to 51 MB and GMM 4 to 4 MB, one dispatch per encode pass or"+
						" EM iteration. The GBM exact grower dispatches once per TREE NODE, thousands of"+
						" times per fit, and the identical code shape took it from 64 MB / 883 allocs to"+
						" 2007 MB / 8965 — a 31x memory regression that shipped hidden behind a 2.80x"+
						" speedup because the commit reported only ns/op. So find the call frame that"+
						" repeats. If the dispatch runs once per call and the enclosing function is not"+
						" itself in a loop, this is per-worker scratch bounded by GOMAXPROCS and is"+
						" SANCTIONED — accept it and report both numbers"+
						" (PERF-PER-WORKER-ALLOCS-ARE-BOUNDED-001); most findings here are that case. If the"+
						" dispatch repeats, hoist to one buffer per chunk on the receiver and select it with"+
						" the chunk index. Report B/op and allocs/op either way, since ns/op alone hid the"+
						" only regression this check exists to catch",
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
		want := map[string]string{"Slice": "Sort", "SliceStable": "SortStable"}[sel.Sel.Name]
		risk := " slices." + want + "Func passes VALUES where sort." + sel.Sel.Name +
			" passes INDICES, so re-derive the comparator rather than transcribing it: here the" +
			" element is not an index into anything, so a transcribed version will not compile and" +
			" the compiler catches the mistake for you."
		if outer, inner, ok := permutationSortComparator(call); ok {
			risk = " THIS ONE IS AN INDEX-PERMUTATION SORT and the conversion is SILENT if done" +
				" wrong. The comparator reads " + outer + "[" + inner + "[param]], so " + inner +
				" holds integer indices. sort." + sel.Sel.Name + " passes POSITIONS while slices." +
				want + "Func passes ELEMENTS, and because both are ints a transcribed " + outer +
				"[" + inner + "[x]] still COMPILES while sorting by a meaningless key — it must" +
				" become " + outer + "[x]. Two occurrences of exactly this shape were caught only" +
				" by deliberately applying the wrong version and watching a gate fail; in one of" +
				" them (a canonicalizing sort feeding an intern table) the whole package still" +
				" passed, because the damage was lost deduplication rather than a wrong answer."
		}
		out = append(out, finding{
			pos:      fset.Position(call.Pos()),
			category: "reflect-swapper-sort",
			msg: fmt.Sprintf("sort.%s allocates a reflect swapper on EVERY call, whatever the slice"+
				" length — switch to slices.%sFunc. Triage by how often this line runs, not by how long"+
				" the slice is, which is the counter-intuitive part: the allocation is per CALL."+
				" Measured in this repo — classic/tree.go's radixByFeature, once per node per feature,"+
				" went 1,095,700 to 352,027 allocs (3.11x); classic/knn.go's three per-node/per-query"+
				" sites went 36,004 to 24,003 (1.50x) sorting SHORT slices of k results. A long sort"+
				" called once allocates one swapper and is not worth touching; five sites were declined"+
				" on exactly that basis. Both forms are UNSTABLE, so ties may land differently — confirm"+
				" the comparator is a total order, or gate the output, before converting.%s",
				sel.Sel.Name, want, risk),
		})
		return true
	})
	return out
}

// permutationSortComparator reports whether a sort.Slice/SliceStable call sorts an INDEX
// PERMUTATION: its comparator indexes some other slice using the sorted slice's own element as
// the index, i.e. OUTER[INNER[param]] where INNER is the slice being sorted. It returns the two
// slice names.
//
// This is the one PS6009 conversion that fails SILENTLY. Everywhere else the element type is not
// an integer, so transcribing the old positional expression into the element-based API does not
// compile and the mistake is caught immediately. Here both the position and the element are ints,
// so the transcription type-checks and sorts by whatever OUTER[INNER[element]] happens to be.
//
// Detection is purely syntactic and needs no type information: finding OUTER[INNER[param]] with
// INNER the sorted slice and param a comparator parameter is itself the proof that INNER's
// elements are used as indices.
func permutationSortComparator(call *ast.CallExpr) (outer, inner string, ok bool) {
	if len(call.Args) < 2 {
		return "", "", false
	}
	slice, isIdent := ast.Unparen(call.Args[0]).(*ast.Ident)
	if !isIdent {
		return "", "", false
	}
	lit, isLit := ast.Unparen(call.Args[1]).(*ast.FuncLit)
	if !isLit || lit.Type == nil || lit.Type.Params == nil {
		return "", "", false
	}
	params := map[string]bool{}
	for _, f := range lit.Type.Params.List {
		for _, nm := range f.Names {
			if nm.Name != "_" {
				params[nm.Name] = true
			}
		}
	}
	if len(params) == 0 {
		return "", "", false
	}
	ast.Inspect(lit.Body, func(n ast.Node) bool {
		if ok {
			return false
		}
		ix, isIx := n.(*ast.IndexExpr)
		if !isIx {
			return true
		}
		outerID, isID := ast.Unparen(ix.X).(*ast.Ident)
		if !isID {
			return true
		}
		// The outer slice is deliberately NOT required to differ from the sorted one. A
		// self-permutation, s[s[a]], has exactly the property this check is about — the element
		// is an integer index — so a transcribed s[s[x]] compiles and is wrong. An earlier
		// version excluded outer == inner, which mutation testing showed to be both unnecessary
		// (a direct s[i] < s[j] is already excluded by the nesting requirement below, since its
		// index is a bare parameter rather than another index expression) and harmful, because
		// it would have skipped the self-permutation case.
		// The index must itself be, or contain, slice[param] — "contain" so that an offset form
		// such as gsc[grp[a]-base] matches, which is the shape that was actually shipped wrong.
		ast.Inspect(ix.Index, func(m ast.Node) bool {
			if ok {
				return false
			}
			inner2, isIx2 := m.(*ast.IndexExpr)
			if !isIx2 {
				return true
			}
			innerID, isID2 := ast.Unparen(inner2.X).(*ast.Ident)
			if !isID2 || innerID.Name != slice.Name {
				return true
			}
			if p, isP := ast.Unparen(inner2.Index).(*ast.Ident); isP && params[p.Name] {
				outer, inner, ok = outerID.Name, innerID.Name, true
			}
			return !ok
		})
		return !ok
	})
	return outer, inner, ok
}

// rootIdentName returns the leftmost identifier of an index/selector chain — `l` for `l[i]`, `m` for
// `m.cov[k]`, `dst` for `dst`. PS4004 needs it because siblingBranchBulkCopies searches for a copy()
// naming the same operands, and that search works on names rather than on rendered expressions.
func rootIdentName(e ast.Expr) (string, bool) {
	for {
		switch x := e.(type) {
		case *ast.Ident:
			return x.Name, true
		case *ast.IndexExpr:
			e = x.X
		case *ast.SelectorExpr:
			e = x.X
		case *ast.ParenExpr:
			e = x.X
		default:
			return "", false
		}
	}
}

// srcText renders a node back to its source form. exprText cannot be used for PS1008: it handles
// only Ident/Selector/Index/Paren, so it renders a composite index as the empty string — `os[i*n+j]`
// came out as `os[]` and an integer loop start came out as nothing at all, which is exactly the
// failure mode already recorded for it on PS6011's index comparison. PS1008's message has to name
// the two runs precisely to be actionable, so it needs the real text.
//
// Scoped to PS1008 on purpose rather than swapped in everywhere: the other checks' messages have
// floors asserting their exact wording, and widening exprText would move all of them at once.
func srcText(fset *token.FileSet, n ast.Node) string {
	var b strings.Builder
	if err := printer.Fprint(&b, fset, n); err != nil {
		return ""
	}
	return b.String()
}

// unitStrideOffset reports whether e is an index expression whose LAST index advances by exactly 1
// per increment of v, returning that index rendered with v in place so the message can name where
// the run starts. Accepts the bare `v` and `X + v` / `v + X` with X free of v; rejects anything
// multiplied, which is a stride rather than a run.
func unitStrideOffset(fset *token.FileSet, e ast.Expr, v string) (string, bool) {
	ix, ok := e.(*ast.IndexExpr)
	if !ok {
		return "", false
	}
	if !unitStrideIn(ix.Index, v) {
		return "", false
	}
	// The BASE must be invariant in v too, or the walk is not a run. `out[j][j] = m.cov[k][j]`
	// passes the last-index test and is a DIAGONAL write across a slice-of-slices, which copy()
	// cannot express at all. Found by reading the 14 reported sites rather than trusting the count:
	// classic/gmm.go's GMMDiag branch is exactly that shape, and it was the one false positive.
	if mentionsVar(ix.X, v) {
		return "", false
	}
	return srcText(fset, e), true
}

// unitStrideIn is the stride test itself: `v`, or a sum of `v` with a term that does not mention v.
// Anything else — a product, a shift, a call, a nested index — is not a unit stride.
func unitStrideIn(e ast.Expr, v string) bool {
	switch x := e.(type) {
	case *ast.Ident:
		return x.Name == v
	case *ast.ParenExpr:
		return unitStrideIn(x.X, v)
	case *ast.BinaryExpr:
		if x.Op != token.ADD {
			return false
		}
		if unitStrideIn(x.X, v) && !mentionsVar(x.Y, v) {
			return true
		}
		return unitStrideIn(x.Y, v) && !mentionsVar(x.X, v)
	}
	return false
}

// mentionsVar reports whether v appears anywhere in e.
func mentionsVar(e ast.Expr, v string) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && id.Name == v {
			found = true
		}
		return !found
	})
	return found
}

// loopExtent renders the half-open interval the loop covers, when it is syntactically apparent:
// `for v := LO; v < HI; v++` exactly, and `v <= HI` as HI+1. A range loop is left unbounded rather
// than guessed at, because `range E` iterates E times for an integer and len(E) times for a slice
// and the AST alone cannot tell which.
func loopExtent(fset *token.FileSet, n ast.Node, v string) (string, string, bool) {
	f, ok := n.(*ast.ForStmt)
	if !ok || f.Cond == nil {
		return "", "", false
	}
	as, ok := f.Init.(*ast.AssignStmt)
	if !ok || len(as.Rhs) != 1 {
		return "", "", false
	}
	cond, ok := f.Cond.(*ast.BinaryExpr)
	if !ok {
		return "", "", false
	}
	if id, ok := cond.X.(*ast.Ident); !ok || id.Name != v {
		return "", "", false
	}
	switch cond.Op {
	case token.LSS:
		return srcText(fset, as.Rhs[0]), srcText(fset, cond.Y), true
	case token.LEQ:
		return srcText(fset, as.Rhs[0]), srcText(fset, cond.Y) + "+1", true
	}
	return "", "", false
}

// perRowMakeSlabFindings flags PS2008 — a loop that allocates one slice per iteration into
// ARR[loopvar], where a single slab plus disjoint capped views would replace n allocations with
// one. Both spellings are matched: a direct ARR[i] = make(...) and the two-step v := make(...)
// followed by ARR[i] = v.
//
// The length must be LOOP-INVARIANT. A length varying with the loop variable makes the rows
// jagged, and a jagged set needs per-row offsets rather than a uniform stride — a different and
// far less mechanical transform, so it is excluded rather than reported with advice that does not
// apply to it.
//
// See the registry entry for the measurements: this is an allocation-count win with no wall-clock
// effect, verified at two independent sites, and the message says so because the transform is
// otherwise easy to mistake for a throughput optimization.
func perRowMakeSlabFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		lv, body, ok := loopVarBody(n)
		if !ok || body == nil {
			return true
		}
		// A loop ranging over a FIXED-SIZE ARRAY is bounded by a compile-time constant, so the
		// slab it would earn is a handful of allocations rather than one per data row. PS2008's own
		// guidance is to prefer data-sized loops, and five of the seven sites this check reported
		// in classic/gmm.go were `for t := range y4` with y4 declared [4][]float64 — four
		// allocations each. Reporting them buries the two that ran thousands of times.
		if rs, ok := n.(*ast.RangeStmt); ok && rangesFixedArray(fn, rs.X) {
			return true
		}
		made := map[string]ast.Node{}
		report := func(base string, at ast.Node, rows ast.Expr) {
			out = append(out, finding{
				pos:      fset.Position(at.Pos()),
				category: "per-row-make-slab",
				msg: fmt.Sprintf("this make() runs once per %s iteration, so %s costs one allocation"+
					" per row. The length does not vary with %s, so all rows are the same width:"+
					" allocate ONE slab and hand out disjoint capped views —"+
					" slab := make([]T, rows*len) then %s[%s] = slab[%s*len : (%s+1)*len : (%s+1)*len]."+
					" Bit-identical (make zeroes, so does a fresh slab; no value or order changes)."+
					" EXPECT NO SPEEDUP: measured at two sites (QR reflectors -39.7%% allocs, GMM"+
					" responsibilities -62.2%%) with NO wall-clock change at either. Worth doing as a"+
					" resource win — less allocator and GC-scan work everywhere — not for throughput."+
					" Preconditions this check cannot see: rows must not be appended to, nor"+
					" individually replaced later. Payoff scales with the ITERATION count, which is a"+
					" runtime fact, so prefer loops bounded by a data size over a small constant.",
					lv, base, lv, base, lv, lv, lv, lv),
			})
		}
		ast.Inspect(body, func(m ast.Node) bool {
			as, isAs := m.(*ast.AssignStmt)
			if !isAs || len(as.Rhs) != 1 || len(as.Lhs) != 1 || !invariantSliceMake(as.Rhs[0], lv) {
				return true
			}
			switch lhs := as.Lhs[0].(type) {
			case *ast.Ident:
				made[lhs.Name] = as
			case *ast.IndexExpr:
				if ix, isID := lhs.Index.(*ast.Ident); isID && ix.Name == lv {
					if base, isB := lhs.X.(*ast.Ident); isB {
						report(base.Name, as, nil)
					}
				}
			}
			return true
		})
		for name, at := range made {
			ast.Inspect(body, func(m ast.Node) bool {
				as, isAs := m.(*ast.AssignStmt)
				if !isAs || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
					return true
				}
				ix, isIx := as.Lhs[0].(*ast.IndexExpr)
				if !isIx {
					return true
				}
				idx, isID := ix.Index.(*ast.Ident)
				if !isID || idx.Name != lv {
					return true
				}
				rhs, isR := as.Rhs[0].(*ast.Ident)
				if !isR || rhs.Name != name {
					return true
				}
				if base, isB := ix.X.(*ast.Ident); isB {
					report(base.Name, at, nil)
				}
				return true
			})
		}
		return true
	})
	return out
}

// rangesFixedArray reports whether e names a variable declared in fn as a fixed-size ARRAY, so a
// range over it iterates a compile-time constant number of times.
//
// No type information is needed: the declaration is in the same function, and an array type carries
// its length in the syntax. A slice ([]T) has no Len and is correctly not matched — its length is a
// runtime fact and a slab over it can be worth having.
func rangesFixedArray(fn *ast.FuncDecl, e ast.Expr) bool {
	id, ok := ast.Unparen(e).(*ast.Ident)
	if !ok {
		return false
	}
	found := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if found {
			return false
		}
		switch d := n.(type) {
		case *ast.DeclStmt: // var y4 [4][]float64
			gd, ok := d.Decl.(*ast.GenDecl)
			if !ok || gd.Tok != token.VAR {
				return true
			}
			for _, sp := range gd.Specs {
				vs, ok := sp.(*ast.ValueSpec)
				if !ok || vs.Type == nil {
					continue
				}
				at, ok := vs.Type.(*ast.ArrayType)
				if !ok || at.Len == nil { // Len == nil means a slice, not an array
					continue
				}
				for _, nm := range vs.Names {
					if nm.Name == id.Name {
						found = true
					}
				}
			}
		case *ast.AssignStmt: // y4 := [4][]float64{...}
			for i, lhs := range d.Lhs {
				lid, ok := lhs.(*ast.Ident)
				if !ok || lid.Name != id.Name || i >= len(d.Rhs) {
					continue
				}
				cl, ok := ast.Unparen(d.Rhs[i]).(*ast.CompositeLit)
				if !ok {
					continue
				}
				if at, ok := cl.Type.(*ast.ArrayType); ok && at.Len != nil {
					found = true
				}
			}
		}
		return !found
	})
	return found
}

// invariantSliceMake reports whether e is make([]T, len, ...) whose size arguments do NOT mention
// lv. A size that varies with the loop variable produces jagged rows, which a uniform-stride slab
// cannot represent.
func invariantSliceMake(e ast.Expr, lv string) bool {
	call, ok := ast.Unparen(e).(*ast.CallExpr)
	if !ok {
		return false
	}
	// The "make" name test has NO floor in ps2008_slab_test.go, deliberately. In valid Go only
	// make accepts a type as its first argument, so the ArrayType test below already implies it
	// and no fixture can distinguish them — mutation testing showed removing the name test
	// reddens nothing. It is kept as an explicit statement of intent, not as a live filter;
	// recorded here so the next reader does not go looking for the missing floor.
	id, ok := call.Fun.(*ast.Ident)
	if !ok || id.Name != "make" || len(call.Args) < 2 {
		return false
	}
	if _, isSlice := call.Args[0].(*ast.ArrayType); !isSlice {
		return false
	}
	varies := false
	for _, a := range call.Args[1:] {
		ast.Inspect(a, func(n ast.Node) bool {
			if idn, ok := n.(*ast.Ident); ok && idn.Name == lv {
				varies = true
			}
			return !varies
		})
	}
	return !varies
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
				remedy := fmt.Sprintf("Interchange the loops so %q runs contiguously (bit-neutral when"+
					" the body is a pure elementwise update), or block four adjacent %q values so one"+
					" fetched line feeds four accumulators.", innerVar, outerVar)
				if readsBackItsOwnWrites(outerBody, buf.Name) {
					remedy = fmt.Sprintf("INTERCHANGE IS NOT AVAILABLE HERE: this nest both READS and"+
						" WRITES %q at differing indices, so the strided accesses are values earlier"+
						" iterations of this same loop produced — a recurrence, not an independent"+
						" traversal, and swapping the loops would read x[j] before it exists (%q is the"+
						" enclosing loop). The remedy that IS"+
						" available is to keep the intermediate in CONTIGUOUS per-iteration scratch and"+
						" scatter the finished row into %q once at the end. RANK BY THE RECURRENCE'S SHARE OF"+
						" RUNTIME, which is what decides this one: shipped on LU back substitution"+
						" (LUSolve_512x512 -8.41%%, 768x768 -7.00%%) where a line-level profile put the"+
						" strided statement at 32%% of the benchmark, and REJECTED on the structurally"+
						" identical back substitution in Lstsq (geomean -0.02%%, only -0.55%% at n=768)"+
						" where the Householder factorization and Q-transpose-b dominate and the"+
						" recurrence is a small fraction. Also check the stride really exceeds 1: the"+
						" single-RHS LU control came out +0.57%% SLOWER, because at one column the access"+
						" was already contiguous and only the scatter remains.", buf.Name, outerVar, buf.Name)
				}
				out = append(out, finding{
					pos:      fset.Position(ix.Pos()),
					category: "strided-inner-walk",
					msg: fmt.Sprintf("the inner loop over %q indexes %q at a stride, so each iteration jumps a"+
						" whole row and touches its own cache line to use one element — and the walk repeats"+
						" once per %q. %s THE PREMISE IS A CACHE CLAIM THIS CHECK CANNOT MAKE: below"+
						" one cache line of stride, consecutive iterations share a line, and a buffer that"+
						" fits L1 pays nothing for striding either — read the stride value and the buffer"+
						" size before acting. 2.40x on NSA P*V and part of KDA's 1.75x were measured AT"+
						" THOSE SITES, not here.",
						innerVar, buf.Name, outerVar, remedy),
				})
				return true
			})
			return true
		})
		return true
	})
	return dedupeByPos(out)
}

// readsBackItsOwnWrites reports whether the nest both READS and WRITES name at index expressions
// that DIFFER — the signature of a recurrence, where the strided reads are values earlier
// iterations of this same loop produced.
//
// That combination changes the remedy completely. When a loop reads back what it wrote,
// interchanging is not merely unprofitable but WRONG: the swapped order would read a value before
// the iteration producing it has run. LU back substitution is the shipped example — x[i] depends
// on every x[j] with j > i — and the fix that works there is to keep the intermediate in
// contiguous scratch and scatter once, which the generic advice does not mention.
//
// The comparison is SYMMETRIC on purpose. PS6011 reports whichever access it matched first, and on
// a three-deep nest (column, row, inner) it reported the WRITE rather than the read — an earlier
// version of this helper assumed the reported access was the read and classified nothing.
//
// Requiring the two index expressions to differ is what separates a recurrence from an
// elementwise in-place update: src[j*cols+c] = src[j*cols+c] * 2 reads and writes the same slot,
// so interchange still applies to it.
func readsBackItsOwnWrites(body *ast.BlockStmt, name string) bool {
	// Compared by the set of variables MULTIPLIED in the index, not by index text. exprText
	// renders only identifiers and selectors, so an index like i*cols+c keys to the empty string
	// and every access collapses to one key — which is what made a first version classify
	// nothing. The multiplied set is the structural fact that matters anyway: the write strides by
	// the recurrence variable and the read by the inner one, and an elementwise in-place update
	// strides both by the same variable.
	var writes, reads []map[string]bool
	collect := func(e ast.Expr, into *[]map[string]bool) {
		ix, ok := e.(*ast.IndexExpr)
		if !ok {
			return
		}
		if id, ok := ast.Unparen(ix.X).(*ast.Ident); ok && id.Name == name {
			*into = append(*into, multipliedVars(ix.Index))
		}
	}
	ast.Inspect(body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok {
			return true
		}
		for _, lhs := range as.Lhs {
			collect(lhs, &writes)
		}
		for _, rhs := range as.Rhs {
			ast.Inspect(rhs, func(m ast.Node) bool {
				if e, ok := m.(ast.Expr); ok {
					collect(e, &reads)
				}
				return true
			})
		}
		return true
	})
	for _, w := range writes {
		for _, r := range reads {
			if len(w) == 0 || len(r) == 0 {
				continue // an unstrided access says nothing about a recurrence
			}
			if !sameStringSet(w, r) {
				return true
			}
		}
	}
	return false
}

// multipliedVars returns the identifiers appearing as an operand of a multiplication anywhere in
// e — the variables the index strides by.
func multipliedVars(e ast.Expr) map[string]bool {
	out := map[string]bool{}
	ast.Inspect(e, func(n ast.Node) bool {
		be, ok := n.(*ast.BinaryExpr)
		if !ok || be.Op != token.MUL {
			return true
		}
		for _, side := range []ast.Expr{be.X, be.Y} {
			if id, ok := ast.Unparen(side).(*ast.Ident); ok {
				out[id.Name] = true
			}
		}
		return true
	})
	return out
}

func sameStringSet(a, b map[string]bool) bool {
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if !b[k] {
			return false
		}
	}
	return true
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

// sortFeedsTruncationFindings flags PS6022 — a slice sorted in full and then immediately
// resliced to a smaller bound.
//
//	slices.SortFunc(cand, byScore)
//	if len(cand) > width { cand = cand[:width] }   // the rest of the order is discarded
//
// This is the third consumer shape in the sort-does-too-much family, and the one the other
// two structurally cannot see. PS6013 requires a COUNTED loop that indexes the slice; PS6001
// requires a consumer that BREAKS on a threshold. Here the consumer is neither a loop nor a
// break — it is a reslice, so no loop exists to match. The gap was found by a survey of the
// decoding path, where both beam search and diverse beam search sort every candidate
// (beams x vocabulary) and keep the top few: at a 2048 vocabulary and width 8 that is a sort
// of 16384 to select 8, once per generated token.
//
// SOUNDNESS. The bound must not be len(target) — that keeps everything and discards nothing.
// Any statement between the sort and the reslice that INDEXES or RANGES over target is
// disqualifying: it may depend on the full order, and it also means one of the other two
// checks already describes the site better. A len(target) guard is not such a read, since
// that is how the idiomatic truncation is written.
//
// Replacing the sort with a selection is bit-safe only when the comparator is a TOTAL order.
// With ties the retained SET is not unique, so a selection and a sort can legitimately keep
// different elements. The message says so, because that is the precondition a reviewer must
// check rather than assume — and in the motivating case the comparator was written as a total
// order precisely so it would reproduce a stable sort's tie order, which is what makes the
// substitution safe there.
func sortFeedsTruncationFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		blk, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, st := range blk.List {
			target, sorter, ok := sortedSliceName(st)
			if !ok {
				continue
			}
			for _, next := range blk.List[i+1:] {
				if bound, isTrunc := truncatesSlice(next, target); isTrunc {
					out = append(out, finding{
						pos:      fset.Position(st.Pos()),
						category: "sort-feeds-truncation",
						msg: fmt.Sprintf("%s orders %q in full and it is then resliced to %s — every comparison"+
							" that ordered the discarded tail was wasted work. A selection (quickselect /"+
							" nth_element) partitions in O(n), after which sorting only the kept prefix"+
							" restores the same order. Bit-safe ONLY when the comparator is a TOTAL order:"+
							" with ties the retained set is not unique, so a selection and a sort can keep"+
							" different elements.", sorter, target, bound),
					})
					break
				}
				// A read of the slice before the truncation may depend on the full order, and
				// means PS6001 or PS6013 describes the site better. A len() guard is not such a
				// read — see readsSliceSlots.
				if readsSliceSlots(next, target) {
					break
				}
			}
		}
		return true
	})
	return out
}

// truncatesSlice reports whether st reslices target to a bound other than len(target), and
// returns that bound's source text. It walks st rather than matching only a top-level assign,
// because the idiomatic form wraps the reslice in an `if len(target) > k` guard.
func truncatesSlice(st ast.Stmt, target string) (string, bool) {
	bound, found := "", false
	ast.Inspect(st, func(n ast.Node) bool {
		if found {
			return false
		}
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Rhs) != 1 {
			return true
		}
		sl, ok := ast.Unparen(as.Rhs[0]).(*ast.SliceExpr)
		if !ok || sl.High == nil || sl.Low != nil || sl.Slice3 {
			return true
		}
		if identName(sl.X) != target {
			return true
		}
		// len(target) keeps every element; nothing is discarded and nothing is wasted.
		if c, isCall := ast.Unparen(sl.High).(*ast.CallExpr); isCall && calleeName(c.Fun) == "len" {
			return true
		}
		bound, found = exprText(sl.High), true
		return false
	})
	return bound, found
}

// readsSliceSlots reports whether st reads target by index or by range — a use that may depend
// on the full sorted order.
//
// A len(target) call needs no exemption and deliberately has none. len takes the slice as a
// bare identifier, which is neither an index nor a range, so the idiomatic
// `if len(target) > k` guard already passes through. An explicit len case was written first
// and removed: mutation testing showed no floor depended on it, and it was worse than
// redundant — by stopping the descent it would also have skipped a genuine indexed read
// nested inside a len argument, such as len(target[i]).
func readsSliceSlots(st ast.Stmt, target string) bool {
	found := false
	ast.Inspect(st, func(n ast.Node) bool {
		if found {
			return false
		}
		switch v := n.(type) {
		case *ast.IndexExpr:
			if identName(v.X) == target {
				found = true
			}
		case *ast.RangeStmt:
			if identName(v.X) == target {
				found = true
			}
		}
		return !found
	})
	return found
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
// that feeds the action that feeds the environment, so it cannot move). Both checks now report per
// CALL SITE: PS1003 used to position its finding at the enclosing loop, which made dedup collapse a
// second batch-1 call in the same loop into the first and hide it entirely — the shape this very
// function was written about. That is fixed, so where a hoistable and a non-hoistable call share a
// loop, PS1003 names both and this check adds the hoist advice for the one it applies to.
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

// allFieldsConst reports whether every field initializer is a compile-time constant. Such a literal
// costs NOTHING to box: the compiler emits a pointer to a static read-only copy rather than calling
// newobject, so hoisting it cannot save an allocation.
//
// MEASURED, because escape analysis says otherwise and that is the trap. `go build -gcflags=-m`
// reports "backend.ConcatAttrs{...} escapes to heap" for a fully constant literal, and PS6016's own
// text used to point at -m as the confirmation. Hoisting two such sites in nlp quant_deepseekv2
// changed allocs/op by ZERO, every sample equal. An isolated benchmark then separated the cases:
// boxing an all-constant literal is 0 allocs and 1.98ns, identical to a pre-boxed package var, while
// a literal with ANY non-constant field — even one that never changes inside the loop — is 1 alloc,
// 24 B and 11.4ns. So the field constness, not the escape verdict, is what decides.
//
// A bare identifier is deliberately NOT treated as constant: the AST cannot tell a const from a var,
// and guessing wrong here would drop a real finding rather than merely keep a harmless one.
func allFieldsConst(cl *ast.CompositeLit) bool {
	for _, el := range cl.Elts {
		v := el
		if kv, ok := el.(*ast.KeyValueExpr); ok {
			v = kv.Value
		}
		if !constFoldable(v) {
			return false
		}
	}
	return len(cl.Elts) > 0
}

// constFoldable reports whether an expression is a literal, or a unary/binary/parenthesized
// combination of literals — the forms the compiler can fold at compile time.
func constFoldable(e ast.Expr) bool {
	switch x := e.(type) {
	case *ast.BasicLit:
		return true
	case *ast.UnaryExpr:
		return constFoldable(x.X)
	case *ast.BinaryExpr:
		return constFoldable(x.X) && constFoldable(x.Y)
	case *ast.ParenExpr:
		return constFoldable(x.X)
	}
	return false
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
		// FALLBACK ARMS ARE INERT. Where an if/else has a devirtualized branch that reaches typed
		// storage, the else is the correctness path for whatever that branch declines — so a literal
		// rebuilt there is not rebuilt on the hot path at all. Verified rather than assumed: a panic
		// in nlp quant_deepseekv2's attnReconstructed else-arm never fires under either
		// BenchmarkQuantDeepSeekV2Reconstructed benchmark, both of which take the typed branch.
		// PS1001 and PS1009 already suppress their own equivalents; this brings PS6016 in line.
		var inert []struct{ lo, hi token.Pos }
		ast.Inspect(body, func(k ast.Node) bool {
			ifs, ok := k.(*ast.IfStmt)
			if !ok || ifs.Else == nil {
				return true
			}
			// The typed access may sit in the BODY (`if ... { s := x.Storage().F64(); ... }`) or in
			// the guard's INIT (`if s := x.Storage().F64(); s != nil {`), and both forms occur.
			guarded := readsTypedStorage(ifs.Body)
			if !guarded && ifs.Init != nil {
				guarded = readsTypedStorage(ifs.Init)
			}
			if !guarded && ifs.Cond != nil {
				guarded = readsTypedStorage(ifs.Cond)
			}
			if guarded {
				inert = append(inert, struct{ lo, hi token.Pos }{ifs.Else.Pos(), ifs.Else.End()})
			}
			return true
		})
		inFallback := func(p token.Pos) bool {
			for _, sp := range inert {
				if p >= sp.lo && p < sp.hi {
					return true
				}
			}
			return false
		}
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
				if !ok || len(cl.Elts) == 0 || cl.Type == nil || inFallback(cl.Pos()) || allFieldsConst(cl) {
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
			"1.25-1.33x with 38-43%% fewer allocations, Gemma2 capped attention 1.21x and DeepSeekV2"+
			"absorbed attention 1.12x were measured AT THOSE SITES, on per-layer per-token decode"+
			" paths. The count above is of STATIC call sites, not iterations, so it says nothing"+
			" about how often this function runs — measure that first.",
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

// setMapFromSliceFindings flags PS3007 — a membership SET (map[K]bool / map[K]struct{}) that is
// built by ranging a slice and then probed inside a loop.
//
// PS3003 deliberately excludes set-shaped maps, on the grounds that a sparse membership set is not
// the dense [0,N) lookup a slice would replace. That is right about DENSIFICATION and wrong about
// this: when the set's contents come from a slice the caller already owns, the fix is not a dense
// table but no map at all — scan the source slice. The map build plus one hash per probe loses to
// len(src) comparisons the compiler keeps in registers, as long as the source stays small.
//
// MEASURED, which is why the check exists. nlp's applyDRY hashed its sequence-breaker set once per
// window position to precompute a dense []bool; runtime.mapaccess1_fast64 was 1.14s of the
// function's 1.99s cumulative profile, 57% of its own time. Scanning DRYBreakers directly took
// BenchmarkApplyDRY from 19.52us to 15.87us, -18.72% at p=0.002 (n=6, interleaved, both arm
// orders), allocations unchanged, and mapaccess1_fast64 left the profile entirely. That measurement
// also fixes the boundary: forced onto each arm, the crossover sits between 8 and 16 elements on an
// M2 Pro (at 8 the scan wins 62.7us vs 64.7us; at 16 it has lost, 68.0us vs 65.4us; at 64 it loses
// badly, 97.7us vs 66.3us). So this is a SMALL-SET transform, and the advice says so.
//
// Two narrowings, both load-bearing rather than cosmetic — without them the check reported three
// sites in this repo and exactly one was real:
//
//   - The set must be READ-ONLY after its build loop. Scanning the source reproduces the predicate
//     only if nothing adds to the set later; autograd's einsum `avail` is written inside the very
//     loop that probes it, so it genuinely needs a map.
//   - A build already guarded by a SIZE THRESHOLD on the source is silent, because such code has
//     already taken this advice and kept the map only as its large-set fallback — applyDRY itself
//     now looks exactly like that. The threshold test must not be confused with an EMPTINESS test:
//     `len(src) > 0` is a nil guard whose branch is the only path, not a fallback, and treating it
//     as a threshold suppressed the one true finding in the repo while it was written that way.
//
// HOTNESS IS NOT VISIBLE HERE. The probe loop's trip count and the source slice's length are both
// runtime facts, so confirm the source really is small and the probe really is on a repeating path,
// then benchmark before converting.
//
// KNOWN BLIND SPOT: only the value form of the build is recognized, `for _, v := range src {
// set[v] = true }`. The index form, `for i := range src { set[src[i]] = true }`, denotes the same
// pattern and is missed. Deliberate — a counting pass over this repo finds no instance, so
// supporting it would add machinery no real code exercises.
func setMapFromSliceFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	// Names declared in this function as a SET map, and the map type's own position.
	sets := map[string]bool{}
	declSet := func(typ ast.Expr, name string) {
		mt, ok := typ.(*ast.MapType)
		if !ok {
			return
		}
		if id, ok := mt.Value.(*ast.Ident); ok && id.Name == "bool" {
			sets[name] = true
			return
		}
		if st, ok := mt.Value.(*ast.StructType); ok && st.Fields != nil && len(st.Fields.List) == 0 {
			sets[name] = true
		}
	}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.AssignStmt:
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
						declSet(r.Args[0], id.Name)
					}
				case *ast.CompositeLit:
					declSet(r.Type, id.Name)
				}
			}
		case *ast.ValueSpec:
			for _, nm := range x.Names {
				declSet(x.Type, nm.Name)
			}
		}
		return true
	})
	if len(sets) == 0 {
		return nil
	}

	// Builds of the shape `for _, v := range SRC { SET[v] = true }`.
	built := map[string]string{}
	buildLoop := map[string]*ast.RangeStmt{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		r, ok := n.(*ast.RangeStmt)
		if !ok || r.Value == nil || r.Body == nil || len(r.Body.List) != 1 {
			return true
		}
		v, ok := r.Value.(*ast.Ident)
		if !ok {
			return true
		}
		as, ok := r.Body.List[0].(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 {
			return true
		}
		idx, ok := as.Lhs[0].(*ast.IndexExpr)
		if !ok {
			return true
		}
		m, ok := idx.X.(*ast.Ident)
		if !ok || !sets[m.Name] {
			return true
		}
		if k, ok := idx.Index.(*ast.Ident); !ok || k.Name != v.Name {
			return true
		}
		src := exprText(r.X)
		if src == "" {
			return true
		}
		built[m.Name] = src
		buildLoop[m.Name] = r
		return true
	})
	if len(built) == 0 {
		return nil
	}

	inBuild := func(name string, n ast.Node) bool {
		bl := buildLoop[name]
		return bl != nil && n.Pos() >= bl.Pos() && n.End() <= bl.End()
	}
	// NARROWING 1: drop any set written to outside its build loop.
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok {
			return true
		}
		for _, l := range as.Lhs {
			ix, ok := l.(*ast.IndexExpr)
			if !ok {
				continue
			}
			m, ok := ix.X.(*ast.Ident)
			if !ok || built[m.Name] == "" || inBuild(m.Name, ix) {
				continue
			}
			delete(built, m.Name)
		}
		return true
	})
	// NARROWING 2: drop any build already guarded by a size threshold on the source.
	for name, src := range built {
		if buildGuardedBySize(fn, buildLoop[name], src) {
			delete(built, name)
		}
	}
	if len(built) == 0 {
		return nil
	}

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
		return true
	})

	var out []finding
	seen := map[token.Pos]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		idx, ok := n.(*ast.IndexExpr)
		if !ok {
			return true
		}
		m, ok := idx.X.(*ast.Ident)
		if !ok || built[m.Name] == "" || inBuild(m.Name, idx) {
			return true
		}
		// A write is a build, not a probe. (Narrowing 1 already dropped such sets, but a
		// comma-ok read must still be distinguished from an assignment target.)
		if as, ok := parent[idx].(*ast.AssignStmt); ok {
			for _, l := range as.Lhs {
				if l == idx {
					return true
				}
			}
		}
		if nearestLoop(parent, idx) == nil || seen[idx.Pos()] {
			return true
		}
		seen[idx.Pos()] = true
		out = append(out, finding{
			pos:      fset.Position(idx.Pos()),
			category: "set-map-from-slice",
			msg: fmt.Sprintf("%q is a membership SET built by ranging %q, then probed in a loop —"+
				" each probe pays a hash to answer a question the source slice already answers."+
				" When %q is small, scanning it directly is faster than building and hashing the"+
				" map (applyDRY: -18.72%%, and mapaccess1_fast64 left the profile); keep the map"+
				" behind a size threshold for large sets, since the scan is O(len(%s)) per probe."+
				" Measured crossover was 8-16 elements. Confirm the source is small and the probe"+
				" repeats, then benchmark.", m.Name, built[m.Name], built[m.Name], built[m.Name]),
		})
		return true
	})
	return out
}

// buildGuardedBySize reports whether bl sits inside an if whose condition compares len(src) against
// a bound other than the literal 0.
//
// The literal-0 exclusion is the whole point rather than a detail: `if len(src) > 0` is an
// emptiness guard, and the build behind it is the function's only path, not a large-set fallback.
// Counting it as a threshold made this check silent on the single genuine site in the repo.
func buildGuardedBySize(fn *ast.FuncDecl, bl *ast.RangeStmt, src string) bool {
	if bl == nil {
		return false
	}
	isLenOfSrc := func(e ast.Expr) bool {
		call, ok := e.(*ast.CallExpr)
		if !ok || len(call.Args) != 1 {
			return false
		}
		return calleeName(call.Fun) == "len" && exprText(call.Args[0]) == src
	}
	isZeroLit := func(e ast.Expr) bool {
		lit, ok := e.(*ast.BasicLit)
		return ok && lit.Kind == token.INT && lit.Value == "0"
	}
	found := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		ifs, ok := n.(*ast.IfStmt)
		if !ok || ifs.Cond == nil || bl.Pos() < ifs.Pos() || bl.End() > ifs.End() {
			return true
		}
		ast.Inspect(ifs.Cond, func(c ast.Node) bool {
			bin, ok := c.(*ast.BinaryExpr)
			if !ok {
				return true
			}
			var other ast.Expr
			switch {
			case isLenOfSrc(bin.X):
				other = bin.Y
			case isLenOfSrc(bin.Y):
				other = bin.X
			default:
				return true
			}
			if !isZeroLit(other) {
				found = true
			}
			return true
		})
		return true
	})
	return found
}

// monotoneBailPerElementFindings flags PS3008 — a loop that accumulates a provably NON-NEGATIVE
// term into a scalar and tests that accumulator against a threshold on EVERY iteration, bailing
// out when it is exceeded.
//
// The accumulator only ever grows, so once it passes the threshold it stays past it. The bail-out
// is therefore a pure heuristic, not semantics: testing it every Nth iteration returns the SAME
// answer, because a run that would have bailed at iteration k still bails at the next checkpoint
// and at the end. What that buys is removing a data-dependent branch the predictor cannot learn.
//
// MEASURED, which is why this check exists rather than being a plausible idea. classic's
// ballTree.within is the L2 leaf test DBSCAN runs for every candidate pair. A line-level profile
// put the bail-out branch at 450ms against 30ms for the subtraction and square it guards — the
// branch, not the arithmetic, was the function. Checking it every fourth dimension took
// BenchmarkDBSCANFit -17.41% at eps=2 (p=0.000) and -8.51% at eps=4 (p=0.010), geomean -13.07%,
// allocations unchanged, with the exact-label DBSCAN goldens still green.
//
// THE MONOTONICITY IS THE CORRECTNESS CONDITION, so the check demands it syntactically rather than
// guessing: the added term must be x*x with identical operands, math.Abs, math.Hypot, or a sum of
// those. An accumulator fed a signed term can dip back below the threshold, and moving its test
// would change the answer, not just the speed.
//
// Two things this cannot see, both stated in the finding. Whether the loop is hot at all — a cold
// bail-out is not worth restructuring. And the NaN edge: with a NaN term the accumulator becomes
// NaN, every `acc > thr` is false, and the original falls out of the loop and returns its
// not-exceeded answer, so the rewritten tail must end with !(acc > thr) and NOT acc <= thr, which
// flips it. That trap is real; it is gated by a test in classic rather than left to reasoning.
func monotoneBailPerElementFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		body := loopBodyOf(n)
		if body == nil || loopStridesByMoreThanOne(n) {
			return true
		}
		// Accumulators in THIS loop's body fed a provably non-negative term.
		accs := map[string]string{}
		for _, st := range body.List {
			as, ok := st.(*ast.AssignStmt)
			if !ok || as.Tok != token.ADD_ASSIGN || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
				continue
			}
			name := identName(as.Lhs[0])
			if name == "" {
				continue
			}
			if term, ok := provablyNonNegative(as.Rhs[0]); ok {
				accs[name] = term
			}
		}
		if len(accs) == 0 {
			return true
		}
		for _, st := range body.List {
			ifs, ok := st.(*ast.IfStmt)
			if !ok || ifs.Init != nil || ifs.Else != nil || !bailsOut(ifs.Body) {
				continue
			}
			bin, ok := ifs.Cond.(*ast.BinaryExpr)
			if !ok || (bin.Op != token.GTR && bin.Op != token.GEQ) {
				continue
			}
			acc := identName(bin.X)
			term, ok := accs[acc]
			if !ok {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(ifs.Pos()),
				category: "monotone-bail-per-element",
				msg: fmt.Sprintf("%q accumulates a non-negative term (%s) and is tested against a"+
					" threshold on EVERY iteration — the accumulator never decreases, so testing it"+
					" every 4th iteration returns the same answer and removes a data-dependent branch"+
					" the predictor cannot learn. MEASURED on classic ballTree.within, the DBSCAN leaf"+
					" test: the branch was 450ms of profile against 30ms for the arithmetic it guarded,"+
					" and checking every 4th dimension gave BenchmarkDBSCANFit -17.41%% at eps=2"+
					" (p=0.000) and -8.51%% at eps=4 (p=0.010). Keep ONE accumulator in the SAME order"+
					" so the sum stays bit-identical, and end the scalar tail with !(%s > thr), NOT"+
					" %s <= thr — with a NaN term the original never bailed and returned its"+
					" not-exceeded answer, and <= flips that. HOTNESS IS NOT VISIBLE HERE: a cold"+
					" bail-out is not worth restructuring, so benchmark the enclosing operation first.",
					acc, term, acc, acc),
			})
		}
		return true
	})
	return out
}

// loopBodyOf returns the body of a for/range statement, or nil.
func loopBodyOf(n ast.Node) *ast.BlockStmt {
	switch l := n.(type) {
	case *ast.ForStmt:
		return l.Body
	case *ast.RangeStmt:
		return l.Body
	}
	return nil
}

// loopStridesByMoreThanOne reports whether the loop advances by a literal step other than 1 —
// evidence the blocking this check recommends has already been applied.
func loopStridesByMoreThanOne(n ast.Node) bool {
	f, ok := n.(*ast.ForStmt)
	if !ok || f.Post == nil {
		return false
	}
	as, ok := f.Post.(*ast.AssignStmt)
	if !ok || as.Tok != token.ADD_ASSIGN || len(as.Rhs) != 1 {
		return false
	}
	lit, ok := as.Rhs[0].(*ast.BasicLit)
	return ok && lit.Kind == token.INT && lit.Value != "1"
}

// provablyNonNegative reports whether e is >= 0 from syntax alone, and names the shape that proves
// it. Only forms that cannot be negative qualify: a square with identical operands, math.Abs,
// math.Hypot, or a sum of those. This is the check's correctness condition, not a heuristic — a
// signed term lets the accumulator fall back below the threshold, and then moving the test changes
// the answer.
func provablyNonNegative(e ast.Expr) (string, bool) {
	switch x := e.(type) {
	case *ast.ParenExpr:
		return provablyNonNegative(x.X)
	case *ast.CallExpr:
		if sel, ok := x.Fun.(*ast.SelectorExpr); ok {
			if identName(sel.X) == "math" && (sel.Sel.Name == "Abs" || sel.Sel.Name == "Hypot") {
				return "math." + sel.Sel.Name, true
			}
		}
	case *ast.BinaryExpr:
		if x.Op == token.MUL && exprText(x.X) == exprText(x.Y) && exprText(x.X) != "" {
			return "a square", true
		}
		if x.Op == token.ADD {
			if a, ok := provablyNonNegative(x.X); ok {
				if b, ok := provablyNonNegative(x.Y); ok {
					return a + " + " + b, true
				}
			}
		}
	}
	return "", false
}

// bailsOut reports whether the block ends in a return or a break/continue.
func bailsOut(b *ast.BlockStmt) bool {
	if b == nil || len(b.List) == 0 {
		return false
	}
	switch b.List[len(b.List)-1].(type) {
	case *ast.ReturnStmt, *ast.BranchStmt:
		return true
	}
	return false
}

// indirectColumnGatherFindings flags PS3009 — a loop reading ONE column of a slice-of-slices
// through an INDIRECT row index: M[idx[k]][f], where f does not vary with the loop.
//
// Every element lands in a different row, so each read dereferences a row header and pulls a whole
// cache line to use eight bytes of it. That is the same cache behavior PS1010 describes, but it is
// NOT the same defect and does not take the same fix. PS1010 requires an interchangeable nest and
// tells you to swap the loops; here the row order is a data-dependent permutation, so there is no
// nest to interchange. The fix is to keep a feature-major copy of the matrix — xT[f*n+row] — so a
// column is contiguous and scattered rows within it share lines.
//
// MEASURED, which is why this exists rather than being a plausible idea. classic's gbmBuilder
// hoists a node's feature column into a buffer before scanning splits, gathering through the
// node's sorted index permutation. That single line was 330ms of bestSplit's 400ms in a profile of
// the classic suite; against a feature-major buffer it took BenchmarkGBMHist_exact_80k -7.74%
// (p=0.002), BenchmarkGBMFit -6.55% (p=0.002) and the 20k fit -2.00% (p=0.007). The win grows with
// n, as the cache argument predicts.
//
// IT TRADES MEMORY FOR TIME, and the finding says so, because the answer is not always yes: the
// transposed copy costs n*d*8 bytes, which raised measured B/op 26.7% at 80k x 20. Worth it there
// because that builder already carries two n*d index arrays; not obviously worth it where the
// matrix is the dominant allocation and the gather is cold.
//
// Reference implementations are the expected false positive. A deliberately simple twin that a
// production path is checked against should not be optimized, and this repo's gbm.go carries
// exactly that, already marked with a perfscan:ignore for its other findings.
func indirectColumnGatherFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	seen := map[int]bool{}
	var stack []ast.Node
	var walk func(ast.Node)
	walk = func(n ast.Node) {
		ast.Inspect(n, func(c ast.Node) bool {
			if c == nil {
				return false
			}
			switch c.(type) {
			case *ast.ForStmt, *ast.RangeStmt:
				if c != n {
					stack = append(stack, c)
					walk(c)
					stack = stack[:len(stack)-1]
					return false
				}
			}
			outer, ok := c.(*ast.IndexExpr)
			if !ok || len(stack) == 0 {
				return true
			}
			row, ok := outer.X.(*ast.IndexExpr)
			if !ok {
				return true
			}
			// The row index must itself be an index expression — that indirection is the whole
			// point, and it is what makes the access a gather rather than a stride PS1006 could see.
			gather, ok := row.Index.(*ast.IndexExpr)
			if !ok {
				return true
			}
			// The COLUMN index must not vary with the innermost loop, or this is not a column walk.
			col := exprText(outer.Index)
			if col == "" || loopAdvancesVar(stack[len(stack)-1], col) {
				return true
			}
			line := fset.Position(outer.Pos()).Line
			if seen[line] {
				return true
			}
			seen[line] = true
			out = append(out, finding{
				pos:      fset.Position(outer.Pos()),
				end:      fset.Position(outer.End()),
				category: "indirect-column-gather",
				msg: fmt.Sprintf("%s reads ONE column (%q) of a slice-of-slices through an INDIRECT"+
					" row index (%s[…]) — every element lands in a different row, so each read"+
					" dereferences a row header and pulls a cache line to use eight bytes. PS1010"+
					" cannot help here: the row order is a data-dependent permutation, so there is no"+
					" nest to interchange. Keep a FEATURE-MAJOR copy instead (xT[col*n+row]) so a"+
					" column is contiguous. MEASURED on classic gbmBuilder.bestSplit, where this line"+
					" was 330ms of the function's 400ms: GBMHist_exact_80k -7.74%% (p=0.002), GBMFit"+
					" -6.55%% (p=0.002), 20k -2.00%% (p=0.007), the win growing with n. IT COSTS"+
					" MEMORY — the copy is n*d*8 bytes, +26.7%% measured B/op at 80k x 20 — so weigh"+
					" that against the gather's hotness rather than converting on sight, and skip it"+
					" for a reference implementation kept simple on purpose.",
					srcText(fset, outer), col, exprText(gather.X)),
			})
			return true
		})
	}
	walk(fn.Body)
	return out
}

// loopAdvancesVar reports whether the loop advances the named variable.
func loopAdvancesVar(n ast.Node, v string) bool {
	switch l := n.(type) {
	case *ast.RangeStmt:
		return exprText(l.Key) == v || exprText(l.Value) == v
	case *ast.ForStmt:
		if as, ok := l.Init.(*ast.AssignStmt); ok {
			for _, lhs := range as.Lhs {
				if exprText(lhs) == v {
					return true
				}
			}
		}
		if inc, ok := l.Post.(*ast.IncDecStmt); ok && exprText(inc.X) == v {
			return true
		}
	}
	return false
}
