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
	{"PS2008", "per-row-make-slab", "a loop that allocates one slice per iteration into ARR[loopvar] — make([]T, len) with len LOOP-INVARIANT — where a single slab plus disjoint capped views would do. THE INVARIANT-LENGTH REQUIREMENT IS REAL BUT IT IS NOT THE END OF THE STORY, and a reader who stops here will leave a win behind: a uniform slab needs uniform rows, so this check stays silent when the per-iteration length VARIES — yet those sites take a block BUMP ALLOCATOR instead, one make per block with each row cut three-index from it, and a row longer than a block getting its own. MEASURED that way on classic DBSCAN, whose core neighbour lists differ in length per point and were cloned one make at a time: -66.18%% and -74.13%% allocs/op with -5.90%% and -7.48%% on time (p=0.000, n=12), against two arms of the same fixture that stayed flat. Note the shape that hid it from this check even in spirit — `dst[i] = append(make([]T, 0, len(src)), src...)`, where the make is nested inside an append. CHECK LIFETIMES BEFORE SLABBING EITHER WAY: a block pins every row in it, so this is only safe when the rows die together, which for DBSCAN they do since the destination is a function local never returned. Replace with one make([]T, n*len) and ARR[i] = slab[i*len : (i+1)*len : (i+1)*len]. Bit-identical by construction: make() zeroes and so does a fresh slab, and no value, order or association changes. EXPECT NO SPEEDUP *WHEN THE LOOP BODY IS SUBSTANTIAL*, and that qualifier was added after a third site broke the unconditional form. Measured with no wall-clock movement: linalg QR's per-column reflectors -39.7% allocs/op and classic GMM's per-sample responsibility rows -62.2% (5751 to 1753 on GMMFit), both p>0.4. Measured WITH movement: classic GMM PredictProba's output rows, -11.17%% at k=4 and -7.30%% at k=8 (p=0.002) alongside allocs 605 to 94. What separates them is how much work the body does per allocation, and the cleanest evidence is INSIDE that third site rather than across sites: the same transform on the same function is -11.17%% at d=4 and INDISTINGUISHABLE at d=16, 32 and 64, where the density evaluation dominates and the allocation never was a meaningful share. Position in the loop nest matters for the same reason — per-ROW scratch can move the clock, per-WORKER scratch cannot, since it is amortized over the whole chunk (classic GMM's pooled worker buffers: allocs -63%%, time -0.61%% geomean). So: it is always a RESOURCE optimization — fewer allocations means less allocator work and less GC scan pressure on every machine, which is why it holds across systems — and it is ALSO a throughput one exactly when the allocation is a real fraction of the per-iteration work. Measure the enclosing operation rather than assuming either way. Bytes barely move: the [][]T header stays and one slab rounds to a size class (+0.1% observed). PRECONDITIONS the check cannot verify: rows must not be appended to (the 3-index cap makes that safe but it then reallocates, losing the benefit) and must not be individually replaced later, since the views are only valid while the slab is. Jagged rows are already excluded — a length that varies with the loop variable needs per-row offsets, a different transform. Cannot be ranked syntactically: the payoff scales with the ITERATION COUNT, which is a runtime fact (PERF-HOTNESS-IS-NOT-SYNTAX-001), so prefer sites whose loop bound is a data size over ones bounded by a small constant", false},
	// PS3xxx — indirection / reflection overhead
	{"PS3001", "reflection-in-loop", "a reflection-based fmt scan (Sscanf/Sscan/Fscanf) in a loop", false},
	{"PS3002", "closure-comparator-sort", "a package sort (sort.Slice/SliceStable) with a comparator closure", false},
	{"PS3003", "int-key-map-in-loop", "a read of an integer-keyed map inside a loop", false},
	{"PS3005", "indirect-key-comparator", "a sort of an index slice whose comparator dereferences the sorted element into a 2-D structure — hoist the key into a flat column first", false},
	{"PS3007", "set-map-from-slice", "a membership SET (map[K]bool / map[K]struct{}) built by ranging a slice and then probed inside a loop. PS3003 excludes set-shaped maps because a sparse set is not the dense [0,N) lookup a slice would replace — true of DENSIFICATION, but this is a different transform: when the set's contents come from a slice the caller already owns, the fix is no map at all. MEASURED: nlp applyDRY hashed its sequence-breaker set once per window position, with runtime.mapaccess1_fast64 at 1.14s of the function's 1.99s cumulative (57%% of its own time); scanning DRYBreakers directly took BenchmarkApplyDRY 19.52us to 15.87us, -18.72%% at p=0.002 (n=6, interleaved, both arm orders), allocations unchanged, and mapaccess1_fast64 left the profile. The same measurement bounds the transform: forced onto each arm the crossover is 8-16 elements on an M2 Pro, so this is a SMALL-SET fix and large sets should keep the map. Silent on a set written after its build loop (a mutable working set genuinely needs a map) and on a build already guarded by a size THRESHOLD on the source, which is code that has taken this advice already — but NOT on an emptiness guard (len(src) > 0), whose branch is the only path rather than a fallback. Hotness is not visible to the AST: confirm the source is small and the probe repeats, then benchmark", false},
	{"PS3008", "monotone-bail-per-element", "a loop accumulating a provably NON-NEGATIVE term (x*x with identical operands, math.Abs, math.Hypot, or a sum of those) into a scalar that is tested against a threshold on EVERY iteration. The accumulator never decreases, so once it passes the threshold it stays past it: testing every 4th iteration returns the SAME answer and removes a data-dependent branch the predictor cannot learn. MEASURED on classic ballTree.within, the leaf test DBSCAN runs per candidate pair, where a line-level profile put the branch at 450ms against 30ms for the subtraction and square it guarded — checking every 4th dimension gave BenchmarkDBSCANFit -17.41%% at eps=2 (p=0.000) and -8.51%% at eps=4 (p=0.010), geomean -13.07%%, allocations unchanged, exact-label goldens green. The non-negativity is the CORRECTNESS condition and is required syntactically, since a signed term can dip back under the threshold and moving its test would change the answer. Keep one accumulator in the same order so the sum stays bit-identical, and end the scalar tail with !(acc > thr) rather than acc <= thr — with a NaN term the original never bailed and returned its not-exceeded answer, and <= flips it. Silent once the loop strides by more than 1, which is the applied form. Hotness is not visible: benchmark the enclosing operation before restructuring a cold bail-out", false},
	{"PS3056", "serial-permutation", "a multi-level nest that only COPIES elements between buffers — every write a read from somewhere else, no accumulation, no arithmetic folding the destination back in — in a package that declares a fan-out helper the function never calls. A permutation has NO dependence between its elements, so splitting the outer loop is race-free and BIT-IDENTICAL at any band count; there is no summation order to preserve because nothing is summed. It is the cheapest parallelization to justify and the easiest to overlook, since it carries no arithmetic to appear in a profile as a kernel. CHECK THE BAND OWNS DISJOINT OUTPUT — a transpose writes COLUMNS of its destination for a band of source rows, disjoint but not obvious, and a data-dependent scatter is not this shape. Gate it with BOTH a value comparison and -race: an overlapping band writes the same values and only the race detector sees it. MEASURED on the GGUF weight transpose, already cache-blocked and still 84%% of its own benchmark at one core: BenchmarkTiedHeadTransposePerCall went 154.2 ms to 49.8, a 66.3%% cut, on the model LOAD path; and on the transpose VJP, -45.7%% F64 and -50.8%% F32", false},
	{"PS3057", "column-read-through-a-jagged-matrix", "a loop that reads ONE column of a [][]T — an index expression x[row][col] whose ROW varies with the loop and whose COLUMN does not. Each read is a pointer chase into a separate length-d row, so n reads touch n cache lines spread across the whole n*d matrix and use 8 bytes of each; the same access against a feature-major mirror xt[col*n+row] is a scattered read inside ONE contiguous length-n array. Mirror the matrix once where it is already being walked and read the mirror everywhere. It is a PURE COPY, so nothing downstream moves by a bit — which also means only a bit-exact digest can gate it, and that the mirror COSTS n*d elements of memory: state the trade, do not hide it. MEASURED on the GBM exact-split builder, where the gather was 51%% of scanFeatures and 16%% of the package: BenchmarkGBMHist_exact_80k -19.5%%, _20k -13.6%% for +15.6%% bytes, with ForestFit flat as a control. RANK BY THE PRODUCT of the loop bound and how many times the same column is re-read — a column read once is a column that does not repay a mirror", false},
	{"PS3058", "per-iteration-scratch-allocation", "a value whose type has a SCRATCH INITIALIZER — a method that assigns 3 or more make() results to its own fields — constructed inside a loop or inside the callback of a fan-out helper. Every iteration allocates the whole working set and throws it away. Recycle it through a sync.Pool: growing each buffer only when the pooled one is too small, and returning it when the iteration finishes. THE SAFETY ARGUMENT IS USUALLY ALREADY MADE: a scratch buffer reused across the thousands of inner steps of ONE iteration is, by that fact, written before it is read, and nothing distinguishes the first step of a new iteration from the hundredth step of the old one. Check for the exception — a buffer the finished product still points into cannot be recycled. EXPECT MEMORY, NOT NECESSARILY TIME. MEASURED on the CART builder inside a random forest fit: ForestFit allocated bytes -42.4%% and allocations -34.4%%, with the wall clock FLAT (111.2 to 108.0 ms, inside the run-to-run spread) because the fit is already parallel and was not allocation-bound. Report it as a resource win or not at all", false},
	{"PS3059", "serial-nest-writing-through-a-derived-base", "a serial nest, in a package that declares a fan-out helper it never calls, whose every indexed write lands through a base DERIVED from the outermost loop variable — obase := b*out, then dst[obase+j] — with at least one write not naming that variable directly. This is PS3034's blind spot, and it has now cost two finds: PS3034 asks whether each write names the OUTERMOST loop variable, and a nest that hoists its row offset into a local does not. Both misses were large. The GGUF weight transpose was 84%% of its own benchmark at one core (-66.3%% once banded, T1117), and the KAN fused spline was 84%% of its layer's (-83.8%% at 256x256 and -79.0%% at 128x128, a 6.2x, T1123) — in a file whose OTHER hot loop had fanned out since it was written. WIDENING PS3034 WAS TRIED AND REVERTED: following derived names there broke three of its own fixtures and took its tree-wide count from 23 to 33 without flagging the transpose. A separate check with the narrow condition is what works. Check the band owns disjoint output before converting, and gate with BOTH a bit-exact digest and -race — which gate catches what depends on the destination: an accumulated one (+=) double-counts on an overlap so both fire, while a pure permutation writes the same value twice and only -race sees it", false},
	{"PS3060", "serial-loop-over-parallel-work", "a loop whose body calls a function that ITSELF fans out, inside a function that does not fan out at all. The inner level is parallel and the outer one is not, so the items run strictly one after another and each pays its own fork and join. THE WIN IS NOT ONLY THE WORK, IT IS THE STALLS: a Muon optimizer step spent 62%% of its profile samples in pthread_cond_wait and pthread_cond_signal and only 32%% in the matmul, because each parameter's five Newton-Schulz iterations fork three times each and nothing overlaps them. Banding the parameter loop returned -34.1%% with only TWO parameters, far more than two-way overlap of the work alone can explain. CALL ANY CALLER-SUPPLIED CALLBACK SERIALLY FIRST: a gradient or loss function belongs to the caller and is not documented as safe to call from several goroutines, so hoist it into a serial pass and fan out only what follows. And size the test above the fan-out helper's work gate, or both arms run serially and an overlapping-band mutation passes", false},
	{"PS3061", "fanout-not-sized-to-the-work", "a fan-out helper that always splits into GOMAXPROCS workers, with only an on/off work gate and no term that scales the WORKER COUNT to the work. Above the gate a small item still fans twelve ways, and the wakeups cost more than the split saves. MEASURED on the quantized decode matmul, which is called thousands of times per generation with one activation row: a profile of a 500-token quantized Llama generation spent 88%% of its samples in pthread_cond_signal and pthread_cond_wait and 1.5%% in the kernel itself. One worker per 1<<15 of element-work, capped by GOMAXPROCS, beat BOTH the fixed twelve and no fan-out at all — QuantLlamaGenerate500 549.7 to 527.4 ms, system CPU -27%%, with the prefill cell flat as a control. The on/off gate alone cannot express this: forcing the same call serial cost 8%% wall, and leaving it at twelve burned 42%% more user CPU for that 8%%. Pick the grain by measuring BOTH the clock and the CPU — a coarser grain traded 4%% of the clock for another 44%% of the system time here, and which side of that is right depends on whether the machine serves one request or many", false},
	{"PS3062", "op-with-no-optimized-kernel", "an operation registered in the REFERENCE backend and in no optimized one, so every caller on the default backend runs the correctness implementation. The reference is written to be obviously right, not fast, and the gap is routinely large: OpCholesky had no cpu kernel at all, and giving it one — ref's arithmetic line for line, with four rows taken per pass so the pivot row is loaded once and four independent accumulator chains run instead of one — measured 21.1 ms to 7.0 ms at n=512, a 3.0x, BIT-IDENTICAL to ref in both dtypes. THE FIRST ATTEMPT AT THAT KERNEL WAS SLOWER THAN REF. Fanning out the row update is the classical-factorization shape PS3040 describes and it lost, 24.1 ms against 22.0 with allocations going 6 to 878, because there are n columns and the fork is paid once per column for a row update that shrinks as the factorization proceeds. Arithmetic, not parallelism, was the lever. GATE A NEW KERNEL BIT-FOR-BIT AGAINST REF, not to a tolerance: a tolerance test passes a blocked or reassociated version too, and once one kernel disagrees every cross-backend golden silently becomes a tolerance test", false},
	{"PS3063", "one-loop-left-serial-in-a-fanning-function", "a serial nest inside a function that ALREADY fans out somewhere else. PS3034 and PS3059 both suppress this case — they require the function to never call a fan-out helper — and that suppression hides the most valuable shape there is, because a function that fans out has already proven the transform is available to it and simply left one loop behind. TWICE MEASURED, both large. The KAN forward fanned out buildBasis and not fusedSpline, which was 84%% of the layer: banding it went 67.28 to 10.90 ms, a 6.2x. The eigendecomposition adjoint fanned out its first two n^3 products and not the triangular third: banding that, plus mirroring the intermediate it reads down its columns, went 18.63 to 8.17 ms at n=256, a 2.3x, of which the banding was about 2x and the mirror about 15%%. CHECK THE WRITES OWN DISJOINT OUTPUT — a triangular loop writing (i,j) and (j,i) does, since a clash would force the two indices equal — and gate with BOTH a bit-exact digest and -race", false},
	{"PS3064", "jagged-matrix-allocated-row-by-row", "a [][]T allocated as one outer slice and then one make() per ROW, so an r-by-c matrix costs r+1 allocations and its rows land wherever the heap puts them. Back the rows with ONE block and window into it — d[i] = base[i*c:(i+1)*c:(i+1)*c] — and the call sites do not change at all, because the type does not. MEASURED across the three autograd factorization adjoints: allocations per call fell 1886 to 356 on the eigendecomposition, 1762 to 491 on the SVD and 1447 to 429 on the QR, a 70 to 81%% cut. THE CLOCK IS THE SMALLER HALF AND IS SHAPE-DEPENDENT: SVD -8.6%%, eigh -5.4%%, QR flat, with an untouched sibling adjoint flat as a control. Report it as a resource win and let the time be a bonus. CAP THE ROW WINDOW at its own length so an append copies instead of reaching into the next row — nothing in the converted code appends, and the cap is what keeps that from mattering later", false},
	{"PS3065", "serial-loop-over-an-expensive-call", "a SINGLE loop — no inner loop of its own — that writes an indexed destination from a call to a function which itself loops, in a package that declares a fan-out helper this function never calls. Every nest-based check misses this: PS3034, PS3059 and PS3063 all require depth 2 or more, and a loop whose per-item work lives inside a CALLEE has depth 1 however expensive it is. MEASURED on the RBF kernel cache, where a column is n independent kernel evaluations: BenchmarkSVCFit/n4000_rbf 6.99 to 5.48 ms, -21.5%%, with the GBM fit flat as a control. The fit had scaled at 1.02x on twelve cores before this. BIT-IDENTICAL — each entry performs exactly the arithmetic it did before and only which goroutine performs it moves. EXPECT AMDAHL, NOT THE CORE COUNT: the parallelized part was 40.6%% of a serial profile, so 6x on it is 1.27x overall, and the SMO iteration it feeds is sequential by construction", false},
	{"PS3066", "consecutive-loops-over-one-buffer", "three or more sibling loops in one block, all over the same bound, all indexing the same buffer — each streams the whole thing and evicts it before the next one starts. If every index is independent ACROSS the loops, merge them: the buffer is then touched once and stays in cache through all the stages. MEASURED on the Kimi delta-attention step, whose four per-step passes over the dv-by-dk state (scale, S dot k, the rank-1 delta write, S dot q) became one: BenchmarkKDA_F64_256x128 8.71 to 7.33 ms, -15.9%%. BIT-IDENTICAL because merging changes only WHEN a row is visited, never how — each row's stages already ran in that order relative to each other. THE WIN IS THE STATE SIZE, NOT THE LOOP COUNT: the same merge on a 64-by-64 state, 32 KB and already L1-resident, measured -1.8%%. Check for a cross-index dependency first — a later loop that needs ALL of an earlier loop's output cannot merge", false},
	{"PS3067", "sequential-outer-with-an-independent-inner-loop", "a nest whose OUTER loop carries a dependence and whose inner loop is independent — every write in it indexed by the inner variable, none by the outer. Banding the inner loop where it is pays one fork per OUTER step, which has now failed on fork count four times; INTERCHANGE FIRST so the independent index is outermost, then band it once. PS3040 describes the same nest and recommends splitting the inner loop in place, which is right for a factorization whose inner work SHRINKS as the outer index advances and wrong when the inner extent is constant. MEASURED TWICE on constant-extent sweeps: the SparseGPT pruner went 52.27 to 41.83 ms (-20.0%%) and the GPTQ quantizer 55.18 to 44.34 ms (-19.6%%), both with an untouched sibling flat as a control. BIT-IDENTICAL, because the rows do not observe each other and within a row the outer index is still visited in order. TWO THINGS TO CHECK. Any scratch the body reuses must become PER WORKER — a shared sort buffer reddened both the digest and -race. And a CALLER-SUPPLIED CALLBACK invoked inside the nest is now called concurrently: GPTQ's quantizer is a pure function of one value, but that had to become a documented requirement on the exported API rather than an implementation detail", false},
	{"PS3068", "serial-best-of-scan", "a loop over independent items that keeps a running BEST — a cost compared against an accumulator declared outside the loop, with the winner's fields assigned under a strict < or > — in a package that declares a fan-out helper this function never calls. Chunk the items, let every chunk apply the same rule within itself, then FOLD THE CHUNKS IN ASCENDING ORDER WITH THE SAME STRICT COMPARISON: that reproduces the serial winner exactly, ties included, because first-wins survives both levels. MEASURED on the CART feature scan, which ran at 1.01x on twelve cores: BenchmarkTreeFit 10.20 to 9.69 ms with the forest fit flat as a control. Modest because a tree's work sits in a few large nodes and the rest is below the fork gate — expect the gate to matter more than the core count. A DIGEST WILL NOT GATE THE TIE RULE. Mutations flipping either strict comparison to <= left a bit-exact prediction digest green, because ordinary data produces no exact-cost tie between two items. Manufacture one — duplicate an item so two are exactly equal — and assert the LOWER index wins, in BOTH placements: adjacent items share a chunk and exercise only the inner comparison, distant ones land in different chunks and exercise only the fold", false},
	{"PS3069", "fanout-queues-jobs-it-does-not-need-to", "a fan-out helper that hands work out through an UNBUFFERED channel, with no path that skips the queue when the job count is already within the worker count. The queue exists so more jobs than workers can be served one at a time; when there are no more, every job pays a rendezvous on top of its goroutine. MEASURED on the classic tree builders, which call their helper with exactly one job per chunk once per NODE, thousands of times per fit: adding a direct fan-out for n <= workers took BenchmarkGBMFit 72.06 to 68.31 ms, -5.2%%, with system CPU 0.43 to 0.37 s and the forest and single-tree cells flat. EXPECT THE SMALLER NUMBER, NOT THE PROFILE'S: 95.6%% of that benchmark's samples were in pthread_cond_wait, pthread_cond_signal and usleep against 1.75%% in the split scan, and the fix was worth 5%% — parked threads are sampled, and sampled is not the same as costing wall clock", false},
	{"PS3070", "one-threshold-two-regimes", "a tuning constant compared against DIFFERENT quantities in different functions — one threshold serving two callers whose costs do not move together. Whatever value suits one is wrong for the other, and tuning it looks like a tradeoff when it is really a missing second constant. MEASURED: classic's treeRadixCutoff gated both a PER-NODE sort of a shrinking range and a ONE-TIME presort of every row. Lowering it for the first took BenchmarkForestFit 121.5 to 101.3 ms, -16.6%%, and simultaneously cost GBMFit about 25%% and moved the GBM bit-exact digest — which read as a model-behavior change until the constants were split. Split, both are free: ForestFit -16.6%%, GBMFit flat, and BOTH digests pass unchanged. SPLIT FIRST, THEN SWEEP — a sweep of a shared constant measures the sum of two answers and finds neither", false},
	{"PS3071", "local-buffer-escapes-per-call", "a METHOD that declares a local fixed-size byte array and hands a slice of it to another call. If that call takes an interface — io.ReadFull and friends — the slice escapes and the array is HEAP-allocated on every invocation, which a reader primitive pays once per scalar it decodes. Hang the buffer on the receiver, which is already on the heap, and the per-call allocation disappears. MEASURED on the GGUF header reader: u32 and u64 were 1.34M objects of a 4.0M allocation profile, and moving their arrays onto the reader plus reusing one scratch for string bodies took BenchmarkReadFileSynth/header-heavy from 223892 to 95804 allocations, -57.2%%, and 5.45 to 4.50 ms, -17.3%%, with the tensor-heavy and skewed cells cutting allocations by the same proportion. SAFE ONLY IF NOTHING KEEPS THE BUFFER: the string case works because string(b) COPIES. Check every caller before sharing one scratch, and check the type is not used concurrently", false},
	{"PS3072", "serial-loop-reseeded-from-the-item", "a loop that RESEEDS, from the current item, the state it appears to carry. The carried object is overwritten before it is read, so the iteration is a pure function of its own item and the loop is bandable despite reading as a chain — a shape a scaling probe finds only after the fact. Give every worker its own copy of the reseeded object and of the scratch the body reuses, accumulate any count per worker and fold afterwards, since integer addition is order-free. MEASURED on the watermark detector, whose per-token partial Fisher-Yates reseeds a PCG from (key, previous token) and restores its permutation before the next: BenchmarkWatermarkDetect 39.17 to 7.17 ms, -81.7%%, bit-identical, from a 0.99x scaling ratio. CHECK THE RESTORE — the win depends on the body leaving its scratch as it found it", false},
	{"PS3073", "invariant-operand-reloaded-per-iteration", "a loop calling a REDUCTION HELPER with one operand that does not vary with the loop variable, so the same memory is re-streamed on every iteration. Jam the loop — 4 iterations per pass, one pass over the shared operand, each result keeping the helper own accumulator count, order and combination, which makes it bit-identical because the jammed dimension is the free one. MEASURED on the linalg Cholesky factorization, whose row update called a 4-accumulator dot with the pivot row shared: Cholesky512 8.99 to 6.25 ms, -30.5%%, Cholesky256 -24.7%%, CholSolve256x128 -13.7%%, bit-identical in both dtype arms, SVD flat as a control. THE SHARING IS THE SINGLE PASS, NOT THE UNROLLING — the same 4 rows written as two 2-row calls reloaded the operand twice and measured nothing. Also measured on the F32 arm of the cpu attention forward, whose F64 sibling had jammed its keys for a long time: MHA512/fwd/cpu 8.42 to 6.73 ms, -20.1%%. THAT ONE FIRST MEASURED FLAT — the five obvious F32 attention cells route to a GEMM kernel and never reach the arm; an execution counter read from a TestMain AFTER m.Run (tests run before benchmarks) found the one cell that did. Written INLINE it also moved an untouched F64 sibling arm by a ULP through a codegen shift; put the jam behind its own function. PS6010 is this transform for an INLINE accumulator loop; this check exists because a reduction behind a CALL is invisible to it", false},
	{"PS3074", "inner-loop-ranges-over-a-shared-operand", "an item loop whose INNER loop ranges over an operand that does not vary with the item, so the whole operand is re-streamed once per item while the body writes per-item output. Jam the item loop — 4 items per pass over one traversal of the shared operand, each item keeping its own accumulators; a shared accumulator stays bit-identical by being held in a local across the 4 additions in the same ascending item order. MEASURED on the cpu attention backward, whose two key loops re-streamed the gradient row and the query row: BenchmarkMHA512/bwd/cpu 24.73 to 13.40 ms, -45.8%% (1.85x), forward and masked cells flat as controls. THIS IS THE SHAPE PS6010 CANNOT SEE — it requires the shared operand to appear as an INDEX expression, and a range subject never does, which is why the line holding 39%% of that benchmark was reported by nothing", false},
	{"PS3075", "inner-loop-accumulates-into-a-shared-buffer", "an item loop whose INNER loop accumulates into a buffer that does not vary with the item, so every item makes a full load-store round trip through that buffer for one addition each. Jam the item loop — 4 items per pass, holding the accumulator element in a local across their four additions and storing once. BIT-IDENTICAL when the additions keep the same ascending item order. MEASURED on the cpu selective-attention kernel, where this weighted-sum loop was 26%% of the profile and a score loop in the same function 46%%: MHASelectF32CPU_1024x1024x64x16 191.3 to 100.9 ms, -47.3%% (1.90x), the 512 cell -42.7%%, the F64 arm -40.1%%, masked cell flat as a control. THE MIRROR OF PS3074 — there the inner loop SUBJECT is shared and the outputs per item; here the outputs are shared and the subject per item — and neither is reachable from PS6010", false},
	{"PS3076", "unroll-factor-fixed-at-two", "a register-blocked loop taking TWO steps per pass. Two is what a register-pressure argument produces; the optimum is what a sweep produces. MEASURED on both gemm band kernels, each carrying a comment reasoning that four would spill: sweeping 3, 4, 6 and 8 found SIX best and eight back at baseline. MTAForward_ch16 277.2 to 250.3 ms (-9.7%%), ch8 -9.5%% on the f32 band; GemmDirF64_1024 19.10 to 15.74 ms (-17.6%%), 512x2048x2048 -11.7%% on the f64 band. BIT-IDENTICAL AT ANY FACTOR — each accumulator takes one rounding per step in ascending order, never a summed pair — so the existing bit-exact oracle gates every arm and the whole sweep costs one measurement each. SWEEP, DO NOT ARGUE", false},
	{"PS3077", "minmax-clamp-in-a-loop", "math.Min wrapped around math.Max (or the reverse) inside a loop — a two-bound CLAMP written as two function calls that carry the whole NaN and signed-zero contract every iteration. MEASURED on the HQQ quantizer, 29%% of whose profile was archMin and archMax against 35%% of its own arithmetic: a comparison chain took BenchmarkHQQuantize 77.14 to 37.79 ms, -51.0%% (2.04x), an optimizer cell flat as a control. TRY internal/fmath FIRST (see PS3082) — it keeps the exact contract, is a smaller edit and measured faster than a chain (40.5 us against 63.6); the chain is the fallback for when the call is already gone. WRITE THE EQUIVALENT CHAIN, NOT THE OBVIOUS ONE: `if r <= lo` and not `<`, because `<` lets a negative zero through where math.Max(0,-0) returns +0; NaN must fall through both bounds untouched. GATE IT TWICE — a bit-for-bit table over -0, +0, NaN, both infinities and each boundary, AND a digest of the caller, since ordinary data never produces the cases that make the naive rewrite wrong", false},
	{"PS3082", "minmax-call-in-a-loop", "math.Min or math.Max called inside a loop. On arm64 math.Min compiles to CALL math.archMin inside a 48-byte frame with a stack-growth check; the min builtin compiles to a single FMIND in a leaf with no frame. THE TWO ARE NOT THE SAME FUNCTION and substituting raw is a real bug: math.Max documents +Inf as beating NaN and math.Min documents -Inf as beating NaN, while the builtins propagate NaN, so they disagree on exactly four ordered pairs. USE internal/fmath, which takes the instruction and consults math only when the instruction returns NaN — the only result the two can disagree on. MEASURED on the reference RL surrogates: PPOClip at batch 4096 100.5 to 41.6 us (-58.7%%), GRPO 126.7 to 79.8 (-37.0%%), GSPO 21.6 to 19.8 (-8.4%%, one clamp per sequence against a 256-token inner loop). IT DOES NOT BEAT AN EXISTING COMPARISON CHAIN — converting the PPO VJP's chain to fmath went 51.8 to 58.6 us, +13%%, and was reverted: this replaces CALLS, not branchless code. GATE IT ON ONE PLANTED VALUE PER CALL: a kernel that reduces to a scalar cannot see the divergence, because one NaN poisons the sum and both formulations then agree on NaN", false},
	{"PS3078", "radix-pass-cannot-be-skipped", "a byte-wise radix that builds its histogram INSIDE each pass, so it can never learn that a pass is the identity. A counting pass whose keys all land in one bucket emits them in read order and can be skipped: build all 8 histograms in ONE traversal, skip the uniform passes, and copy home when the surviving count is ODD (the fixed 8-pass form never had to). MEASURED on the CART builder, where the per-feature radix was 24%% of the profile: BenchmarkForestFit 88.6 to 79.4 ms, -10.4%%, every paired round, GBMFit and SVC flat. IT PAYS ON THE DATA, NOT THE CODE — float64 bit-keys of one feature column barely move in their sign and exponent bytes within a node. Full-entropy keys skip nothing, so measure the caller", false},
	{"PS3079", "per-job-whole-input-allocation", "a fan-out body calling a function that ALLOCATES and returns slices sized by its input, so every job pays a whole input\u0027s worth of allocation. Recycle through a sync.Pool taken at the top of the job and returned at its end, resizing only when the job needs more. MEASURED on the random forest, where each tree materialized its own row-pointer slice and label copy: BenchmarkForestFit 33.70 to 20.14 MB/op, -40.2%%, and 1883 to 1666 allocations, with the wall clock FLAT (78.4 vs 77.6 ms). EXPECT BYTES, NOT TIME. THE SAFETY QUESTION IS RETENTION and it is answerable by reading what the callee stores — the tree fitter keeps only its root, class set and feature count — plus whether the buffers are fully overwritten before being read. If either is false the pool is a correctness bug", false},
	{"PS3080", "one-dimensional-accessor-walk", "a loop making 3 or more AtF64/SetF64 calls per element, each indexed by the loop variable ALONE. PS1005 reports the multi-dimensional version and declines this one — it requires 2 or more index arguments — so a rank-1 walk is invisible to it. MEASURED on the PPO clipped-surrogate backward, 4 such calls per element and NO benchmark until one was written: BenchmarkPPOVJP_65536 2000 to 680 microseconds and the 4096 cell 124 to 42, both -66%% (2.9x). Take the typed slice once when every operand is already the right dtype and KEEP the accessor arm, because the output dtype follows the input. The 2 arms cannot be compared as equal bits — the accessor arm stores float32 where the typed one stores float64 — so what must hold is that the accessor result equals the typed one rounded once", false},
	{"PS3081", "operand-streamed-once-per-output-unit", "a per-output-unit closure whose loop cuts a slice by an index unrelated to the unit — so that whole operand is streamed again for every output. Block the unit by 3: derive 3 per-unit operands and compute 3 outputs from ONE pass over the shared one. MEASURED on the quantized matmul, 8 activation elements and 1 weight element per 8 FMAs once per output column: BenchmarkQuantMamba2Prefill_512 276.2 to 216.4 ms, -21.6%%, decode flat. THREE, AND SWEPT — 2, 3 and 4 read 237.2, 220.3 and 261.3 against 270.8, so four spills. PASS THE PER-UNIT SCRATCH AS NAMED PARAMETERS, NOT A SLICE OF SLICES: identical arithmetic measured 237.2 against 222.5 at two columns, so an indexed harness understates every arm of the sweep. BIT-IDENTICAL when each output keeps its own accumulator over the same ascending index", false},
	{"PS3055", "sort-then-truncate", "a slice sorted in FULL and then cut to a small prefix. Everything past the cut was ordered for nothing. SELECT INSTEAD OF SORTING: a bounded worst-at-root heap keeps the best k in O(n log k), and sorting just those reproduces the prefix exactly. BIT-IDENTICAL WHEN THE COMPARATOR IS A STRICT TOTAL ORDER — check that first, since with genuine ties a heap and a sort can disagree about which equal element is kept. MEASURED on diverse beam search, which sorted every beam whole vocabulary expansion to keep a handful: BenchmarkDiverseBeamSearch/cheap fell 90.5%% and /realistic 42.7%%, with plain beam search unmoved. GATE THE ORDER, NOT JUST THE SET — leaving the survivors in heap order passed every existing test of the measured site, because the result is re-sorted before return and the permutation only shows up in the NEXT step. BUILD THE HEAP FROM A COPY if it reuses the input array", false},
	{"PS3054", "asymmetric-dtype-arm", "an if/else on a TYPE-ASSERTION flag whose arms are not the same shape: one spells its reduction out as a scalar loop while the other hands the same work to a helper. That is a half-finished optimization — one dtype unrolled or vectorized, the twin left — and it survives because the suite usually has a cell for the optimized dtype only. BRING THE ARMS LEVEL, AND ADD THE MISSING CELL FIRST: a change to one arm reads as NOISE against a benchmark entering the other. MEASURED TWICE — the flash attention f64 scores read 7.38/7.22/7.63 ms against the f32 cell and -35.7%% against an f64 cell added for them; the retention backward had the same split in two places and matching them took BenchmarkRetentionBwdF64 down 25.3%% with the f32 cell unmoved. CHECK WHICH ARM IS BEHIND: a helper that reassociates may be gated to a tolerance the other dtype lacks, in which case the scalar arm is correct and needs an EXACT grouping rather than the same call", false},
	{"PS3053", "independent-reductions-one-at-a-time", "a loop over items where each computes its own SCALAR reduction over the same shared source. A single-accumulator reduction is a DEPENDENT add chain, so the loop is bound by add LATENCY not throughput, and running the items one at a time leaves the chains end to end when they could interleave. TAKE FOUR ITEMS PER PASS with four separate accumulators, loading each source element once for all four. BIT-IDENTICAL, which is what separates this from PS3010: there the fix splits ONE sum into partials and reassociates, here the accumulators belong to DIFFERENT results and each keeps its own ascending order. MEASURED on the memorizing-attention k-nearest-neighbour scan, whose per-query dot was 44.5%% of the benchmark: BenchmarkMemForward_512 fell 34.6%%, the 128 cell 22.6%%, BenchmarkMemGatherLarge 44.6%%. DESIGN THE GATING MUTATION WITH CARE — the observable is which items get SELECTED, so a perturbation that scales or shifts one item's scores uniformly changes no ranking and leaves the oracle green (a 1%% scale and a constant offset both did); it must depend on the shared source's index and be large enough to reorder", false},
	{"PS3052", "staged-matrix-reduced-against-one-column", "a buffer filled by one call and consumed by the very next, where the consumer output width is a variable the function never branches on. At width one the staged buffer exists only to be reduced against a SINGLE vector — every value written once and read once, when it could have come straight from the source. ADD A FUSED PATH FOR WIDTH ONE, visiting the same elements in the same order into one accumulator. MEASURED on conv2d, whose im2col matrix writes c*kh*kw values per output pixel: with one output filter the fill was 12.7%% of a multi-token-attention forward against 6.7%% for the GEMM it fed, and fusing took a 256x256 3x3 single-filter convolution down 22.1%% and the forward 3.6%%. MIND THE ZEROS THE STAGING HOLDS — conv2d columns carry zeros for padded taps and the GEMM adds them, so the fused path is exact only without padding; gate on that and test BOTH sides. THE GAIN IS IN THE SHORT KERNELS: 22.1%% at 3x3 against 4.3%% at 6x11, where the staged read was already long enough to amortize", false},
	{"PS3051", "blocked-kernel-without-a-degenerate-shape-guard", "a kernel that BLOCKS one dimension while iterating another innermost, with nothing special-cased for that inner dimension being 1. At width one the innermost loop runs a single iteration, so the block pays all its slicing and loop machinery to move one element per pass and the accumulators sit in memory when they would fit in registers. ADD THE DEGENERATE PATH: same blocking, accumulators held as SCALARS across the whole reduction, stored once — bit-identical, since each still takes its terms in the same order. MEASURED on the CPU band GEMM, which a conv2d with one output filter reaches as n == 1: a 2048-square matrix-vector product fell 23.4%% in f64 and 36.7%% in f32, the conv-shaped 262144x66 case 17.7%%, the attention forward 6.3%%. MEASURE THE OBVIOUS FORM TOO AND EXPECT IT TO LOSE: a plain per-row dot was tried first and was slightly WORSE than the block it replaced, because the block amortizes its loop over four rows — the win is the registers, not the simpler loop", false},
	{"PS3050", "serial-tail-after-fanout", "a fan-out call followed, in the same function, by a serial elementwise pass over a whole buffer the bands already own. Every worker finishes and then one goroutine walks the entire output again — an Amdahl term that grows with the output rather than with the work. FOLD IT INTO THE BAND: each band owns rows [lo,hi), so doing its own slice there is disjoint, elementwise and BIT-IDENTICAL, because nothing accumulates. MEASURED on the portable f32 matmul, which accumulates into an f64 scratch and narrowed the whole result afterwards: the tail was 6.4%% of a batched vision forward at one worker, and folding it took three of those forwards down 4.7%% to 7.6%% with the f64 matmul beside them flat. GATE IT ON A SENTINEL — pre-fill the output with a value the correct result cannot produce, so a band that narrows the wrong range shows as an untouched cell", false},
	{"PS3049", "axpy-reloads-its-destination", "a nest whose inner loop accumulates into a destination the OUTER loop does not choose, from operands that move with it. Every outer iteration reads and writes the whole destination again, so each element carries a load-modify-store chain through memory once per outer step. UNROLL THE OUTER LOOP AND HOLD THE RUNNING VALUE: FOUR outer steps per pass, load the element once, add each contribution, store once. Four is measured, not as many as fit: 1, 4 and 8 steps per pass on the decode kernel came to 823 ms, 682 ms and 937 ms. THE INNER PASS MUST BE LONG, because the extra scalars and row bases are set up once per pass — the identical transform on an attention value accumulation of one head width cost 6 to 9%% on three attention cells and was reverted. ADD THEM ONE AT A TIME, NOT AS A SUM — v += a0*x0; v += a1*x1 keeps the ascending accumulation order and is BIT-IDENTICAL, while summing the products first reassociates. MEASURED on the decode matrix-vector kernel, 46.6%% of a generate loop serial profile: four rows per pass took BenchmarkGPTGenerate500RowBuf down 12.6%%, CLA decode 10.8%%, T5 decode 13.5%% and the Llama prompt 26.7%%, with the blocked matmul cells untouched. TEST THE REMAINDER AND A NON-ZERO WINDOW: an unrolled body that rebuilds its source offsets from the loop variable alone agrees with the original on a full-width call and on nothing else", false},
	{"PS3048", "fanout-without-a-work-floor", "a fan-out helper that takes its worker count from GOMAXPROCS and gates only on the TOTAL work, with nothing bounding the work each worker receives. An op just over the total threshold is then split every way the machine allows and each band carries a fraction of the amount that justified splitting at all. DERIVE THE COUNT FROM THE WORK: workers = min(GOMAXPROCS, total/floor), falling back to the serial body at one. MEASURED on this repository CPU pool, where a per-token decode step issues a long run of ops just above the threshold: the profile was 42%% runtime.usleep, 22%% cond_wait and 9%% cond_signal against 14%% arithmetic, and BenchmarkLlamaPromptStepwise ran 2.3x SLOWER at twelve cores than at one. A floor equal to the fan-out threshold took Llama down 36%%, Mixtral 44.7%% and Mamba prefill 27.6%% with the large-op cells unchanged. THE CURVE IS NOT MONOTONE — swept at 2^14 to 2^17 the times were 153, 85, 110 and 127 ms, so pick the floor by measurement. CHANGING THE BAND COUNT MUST NOT CHANGE A VALUE, which the GOMAXPROCS parity tests are what prove", false},
	{"PS3047", "one-shared-accumulator-blocks-split", "a loop over a dimension whose accumulating writes are ALL indexed by that dimension except ONE. That single exception is the only thing keeping the loop serial. RECORD AND FOLD: run the iterations in parallel, storing the shared accumulator per-item FACTOR into a buffer instead of adding it, then fold that buffer afterwards in the original iteration order — every add then happens in the sequence the serial loop used, so the result is BIT-IDENTICAL, which per-iteration partial sums merged at the end are NOT. MEASURED on the MLA attention backward, where four of five gradients are written at head-chosen columns and the fifth is the shared decoupled-key gradient: BenchmarkMLAVJPSeq256 fell 67.9%% and Seq128 66.7%%. SIZE THE RECORDING BUFFER FIRST — it holds one value per (iteration, inner index) pair, so process iterations in GROUPS that keep it under a budget; a group of one degrades to the serial form and stays correct. THE FOLD MUST REPRODUCE THE ORIGINAL BOUNDS: a triangular inner range folded with the rectangular bound agrees with the serial form on the rectangular case and nowhere else", false},
	{"PS3046", "item-reduction-into-partitioned-windows", "a loop over items whose inner loops accumulate into a WINDOW of a shared destination, cut at an offset the item does not appear in. The item loop is a REDUCTION and cannot fan out, but the loops that CHOOSE the window already partition the destination into disjoint pieces, and splitting one of those is safe: each worker owns whole windows, every window still sums its items in ascending order, and the result is BIT-IDENTICAL. MEASURED on the softmax regression Hessian, where every sample contributes to every (class pair, feature) window: BenchmarkSoftmaxRegressionFit fell 43.9%%, the two smaller softmax cells 24.5%% and 26.6%%. CUT THE BANDS ON CUMULATIVE WORK WHEN THE INNER RANGE IS TRIANGULAR — a loop whose iteration a writes m-a columns gives its first band about 2m/workers times the last band under an equal-count split. CHECK THE GATE AGAINST THE REAL SHAPE: this one first measured as no change at all because the work estimate fell 4%% short of the threshold and the split never ran. PS3045 is the sibling for a scatter whose offset within the window is chosen by the DATA", false},
	{"PS3045", "colliding-scatter-with-partitionable-destination", "an item loop whose inner loop over a second dimension accumulates into a destination indexed by that dimension PLUS a data-dependent offset. Two items can land on the same slot, so the ITEM loop cannot fan out — but the inner dimension partitions the destination into disjoint windows, and splitting THERE is safe: each worker owns whole windows, every slot still accumulates its items in ascending item order, and the result is BIT-IDENTICAL. Per-worker partial copies merged afterwards would reassociate every sum and are not. MEASURED on the histogram gradient-boosting builder, where every sample updates one bin of every feature: BenchmarkGBMHist_hist_80k fell 19.0%%, the 20k cell 11.3%%. FLOOR THE WINDOWS PER WORKER — every worker re-walks the WHOLE item list, so that walk is paid once per worker instead of once in total; swept at 20 features on 12 cores, 4 per worker was best and 8 was 10%% worse because it left only two workers. GATE ON items times windows so small calls stay serial", false},
	{"PS3044", "serial-reduction-blocks-parallel-map", "a loop that computes a per-item value with a call and then folds it into shared state at an index the item does not determine. The fold is a REDUCTION — two items can land on the same slot — so the loop cannot fan out as written, and the call in front of it usually holds all the time. SPLIT THE MAP FROM THE REDUCE: run the call over the items in a parallel pass writing a per-item array, then fold that array in ascending item order exactly as before. No partial sums are merged and every accumulator takes the same terms in the same order, so the result is BIT-IDENTICAL and a golden from the previous implementation gates it. MEASURED on the AQLM encoder k-means assignment, where the nearest-centroid search costs k*dim and the fold costs dim: BenchmarkEncodeAQLM fell 37.2%%. RANK IT AGAINST WALL CLOCK — a serial stretch pays its full CPU time while the rest of the path is parallel. If the map is CHEAP relative to the fold there is nothing here: the extra array and the second pass are then the whole cost", false},
	{"PS3043", "source-rebound-per-output", "a nest whose OUTER loop owns a destination and whose INNER loop rebinds a source element from a collection. The set of sources does not depend on the outer variable, so it is re-read once per outer iteration and the pass moves outer-count times the source volume through the caches. INTERCHANGE THE LOOPS: bind each source once and update every destination while it is loaded. The per-destination accumulation order is unchanged, so the result stays BIT-IDENTICAL and an existing exact-equality gate still applies. MEASURED on the multi-token-attention head mix, out[o] = sum_p w[o,p]*maps[p] written as a loop over outputs re-reading every map — 16 passes over 33 MB per group; BenchmarkMTAForward_ch16 fell 11.7%% once the source loop moved out and the element range was split across workers. THE INTERCHANGE ALONE IS RARELY THE WIN: it makes the destinations live at once, so hold them for a BAND of elements, and check whether the loop was serial while the rest of the path was parallel — a serial stretch costs its full CPU time in wall clock however small its profile share looks", false},
	{"PS3042", "whole-tensor-staging-buffer", "a scratch buffer allocated before a fan-out call, sized by the fan-out's ITEM COUNT times a width, and touched only inside the callback. Each band writes and reads only its own rows, so the buffer's SIZE is set by the whole tensor while its WORKING SET is one band: every element the producing stage writes goes out to memory and comes back for the consuming stage. Size it per band or per chunk and hand each band its own window. MEASURED on conv2d, whose im2col column matrix was rows x k — one 512x512 head convolution materialized 138 MB to multiply it by a 66-element weight vector; chunked to an L2-resident window the largest torch conv shape went -11%%, the attention forward -12%%, B/op fell 62-88%%. CHECK WHETHER THE CONSUMER ACCUMULATES: a whole-tensor buffer writes each row's slot once, so an accumulating consumer reads as a store and the pool's zeroing is invisible; a reused window must be cleared between chunks. CAP THE CHUNK AT ONE BAND, or the total becomes workers x chunk — more memory than before on inputs too small to have had a problem", false},
	{"PS3041", "per-item-rescan-of-shared-collection", "an outer loop over items whose body walks a collection held on the receiver — directly, or one same-type method call deep — without that walk depending on the item. Every item re-reads the same memory and reuses none of it, so the pass moves items x collection bytes through the caches and is BANDWIDTH-bound however cheap the arithmetic is. Batch the item loop into TILES: load each element once and do all B items' work on it while it is in cache. MEASURED on the memorizing-attention neighbour search, where each query token scanned the whole key bank alone — tiles of 16 cut BenchmarkMemForward_512 by 24%% and BenchmarkMemGatherLarge by 30%%. CONFIRM THE DIAGNOSIS FIRST: if the collection already fits in L2 the traffic was never the cost and tiling buys nothing. THE TILE MUST NOT REASSOCIATE — give each item its own accumulator and its own result state and visit the collection in the same order, and the output stays bit-identical, which is what lets the existing goldens gate the rewrite. PS3034 and PS3040 are about UNUSED PARALLELISM in a nest; this one fires on a loop that may already be parallel and is still re-streaming its data", false},
	{"PS3040", "inner-independent-under-sequential-outer", "a three-deep nest whose outer loop carries a real dependence — it is read, never written — while the MIDDLE loop is independent: every write is indexed by the middle variable and none by the outer. The outer cannot be split and the middle can, so the fan-out belongs one level in. That is the shape of every classical factorization: pivot in order, update the remaining rows in parallel. MEASURED on an LU rank-1 update that was 92%% of its own benchmark on ONE line: -40.8%% at 512 wide, -11.1%% at 256, 128 unchanged below the gate. PS3034 does not cover this and should not — it asks whether the OUTER loop can be split. Two conversion requirements: gate on the work at THIS step (rows times columns, not the row count) or mid-sized inputs stay serial and it reads as a size effect; and keep the below-gate path a PLAIN duplicated loop, because routing a 128-wide factorization through the callback cost 3 to 4%% that hoisting the gate did not recover. Gate it with an oracle blind to the internals — a solve residual caught a dropped row that every existing test in the package missed. EXPECT LESS THAN THE SERIAL SHARE PROMISES, and by a lot when the outer loop is long: a Gauss-Jordan solve that was 40%% of an AQLM encode's wall clock returned only 6.6%% end to end, because EVERY pivot step pays its own fork and there are n of them. Widening or narrowing the gate did not recover it (1<<12, 1<<14, 1<<16 and 1<<18 all measured, best at 1<<14). Divide the serial share by the fork count before promising anything", false},
	{"PS3039", "recursive-split-alloc", "a self-recursive function that allocates TWO slices sized by its input, fills them by appending each element to one or the other, and passes them to its own recursive calls — a divide-and-conquer partition written the allocating way. The cost is per NODE of the recursion, so it scales with the tree rather than with the data. Partition in place against one reused buffer instead. MEASURED on a CART builder\u0027s subsampled path: 352029 to 192021 allocs/op (-45.5%%), bytes -63.9%%, ns/op -6.8%% to -9.2%% against a control drifting under 2%%. Safe because writing dst[mid] while ranging over dst cannot clobber an unread element (mid advances only on a write, and every write consumes a value already read), and copying the second side back in order preserves what both appends produced. GATE IT WITH AN EXACT GOLDEN GENERATED FROM THE OLD CODE: the property tests that usually cover a tree builder stay green for a DIFFERENT tree, and on the measured site they stayed green with the copy-back deleted", false},
	{"PS3038", "dispatch-literal-slice", "a direct backend.Execute building its input slice inline, in a package that already declares a pooled helper of that arity. The literal is one allocation per dispatch, and Execute drops the slice the instant it returns unless a recorder is attached — which is exactly what the pooled helper checks before borrowing. MEASURED on nn.Linear.Forward, the most-called forward in the package: two literals became two pooled borrows and a per-image MLP-Mixer forward went 3944 to 3687 allocs/op (-6.5%%), a ViT forward -2.0%%. Judge on allocs/op; the time was flat everywhere, since these forwards are dominated by the kernels the slice merely names. THE RECORDER GUARD IS THE CONTRACT: Execute\u0027s tape node stores that exact slice, so a pooled one would be overwritten by the next op and a training run would silently get wrong gradients — use the helper, never inline the borrow. 214 sit in nn alone, so rank by call frequency and convert where a benchmark can see it", false},
	{"PS3037", "mis-sized-append-buffer", "a slice made with a stated capacity inside a loop and then appended to from a NESTED loop, so the hint is sized per outer pass while the appends run outer times inner. The hint guarantees the opposite of what it looks like: the slice doubles its way up from the hint to its true size on EVERY outer pass, copying everything it holds each time. MEASURED on a beam search whose hint was 8 per live beam against a true size of one candidate per beam per VOCABULARY ENTRY — that one append line was 2.45 GB of a 2.90 GB benchmark, and hoisting the buffer above the loop with a per-pass truncation took bytes -85.0%% on beam search and -98.0%% on its diverse variant. Judge on B/op first: the time win was -8.8%% and -2.9%%, real but far smaller. Prove nothing survives the reset — dropping the truncation must redden the suite, and on the measured site it did. PS3035 does not cover this: it wants a size the loop does not vary and only sees loops with a range clause or an init statement, and the measured site is a bare condition loop whose hint mentions the collection it iterates", false},
	{"PS3036", "self-comparison-oracle", "a test that computes BOTH sides of its comparison with the same function and asserts they agree, so the expected value is produced by the code under test. Such a gate can only see state carried BETWEEN calls: any mistake INSIDE the computation changes both sides identically and it stays green. Found exactly that way — a Newton-Schulz orthogonalization gate of this shape passed with two intermediate buffers wired to one slice, and only an independently written reference caught it. Add a slow, obvious implementation with its own buffers and compare to a tolerance when the summation orders differ. SOMETIMES DELIBERATE: comparing one function across a CONFIG difference (GOMAXPROCS 1 against many, a fast path against a fallback) is a real differential test — it still gates only that difference, so document it and keep a separate reference for the arithmetic. Matters because every optimization here is defended by a bit-identity gate, and a gate that cannot fail reports coverage that does not exist. Requires -tests", false},
	{"PS3035", "loop-hoistable-scratch", "a slice allocated with make at the top of a loop body, sized by something the loop does not vary, that never leaves the iteration — one buffer made and thrown away per pass. Hoist it above the loop. MEASURED on a Cholesky solve whose forward-substitution buffer was allocated per right-hand side: at 128 columns, 133 allocations became 43 and bytes fell 18.1%%. PS2001 does NOT cover this, since it fires only on the configured tensor allocators, so a plain make of a slice in a loop is invisible to every other check in this table. PROVE THE OVERWRITE BEFORE HOISTING: a fresh make is zeroed and a reused buffer is not, so an iteration that reads a slot before writing it would silently start seeing the previous pass\u0027s value — poison with NaN between iterations, confirm green, then delete one write and confirm red. JUDGE ON allocs/op AND B/op; the time win is usually nil and was here at both shapes. A buffer allocated once per WORKER BAND is already in the right place (PS6008) — bounded by GOMAXPROCS", false},
	{"PS3034", "serial-nest-with-idle-fanout", "a three-deep nest filling a destination the function itself allocated, indexed by the OUTERMOST loop variable so the iterations are independent, running on one core in a package that already declares a fan-out helper. Split the outer loop into bands: each owns whole rows of the destination and every element accumulates in the same order, so the result is BIT-IDENTICAL and the gate asserts exact equality rather than a tolerance. MEASURED on a Muon optimizer step whose flat matmul was 48.8%% of the profile and serial: 195.8ms to 77.4ms, -60.5%%, confirmed in both benchmark orders. GATE ON THE PRODUCT of the loop bounds, not the outer count — Newton-Schulz drives a few rows through a great deal of work each, and a row-count gate leaves exactly that shape serial. Expect allocations to RISE (46 to 568 per step, one closure per band) with bytes flat; that trade is the point. Moving the first band onto the calling goroutine was measured here and did nothing. Silent when any write is indexed without the outer variable, which is a cross-iteration accumulator a split would race on", false},
	{"PS3033", "per-item-alloc-helper", "a package-local helper that allocates its result with make and returns it, whose every in-file call site is PER ITEM — inside a loop, or inside a helper that is itself only called per item. The buffer outlives nothing, so it belongs to the WORKER rather than the item: take a scratch parameter, grow it only when cap is short, reslice it to the length the caller needs. MEASURED on a k-NN predict where the k-best heap and its backing array, the neighbour weights and the per-class vote accumulator were all per query: 36020 allocations per batch became 92, -99.7%%, bytes -97.7%%, ns/op -4.9%%. JUDGE ON allocs/op AND B/op, NOT ns/op — the time win is the collector not walking those objects, so it is small where memory bandwidth is plentiful and larger where it is not. The reachability is a FIXED POINT, not a single pass: the measured allocator was called from another helper which was called from the per-row callback, so no lexical test sees it. Two conversion traps: every reused buffer must be fully overwritten before it is read (truncate, reslice, CLEAR), and a result the caller KEEPS must still be copied out or every row of a chunk shares one array. Prefer sizing a staging buffer by what the caller will WRITE rather than by the maximum — sized to the write there is no stale tail and no clearing pass, and a mismatched count fails a length check instead of reading the previous item. Silent when a call site stores the result into an index or a field, and on exported helpers, whose callers this file cannot see", false},
	{"PS3032", "closure-accessor-in-loop", "a function VALUE obtained from a factory call and then invoked inside a loop, so every element pays an indirect call that cannot be inlined. This is the per-element dispatch anti-pattern one level shallower, and it hides better: a helper handing back readers and writers reads like setup, and the cost is in the calls rather than in the helper. Add typed arms walking raw storage and keep the closure form as the fallback for dtypes the typed arms cannot serve. MEASURED on two pooling backward rules: -48.2%% to -53.2%% across four cells. TWO TRAPS IN THE CONVERSION, both hit while making that change: the closure boundary BLOCKS FMA CONTRACTION that a typed arm allows — a scale product and an accumulating add in one function fuse where a call between them cannot — so wrap the product in an explicit conversion or the arms drift an ulp; and the parity fixture must make an element receive SEVERAL accumulations, or f32 narrowing differences cannot appear and a wrong arm passes. No type information is needed to find this: a name can only be CALLED if it holds a function", false},
	{"PS3031", "symmetric-pair-computed-twice", "a full i,j nest accumulating a term AND its mirror in the same body, so every pair is formed twice over the full range and the diagonal forms the identical sum twice. Run the inner loop from the outer index and write both positions. BIT-IDENTICAL when the store is a SYMMETRIC combination of the two sums: the full loop stored f(b,a) at the mirrored position where the triangle stores f(a,b), and IEEE addition is commutative, so a+b and b+a have the same bits for every non-NaN operand; each sum keeps its own operands and ascending order, so nothing is reassociated. MEASURED TWICE, both about a third: a Cholesky VJP at -34.33%% and an eigh VJP at -33.9%%. CHECK THE STORE FIRST — if what is written is not symmetric in the two sums the mirror is not free and this does not apply", false},
	{"PS3030", "fixed-offset-stores-not-windowed", "a counted loop touching ONE slice at three or more distinct CONSTANT offsets from an invariant base plus the loop variable. Each access carries its own bounds check, in a body that may be only a few operations wide. Cut one fixed-length window above the loop and index it by the offsets alone, leaving a single slice check per group. MEASURED on a Q6_K dequantizer with four stores per iteration: -16.5%%, with the compiler's BCE diagnostic confirming four per-store checks gone and one slice check left. PURE ADDRESSING — no value changes, so existing goldens are the right gate. Look for siblings before assuming novelty: that site was the LAST of its family to be cut and the same file's dot-product twin had already done it. Distinct from PS3019, which is about an unrolled loop whose lanes sit at i+0..i+K-1 under a len bound; here the loop steps by one and the offsets are the strides of a packed group", false},
	{"PS3029", "unbuffered-file-to-parser", "a file handle opened in this function and passed straight to a callee with no buffering in between. If that callee reads FIELD BY FIELD — a length, then the bytes, for every string — each is its own read syscall. Wrap it in bufio.NewReaderSize. MEASURED on a GGUF loader whose header is dominated by tokenizer arrays: a 32k-token vocabulary cost on the order of 160k syscalls before a single tensor was touched, and buffering took the load from 66.0ms to 5.5ms, -91.7%%. The tensor-heavy shape of the same benchmark moved only -18.9%%, which is the tell — THIS COST IS CONSTANT IN FILE SIZE, so it is worst where the file is smallest and it hides completely behind a benchmark that only loads large payloads. Cost is one allocation of the buffer size per open. Silent when the handle goes to a bulk consumer (io.ReadAll, io.Copy, io.ReadFull, an existing bufio wrapper), where buffering buys nothing and costs a copy", false},
	{"PS3028", "unpooled-fully-overwritten-scratch", "a per-call scratch buffer sized by a product of three or more dimensions, where every write in the function is a plain assignment rather than an accumulate — so no slot is read before it is written and the runtime's zeroing of a fresh allocation buys nothing. Recycle it through a sync.Pool. MEASURED: an attention backward's per-head contribution buffer was 16.7 MB at 8 heads and 512x512; pooling cut allocation bytes 53%%, 25.2 MB to 11.8 MB per call. EXPECT NO SPEEDUP — ns/op did not move at all on that kernel, since a memset is a few percent of a body doing heads*sq*sk*dk MACs. This is a resource finding: its value is that the buffer grows with the square of the sequence while the compute grows with the square times the head dimension, so the ratio worsens with length. BEFORE SHIPPING, PROVE THE OVERWRITE instead of trusting the check: poison every borrowed buffer with NaN and confirm the suite stays green, then delete one write and confirm it reddens — this scanner sees the SHAPE of the writes, not their coverage. Silent when the function already mentions a pool, and when any write to the buffer accumulates", false},
	{"PS3027", "input-view-on-output-tensor", "a READ-ONLY view helper (configured inputViewFuncs) applied to a tensor the function ALLOCATED as an output. These helpers return the live storage when the dtype matches their element type and a WIDENED COPY when it does not, so on the mismatched dtype a kernel accumulates into a buffer nobody reads and the output comes back untouched — right shape, no error, all zeros. FOUND EXACTLY THIS WAY: a masked-attention backward returned four all-zero gradient tensors on F32 while F64 was correct, so an F32 fine-tune of a trainable attention bias propagated no gradient at all; it survived because every test touching the op built F64 tensors. Use the output counterpart (configured outputViewFuncs), which returns a buffer plus a flush, and call the flush before every fast-path return. Then add a test in the OTHER dtype — this class hides precisely because the fast path is exercised in the dtype where the view happens to alias. Not a performance finding: it is the correctness cost of a devirtualization, which is why it lives with the checks that guard those", false},
	{"PS3026", "full-fanout-under-topk-gate", "a function that picks a SUBSET of branches with a top-k gate and then evaluates EVERY branch anyway, leaving the unselected ones to be multiplied by a zero weight. Skip them: mark the chosen indices in the selection loop that already exists and continue past the rest. The result is the same BITS, not merely close — an unselected branch contributes output times exactly zero, adding an exact zero returns a finite accumulator unchanged, and the surviving addends keep their relative order; state the two exceptions in the doc, a negative-zero accumulator sign and a NaN or Inf escaping a branch nobody routed to. MEASURED on a mixture-of-experts decode step: -23.7%% ns/op, -20.8%% allocs, 8 samples per arm interleaved in both orders. These fan-outs are usually GEMVs, so the step is bound by the weight bytes it streams and skipping k of E branches removes that fraction of the footprint directly. BENCHMARK PROTOCOL: when the benchmark carries state across iterations — a growing KV cache is the common case — pin -benchtime to a fixed count and interleave the arms in BOTH orders; a single-order run of this very change reported it as 239%% SLOWER. Selector names come from the configured topKSelectorFuncs list, since the gate and the fan-out are a project's own vocabulary", false},
	{"PS3025", "unrounded-product-under-exactness-claim", "a function whose doc claims bit-identity with a DIFFERENT implementation while its body contains a bare multiply feeding an add. Go contracts that into an FMA on arm64 and not on amd64, so the product rounds once here and twice there and the two paths differ by an ulp on one architecture only — invisible to amd64 CI. Wrap the product in an explicit float64/float32 conversion, the only construct the spec guarantees forces the rounding; an intermediate VARIABLE does not work and left all 32 FMADDD in place in the measured case. Found exactly this way: 4 fused inference paths (TPA, KAN, MemorizingAttention, MTA) merged green on amd64 and failed on every arm64 machine, 2 of them carrying comments asserting no FMA while the code contracted. If the peer is a backend kernel that also contracts — the cpu matmul emits 202 FMADDD — exact equality is unreachable from this side and the pin belongs at a tolerance. Claims about a PARALLEL split of the same loop are unaffected and not reported", false},
	{"PS3024", "fixed-arity-variadic-call", "a call to a VARIADIC dispatch wrapper that passes a fixed number of arguments. Go builds a fresh slice for the variadic pack at every such site, so a wrapper existing only to forward its pack into a dispatch costs one allocation per dispatch, at every caller, forever. Call a fixed-arity sibling that borrows a pooled slice. MEASURED on nlp, where an MHA method was a byte-for-byte clone of the package's own variadic helper, never touched its receiver, and had 13 call sites while recorder-guarded pooled siblings for arities 1 through 4 sat unused beside it: routing each site to its sibling took a 500-token generate from 235.3k to 225.3k allocs/op, -4.26%% (p=0.000, n=10), control identical. JUDGE ON allocs/op, NOT ns/op — the same change did not separate on time (p=0.143), since those benchmarks are dominated by backend worker-pool park and wake. Check for an existing pool before adding one: nn already ships nnIns1Pool through nnIns3Pool that 43 of its own wrappers bypass. The pooled form MUST defer to the variadic one under a recorder, or the tape retains a slice about to be reused. Wrapper names come from the configured variadicDispatchWrappers list, since the call and the declaration sit in different files and this scanner has no package view. Silent on a genuine spread", false},
	{"PS3023", "transpose-pass-over-built-matrix", "a nested loop that materializes a TRANSPOSED COPY of a matrix this function built itself. This is deliberately the case PS1010 excludes — a transpose writes the inner variable on the left, so interchange only moves the stride — and the remedy is different in kind: DELETE the pass by having the producer write the layout the consumer wants, which also removes the intermediate and its per-row allocations. MEASURED on the autograd logdet VJP, which solved its triangular inverse row-major then transposed it because the contraction needs columns: solving straight into column-major went -10.37%% at n=512 and -6.93%% at n=256 (p=0.000, n=12), allocs down about a third, control flat, bit-identical over 5547 values. Two further costs no line profile shows: the consumer stops walking down a column of a slice-of-slices, losing a row-pointer load and a bounds check per element; and a PARALLEL producer that wrote columns had every worker contending for cache lines with its neighbours, where row-major gives each its own row. Check the source is not ALSO consumed in the original layout, or flipping the producer just moves the transpose", false},
	{"PS3021", "monotone-guard-in-loop", "a counted loop whose entire body sits behind a guard that moves monotonely with the loop variable and is compared against something invariant. That is not a per-iteration decision — the guard is false for a RUN of iterations at one end and true for the rest — so it belongs in the loop BOUNDS. Computing the crossing point once removes the branch from every iteration AND frees any loop-invariant the branch was trapping, since it splits the body into its own basic block and Go SSA will not hoist across it. MEASURED on the autograd conv1d backward, whose per-tap guard was false for only 3 of 2048 positions and whose loop HEADER profiled larger than either line it protected: F32 -7.92%% (p=0.000, n=16). The F64 arm of the same edit was directionally -4.7%% but did NOT separate at n=16 (p=0.210) — so expect a win when the skipped run is small and the body cheap, and nothing measurable otherwise; apply it anyway when bit-identical, since it strictly removes instructions. Bit-identical by construction: only iterations whose body never ran are skipped. Equality guards are excluded — they select one iteration, not a run. Check the DIRECTION before rewriting; getting it backwards silently drops work", false},
	{"PS3020", "invariant-behind-bounds-check", "a counted loop that indexes slices with its loop variable AND recomputes a loop-invariant value every iteration. These are ONE defect: each indexed read is a bounds check, a bounds check is a panic edge that splits the body into separate basic blocks, and Go SSA will not hoist across a block that can panic — so the checks cost more than their own instructions because they also trap the invariant. Fix both halves: range over the destination and cut every companion to its length, then lift the invariant above the loop. MEASURED on the rl Polyak soft update, where (1-tau) was rematerialized per element behind two bounds checks — body from 14 instructions to 5, -21.30%% F64 and -18.62%% F32 (p=0.000, n=12), with an untouched sibling benchmark flat as the control. Restricted to invariants used as a VALUE against an indexed element; addressing arithmetic folds into an addressing mode and is excluded. Verify the FMA fusion did not move — merging blocks can change which multiply contracts, and a 1-ulp failure from exactly that is already on record here", false},
	{"PS3019", "unrolled-index-not-windowed", "a manually unrolled loop bounded by `i+K <= len(x)` that reads x at K constant offsets and never cuts it to the K-wide window. The bound does NOT discharge those reads — i+K can overflow, so the prove pass keeps a check on every one. Cutting a window once per iteration replaces K checks with ONE slice check. MEASURED on nlp dotAndNorm, eight reads to two checks: -16.15%% and -18.55%% (p<=0.001, n=12), geomean -17.36%%, bit-identical. IT DID NOT PAY at the site it was found on, which is the load-bearing half of this advice: the classic ballTree L2 leaf test went four checks to one for -1.11%% against an UNTOUCHED sibling arm that moved -1.06%% in the same run, so nothing was attributable — that loop has a data-dependent early exit whose misprediction dominates and hides the checks. Require a branchless body with no loop-carried dependency, and run an untouched control, because this class yields a plausible small win that is really drift. Reordering the reads to touch the highest offset first is NOT a substitute and left all four checks in place. Clamping a second operand to the first outside the loop makes its window free", false},
	{"PS3018", "max-normalized-exp", "math.Exp(x - m) where m is a max that INCLUDES x, so the call is exp(0) whenever the max picked x and exp(0) is exactly 1. MEASURED TWICE: on the RWKV WKV forward scan, where both stabilized pairs had this shape and half of four calls per element computed a constant, -12.42%% and -11.93%% (p=0.000, n=8) on a kernel that was 75.6%% math.archExp; and on the WKV backward, -13.59%% and -14.87%% (p=0.000, n=12), bit-identical over 6540 gradients across five decay regimes including negative and zero. Check for REPEATS at the same time: each call SITE is reported, and in the backward each of the two exponents appeared twice — math.Exp is not inlined, so a repeat is a second evaluation rather than a subexpression the compiler folds. Test the max against the ARGUMENT rather than branching on the original comparison, so a NaN operand still evaluates both exponentials exactly as before. The win scales with 1/N — over a max of N terms it saves one call in N and is not worth the branch; it paid here because N is two — and the measured saving was a third of what the exp profile share predicted", false},
	{"PS3017", "companion-not-sliced", "a loop that already ranges over a row but still indexes a SECOND slice with the range key, so only half the bounds checks are gone. Ranging proves the row index in range and says nothing about the companion, whose length the compiler cannot relate to the ranged slice. Cut both to the same length, writing the relation explicitly for a trailing segment. MEASURED as the gap between a half-applied and a finished conversion: linalg Cholesky ranged only the row for geomean -2.08%% and gained a FURTHER -1.59%% (three of four cells at p=0.000) when the companion was sliced, while linalg LU done with both from the start went -6.61%% geomean over nine cells. Bit-identical. THE COST IS OFTEN REGISTER PRESSURE RATHER THAN THE CHECKS, which changes how to rank a site: every surviving check keeps a loop-invariant LENGTH live, and enough of them crowd out the loop's own state. MEASURED on the classic GMM full-covariance solve, where four L rows and four y vectors were indexed inside a loop bounded by an integer — eight live lengths, which spilled the induction variable itself and forced a slice pointer to reload every iteration. Cutting all eight to one length took GMMFitFull -5.27%% (p=0.000, n=12), with the diagonal arm flat as a control. So count the OPERANDS, not just the checks. That site also shows this check's own blind spot: it requires a loop that ranges over a SLICE, and a loop written `for j := range i` over an integer looks identical without types while proving nothing (RANGE-OVER-AN-INT-LOOKS-LIKE-RANGE-OVER-A-SLICE-001), so integer-bounded loops with many indexed companions are NOT reported and must be found by reading. A dedicated check for them was built and withheld: at four-or-more operands it matched 141 sites, nearly all of them the ordinary outer loop that indexes several parallel arrays, which is not a defect. Silent when the companion name is cut from a slice expression anywhere in the function; the precondition it cannot check is that the two really span the same extent, so an offset or shorter companion needs its own slice", false},
	{"PS3016", "two-deep-index-not-ranged", "an inner loop reading m[i][k] with the OUTER index invariant, so every step re-loads the row pointer and bounds-checks against it. Hoist the row and RANGE over it — and the range is the half that pays. On linalg Cholesky forward substitution, hoisting the row while keeping the integer-bounded loop measured geomean -0.53%% over eleven benchmarks with one cell at +0.41%% (p=0.038) and did not reach significance at n=12 (p=0.060); converting the same site to range over the row gave -2.82%%, -3.50%% and -0.79%% (p<=0.043) on three of five cells, geomean -2.08%%. The mechanism is bounds-check elimination, not the pointer reload. MEASURED FAR LARGER on classic GaussianNB.Fit, which is the instance to reason from: -32.07%% and -28.24%% (p=0.000, n=12) with two untouched sibling benchmarks flat. The reason that site pays 15x what Cholesky did is worth understanding before ranking a candidate — the checks were not just costing their own instructions, they were BLOCKING A HOIST. Each check is a panic edge that splits the body into its own basic block, and Go SSA will not hoist across a block that can panic, so the outer index times the row stride and BOTH slice headers were rematerialized on every inner step even though all of them are invariant: 14 of the 19 instructions in that loop were address arithmetic that should have been loop-invariant. So rank a two-deep site by how much invariant work is trapped behind its checks, not by the check count alone, and read -gcflags=-S to see it. Cut the row and the companion to ONE length so the range proves both indexes; ranging only the row leaves the second check and most of the win. Bit-identical either way. Silent when the loop already ranges over a slice, which is the applied form; a loop whose OUTER index moves instead has no row to range and is a different problem", false},
	{"PS3015", "write-only-alloc-field", "a struct field handed a fresh allocation and read nowhere in the package, so every construction pays for a buffer nothing uses. The compiler will not report it: an unused local is an error, but a field assigned in a constructor counts as a use. MEASURED on the autograd WKV backward, where a linear-time rewrite stopped needing the loga and p exponent buffers the quadratic path required and they kept being allocated per worker per call; removing them with a pooled scratch took that kernel from 134 to 46 allocs/op and 434.7 to 278.2 KiB, worth a further 15.82%% geomean on time. A KERNEL REWRITE IS THE USUAL CAUSE, since the scratch a kernel inherits describes the algorithm it replaced. Restricted to UNEXPORTED fields, where absence of reads in the package is conclusive, and matched by NAME without types so a same-named field read anywhere suppresses it; a read through reflection is the remaining blind spot", false},
	{"PS3014", "coupled-index-weight", "a doubly-nested reduction whose accumulated term is scaled by an arithmetic combination of BOTH loop indices, used as a VALUE rather than as an index. That coupling is what makes such a sum look irreducibly quadratic, and a DIFFERENCE usually is not: (t-1-i) rewrites as (t-1) minus i, turning one distance-weighted sum into (t-1)*S minus S1 over two ordinary prefix sums maintained in O(1) per step. MEASURED on the autograd WKV backward, where dw was the only gradient with a distance weight and the only reason the pass stayed quadratic; splitting it took seq=512 from 10335us to 580us (17.8x) and the cost per doubling of seq from 3.85x to 1.81x. PRECONDITION THE CHECK CANNOT SEE: the remaining factors must separate into a t-only part and an i-only part or nothing hoists. Products and moduli of the two indices generally do not split and are reported only so they can be ruled out deliberately. Index arithmetic is excluded, since a[t*n+i] couples the indices to address memory rather than to weight a value", false},
	{"PS3013", "leaking-format-param", "a pointer-carrying parameter handed to a fmt call. Passing it as an interface argument makes escape analysis mark the PARAMETER as leaking, and that verdict belongs to the function rather than to the branch, so every caller heap-allocates its argument even though the formatting usually sits on a panic or error path that never runs. MEASURED on tensor.NewOn, whose invalid-shape panic formatted its shape with a %%v verb: the shape literal at every call site escaped, costing one allocation per tensor created anywhere in the tree. Swapping in shape.String(), which escape analysis already proves non-escaping, took a Jamba decode step -5.96%% allocs/op and -0.19%% B/op with time unchanged (p=1.000, n=12) and QuantMamba2 decode -8.70%% allocs. Fix with a non-escaping formatter, never by deleting the message, and VERIFY with go build -gcflags=-m that the parameter flips from leaking param to does not escape — another leak in the same function keeps the old verdict and the change buys nothing. Named parameter types need the configured pointerTypeNames list, since with no type checker a named slice and an int alias are indistinguishable", false},
	{"PS3012", "slice-built-for-one-element", "a package-level function call immediately indexed by a constant, f(x)[0] — the callee builds a whole collection and the caller keeps one item, so where the callee allocates, the rest of it and the slice header are waste that repeats every call. MEASURED on nlp QuantMamba2 decode: rows2D materializes the in_proj output as [][]float64 once per LAYER per token and the caller takes row 0, which at seq=1 was 37%% of all allocation OBJECTS in the step; threading a scratch buffer through the per-stream layer state instead went -18.37%% B/op, -3.67%% allocs/op and -0.54%% sec/op across all seven quantization formats (p<=0.01 every cell). The scratch must live on a PER-STREAM or per-worker object, never on a shared model, and the element must be read-only at the call site since the original returns an independent copy. Restricted to a bare identifier callee: method chains like t.Shape()[0] return a view and allocate nothing. Cannot see whether the callee allocates — confirm before acting", false},
	{"PS3011", "static-chunk-barrier", "work split into equal chunks sized by the worker count, one goroutine per chunk, joined at a barrier — the slowest worker sets the wall clock. On a heterogeneous CPU that is the common case, not a tail: an M2 Pro has 8 performance and 4 efficiency cores, so a chunk landing on an E core can take several times as long. MEASURED on the autograd WKV VJP, where the static split made MORE CORES SLOWER (GOMAXPROCS=8 at 3.36ms against 3.76ms at 12) and pthread_cond_wait was 47.96%% of the profile, more than every line of the kernel combined; claiming units through an atomic cursor went -28.73%% and -29.58%% (p=0.000, n=8) and was BIT-IDENTICAL, since which worker runs a unit cannot change that unit arithmetic. Diagnose with a GOMAXPROCS sweep and a FUNCTION profile — a line profile ranks the kernel and hides the waiting. TWO CONDITIONS DECIDE WHETHER CLAIMING PAYS, both learned by converting siblings and reverting them. (1) A CLAIM'S WORKING SET MUST BE THE CLAIM, not the whole input: a conv1d backward whose body walks the entire sequence per call re-streamed its input once per claim instead of once per worker — about 42x the memory traffic at grain 8 — and ran 126%% SLOWER than the split it replaced; even at four claims per worker it stayed 8%% behind, so that one keeps its static split. (2) THE BODY MUST NOT ALLOCATE PER INVOCATION: a distillation VJP that allocates its softmax scratch per call went -7.4%% on time but +26%% bytes and +50%% allocations, because a cursor invoked it 32 times where the split invoked it 12 — convert the helper to carry PER-WORKER scratch first, then claim. Where both hold it is worth a lot: an MoE combine backward went -34.5%% and -37.4%% in the two orders. Silent once the function reaches for sync/atomic, which is what claiming looks like", false},
	{"PS3010", "serial-reduction-chain", "a single-accumulator floating-point reduction whose every add depends on the previous one, so the loop is bound by add LATENCY rather than throughput. Four independent partials summed at the end measured 537.8 -> 177.3 ns at d=512 (3.03x) and 89.3 -> 43.6 ns at d=128 (2.05x) on Apple M2 Pro darwin/arm64, matching the hardware prediction for a dependent-add chain. NOT BIT-IDENTICAL: it reassociates the sum, so the value moves in the last ulp, and the two hottest instances in this tree are blocked on exactly that — nlp randomOrthogonal is regenerated at dequantization time by TurboQuant, and classic ballTree.within decides exact-label DBSCAN goldens. Requires a pure reduction: one accumulator, written once, read nowhere else in the loop, no branching. An accumulator that is also TESTED is PS3008 territory and is excluded, since four partials cannot be compared against a threshold without summing them first", false},
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
	{"PS6023", "threshold-path-uncovered", "a package-level tuning constant that gates two code paths through a relational comparison, where NO test file in the package names it — so nothing PINS which arm the tests take. Sometimes the path is entirely unexercised (all three hand-found cases below were); sometimes it is covered incidentally by a test aimed at something else, which is weaker than it looks: the diagnosis misleads, and the coverage vanishes silently the moment the constant is retuned. Measured example of the second kind: both radix sorts behind nlp's radixSortCutoff turned out to be reachable from existing tests, but an ascending-radix STABILITY break surfaced only as a failure in TestDistQuickselectParity, a test named for something else, and nothing pinned the tie-break that diffusionRefillOrder documents. Found by hand three times in three packages before this check existed: linalg goldens ran 384 elements against a 65536 bound and a one-ulp perturbation of the parallel arm left them green; every SnapKV test used 2-3 rows against a group of 4; WandaPrune's fan-out needs 2+ panels and every test capped cout below the panel width, so making the worker scratch SHARED left all eleven Wanda tests passing with the race detector firing. REMEDY: one gate whose two arms are the SAME source selected by the threshold — never a separately written reference, which contracts to FMA differently and fails by an ulp on arithmetic that never changed. If the package's tests are EXTERNAL (package X_test) and the constant is unexported they cannot name it at all, so add an internal test file that asserts the geometry clears the bound, or select the arms through an exported knob such as GOMAXPROCS. VERIFICATION-GAP check: it reports missing evidence, not a defect — a hit is a test to write, never a code change. MEASURED CONSEQUENCE, and this check called it correctly before anyone acted on it: backend/cpu's matmulInlineWork, which decides whether an F64 matmul runs serial or fans out, sat in this check's output as gemm.go:95 while it went stale. The band kernels beneath it were register-tiled and became roughly twice as fast; fork/join cost did not change, so the crossover moved up and the untouched constant kept fanning out shapes that had stopped paying for it — measured at +37.26%% slower at the gate value itself, +22.85%% and +9.41%% just above it. Re-swept, the crossover had moved from 262144 to between 592704 and 681472. The finding was sitting here the whole time and was re-derived from first principles instead of read. TREAT THIS CHECK'S OUTPUT AS A LIVE WORKLIST: an untested threshold is not merely under-covered, it is a number nobody can re-sweep when the code it balances changes, and this repo has now had four of those go stale in one session MEASURED CASE OF A GATE SET TOO HIGH: a gradient-boosting fit gated its parallel feature scan at d*n >= 1<<17, four times higher than the fork it avoids costs. Halving it twice took BenchmarkGBMFit from 138.2ms to 80.2ms (-42.0%%) and an exact-grower cell -11.7%%, against a control drifting under 1%%; raising it instead to 1<<19 cost 2.4x. So when a threshold has no test on both sides, suspect the VALUE as much as the coverage, and sweep it with three interleaved arms before believing any single reading — the same sweep read +5.3%% and -6.8%% on one cell across two runs", false},
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
	{"PS6024", "receiver-scratch-buffer", "a method using a receiver SLICE FIELD as a per-call temporary (indexed write, then indexed read, same call) — unsafe to call concurrently, and every caller contends on one cache line", false},
	{"PS6007", "search-feeds-reduction", "a loop whose expensive per-item CALL produces an index into an accumulation — split it into a parallel search pass and a sequential fold, since chunked partials would reassociate the sums", false},
	{"PS6008", "alloc-in-parallel-body", "a buffer allocated INSIDE a parallel dispatch body — free when the dispatch is infrequent, ruinous when it sits in a hot loop; hoist to per-chunk buffers indexed by the chunk", false},
	{"PS6009", "reflect-swapper-sort", "sort.Slice/SliceStable reaches its swap through reflectlite.Swapper, which ALLOCATES on every call — slices.SortFunc/SortStableFunc is the same comparator with a monomorphized swap and no allocation. MEASURED on the CART builder\u0027s per-node feature sort, called once per candidate feature per node: that ONE line was 68%% of a random-forest fit\u0027s allocations, and converting it took the fit from 1095702 to 352017 allocs/op (-67.9%%), bytes -11.5%%, and time -4.9%% to -6.1%% against a control drifting -1.8%%. So this is not only an allocation finding where the sort is called often — it shows in ns/op too, because the allocation is per CALL rather than per element. TWO THINGS TO GET RIGHT: the comparator takes ELEMENTS, not indices, so a body written as less(i, j) over the slice must be rewritten in terms of the values; and the tie order of an unstable sort is unspecified in both forms, so confirm the caller does not depend on it — here a sklearn-parity golden proves it does not, and inverting the comparator reddens three tests including that golden", false},
	{"PS6003", "partial-fast-path-coverage", "a fast path that bypasses the general path for only SOME members of a variant family a switch in the same function enumerates — the uncovered variants silently pay the slow path", false},
	{"PS6006", "cross-backend-dtype-gap", "an op registered via std.add(backend.Op…, tensor.<dtype>, kernel) in one backend package for a STRICT SUBSET of the dtypes a sibling backend package registers for the SAME op — the missing dtype(s) silently fall through to the sibling (typically the serial reference) kernel. Register the fast backend's kernel for the missing dtype too: mirror the reference arithmetic (widen reads to F64, accumulate in F64, narrow only on store → byte-identical) parallelized over the op's independent rows/channels/heads. Shipped: WKV #692 (7.5-9.1x), distill #693 (6.8-10.5x), SSM #694 (9.3-10.3x), masked-attn #695 (8.1x), select-attn #696 (7.8-8.0x) — all bit-exact F32 cpu registrations closing an F32→serial-ref fall-through. Whole-repo fact (needs both backend packages parsed); silent when a single package is scanned.", false},
	{"PS6004", "unverified-dual-path", "a devirtualized fast path with a generic fallback — a bit-identity claim needing a bit-exact test", true},
	{"PS4010", "vectorizable-butterfly", "an in-loop butterfly p,q = x+y,x-y (add and subtract of the SAME operand pair written to two indexed slots) — a scalar FWHT/FFT/Hadamard stage a SIMD Add/Sub over the contiguous stride-separated runs would vectorize", false},
	{"PS4011", "op-dispatch-recurrence", "a sequential loop dispatching 2+ backend ops (calls passing a backend.Op* constant) per iteration in a function with NO fused typed fast path (no flatF64 guard) — O(seq) dispatch+alloc overhead on tiny per-step tensors; add a raw-slice fused path", false},
	{"PS4012", "scaled-serial-dot", "a serial scalar dot accumulator whose result is SCALED/dequantized (acc*scale…) before being stored — a quantized/dequant GEMM inner loop; latency-bound like PS4008 but missed by it (acc isn't stored raw). Break the chain with independent accumulators; bit-identical when the products are integer-valued (int8·int8 partials < 2^53 reassociate exactly), else tolerance-gated", false},
	{"PS4013", "einsum-no-inference-fusion", "a function dispatching backend.OpEinsum (and taking a *backend.Context) with NO ctx.Recorder == nil fused inference branch. The generic einsum engine decodes every contraction combo with a per-index modulo (~38 ns/combo) and materializes the whole output tensor; for a large contraction that dominates the layer's forward. Add a typed fused inference path gated on ctx.Recorder == nil (training keeps the differentiable einsum so gradients still reach the operands) reproducing the einsum's index order for bit-identity. Shipped: CoPE gather #660 (37-133x, [T,T,MaxPos+1] one-hot einsum → direct z gather), KAN spline #661 (45-75x, bic,ijc->bij + bij,ij->bj → fused typed contraction). Bench the ENCLOSING forward at production dims; wins scale with the contraction size (combos = product of all einsum index sizes)", false},
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
	// PS3062 — the package name of the REFERENCE backend and of the optimized backends that
	// should shadow it, plus the method that registers a kernel. Empty by default: like the
	// other domain checks, this one stays silent until a project names its own vocabulary, and
	// the names must come from here rather than from literals in the engine — perfscan never
	// imports the tree it scans, so it cannot read a project's typed backend constants.
	ReferenceBackendPkg  string   `json:"referenceBackendPkg,omitempty"`
	OptimizedBackendPkgs []string `json:"optimizedBackendPkgs,omitempty"`
	KernelRegisterFuncs  []string `json:"kernelRegisterFuncs,omitempty"`
	// PointerTypeNames lists NAMED types that carry a pointer (a named slice, map, or struct
	// holding one). PS3013 needs to know whether a parameter can leak, and with no type checker a
	// bare identifier like Shape is indistinguishable from an int alias.
	PointerTypeNames []string `json:"pointerTypeNames,omitempty"`
	// PS3024 — variadic dispatch wrappers whose call sites usually pass a fixed arity.
	// Configured because the call and the declaration sit in different files and this scanner
	// has no package-level view.
	VariadicDispatchWrappers []string `json:"variadicDispatchWrappers,omitempty"`
	// PS3026 — top-k / gating selector helpers: a call to one proves the function chose a SUBSET
	// of branches, which is what makes a later full-range fan-out over all of them reportable.
	// Configured because the selector and the fan-out are a project's own vocabulary. Empty by
	// default: without it PS3026 cannot report.
	TopKSelectorFuncs []string `json:"topKSelectorFuncs,omitempty"`
	// PS3027 — helpers that return a READ-ONLY view of a tensor's values, widening a copy when the
	// dtype does not match the view's element type. Applying one to a tensor the function ALLOCATED
	// as an output means the kernel writes into a detached copy. Empty by default.
	InputViewFuncs []string `json:"inputViewFuncs,omitempty"`
	// PS3027 — the output counterpart of an input view: returns a buffer plus a flush that narrows
	// it back into the tensor's storage. Named so the advice can point at the right helper.
	OutputViewFuncs []string `json:"outputViewFuncs,omitempty"`
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
	pointerTypes                                   map[string]bool
	variadicWrappers                               map[string]bool
	topKSelectors                                  map[string]bool
	refBackendPkg                                  string
	optBackendPkgs                                 map[string]bool
	kernelRegisterFuncs                            map[string]bool
	inputViews, outputViews                        map[string]bool
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
		refBackendPkg:       c.ReferenceBackendPkg,
		optBackendPkgs:      toSet(c.OptimizedBackendPkgs),
		kernelRegisterFuncs: toSet(c.KernelRegisterFuncs),
		accessors:           toSet(c.ElementAccessors),
		fastPath:            toSet(c.FastPathHelpers),
		elemCount:           toSet(c.ElementCountMethods),
		shapeMethods:        toSet(c.ShapeMethods),
		indexDecompose:      toSet(c.IndexDecomposeFuncs),
		allocators:          toSet(c.AllocatorFuncs),
		visitors:            toSet(c.PerElementVisitors),
		bulkCopy:            toSet(c.BulkCopyHelpers),
		vectorized:          toSet(c.VectorizedSiblingFuncs),
		pureCompute:         toSet(c.PureComputeFuncs),
		layoutOps:           toSet(c.LayoutOpConstants),
		pointerTypes:        toSet(c.PointerTypeNames),
		variadicWrappers:    toSet(c.VariadicDispatchWrappers),
		topKSelectors:       toSet(c.TopKSelectorFuncs),
		inputViews:          toSet(c.InputViewFuncs),
		outputViews:         toSet(c.OutputViewFuncs),
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

// opRegistry records, per backend package, which dtypes each op is registered for
// (via std.add(backend.Op…, tensor.<dtype>, kernel)) and the source position of the
// first such registration — the anchor for a PS6006 cross-backend-dtype-gap finding.
type opRegistry struct {
	dtypes map[string]map[string]map[string]bool // pkg → op → dtype-suffix → true
	pos    map[string]map[string]token.Position  // pkg → op → first std.add position
}

// selName returns the selector field of `X.Field` when X is the bare ident `pkg`
// (e.g. constNameFor(expr,"backend") on `backend.OpWKV` → "OpWKV"); "" otherwise.
func constNameFor(e ast.Expr, pkg string) string {
	sel, ok := e.(*ast.SelectorExpr)
	if !ok {
		return ""
	}
	id, ok := sel.X.(*ast.Ident)
	if !ok || id.Name != pkg {
		return ""
	}
	return sel.Sel.Name
}

// collectOpRegistrations harvests every std.add(backend.Op…, tensor.<dtype>, …) call
// across the parsed files, grouped by the enclosing package name. It is REPO-scoped:
// PS6006 compares an op's dtype coverage across sibling backend packages, so every
// backend's registrations must be seen before any gap is judged.
func collectOpRegistrations(fset *token.FileSet, files []*ast.File) *opRegistry {
	r := &opRegistry{
		dtypes: map[string]map[string]map[string]bool{},
		pos:    map[string]map[string]token.Position{},
	}
	for _, f := range files {
		if f.Name == nil {
			continue
		}
		pkg := f.Name.Name
		ast.Inspect(f, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok || len(call.Args) < 2 {
				return true
			}
			// Match the receiver-method call `std.add(...)` — the backend kernel registry.
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok || sel.Sel.Name != "add" {
				return true
			}
			if recv, ok := sel.X.(*ast.Ident); !ok || recv.Name != "std" {
				return true
			}
			op := constNameFor(call.Args[0], "backend")
			dt := constNameFor(call.Args[1], "tensor")
			if op == "" || dt == "" || !strings.HasPrefix(op, "Op") {
				return true
			}
			if r.dtypes[pkg] == nil {
				r.dtypes[pkg] = map[string]map[string]bool{}
				r.pos[pkg] = map[string]token.Position{}
			}
			if r.dtypes[pkg][op] == nil {
				r.dtypes[pkg][op] = map[string]bool{}
				r.pos[pkg][op] = fset.Position(call.Pos())
			}
			r.dtypes[pkg][op][dt] = true
			return true
		})
	}
	return r
}

// dtypeGapFindings reports PS6006: for each (package, op), any dtype that a SIBLING
// backend package registers for the same op but this package does not — the missing
// dtype dispatches to the sibling (typically serial reference) kernel. Repo-agnostic:
// it never hard-codes package names, it flags whichever registry is the strict subset.
func (r *opRegistry) dtypeGapFindings() []finding {
	var out []finding
	for pkg, ops := range r.dtypes {
		for op, have := range ops {
			// Union of the same op's dtypes across every OTHER backend package.
			missing := map[string]bool{}
			var siblings []string
			for other, oops := range r.dtypes {
				if other == pkg {
					continue
				}
				sset, ok := oops[op]
				if !ok {
					continue
				}
				sib := false
				for dt := range sset {
					if !have[dt] {
						missing[dt] = true
						sib = true
					}
				}
				if sib {
					siblings = append(siblings, other)
				}
			}
			if len(missing) == 0 {
				continue
			}
			out = append(out, finding{
				pos:      r.pos[pkg][op],
				category: "cross-backend-dtype-gap",
				msg: fmt.Sprintf("backend %q registers %s only for {%s}, but sibling backend(s) %s also handle {%s} — those dtypes fall through to the sibling (serial reference) kernel. Register %s here for {%s} too (mirror the reference arithmetic bit-exactly, parallel over the op's independent rows/channels/heads).",
					pkg, op, sortedKeys(have), fmtStrs(siblings), sortedKeys(missing), op, sortedKeys(missing)),
			})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].pos.Filename != out[j].pos.Filename {
			return out[i].pos.Filename < out[j].pos.Filename
		}
		return out[i].pos.Line < out[j].pos.Line
	})
	return out
}

// sortedKeys renders a string-set as a stable comma-joined list.
func sortedKeys(m map[string]bool) string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return strings.Join(ks, ",")
}

// fmtStrs renders a slice as a stable comma-joined list.
func fmtStrs(xs []string) string {
	cp := append([]string(nil), xs...)
	sort.Strings(cp)
	return strings.Join(cp, ",")
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
	// File-scoped fact for PS6024: receiver fields that are INDEXED in exactly one
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
	out = append(out, writeOnlyAllocFieldFindings(fset, f)...)
	for name := range intMapReg[curPkg] { // cross-file dispatch registries
		intKeyMaps[name] = true
	}
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		out = append(out, scanFunc(fset, fn, wrappers, intKeyMaps, ns)...)
		out = append(out, serialNestWithIdleFanoutFindings(fset, f, fn)...)
		out = append(out, perItemRescanFindings(fset, f, fn)...)
		out = append(out, wholeTensorStagingFindings(fset, f, fn)...)
		out = append(out, sourceReboundPerOutputFindings(fset, fn)...)
		out = append(out, mapBlockedByReductionFindings(fset, f, fn)...)
		out = append(out, collidingScatterFindings(fset, f, fn)...)
		out = append(out, itemReductionWindowFindings(fset, f, fn)...)
		out = append(out, oneSharedAccumulatorFindings(fset, f, fn)...)
		out = append(out, fanoutWorkFloorFindings(fset, f, fn)...)
		out = append(out, axpyReloadFindings(fset, fn)...)
		out = append(out, serialTailAfterFanoutFindings(fset, f, fn)...)
		out = append(out, degenerateShapeGuardFindings(fset, fn)...)
		out = append(out, stagedSingleColumnFindings(fset, fn)...)
		out = append(out, independentReductionsFindings(fset, fn)...)
		out = append(out, asymmetricDtypeArmFindings(fset, fn)...)
		out = append(out, sortThenTruncateFindings(fset, fn)...)
		out = append(out, serialPermutationFindings(fset, f, fn)...)
		out = append(out, columnReadFindings(fset, fn)...)
		out = append(out, perIterationScratchFindings(fset, f, fn)...)
		out = append(out, derivedBaseNestFindings(fset, f, fn)...)
		out = append(out, serialLoopOverParallelWorkFindings(fset, f, fn)...)
		out = append(out, unsizedFanoutFindings(fset, f, fn)...)
		out = append(out, refOnlyKernelFindings(fset, f, fn, ns)...)
		out = append(out, serialLoopInFanningFuncFindings(fset, f, fn, ns)...)
		out = append(out, jaggedMatrixFindings(fset, fn)...)
		out = append(out, serialLoopOverCallFindings(fset, f, fn)...)
		out = append(out, consecutiveLoopFindings(fset, fn)...)
		out = append(out, interchangeBeforeBandFindings(fset, f, fn)...)
		out = append(out, serialBestOfScanFindings(fset, f, fn)...)
		out = append(out, queueingFanoutFindings(fset, f, fn)...)
		out = append(out, sharedThresholdFindings(fset, f, fn)...)
		out = append(out, escapingLocalBufferFindings(fset, fn)...)
		out = append(out, reseededSerialLoopFindings(fset, f, fn)...)
		out = append(out, invariantOperandReloadFindings(fset, f, fn)...)
		out = append(out, sharedRangeSubjectFindings(fset, fn)...)
		out = append(out, sharedAccumulatorFindings(fset, fn)...)
		out = append(out, narrowUnrollFindings(fset, fn)...)
		out = append(out, clampInLoopFindings(fset, fn)...)
		out = append(out, minMaxCallInLoopFindings(fset, fn)...)
		out = append(out, radixPassFindings(fset, fn)...)
		out = append(out, perJobGatherFindings(fset, f, fn)...)
		out = append(out, accessorWalk1DFindings(fset, f, fn)...)
		out = append(out, perUnitStreamFindings(fset, fn)...)
		out = append(out, innerIndependentUnderSequentialOuterFindings(fset, f, fn)...)
		out = append(out, loopHoistableScratchFindings(fset, fn)...)
		out = append(out, selfComparisonOracleFindings(fset, fn)...)
		out = append(out, misSizedAppendBufferFindings(fset, fn)...)
		out = append(out, dispatchLiteralSliceFindings(fset, f, fn)...)
		out = append(out, recursiveSplitAllocFindings(fset, fn)...)
	}
	// PS5002 is a whole-file structural check (consecutive sibling loops), not a
	// per-function trigger attribution, so it runs once over the file's blocks.
	out = append(out, perQueryAllocHelperFindings(fset, f)...)
	out = append(out, scanFusableLoops(fset, f)...)
	out = append(out, einsumNoInferenceFusionFindings(fset, f)...)
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
			if loopArgs >= 2 && !inDeclinedTypedFallback(parent, call, ns) {
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
	out = append(out, serialReductionChainFindings(fset, fn)...)
	out = append(out, staticChunkBarrierFindings(fset, fn)...)
	out = append(out, sliceBuiltForOneElementFindings(fset, fn)...)
	out = append(out, leakingFormatParamFindings(fset, fn, ns)...)
	out = append(out, coupledIndexWeightFindings(fset, fn)...)
	out = append(out, twoDeepIndexNotRangedFindings(fset, fn)...)
	out = append(out, companionNotSlicedFindings(fset, fn)...)
	out = append(out, unrolledIndexNotWindowedFindings(fset, fn)...)
	out = append(out, invariantBehindBoundsCheckFindings(fset, fn)...)
	out = append(out, monotoneGuardInLoopFindings(fset, fn)...)
	out = append(out, fixedArityVariadicCallFindings(fset, fn, ns)...)
	out = append(out, unroundedProductUnderExactnessClaimFindings(fset, fn)...)
	out = append(out, fullFanoutUnderTopKGateFindings(fset, fn, ns)...)
	out = append(out, inputViewOnOutputTensorFindings(fset, fn, ns)...)
	out = append(out, unpooledFullyOverwrittenScratchFindings(fset, fn)...)
	out = append(out, unbufferedFileToParserFindings(fset, fn)...)
	out = append(out, fixedOffsetStoresNotWindowedFindings(fset, fn)...)
	out = append(out, symmetricPairComputedTwiceFindings(fset, fn)...)
	out = append(out, closureAccessorInLoopFindings(fset, fn)...)
	out = append(out, maxNormalizedExpFindings(fset, fn)...)
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
	out = append(out, transposePassFindings(fset, fn)...)
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
	collectFanoutHelpers(parsed)
	collectScratchTypes(parsed)
	collectFanningFuncs(parsed)
	collectKernelRegistrations(parsed, ns)
	collectLoopyFuncs(parsed)
	collectThresholdComparisons(parsed)
	collectExecPoolHelpers(parsed)
	collectThresholdUse(fset, parsed)
	collectReductionHelpers(parsed)
	collectAllocGathers(parsed)
	collectTypedExposers(parsed)
	collectWriteOnlyFields(fset, parsed)
	for _, f := range parsed {
		for _, fd := range scanFile(fset, f, ns) {
			if enabled[fd.category] {
				fd.id = catToID[fd.category]
				all = append(all, fd)
			}
		}
	}
	// PS6006 cross-backend-dtype-gap is a REPO-scoped fact (compares an op's dtype
	// coverage across sibling backend packages), so it runs once over all parsed files.
	for _, fd := range collectOpRegistrations(fset, parsed).dtypeGapFindings() {
		if enabled[fd.category] {
			fd.id = catToID[fd.category]
			all = append(all, fd)
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
					msg: fmt.Sprintf("nested %s/%s loop reduces-over or updates-in-place %s[%s*%s + %s] — the INNER"+
						" var %s is the high-stride (×%s) part while the OUTER var %s is contiguous, so the inner"+
						" loop strides %s by %s every step (cache-thrashing). Interchange to %s-outer/%s-inner so %s"+
						" is walked contiguously in %s — bit-identical (reductions keep the same ascending-%s order;"+
						" element-wise updates like S[r]*=a[c] are trivially bit-exact). Shipped: MLA value-mix"+
						" (reduction), KimiDeltaAttention decay #658 (update). Benchmark the kernel.",
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
	// Gate on a compute op-assign in the loop so pure strided reads (intentional gathers,
	// dst[o]=ARR[strided]) are not flagged: += is the reduction case (acc += ARR[strided]);
	// *= / -= catch the in-place strided UPDATE case (ARR[strided] *= expr — e.g. a per-
	// channel decay S[r*dk+c] *= a[c]), which the interchange fixes just as well and, being
	// element-wise, is trivially bit-exact. Shipped update case: KimiDeltaAttention decay #658.
	hasReduce := false
	ast.Inspect(root, func(n ast.Node) bool {
		if as, ok := n.(*ast.AssignStmt); ok {
			switch as.Tok {
			case token.ADD_ASSIGN, token.MUL_ASSIGN, token.SUB_ASSIGN:
				hasReduce = true
			}
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
		// Precision guard (§step6, 2026-07-30): PS4011's fused-flatF64 lever only pays on a
		// per-TIMESTEP recurrence whose per-step tensors are microscopic. It over-fired on
		// per-HEAD attention loops (fox_block/aaren/cope: `for h := range X.Heads`, each head a
		// full [T,T] Q·Kᵀ→softmax→·V) where the dispatch overhead is negligible against the
		// GEMMs — a matmul-dominated body, no fused-recurrence win (measured false positives).
		// Suppress two unambiguous "this is attention, not a scalar recurrence" markers:
		// ranging over a `.Heads` field, or dispatching OpSoftmax over keys (a linear-attention
		// recurrence — the real target — never softmaxes per step). Zero false-negative risk.
		if rangesOverField(n, "Heads") || loopBodyDispatchesOp(body, "OpSoftmax") ||
			loopBodyCallsMethod(body, "Forward") || loopBodyCallsMethod(body, "Route") {
			return false // per-head/per-expert/per-recursion composition, not a scalar recurrence
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

// rangesOverField reports whether n is `for k := range <expr>.<field>` — a loop over a
// named struct field (e.g. `range b.Heads`), the marker of a per-head/per-slot loop rather
// than a sequence scan. Used to suppress PS4011 on per-head attention loops.
func rangesOverField(n ast.Node, field string) bool {
	rs, ok := n.(*ast.RangeStmt)
	if !ok {
		return false
	}
	sel, ok := rs.X.(*ast.SelectorExpr)
	return ok && sel.Sel.Name == field
}

// loopBodyDispatchesOp reports whether body contains a backend-op call passing the named
// Op* constant (e.g. "OpSoftmax") — used to suppress PS4011 when the loop is quadratic
// attention (softmax over keys) rather than a scalar recurrence.
func loopBodyDispatchesOp(body ast.Node, opName string) bool {
	found := false
	ast.Inspect(body, func(m ast.Node) bool {
		if found {
			return false
		}
		call, ok := m.(*ast.CallExpr)
		if !ok {
			return true
		}
		for _, a := range call.Args {
			if sel, ok := a.(*ast.SelectorExpr); ok && sel.Sel.Name == opName {
				if _, ok := sel.X.(*ast.Ident); ok {
					found = true
					return false
				}
			}
		}
		return true
	})
	return found
}

// loopBodyCallsMethod reports whether body contains a method call `X.<name>(...)` — used to
// suppress PS4011 when each iteration runs a whole sub-module (e.g. `.Forward`/`.Route` on a
// per-expert / per-recursion / per-block loop), which is composition, not a scalar recurrence.
func loopBodyCallsMethod(body ast.Node, name string) bool {
	found := false
	ast.Inspect(body, func(m ast.Node) bool {
		if found {
			return false
		}
		if call, ok := m.(*ast.CallExpr); ok {
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == name {
				found = true
				return false
			}
		}
		return true
	})
	return found
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
// funcHasContextParam reports whether fn takes a *T param whose type name is "Context"
// (i.e. *backend.Context) — the marker of a backend-dispatching / forward function, used to
// scope PS4013 to layer forwards and skip backend-internal kernels/registration.
func funcHasContextParam(fn *ast.FuncDecl) bool {
	if fn.Type.Params == nil {
		return false
	}
	for _, f := range fn.Type.Params.List {
		star, ok := f.Type.(*ast.StarExpr)
		if !ok {
			continue
		}
		if sel, ok := star.X.(*ast.SelectorExpr); ok && sel.Sel.Name == "Context" {
			return true
		}
	}
	return false
}

// einsumNoInferenceFusionFindings flags PS4013 at the FILE level: an OpEinsum dispatched in a
// *backend.Context forward, reported ONLY when the file has no ctx.Recorder reference anywhere —
// i.e. the layer has no fused inference path at all. A file whose Forward already gates a
// Recorder==nil fast path (even if the einsum itself sits in a training-only helper, as in
// griffin/hgrn) is fused for inference and is NOT flagged. The generic einsum engine decodes
// every contraction combo with a per-index modulo (~38 ns/combo) and materializes the whole
// output tensor, so a large contraction dominates the layer's forward; a typed fused path gated
// on ctx.Recorder == nil (training keeps the differentiable einsum so gradients still reach the
// operands) reproducing the einsum's index order is often bit-identical and 1-2 orders of
// magnitude faster. Shipped: CoPE #660 (37-133x), KAN #661 (45-75x).
func einsumNoInferenceFusionFindings(fset *token.FileSet, f *ast.File) []finding {
	// File-level: if any function gates on ctx.Recorder, the layer already has a fused
	// inference path — the einsum is its differentiable/training branch, not a candidate.
	hasRecorder := false
	ast.Inspect(f, func(n ast.Node) bool {
		if sel, ok := n.(*ast.SelectorExpr); ok && sel.Sel.Name == "Recorder" {
			hasRecorder = true
			return false
		}
		return true
	})
	if hasRecorder {
		return nil
	}
	var out []finding
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil || !funcHasContextParam(fn) {
			continue
		}
		var einsumPos token.Pos
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			if sel, ok := n.(*ast.SelectorExpr); ok && sel.Sel.Name == "OpEinsum" && !einsumPos.IsValid() {
				einsumPos = sel.Pos()
			}
			return true
		})
		if einsumPos.IsValid() {
			out = append(out, finding{
				pos:      fset.Position(einsumPos),
				category: "einsum-no-inference-fusion",
				msg: "OpEinsum dispatched in a *backend.Context forward with no ctx.Recorder == nil fused" +
					" inference path — the generic einsum engine decodes every contraction combo with a" +
					" per-index modulo (~38 ns/combo) and materializes the whole output. For a large" +
					" contraction add a typed fused path gated on ctx.Recorder == nil (training keeps the" +
					" differentiable einsum for gradients), reproducing the einsum's index order for" +
					" bit-identity. Shipped: CoPE #660 (37-133x), KAN #661 (45-75x).",
			})
		}
	}
	return out
}

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
						" rebuilt N times per inner index."+
						" REDUNDANT ARITHMETIC IS NOT AUTOMATICALLY COST, and this check cannot tell the"+
						" difference: when the recompute is a REDUCTION sharing a loop with another"+
						" reduction, it runs in the latency shadow of the chain beside it and is very"+
						" nearly free. MEASURED on nlp MaxContextCosine, where the ‖ctx‖² sum is rebuilt"+
						" once per (candidate, context) pair: the inner loop costs 822 ns with that term"+
						" and 813 ns without it at dim=768, so the whole redundancy is about 1%% despite"+
						" being half the arithmetic. Hoisting it into a prepass REGRESSED the benchmark"+
						" +98.51%% at 8 candidates and +34.70%% at 64 (p=0.000, n=8), because the prepass"+
						" is serial O(numContext·dim) while the work it saves was spread across workers"+
						" and nearly free to begin with. Before hoisting a reduction out of a loop that"+
						" contains another one, measure the loop WITH and WITHOUT the term: if removing it"+
						" does not speed the loop up, hoisting it cannot speed the function up"+
						" (PERF-REDUNDANT-WORK-IN-A-LATENCY-SHADOW-IS-FREE-001, PS3010)",
						innerVar, outerVar, innerVar),
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
		// SECOND KNOWN GENUINE CASUALTY, MEASURED: the NSA masked-attention score loop calls a
		// keep(j) PREDICATE to skip masked keys, and its dot is otherwise the exact shape this
		// check exists for — one accumulator per key over a query row re-streamed once per key.
		// The predicate is not the bottleneck: jamming that loop four keys per pass, together
		// with the P·V accumulation PS3075 reported beside it, took BenchmarkNSABranches from
		// 29.49 to 14.82 ms, -49.7%%, with a CoPE cell flat as a control. A MASK PREDICATE IS
		// NOT PER-ELEMENT WORK — it is one branch per key guarding a continue — so a loop whose
		// only call is a boolean guard is worth reading by hand even though this check declines
		// it. THAT WIDENING IS NOW MADE, IN ITS OWN ROUND AND RE-COUNTED: a call reached only
		// through an if whose taken branch just skips the item is exempt, and the effect on the
		// whole tree is 47 to 49 findings — EXACTLY the two backend/ref/mha_masked_backward
		// sites this note names as the genuine casualty, with nothing lost. Applied to the NSA
		// score loop as it stood before it was jammed by hand, it reports that too.
		//
		// The exemption is narrow on purpose: the else arm still counts, a branch that COMPUTES
		// rather than skips still counts, and a call inside the sentinel store still counts —
		// except for math.Inf and math.NaN, which this same note records as constant loads and
		// which are exactly what a mask bail-out stores. An arbitrary cap on the branch length
		// was written first and removed: no fixture could isolate it, and it left a skip branch
		// calling something exempt at any length.
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
		// APPLYING THIS CHECK LEAVES A BY-ONE TAIL, and that tail is the reported shape exactly.
		// The first site converted on its advice — the Titans deep scan — went on being reported
		// twice afterwards, from the remainder loops of its own fix. Operands a wide-stride loop
		// in the same function already amortizes are done.
		jammedOperands := wideStrideOperands(fn)
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
			if shared && perOutput && storedToIndexOf(outBody, acc, outVar) &&
				!sharedOperandIsJammed(outBody, acc, derived, outVar, jammedOperands) {
				out = append(out, finding{
					pos:      fset.Position(n.Pos()),
					end:      fset.Position(n.End()),
					category: "output-invariant-operand-reload",
					msg: fmt.Sprintf("accumulator %q re-reads an operand that does not vary with output index %q — unrolling"+
						" the output loop by 4 would let one load feed 4 accumulators (register blocking)."+
						" FIRST MEASURED RESULT, AND IT IS LARGE: the Titans deep scan carried this"+
						" shape in four inner loops — a hidden-unit dot re-reading the key row and"+
						" an output dot re-reading the hidden vector, in each of the key and query"+
						" passes. Jamming all four four units per pass took"+
						" BenchmarkTitansScanDeep_512x96h192 from 98.97 to 38.73 ms, -60.9%%"+
						" (2.56x), and the 256x64h128 cell -58.0%%, with an optimizer benchmark"+
						" flat as a control. THE GATE WAS A TOLERANCE AND HAD TO BE REPLACED: the"+
						" fused path and its dispatch oracle already disagreed by an ulp on this"+
						" host before any jam, so bit-identity had to be frozen against the path"+
						" ITSELF with a digest, at shapes whose dim and hid are NOT multiples of"+
						" four — every existing shape was, so neither jam's by-one tail ran."+
						" TWO THINGS MUST STILL BE CHECKED BEFORE ACTING. Confirm the"+
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
		// A GUARD IS NOT THE BOTTLENECK. Both genuine casualties this exclusion is known to
		// have cost were mask guards — backend/ref/mha_masked_backward.go, where math.Inf and
		// math.IsInf sit in a bail-out branch, and the NSA score loop, whose keep(j) predicate
		// gates a continue. A call reached only through an if whose taken branch just SKIPS the
		// item runs once per item and never on the arithmetic path, so it says nothing about
		// what the loop is bottlenecked on. Everything else about the if — its else branch, and
		// any branch that does real work — is still descended into.
		if ifs, ok := n.(*ast.IfStmt); ok && branchOnlySkips(ifs.Body) {
			if ifs.Else != nil {
				ast.Inspect(ifs.Else, func(m ast.Node) bool {
					if found {
						return false
					}
					if c, ok := m.(*ast.CallExpr); ok {
						if id, isID := ast.Unparen(c.Fun).(*ast.Ident); !isID || !trivial[id.Name] {
							found = true
							return false
						}
					}
					return true
				})
			}
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

// constantCalls are calls that compile to a constant load rather than a call — the same fact the
// calibration note above records about math.Inf. A sentinel store using one is not work.
var constantCalls = map[string]bool{"math.Inf": true, "math.NaN": true}

// callsBeyondConstants is loopCallsNonTrivial with those constants also exempt. It is separate
// so the main predicate keeps its measured calibration untouched.
func callsBeyondConstants(n ast.Node) bool {
	found := false
	ast.Inspect(n, func(m ast.Node) bool {
		if found {
			return false
		}
		c, ok := m.(*ast.CallExpr)
		if !ok {
			return true
		}
		if sel, ok := c.Fun.(*ast.SelectorExpr); ok {
			if constantCalls[identName(sel.X)+"."+sel.Sel.Name] {
				return true
			}
		}
		if loopCallsNonTrivial(c) {
			found = true
			return false
		}
		return true
	})
	return found
}

// branchOnlySkips reports whether a branch body does nothing but abandon the current item — a
// continue or a break, optionally after assignments. A branch that computes something is real
// work and its calls still count.
func branchOnlySkips(b *ast.BlockStmt) bool {
	if b == nil || len(b.List) == 0 {
		return false
	}
	skips := false
	for _, st := range b.List {
		switch t := st.(type) {
		case *ast.BranchStmt:
			if t.Tok == token.CONTINUE || t.Tok == token.BREAK {
				skips = true
				continue
			}
			return false
		case *ast.AssignStmt:
			// A bail-out often records a sentinel before skipping, and a store is not work.
			// A CALL inside that store IS: an arbitrary length cap was written here first and
			// no fixture could isolate it, while it left `if !ok { x = expensive(); continue }`
			// exempt at any length. Testing the assignment itself is the condition that was
			// meant.
			for _, rhs := range t.Rhs {
				if callsBeyondConstants(rhs) {
					return false
				}
			}
		default:
			return false
		}
	}
	return skips
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
						" order, but CONFIRM that at the site rather than assuming it."+
						" INTERCHANGE BEFORE TRANSPOSE, measured head to head on ONE kernel: a QR"+
						" VJP whose two dominant terms both read down a column went -37.7%% and"+
						" -38.2%% by moving the accumulating loop outermost, at ZERO extra memory,"+
						" while transposing the same operands — the remedy that paid on three"+
						" other kernels — gave only -7.3%% and -12.6%% and cost 38%% more bytes and"+
						" 195 more allocations for the intermediates. The discriminator is whether"+
						" the loop carrying the strided index is INDEPENDENT: if it merely"+
						" accumulates, it can move outermost and no copy is needed at all."+
						" Transpose only when that loop cannot move.",
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
func inDeclinedTypedFallback(parent map[ast.Node]ast.Node, n ast.Node, ns nameSets) bool {
	for cur := n; cur != nil; cur = parent[cur] {
		switch s := parent[cur].(type) {
		case *ast.IfStmt:
			// Only the ELSE side is inert. The taken side is the fast path itself, and an
			// accessor loop there would be a real finding.
			if s.Else == cur && (reachesFastPath(s.Body, ns) ||
				(s.Init != nil && reachesFastPath(s.Init, ns)) ||
				(s.Cond != nil && reachesFastPath(s.Cond, ns))) {
				return true
			}
		case *ast.CaseClause:
			// A default clause carries no expression list; a converted sibling is any other
			// clause that reaches typed storage.
			if len(s.List) == 0 && switchHasConvertedClause(parent, s, ns) {
				return true
			}
		case *ast.BlockStmt:
			// EARLY-RETURN SPELLING. `if fast { ...; return }` followed by the accessor loop at
			// function level is the same construct as if/else — linalg NormFro is written this
			// way — and keying only on Else misses it. The guard has to TERMINATE for the tail to
			// be a fallback: without the return both run, and the loop is on the common path.
			if precededByTerminatingFastPath(s, cur, ns) {
				return true
			}
		}
	}
	return false
}

// precededByTerminatingFastPath reports whether some statement before target in block is an
// else-less `if <fast path> { ...; return }`, which makes everything after it the declined arm.
func precededByTerminatingFastPath(block *ast.BlockStmt, target ast.Node, ns nameSets) bool {
	for _, st := range block.List {
		if st == target {
			return false // reached the loop without passing a terminating guard
		}
		ifs, ok := st.(*ast.IfStmt)
		if !ok || ifs.Else != nil || !blockTerminates(ifs.Body) {
			continue
		}
		if reachesFastPath(ifs.Body, ns) ||
			(ifs.Init != nil && reachesFastPath(ifs.Init, ns)) ||
			(ifs.Cond != nil && reachesFastPath(ifs.Cond, ns)) {
			return true
		}
	}
	return false
}

// blockTerminates reports whether b ends in a return or a branch, so control cannot fall through
// to the statements after the guard.
func blockTerminates(b *ast.BlockStmt) bool {
	if b == nil || len(b.List) == 0 {
		return false
	}
	switch b.List[len(b.List)-1].(type) {
	case *ast.ReturnStmt, *ast.BranchStmt:
		return true
	}
	return false
}

// reachesFastPath reports whether a subtree acquires a contiguous typed view, either directly via
// Storage().<Typed>() or through one of the package's configured fast-path helpers. NormFro's guard
// is `if d, ok := flatRowMajor(a); ok`, and flatRowMajor is already in fastPathHelpers — the
// config knew about it before this suppression did.
func reachesFastPath(root ast.Node, ns nameSets) bool {
	if readsTypedStorage(root) {
		return true
	}
	found := false
	ast.Inspect(root, func(n ast.Node) bool {
		if found {
			return false
		}
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		switch fn := call.Fun.(type) {
		case *ast.Ident:
			if ns.fastPath[fn.Name] {
				found = true
			}
		case *ast.SelectorExpr:
			if ns.fastPath[fn.Sel.Name] {
				found = true
			}
		}
		return !found
	})
	return found
}

// switchHasConvertedClause reports whether the switch owning def has some OTHER, non-default clause
// that reaches typed storage — the sibling that makes def the declined arm.
func switchHasConvertedClause(parent map[ast.Node]ast.Node, def *ast.CaseClause, ns nameSets) bool {
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
		if reachesFastPath(cc, ns) {
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
						" THOSE SITES, not here. A THIRD MEASUREMENT, and the cheapest shape to look"+
						" for: a triangular-solve substitution ran its right-hand-side column index"+
						" OUTERMOST, so each step jumped k elements through two buffers and re-fetched"+
						" the same factor element once per column; making that index innermost went"+
						" -35.6%% and made the factor element loop-invariant, loaded once per (i,p)"+
						" instead of once per (i,p,c). When the strided operand is indexed by an outer"+
						" loop that carries no dependence, interchange is the whole fix and costs"+
						" nothing. THE SIZE DEPENDENCE IS NOT THEORETICAL, and the same kernel"+
						" measures both ways: a Householder QR interchanged exactly this way went"+
						" -35.0%% at 128x64 and NOTHING at 32x16, where the whole factorization is"+
						" L1-resident and the layout it walks stops mattering. Measure the size you"+
						" care about; a small cell will report a real transform as noise, and a"+
						" large one will oversell it for a caller who never gets there.",
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

// serialReductionChainFindings flags PS3010 — a reduction loop with ONE accumulator, where every
// iteration's add depends on the previous one. On an out-of-order core the loop is bound by
// floating-point add/FMA LATENCY rather than throughput, and splitting the accumulator into four
// independent partials removes that dependency.
//
// The shape is required to be a pure reduction: exactly one `acc += <indexed expr>` in the body,
// acc read nowhere else in the loop, and no branching or control flow. A loop that also TESTS its
// accumulator (PS3008's early-bail shape) is excluded, because four partials cannot be compared
// against the threshold without summing them first, which changes when the loop bails.
//
// The applied form needs no special case: it has four ADD_ASSIGN statements, so the count test
// already declines it.
func serialReductionChainFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		body := loopBody(n)
		if body == nil {
			return true
		}
		// A loop that already strides by more than one has had this transform applied: the four
		// partials of an unrolled dot product are four distinct names each written once, which
		// otherwise satisfies every test below and would report the fix as the defect.
		if loopStrideExceedsOne(n) {
			return true
		}
		// One to four FUSED accumulators, each written exactly once. Requiring exactly one was the
		// first cut and it was wrong in the direction that costs the most: a fused dot-plus-norm
		// loop is the canonical hot reduction, and restricting to a single chain flagged the COLD
		// norm loop in nlp MaxContextCosine (0.00s) while missing the hot pair loop right below it
		// (0.37s, the second-hottest own-package line in nlp). Two chains already give the core two
		// independent streams, and the split still measured 2.90x at dim=768.
		//
		// The ceiling is four because past that the partials themselves compete for registers, and
		// a loop already carrying four independent chains has the parallelism this check exists to
		// recommend.
		accs := map[string]ast.Expr{}
		adds := 0
		simple := true
		for _, st := range body.List {
			as, ok := st.(*ast.AssignStmt)
			if !ok {
				simple = false // an if/for/switch/call in the body: not a pure reduction
				continue
			}
			if as.Tok == token.ADD_ASSIGN && len(as.Lhs) == 1 && len(as.Rhs) == 1 {
				if nm := identName(as.Lhs[0]); nm != "" {
					accs[nm] = as.Rhs[0]
					adds++
					continue
				}
			}
			if as.Tok != token.ASSIGN && as.Tok != token.DEFINE {
				simple = false
			}
		}
		// adds != len(accs) means some accumulator is written twice — that is the applied,
		// already-unrolled form, not a candidate.
		if !simple || len(accs) == 0 || len(accs) > 4 || adds != len(accs) {
			return true
		}
		// Every accumulator must be write-only inside the loop. Any other mention is a read — a
		// threshold test, a running comparison — and the split would change what it observes.
		// A loop that TESTS its accumulator is PS3008's early-bail shape and is excluded here,
		// because four partials cannot be compared against a threshold without summing them first.
		mentions := map[string]int{}
		ast.Inspect(body, func(k ast.Node) bool {
			if id, ok := k.(*ast.Ident); ok {
				if _, isAcc := accs[id.Name]; isAcc {
					mentions[id.Name]++
				}
			}
			return true
		})
		acc := ""
		for nm := range accs {
			if mentions[nm] != 1 {
				return true
			}
			if acc == "" || nm < acc {
				acc = nm // deterministic name for the message; map order is not stable
			}
		}
		// Every term has to vary per iteration; summing a loop-invariant is a different (and much
		// better) fix, and PS4006 owns it.
		for _, rhs := range accs {
			varies := false
			ast.Inspect(rhs, func(k ast.Node) bool {
				if _, ok := k.(*ast.IndexExpr); ok {
					varies = true
				}
				return !varies
			})
			if !varies {
				return true
			}
		}
		// Integer addition is exactly associative, so for an integer accumulator the split is
		// bit-identical by construction and the contract check below does not apply at all. Say
		// which case this is rather than making every reader re-derive it.
		safety := "THE TRANSFORM IS NOT BIT-IDENTICAL, and that is the whole risk: it reassociates" +
			" the sum, so the result moves in the last ulp. DO NOT apply it where the exact value is" +
			" pinned."
		switch accumulatorKind(fn, acc) {
		case "integer":
			safety = "THIS ONE IS AN INTEGER ACCUMULATOR, so the usual objection does not apply:" +
				" integer addition is exactly associative and the split is bit-identical BY" +
				" CONSTRUCTION, with no contract to clear. The payoff is smaller than the float case" +
				" because an integer add is a single cycle and the loop is closer to load-bound —" +
				" measured 271 -> 168 ns at d=768 (1.61x) — but it is unconditionally safe."
		case "float":
			safety = "THE TRANSFORM IS NOT BIT-IDENTICAL, and that is the whole risk: it reassociates" +
				" the FLOAT sum, so the result moves in the last ulp. DO NOT apply it where the exact" +
				" value is pinned. f32 gains MORE than f64, not less — 610 -> 167 ns at d=768 (3.65x)" +
				" against 2.90x for f64 — because the same add latency covers half the memory traffic."
		}
		out = append(out, finding{
			pos:      fset.Position(n.Pos()),
			category: "serial-reduction-chain",
			msg: fmt.Sprintf("%q is a single-accumulator reduction: each iteration's add waits on the"+
				" previous one, so the loop runs at floating-point add LATENCY, not throughput."+
				" Four independent partials summed at the end removed that stall in a direct"+
				" measurement on this host (Apple M2 Pro, darwin/arm64, go1.26, f64 dot product):"+
				" 537.8 -> 177.3 ns at d=512 (3.03x) and 89.3 -> 43.6 ns at d=128 (2.05x). The 512"+
				" figure matches the hardware prediction — 512 dependent adds at about four cycles"+
				" each is roughly 600 ns at 3.4 GHz — so this is a latency stall, not a measurement"+
				" artifact. %s Two of the hottest reductions in this repo are blocked for"+
				" precisely that reason and are NOT candidates — nlp randomOrthogonal, whose matrix"+
				" TurboQuant regenerates at dequantization time so a changed draw would break every"+
				" model quantized with the old one, and classic ballTree.within, whose sum decides"+
				" exact-label DBSCAN goldens. Check for a bit-stability test and a reproducibility"+
				" contract BEFORE measuring, then benchmark: the win is real but it is worth nothing"+
				" if the loop is cold (PERF-HOTNESS-IS-NOT-SYNTAX-001)", acc, safety),
		})
		return true
	})
	return out
}

// loopStrideExceedsOne reports whether a 3-clause for loop advances its index by more than one per
// iteration (`i += 4`). A range loop and a plain `i++` both stride by one. Used to recognize an
// already-unrolled loop, which is the applied form of PS3010 rather than a candidate for it.
func loopStrideExceedsOne(n ast.Node) bool {
	fs, ok := n.(*ast.ForStmt)
	if !ok || fs.Post == nil {
		return false
	}
	as, ok := fs.Post.(*ast.AssignStmt)
	if !ok || as.Tok != token.ADD_ASSIGN || len(as.Rhs) != 1 {
		return false
	}
	lit, ok := as.Rhs[0].(*ast.BasicLit)
	if !ok || lit.Kind != token.INT {
		return false
	}
	return lit.Value != "1"
}

// accumulatorKind infers whether an accumulator is an integer or a float from its declaration in
// the enclosing function. AST only — there is no type checker here — so it recognizes the two
// spellings that actually occur: an explicit `var s float64` / `var s int`, and a `s := 0` or
// `s := 0.0` whose literal kind gives the type away. Anything else returns "", and the caller
// falls back to advice that covers both.
//
// The distinction is not cosmetic. Integer addition is EXACTLY associative, so splitting an integer
// accumulator is bit-identical by construction and needs no contract check at all; floating-point
// addition is not, and there the split is a value change that has to be cleared first.
func accumulatorKind(fn *ast.FuncDecl, name string) string {
	kind := ""
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch s := n.(type) {
		case *ast.DeclStmt:
			gd, ok := s.Decl.(*ast.GenDecl)
			if !ok || gd.Tok != token.VAR {
				return true
			}
			for _, sp := range gd.Specs {
				vs, ok := sp.(*ast.ValueSpec)
				if !ok || vs.Type == nil {
					continue
				}
				id, ok := vs.Type.(*ast.Ident)
				if !ok {
					continue
				}
				for _, nm := range vs.Names {
					// Only a RECOGNIZED type name settles the question. A named or generic type
					// resolves to "", and overwriting with it would erase a kind already found.
					if nm.Name == name {
						if k := numericKindOfTypeName(id.Name); k != "" {
							kind = k
						}
					}
				}
			}
		case *ast.AssignStmt:
			if s.Tok != token.DEFINE || len(s.Lhs) != len(s.Rhs) {
				return true
			}
			for i, l := range s.Lhs {
				if identName(l) != name {
					continue
				}
				if lit, ok := s.Rhs[i].(*ast.BasicLit); ok {
					switch lit.Kind {
					case token.INT:
						kind = "integer"
					case token.FLOAT:
						kind = "float"
					}
				}
			}
		}
		return true
	})
	return kind
}

// numericKindOfTypeName classifies a builtin numeric type name.
func numericKindOfTypeName(t string) string {
	switch t {
	case "int", "int8", "int16", "int32", "int64",
		"uint", "uint8", "uint16", "uint32", "uint64", "uintptr", "byte", "rune":
		return "integer"
	case "float32", "float64":
		return "float"
	}
	return ""
}

// staticChunkBarrierFindings flags PS3011 — work split into equal chunks sized by the worker count,
// dispatched one goroutine per chunk, and joined at a barrier. The partition assumes every worker
// retires its share at the same rate, and on a heterogeneous CPU it does not.
//
// The signature is three things in one function: a ceil-division of the work by a worker count, a
// `go` inside a loop, and a Wait. Silent once the function reaches for sync/atomic, which is what
// claiming looks like.
func staticChunkBarrierFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var ceil ast.Node
	var spawnInLoop, waits, claims bool
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch s := n.(type) {
		case *ast.BinaryExpr:
			// (X + Y - 1) / Y — the ceil-division idiom. Only the shape is checked; whether Y is
			// really a worker count is left to the reader, and the message says so.
			if s.Op == token.QUO && isCeilNumerator(s.X, s.Y) {
				ceil = s
			}
		case *ast.GoStmt:
			if enclosingLoopOf(fn.Body, s) != nil {
				spawnInLoop = true
			}
		case *ast.SelectorExpr:
			switch s.Sel.Name {
			case "Wait":
				waits = true
			case "Add", "CompareAndSwap", "Load":
				// An atomic cursor is the applied form. Distinguishing it from WaitGroup.Add needs
				// the receiver, so only a package-qualified atomic or an atomic-typed field counts.
				if id, ok := s.X.(*ast.Ident); ok && id.Name == "atomic" {
					claims = true
				}
			}
		case *ast.Ident:
			if s.Name == "atomic" {
				claims = true
			}
		}
		return true
	})
	if ceil == nil || !spawnInLoop || !waits || claims {
		return nil
	}
	return []finding{{
		pos:      fset.Position(ceil.Pos()),
		category: "static-chunk-barrier",
		msg: fmt.Sprintf("%s deals equal chunks to one goroutine each and joins at a barrier, so the"+
			" slowest worker sets the wall clock. THAT IS NOT A TAIL EFFECT ON A HETEROGENEOUS CPU,"+
			" IT IS THE COMMON CASE: an Apple M2 Pro has 8 performance and 4 efficiency cores, so a"+
			" chunk landing on an E core can take several times as long and every P core waits for"+
			" it. MEASURED on the autograd WKV VJP, where the giveaway was that MORE CORES MADE IT"+
			" SLOWER — GOMAXPROCS=8 ran 3.36ms against 3.76ms at 12 — and pthread_cond_wait was"+
			" 47.96%% of the profile, more than every line of the kernel combined. Claiming units"+
			" through an atomic cursor instead went -28.73%% and -29.58%% (p=0.000, n=8), BIT-IDENTICAL,"+
			" because which worker runs a unit cannot change that unit's arithmetic. THE DIAGNOSTIC IS"+
			" CHEAP AND COMES FIRST: sweep GOMAXPROCS and look at a FUNCTION profile, not a line"+
			" profile — a line profile ranks the kernel and hides the waiting, which is how this site"+
			" was nearly missed. ATTRIBUTE THE CURVE BEFORE ACTING ON IT, with a control: a GOMAXPROCS"+
			" curve is a property of the WHOLE benchmark, not of the chunker you happen to be reading."+
			" Force THIS chunker serial and re-run; if the benchmark does not move, the scaling loss is"+
			" somewhere else. nlp quant_mamba2 screened at +4.5%% from GOMAXPROCS 8 to 12 and its curve"+
			" bottomed at SIX, yet forcing its parallelChunks serial changed the decode benchmark by"+
			" 0.06%% — it was not on the hot path at all, and both a claim rewrite and a work-sized"+
			" worker cap measured null-to-negative there before the control was run. Note also that a"+
			" minimum BELOW the performance-core count indicts per-worker dispatch overhead rather than"+
			" efficiency-core imbalance, and claiming does not fix that. Preconditions the check cannot see: the units must be independent"+
			" (they already are, or the static split would be wrong too), the grain must stay large"+
			" enough that the claim is negligible, and any per-chunk scratch indexed by a chunk"+
			" number must be re-keyed to the WORKER instead, since claiming makes the number of units"+
			" exceed the number of workers. TWO CONDITIONS DECIDE WHETHER CLAIMING PAYS AT ALL,"+
			" both learned by converting a sibling and reverting it. A CLAIM'S WORKING SET MUST BE"+
			" THE CLAIM, not the whole input: a conv1d backward whose body walks the entire"+
			" sequence per call re-streamed its input once per claim instead of once per worker,"+
			" about 42x the memory traffic at grain 8, and ran 126%% SLOWER than the split it"+
			" replaced — at four claims per worker it was still 8%% behind, so that one keeps its"+
			" static split. And THE BODY MUST NOT ALLOCATE PER INVOCATION: a distillation VJP"+
			" allocating softmax scratch per call went -7.4%% on time but +26%% bytes and +50%%"+
			" allocations, because a cursor invoked it 32 times where the split invoked it 12 —"+
			" carry per-worker scratch first, then claim. Where both hold the win is large: an MoE"+
			" combine backward went -34.5%% and -37.4%% with the arms measured in both orders",
			fn.Name.Name),
	}}
}

// unparen strips redundant parentheses, which the ceil idiom always carries:
// `chunk := (n + nw - 1) / nw` parses its numerator as a ParenExpr, not a BinaryExpr.
func unparen(e ast.Expr) ast.Expr {
	for {
		p, ok := e.(*ast.ParenExpr)
		if !ok {
			return e
		}
		e = p.X
	}
}

// isCeilNumerator reports whether x is the `a + d - 1` half of a ceil-division by d.
func isCeilNumerator(x, d ast.Expr) bool {
	outer, ok := unparen(x).(*ast.BinaryExpr)
	if !ok || outer.Op != token.SUB || !isIntLit(outer.Y, "1") {
		return false
	}
	inner, ok := unparen(outer.X).(*ast.BinaryExpr)
	if !ok || inner.Op != token.ADD {
		return false
	}
	ds := exprText(unparen(d))
	return exprText(unparen(inner.Y)) == ds || exprText(unparen(inner.X)) == ds
}

func isIntLit(e ast.Expr, v string) bool {
	lit, ok := e.(*ast.BasicLit)
	return ok && lit.Kind == token.INT && lit.Value == v
}

// enclosingLoopOf returns the innermost loop in root containing n, or nil.
func enclosingLoopOf(root ast.Node, n ast.Node) ast.Node {
	var found ast.Node
	ast.Inspect(root, func(k ast.Node) bool {
		if loopBody(k) == nil {
			return true
		}
		if k.Pos() <= n.Pos() && n.End() <= k.End() {
			found = k
		}
		return true
	})
	return found
}

// sliceBuiltForOneElementFindings flags PS3012 — a package-level function call whose result is
// immediately indexed by a constant, `f(x)[0]`. The callee builds a whole collection and the caller
// keeps one item of it, so every other item and the slice header are pure waste.
//
// Restricted to a bare identifier callee. Method chains like t.Shape()[0] or s.Storage().F64()[0]
// return a view or a field and allocate nothing, and including them buried the real class.
func sliceBuiltForOneElementFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		ix, ok := n.(*ast.IndexExpr)
		if !ok {
			return true
		}
		lit, ok := ix.Index.(*ast.BasicLit)
		if !ok || lit.Kind != token.INT {
			return true
		}
		call, ok := ix.X.(*ast.CallExpr)
		if !ok {
			return true
		}
		id, ok := call.Fun.(*ast.Ident)
		if !ok || isBuiltinName(id.Name) {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(ix.Pos()),
			category: "slice-built-for-one-element",
			msg: fmt.Sprintf("%s(...)[%s] builds a collection and keeps ONE element of it. Where the"+
				" callee allocates, that is the whole allocation wasted for one item, and it repeats"+
				" on every call. MEASURED on nlp QuantMamba2's decode, where rows2D materializes the"+
				" in_proj output as [][]float64 once per LAYER per token and the caller takes row 0:"+
				" at seq=1 each call allocates a header slice and a row that live for microseconds,"+
				" and they were 37%% of all allocation OBJECTS in the step. Replacing the two decode"+
				" sites with a scratch buffer threaded through the per-stream layer state went"+
				" -18.37%% B/op and -3.67%% allocs/op across all seven quantization formats, with"+
				" sec/op -0.54%% (p<=0.01 in every cell). THE FIX IS A SCRATCH PARAMETER, not a cache:"+
				" put the buffer on whatever object is already per-stream or per-worker, never on a"+
				" shared model or layer, since those are read concurrently by every stream decoding"+
				" against the same weights. Check first that the caller only READS the element — the"+
				" original returns an independent copy, and a scratch buffer is reused, so a site that"+
				" mutates its row or retains it past the call needs the copy it already has."+
				" This check cannot see whether the callee allocates at all; confirm that before"+
				" acting (PERF-HOTNESS-IS-NOT-SYNTAX-001)", id.Name, lit.Value),
		})
		return true
	})
	return out
}

// isBuiltinName reports whether an identifier is a Go builtin whose result is never a fresh
// collection worth avoiding.
func isBuiltinName(s string) bool {
	switch s {
	case "len", "cap", "make", "new", "append", "copy", "min", "max", "complex", "real", "imag":
		return true
	}
	return false
}

// leakingFormatParamFindings flags PS3013 — a pointer-carrying parameter handed to a fmt call.
//
// Passing it as an interface argument makes Go's escape analysis mark the PARAMETER as leaking,
// and that verdict is a property of the function, not of the path: every caller must then
// heap-allocate its argument, even though the formatting usually sits on a panic or error branch
// that never executes.
func leakingFormatParamFindings(fset *token.FileSet, fn *ast.FuncDecl, ns nameSets) []finding {
	if fn.Type.Params == nil {
		return nil
	}
	leakable := map[string]bool{}
	for _, f := range fn.Type.Params.List {
		if !typeCanLeak(f.Type, ns) {
			continue
		}
		for _, nm := range f.Names {
			if nm.Name != "_" {
				leakable[nm.Name] = true
			}
		}
	}
	if len(leakable) == 0 {
		return nil
	}
	var out []finding
	seen := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		pkg, ok := sel.X.(*ast.Ident)
		if !ok || pkg.Name != "fmt" {
			return true
		}
		for _, arg := range call.Args {
			id, ok := arg.(*ast.Ident)
			if !ok || !leakable[id.Name] || seen[id.Name] {
				continue
			}
			seen[id.Name] = true
			out = append(out, finding{
				pos:      fset.Position(arg.Pos()),
				category: "leaking-format-param",
				msg: fmt.Sprintf("%q is a pointer-carrying parameter of %s handed to fmt.%s, which"+
					" makes escape analysis mark it LEAKING — and that verdict belongs to the"+
					" function, not to this branch, so EVERY CALLER heap-allocates its argument even"+
					" though this path usually never runs. MEASURED on tensor.NewOn, whose"+
					" invalid-shape panic formatted its shape with %%v: the literal at each call site"+
					" escaped, costing one allocation per tensor CREATED anywhere in the tree."+
					" Replacing it with shape.String() — a method escape analysis already proves"+
					" non-escaping — took a Jamba decode step -5.96%% allocs/op and -0.19%% B/op with"+
					" time unchanged (p=1.000, n=12), and QuantMamba2 decode -8.70%% allocs."+
					" THE FIX IS A NON-ESCAPING FORMATTER, not deleting the message: give the type a"+
					" String method that only reads its receiver and build the string from it, or"+
					" format the fields individually. Verify with go build -gcflags=-m that the"+
					" parameter changes from `leaking param` to `does not escape`, since a second"+
					" leak elsewhere in the function keeps the old verdict and the fix buys nothing."+
					" This check cannot tell whether the parameter already leaks for another reason,"+
					" so confirm before acting", id.Name, exprText(paramTypeOf(fn, id.Name)), sel.Sel.Name),
			})
		}
		return true
	})
	return out
}

// paramTypeOf returns the declared type expression of the named parameter.
func paramTypeOf(fn *ast.FuncDecl, name string) ast.Expr {
	for _, f := range fn.Type.Params.List {
		for _, nm := range f.Names {
			if nm.Name == name {
				return f.Type
			}
		}
	}
	return nil
}

// typeCanLeak reports whether a parameter type carries a pointer, so that leaking it forces the
// caller's argument onto the heap. Slices, maps and pointers are syntactic; a NAMED type needs the
// configured pointerTypeNames list, since with no type checker `Shape` and an int alias look alike.
func typeCanLeak(t ast.Expr, ns nameSets) bool {
	switch e := t.(type) {
	case *ast.ArrayType:
		return e.Len == nil // slice; a fixed array is a value
	case *ast.MapType, *ast.StarExpr:
		return true
	case *ast.Ident:
		return ns.pointerTypes[e.Name]
	case *ast.SelectorExpr:
		return ns.pointerTypes[e.Sel.Name]
	}
	return false
}

// coupledIndexWeightFindings flags PS3014 — a nested-loop reduction whose accumulated term is
// scaled by an arithmetic combination of BOTH loop variables, used as a VALUE rather than as an
// index. That coupling is what makes such a sum look irreducibly quadratic, and it is often not.
func coupledIndexWeightFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	seen := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		outer := loopVarOf(n)
		if outer == "" {
			return true
		}
		ast.Inspect(loopBody(n), func(m ast.Node) bool {
			inner := loopVarOf(m)
			if inner == "" || inner == outer {
				return true
			}
			ast.Inspect(loopBody(m), func(st ast.Node) bool {
				as, ok := st.(*ast.AssignStmt)
				if !ok || (as.Tok != token.ADD_ASSIGN && as.Tok != token.SUB_ASSIGN) || len(as.Rhs) != 1 {
					return true
				}
				be := coupledFactor(as.Rhs[0], outer, inner)
				if be == nil {
					return true
				}
				key := fmt.Sprintf("%d", be.Pos())
				if seen[key] {
					return true
				}
				seen[key] = true
				out = append(out, finding{
					pos:      fset.Position(be.Pos()),
					category: "coupled-index-weight",
					msg: fmt.Sprintf("this accumulation is scaled by %q, an arithmetic combination of"+
						" the outer index %q and the inner index %q used as a VALUE. That coupling is"+
						" what makes a doubly-nested reduction look irreducibly quadratic, and a"+
						" DIFFERENCE usually is not: split it. MEASURED on the autograd WKV backward,"+
						" whose dw term is a sum over t and i<t of (t-1-i) times a weight — rewriting"+
						" (t-1-i) as (t-1) minus i turns one distance-weighted sum into (t-1)*S minus"+
						" S1, where S accumulates the weights and S1 accumulates i times the weights,"+
						" both ordinary prefix sums maintained in O(1) per step. dw was the ONLY"+
						" gradient in that kernel whose weight depended on the distance and the only"+
						" reason the whole pass stayed quadratic; splitting it took seq=512 from"+
						" 10335us to 580us, 17.8x, with the cost per doubling of seq falling from"+
						" 3.85x to 1.81x. PRECONDITION THIS CHECK CANNOT SEE: the rest of the term"+
						" must factor into parts that depend on t alone and on i alone, or there is"+
						" nothing to hoist out of the inner sum. A product or a modulus of the two"+
						" indices generally does not split and is reported only so it can be ruled"+
						" out deliberately. Index arithmetic is excluded — a[t*n+i] couples the"+
						" indices to ADDRESS memory, not to weight a value."+
						" THE SPLIT ONLY PAYS WHEN THE INNER SUM COLLAPSES TO A SCALAR, and that is"+
						" how to triage this list rather than by syntax. In WKV the distance-weighted"+
						" term is summed over i into one number per channel, so hoisting (t-1) out"+
						" leaves two running sums and the inner loop disappears. In ATTENTION the very"+
						" same shape appears as an ALiBi bias, slopes[h]*float64(j-i) added to a"+
						" per-pair score that is then used individually — there is no inner reduction"+
						" to hoist, and the enclosing loop is irreducibly quadratic because it must"+
						" produce every pair. Five of the six sites in this tree are that case and"+
						" are NOT candidates (DISTANCE-WEIGHTS-SPLIT-INTO-TWO-PREFIX-SUMS-001)",
						renderExpr(be), outer, inner),
				})
				return true
			})
			return true
		})
		return true
	})
	return out
}

// loopVarOf returns the single index variable a loop advances, or "" if it is not that shape.
func loopVarOf(n ast.Node) string {
	switch s := n.(type) {
	case *ast.RangeStmt:
		if id, ok := s.Key.(*ast.Ident); ok && id.Name != "_" {
			return id.Name
		}
	case *ast.ForStmt:
		if s.Init == nil {
			return ""
		}
		as, ok := s.Init.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE || len(as.Lhs) != 1 {
			return ""
		}
		if id, ok := as.Lhs[0].(*ast.Ident); ok {
			return id.Name
		}
	}
	return ""
}

// coupledFactor returns the sub-expression combining both loop variables as a value, or nil.
//
// The walk is explicit rather than an ast.Inspect with a parent lookup, because the exclusion is
// the entire precision of this check and it has to be structural: an INDEX subtree is skipped
// outright. a[t*n+i] couples the indices to compute an address, which is simply how a 2-D buffer is
// walked and says nothing about the reduction's complexity; only a coupling used as a FACTOR does.
func coupledFactor(root ast.Expr, outer, inner string) *ast.BinaryExpr {
	var walk func(ast.Expr) *ast.BinaryExpr
	walk = func(e ast.Expr) *ast.BinaryExpr {
		switch x := e.(type) {
		case nil:
			return nil
		case *ast.IndexExpr:
			// Descend into the operand (it may itself hold a factor) but never the index.
			return walk(x.X)
		case *ast.ParenExpr:
			return walk(x.X)
		case *ast.CallExpr:
			// A coupling only counts when it is converted to a FLOAT. That is the discriminator
			// between a distance WEIGHT and a flat INDEX: float64(t-1-i) scales a term, whereas
			// r*d+j addresses one, and without a type checker the conversion is the only reliable
			// signal. Requiring it took the finding list from 18 to the handful that are real.
			if isFloatConv(x) && len(x.Args) == 1 {
				if be, ok := unparen(x.Args[0]).(*ast.BinaryExpr); ok &&
					isArithOp(be.Op) && mentionsAsValue(be, outer) && mentionsAsValue(be, inner) {
					return be
				}
			}
			for _, a := range x.Args {
				if r := walk(a); r != nil {
					return r
				}
			}
			return nil
		case *ast.UnaryExpr:
			return walk(x.X)
		case *ast.SelectorExpr:
			return walk(x.X)
		case *ast.BinaryExpr:
			if r := walk(x.X); r != nil {
				return r
			}
			return walk(x.Y)
		}
		return nil
	}
	return walk(root)
}

// mentionsAsValue reports whether name appears in e OUTSIDE every index subtree.
//
// The distinction is the check's precision. l[k][i] * lbar[k][j] mentions both loop indices, but
// only as SUBSCRIPTS: its value skeleton is l * lbar, which couples nothing and is just how a
// matmul-shaped term is written. Searching the whole subtree instead flagged 189 sites, essentially
// all of them this shape.
func mentionsAsValue(e ast.Expr, name string) bool {
	switch x := e.(type) {
	case nil:
		return false
	case *ast.Ident:
		return x.Name == name
	case *ast.IndexExpr:
		return mentionsAsValue(x.X, name) // never x.Index
	case *ast.ParenExpr:
		return mentionsAsValue(x.X, name)
	case *ast.UnaryExpr:
		return mentionsAsValue(x.X, name)
	case *ast.SelectorExpr:
		return mentionsAsValue(x.X, name)
	case *ast.BinaryExpr:
		return mentionsAsValue(x.X, name) || mentionsAsValue(x.Y, name)
	case *ast.CallExpr:
		// An ELEMENT ACCESSOR's arguments are coordinates, not values: t.AtF64(i, j) addresses an
		// element exactly as t[i][j] does, and counting those as a coupling flagged 78 sites of
		// which essentially none were one. A conversion such as float64(t-1-i) is descended, since
		// that is precisely where a real distance weight appears.
		if accessorCallName(x) != "" {
			return false
		}
		for _, a := range x.Args {
			if mentionsAsValue(a, name) {
				return true
			}
		}
		return false
	}
	return false
}

// accessorCallName returns the element-accessor name a call invokes, or "".
func accessorCallName(c *ast.CallExpr) string {
	switch f := c.Fun.(type) {
	case *ast.SelectorExpr:
		if coupledAccessorNames[f.Sel.Name] {
			return f.Sel.Name
		}
	case *ast.Ident:
		if coupledAccessorNames[f.Name] {
			return f.Name
		}
	}
	return ""
}

// coupledAccessorNames are the element accessors whose arguments are coordinates. Kept local and
// explicit rather than read from elementAccessors, which lists only the tensor methods; `at` is
// this repository's package-local two-coordinate helper and reads the same way.
var coupledAccessorNames = map[string]bool{
	"AtF64": true, "AtF32": true, "At": true, "SetF64": true, "SetF32": true, "Set": true,
	"at": true, "idx": true, "index": true,
}

// isFloatConv reports whether a call is a float conversion.
func isFloatConv(c *ast.CallExpr) bool {
	id, ok := c.Fun.(*ast.Ident)
	return ok && (id.Name == "float64" || id.Name == "float32")
}

func isArithOp(op token.Token) bool {
	switch op {
	case token.ADD, token.SUB, token.MUL, token.QUO, token.REM:
		return true
	}
	return false
}

func mentionsIdent(root ast.Node, name string) bool {
	found := false
	ast.Inspect(root, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && id.Name == name {
			found = true
		}
		return !found
	})
	return found
}

// renderExpr prints an expression for a finding message. exprText only renders the x / x.F / x.F[i]
// shapes a cache-slot message needs and returns "" for anything else, which turned PS3014's message
// into a bare empty pair of quotes.
func renderExpr(e ast.Expr) string {
	var b strings.Builder
	if err := printer.Fprint(&b, token.NewFileSet(), e); err != nil {
		return "?"
	}
	return b.String()
}

// --- PS3015: a struct field given an allocation but never read ---

var (
	deadFieldDeclared = map[string]bool{} // unexported struct field names declared in this package
	deadFieldRead     = map[string]bool{} // field names read anywhere in this package
)

// collectWriteOnlyFields records every unexported struct field declared in the package and every
// field name READ anywhere in it. Matching is by NAME, without types, which is deliberately
// conservative: a same-named field read elsewhere suppresses the finding rather than risking a
// false one.
//
// Restricted to UNEXPORTED names because those cannot be read outside the package, which is what
// makes "no reads anywhere here" conclusive.
func collectWriteOnlyFields(fset *token.FileSet, files []*ast.File) {
	deadFieldDeclared = map[string]bool{}
	deadFieldRead = map[string]bool{}
	for _, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			st, ok := n.(*ast.StructType)
			if !ok || st.Fields == nil {
				return true
			}
			for _, fld := range st.Fields.List {
				for _, nm := range fld.Names {
					if !nm.IsExported() {
						deadFieldDeclared[nm.Name] = true
					}
				}
			}
			return true
		})
	}
	for _, f := range files {
		markFieldReads(f)
	}
	// TEST FILES COUNT AS READERS, and they are parsed here rather than taken from the scan set
	// because the default run excludes them. Without this, any field whose only consumer is a
	// test reads as dead: nlp's layerSkipDecodeTrace.blockTokens is written in production and
	// ranged over only by llama_layerskip_decode_internal_test.go, and it was reported until
	// test files were included. Nothing in a test file is ever REPORTED — they are read for
	// field mentions alone.
	dirs := map[string]bool{}
	for _, f := range files {
		dirs[filepath.Dir(fset.Position(f.Pos()).Filename)] = true
	}
	for dir := range dirs {
		ents, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, e := range ents {
			if e.IsDir() || !strings.HasSuffix(e.Name(), "_test.go") {
				continue
			}
			tf, err := parser.ParseFile(fset, filepath.Join(dir, e.Name()), nil, 0)
			if err != nil {
				continue
			}
			markFieldReads(tf)
		}
	}
}

// markFieldReads records field names appearing anywhere OTHER than as a write target: the left side
// of an assignment, or the key of a composite literal element.
func markFieldReads(f *ast.File) {
	writeTargets := map[ast.Node]bool{}
	ast.Inspect(f, func(n ast.Node) bool {
		switch s := n.(type) {
		case *ast.AssignStmt:
			for _, l := range s.Lhs {
				switch t := l.(type) {
				case *ast.SelectorExpr:
					writeTargets[t.Sel] = true
				case *ast.IndexExpr:
					// x.f[i] = v STORES INTO an element. It reads the header to find the
					// backing array, but it is not a use of the DATA, and treating it as one
					// is what let a field written only through its own initialization loop
					// look alive.
					if sel, ok := t.X.(*ast.SelectorExpr); ok {
						writeTargets[sel.Sel] = true
					}
				}
			}
		case *ast.KeyValueExpr:
			if id, ok := s.Key.(*ast.Ident); ok {
				writeTargets[id] = true
			}
		case *ast.RangeStmt:
			// `for i := range x.f` with NO value variable ranges for the indices alone, which
			// an initialization loop does to fill the field. That is not a read of the data
			// either. `for _, v := range x.f` IS, because v carries an element out, so the
			// value variable is the discriminator and dropping it would flag genuinely-read
			// fields.
			if s.Value != nil && identName(s.Value) != "_" {
				return true
			}
			if sel, ok := s.X.(*ast.SelectorExpr); ok {
				writeTargets[sel.Sel] = true
			}
		}
		return true
	})
	ast.Inspect(f, func(n ast.Node) bool {
		if sel, ok := n.(*ast.SelectorExpr); ok && !writeTargets[sel.Sel] {
			deadFieldRead[sel.Sel.Name] = true
		}
		return true
	})
}

// writeOnlyAllocFieldFindings flags PS3015 — a field handed a fresh allocation and never read.
func writeOnlyAllocFieldFindings(fset *token.FileSet, f *ast.File) []finding {
	var out []finding
	seen := map[string]bool{}
	report := func(name string, pos token.Pos, what string) {
		if !deadFieldDeclared[name] || deadFieldRead[name] || seen[name] {
			return
		}
		seen[name] = true
		out = append(out, finding{
			pos:      fset.Position(pos),
			category: "write-only-alloc-field",
			msg: fmt.Sprintf("field %q is given a fresh %s here and is READ NOWHERE in this package,"+
				" so every construction pays an allocation for a buffer nothing uses. THE COMPILER"+
				" WILL NOT SAY SO: an unused local is an error, but a struct field assigned in a"+
				" constructor is a use as far as it is concerned. MEASURED on the autograd WKV"+
				" backward, where a linear-time rewrite stopped needing the loga and p exponent"+
				" buffers the quadratic path required; they stayed in the scratch struct and kept"+
				" being allocated per worker per call. Removing them, together with pooling the"+
				" scratch, took that kernel from 134 to 46 allocs/op and 434.7 to 278.2 KiB, and the"+
				" reduced allocator work showed up as a further 15.82%% geomean on TIME. A KERNEL"+
				" REWRITE IS THE USUAL CAUSE, because the scratch a kernel inherits describes the"+
				" algorithm it replaced (A-REWRITE-LEAVES-DEAD-SCRATCH-BEHIND-001). Matching is by"+
				" field NAME and without types, so a same-named field read anywhere in the package"+
				" suppresses this; the remaining blind spot is a read through reflection, which no"+
				" AST pass can see", name, what),
		})
	}
	ast.Inspect(f, func(n ast.Node) bool {
		switch s := n.(type) {
		case *ast.KeyValueExpr:
			id, ok := s.Key.(*ast.Ident)
			if ok && allocatingExpr(s.Value) {
				report(id.Name, id.Pos(), allocKind(s.Value))
			}
		case *ast.AssignStmt:
			for i, l := range s.Lhs {
				if i >= len(s.Rhs) || !allocatingExpr(s.Rhs[i]) {
					continue
				}
				switch t := l.(type) {
				case *ast.SelectorExpr:
					report(t.Sel.Name, t.Sel.Pos(), allocKind(s.Rhs[i]))
				case *ast.IndexExpr:
					// x.f[i] = make(...) fills a field ELEMENT by element, which is how an
					// array-of-slices scratch is initialized. The allocation is just as wasted
					// when nothing reads f, and reporting only whole-field assignment missed
					// exactly that shape.
					if sel, ok := t.X.(*ast.SelectorExpr); ok {
						report(sel.Sel.Name, sel.Sel.Pos(), allocKind(s.Rhs[i]))
					}
				}
			}
		}
		return true
	})
	return out
}

// allocatingExpr reports whether an expression produces a fresh heap object worth paying for.
func allocatingExpr(e ast.Expr) bool { return allocKind(e) != "" }

func allocKind(e ast.Expr) string {
	switch x := e.(type) {
	case *ast.CallExpr:
		if id, ok := x.Fun.(*ast.Ident); ok {
			switch id.Name {
			case "make":
				return "make"
			case "new":
				return "new"
			}
		}
	case *ast.CompositeLit:
		return "composite literal"
	case *ast.UnaryExpr:
		if x.Op == token.AND {
			if _, ok := x.X.(*ast.CompositeLit); ok {
				return "composite literal"
			}
		}
	}
	return ""
}

// twoDeepIndexNotRangedFindings flags PS3016 — an inner loop reading m[i][k] where the OUTER index
// is invariant, so the row could be ranged over instead of indexed.
//
// Reported only when the loop does NOT already range over a slice: ranging is the fix, so a loop
// that already does it is the applied form.
func twoDeepIndexNotRangedFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	seen := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		body := loopBody(n)
		if body == nil {
			return true
		}
		k := loopVarOf(n)
		if k == "" || rangesOverSlice(n) {
			return true
		}
		ast.Inspect(body, func(m ast.Node) bool {
			ix, ok := m.(*ast.IndexExpr)
			if !ok {
				return true
			}
			inner, ok := ix.X.(*ast.IndexExpr)
			if !ok {
				return true
			}
			// COLUMN WALK: m[k][i], the outer subscript moving and the inner invariant. There is
			// no row to walk ALONG, which is why this was declined for two rounds — but the SLICE
			// OF ROWS is itself rangeable, and pairing a companion to it removes every check but
			// the one on the row that comes out. Measured twice: linalg Cholesky back
			// substitution went 3 checks to 1 for -2.26%% and -2.13%% (p<=0.017, n=12), and
			// autograd's logdet forward solve went 4 to 1 for -2.23%% and -1.33%% (p<=0.015).
			if identName(inner.Index) == k && identName(ix.Index) != k {
				col := identName(ix.Index)
				base := identName(inner.X)
				// Keyed by the LOOP as well as the base: the same matrix is usually walked by
				// both an outer and an inner loop, and a per-function key let the outer
				// instance mask the inner one — which is the hotter of the two, being O(n)
				// deeper. autograd's logdet solve reported only its outer write until this.
				if col == "" || base == "" || assignedWithin(body, col) || seen[base+"#col#"+k] {
					return true
				}
				seen[base+"#col#"+k] = true
				out = append(out, finding{
					pos:      fset.Position(ix.Pos()),
					category: "two-deep-index-not-ranged",
					msg: fmt.Sprintf("%s[%s][%s] walks DOWN a column: the row subscript moves with"+
						" this loop and %q is fixed. There is no row to range along, but the SLICE"+
						" OF ROWS is rangeable — range %s[lo:hi] and pair any companion to the same"+
						" length, which leaves only the [%s] on the row that comes out. Measured"+
						" twice: linalg Cholesky back substitution 3 bounds checks to 1 for -2.26%%"+
						" and -2.13%% (p<=0.017, n=12), autograd logdet forward solve 4 to 1 for"+
						" -2.23%% and -1.33%% (p<=0.015). Bit-identical — the operands and their"+
						" ascending order do not change. VERIFY FIRST with"+
						" -gcflags=-d=ssa/check_bce/debug=1 and count the checks INSIDE the loop"+
						" against the multiply-adds beside them: the same conversion on a 4x4"+
						" register tile measured null, because sixteen accumulators amortize a"+
						" predicted branch to nothing — and so did autograd's conv1d backward, at"+
						" the TOP of that ratio ranking, because it streams about 200MB per call"+
						" and a bandwidth-bound loop hides a removed branch just as thoroughly"+
						" (RANK-BCE-CANDIDATES-BY-CHECKS-OVER-FMA-001,"+
						" MEMORY-BOUND-HIDES-CHECK-REMOVAL-TOO-001)",
						base, k, col, col, base, col),
				})
				return true
			}
			if identName(ix.Index) != k {
				return true
			}
			// The outer subscript must be INVARIANT for the whole loop, not merely different
			// from this loop's variable. Checking only the latter let the OUTER loop of a nested
			// pair match l[k][i]: from its point of view i is the moving subscript and k the row,
			// when k is the inner loop's own variable and there is no fixed row at all.
			row := identName(inner.Index)
			if row == "" || row == k || assignedWithin(body, row) {
				return true
			}
			_ = row
			base := identName(inner.X)
			if base == "" || seen[base] {
				return true
			}
			seen[base] = true
			out = append(out, finding{
				pos:      fset.Position(ix.Pos()),
				category: "two-deep-index-not-ranged",
				msg: fmt.Sprintf("%s[%s][%s] indexes two deep with %q invariant in this loop, so every"+
					" step re-loads the row pointer AND bounds-checks against it. Hoist the row and"+
					" RANGE over it. THE RANGE IS THE PART THAT PAYS, which is worth stating because"+
					" the obvious half does not: on linalg Cholesky's forward substitution, hoisting"+
					" li := l[i] while keeping the integer-bounded loop measured geomean -0.53%%"+
					" over eleven benchmarks with one cell at +0.41%% (p=0.038), and re-running the"+
					" two largest shapes at n=12 still failed to reach significance at p=0.060."+
					" Converting the same site to `for k, lik := range li[:i]` then gave -2.82%%,"+
					" -3.50%% and -0.79%% (p<=0.043) on three of five cells, geomean -2.08%%. The"+
					" mechanism is bounds-check elimination, not the pointer reload, and a loop"+
					" bounded by an int keeps the check however the pointer is held"+
					" (HOISTING-A-ROW-PAYS-ONLY-VIA-THE-RANGE-001). Bit-identical either way: the"+
					" operands and their order do not change. PRECONDITION THIS CHECK CANNOT SEE:"+
					" the range bound must equal the loop's own bound, so a loop running to something"+
					" other than the row length needs a slice expression rather than a bare range,"+
					" and one whose OUTER index moves instead — m[k][i] — has no row to range at all"+
					" and is a different problem", base, identName(inner.Index), k,
					identName(inner.Index)),
			})
			return true
		})
		return true
	})
	return out
}

// assignedWithin reports whether a name is defined or assigned anywhere inside a block, which
// includes serving as a nested loop's variable. Such a name is not invariant for the outer loop.
func assignedWithin(body *ast.BlockStmt, name string) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}
		switch s := n.(type) {
		case *ast.AssignStmt:
			for _, l := range s.Lhs {
				if identName(l) == name {
					found = true
				}
			}
		case *ast.RangeStmt:
			if identName(s.Key) == name || identName(s.Value) == name {
				found = true
			}
		case *ast.IncDecStmt:
			if identName(s.X) == name {
				found = true
			}
		}
		return !found
	})
	return found
}

// rangesOverSlice reports whether a loop already ranges over A ROW, which is the applied form.
//
// A BARE IDENTIFIER does not count, and that is the whole subtlety: `for k := range i` over an int
// is syntactically indistinguishable from `for k := range xs` over a slice, and treating both as
// applied made this check miss the very site it was built from — Cholesky's forward substitution
// loops `for k := range i`. Only a row EXPRESSION — a slice of one, an element of a slice-of-slices,
// or a field — is evidence the fix is already in. Ranging over some other slice is still a
// candidate, since it is the ROW whose bounds check the range has to eliminate.
func rangesOverSlice(n ast.Node) bool {
	rs, ok := n.(*ast.RangeStmt)
	if !ok {
		return false
	}
	switch rs.X.(type) {
	case *ast.SliceExpr, *ast.IndexExpr, *ast.SelectorExpr:
		return true
	}
	return false
}

// companionNotSlicedFindings flags PS3017 — a loop that DOES range over a row but still indexes a
// second slice with the range key, so only half the bounds checks were removed.
//
// This is the shape PS3016 suppresses as "applied", and it is applied only halfway: ranging proves
// the ROW's index in range and says nothing about the companion's.
func companionNotSlicedFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		rs, ok := n.(*ast.RangeStmt)
		if !ok {
			return true
		}
		key := identName(rs.Key)
		if key == "" || key == "_" {
			return true
		}
		// Names cut from a slice expression anywhere in this function are treated as already
		// paired with the row. Conservative: a same-named local sliced elsewhere suppresses.
		sliced := slicedNames(fn.Body)
		// THE RANGED THING MUST BE A DELIBERATELY CUT ROW, not any collection. Ranging a field or
		// a plain slice and indexing a parallel one by the key is ordinary Go — `for i, t := range
		// obj.Items { out[i] = ... }` — and matching it reported 199 sites, essentially none of
		// them a numeric kernel. Requiring an explicit cut, or a name that was cut, keeps this to
		// loops someone converted on purpose.
		rows := rowNames(fn.Body)
		if _, isCut := rs.X.(*ast.SliceExpr); !isCut && !sliced[identName(rs.X)] && !rows[identName(rs.X)] {
			return true
		}
		// And it must be a REDUCTION: a compound assignment in the body. Without that there is no
		// hot inner accumulation for the second bounds check to matter to.
		if !hasCompoundAssign(rs.Body) {
			return true
		}
		// The companion and the range VALUE must meet in one accumulation — dot += qk[j]*v — which
		// is the two-operand reduction this transform is about. Requiring only a compound assign
		// somewhere in the body matched 260 sites once rows were recognized, because `x := y[i]`
		// is any slice element and most such loops are not numeric kernels at all.
		val := identName(rs.Value)
		if val == "" || val == "_" {
			return true
		}
		seen := map[string]bool{}
		ast.Inspect(rs.Body, func(m ast.Node) bool {
			ix, ok := m.(*ast.IndexExpr)
			if !ok || identName(ix.Index) != key {
				return true
			}
			base := identName(ix.X)
			if base == "" || sliced[base] || seen[base] || !pairedInReduction(rs.Body, base, val) {
				return true
			}
			seen[base] = true
			out = append(out, finding{
				pos:      fset.Position(ix.Pos()),
				category: "companion-not-sliced",
				msg: fmt.Sprintf("%s[%s] is indexed by the range key of a loop that already ranges a"+
					" row, so the ROW's bounds check is gone and this one is NOT. Ranging proves only"+
					" its own index in range; the compiler cannot relate %q's length to the ranged"+
					" slice. Cut both to the same length — and for a trailing segment write the"+
					" relation down, as in yr = yr[:len(lr)], or it still cannot see it. MEASURED as"+
					" the difference between a half-applied and a finished conversion: linalg"+
					" Cholesky was converted with only the row ranged for geomean -2.08%%, and adding"+
					" the companion slice afterwards gave a FURTHER -1.59%% with three of four cells"+
					" at p=0.000. linalg LU, done with both from the start, went -6.61%% geomean over"+
					" nine cells. Bit-identical: operands and order do not change, only how the"+
					" indices are proven (SLICE-BOTH-OPERANDS-NOT-JUST-THE-ROW-001). PRECONDITION"+
					" THIS CHECK CANNOT SEE: the two must genuinely have the same length over the"+
					" loop's extent — a companion indexed with an OFFSET, or shorter than the row,"+
					" needs its own slice expression rather than a bare cut."+
					" VERIFY WITH THE COMPILER, not by inference: go build"+
					" -gcflags=-d=ssa/check_bce/debug=1 lists every remaining check with a line, and"+
					" what matters is the count INSIDE the inner loop, not in the file. Cholesky's"+
					" accumulate went 3 inner checks to 1 to 0 across the two steps of this"+
					" conversion, matching -2.08%% then a further -1.59%%; a whole-file count showed"+
					" 35 to 34 to 34 and would have said the second step did nothing. AND THE WIN"+
					" SCALES WITH CHECKS PER FMA: the same conversion on backend/cpu's 4x4 register"+
					" tile took its inner loop from 2 checks to 1 and measured NULL across twelve"+
					" cells, because sixteen accumulators amortize one predicted branch to nothing."+
					" A scalar dot product does not", base, key, base),
			})
			return true
		})
		return true
	})
	return out
}

// pairedInReduction reports whether some += or -= in the block has both the companion and the
// range value on its right-hand side, which is what a two-operand inner product looks like.
func pairedInReduction(b *ast.BlockStmt, base, val string) bool {
	found := false
	ast.Inspect(b, func(n ast.Node) bool {
		if found {
			return false
		}
		as, ok := n.(*ast.AssignStmt)
		if !ok || (as.Tok != token.ADD_ASSIGN && as.Tok != token.SUB_ASSIGN) || len(as.Rhs) != 1 {
			return true
		}
		// Either the companion is an OPERAND — dot += qk[j]*v — or it is the accumulation
		// TARGET — qi[j] -= dot*v. Both are the same transform; matching only the operand form
		// missed the second Gram-Schmidt loop in randomOrthogonal, which is the mirror of the
		// first and was fixed alongside it.
		if mentionsIdent(as.Rhs[0], val) &&
			(mentionsIdent(as.Rhs[0], base) || mentionsIdent(as.Lhs[0], base)) {
			found = true
		}
		return !found
	})
	return found
}

// hasCompoundAssign reports whether a block accumulates with += or -=.
func hasCompoundAssign(b *ast.BlockStmt) bool {
	found := false
	ast.Inspect(b, func(n ast.Node) bool {
		if as, ok := n.(*ast.AssignStmt); ok &&
			(as.Tok == token.ADD_ASSIGN || as.Tok == token.SUB_ASSIGN) {
			found = true
		}
		return !found
	})
	return found
}

// slicedNames collects identifiers assigned from a slice expression — the finished pairing.
func slicedNames(body *ast.BlockStmt) map[string]bool {
	return assignedFrom(body, func(e ast.Expr) bool {
		_, ok := e.(*ast.SliceExpr)
		return ok
	})
}

// rowNames collects identifiers assigned a ROW of a matrix, qi := q[i].
//
// These count as ranged rows for PS3017 even though no slice expression is involved, and leaving
// them out was a real false negative: nlp randomOrthogonal's Gram-Schmidt takes qi := q[i] and then
// ranges it while indexing a second row, which is the hottest instance of this shape in the tree
// and was invisible until rows were recognized alongside cuts.
func rowNames(body *ast.BlockStmt) map[string]bool {
	return assignedFrom(body, func(e ast.Expr) bool {
		_, ok := e.(*ast.IndexExpr)
		return ok
	})
}

func assignedFrom(body *ast.BlockStmt, want func(ast.Expr) bool) map[string]bool {
	out := map[string]bool{}
	ast.Inspect(body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok {
			return true
		}
		for i, r := range as.Rhs {
			if !want(r) || i >= len(as.Lhs) {
				continue
			}
			if nm := identName(as.Lhs[i]); nm != "" {
				out[nm] = true
			}
		}
		return true
	})
	return out
}

// maxNormalizedExpFindings flags PS3018 — math.Exp(x - m) where m was assigned from a max that
// INCLUDES x, so the call is exp(0) whenever the max picked x, and exp(0) is exactly 1.
// maxBinding is one assignment of a math.Max to a name, with the position that makes it possible
// to tell which binding a later call actually sees.
type maxBinding struct {
	pos  token.Pos
	args []ast.Expr
}

// bindingAt returns the arguments of the last max bound to the name before pos, or nil if the name
// was not bound to a max by then. A call ahead of every binding sees none of them.
func bindingAt(bs []maxBinding, pos token.Pos) []ast.Expr {
	var args []ast.Expr
	for _, b := range bs {
		if b.pos < pos {
			args = b.args
		}
	}
	return args
}

func maxNormalizedExpFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	seen := map[token.Pos]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		blk, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		// maxOf[name] = every max assigned to that name, in source order. A name is routinely
		// REBOUND — a scan writes `q := max(a, b)` for one stabilized pair and then `q = max(c, d)`
		// for the next, in the same block — so a map of name to the LAST arguments answers the
		// wrong question for every call before the rebinding. Keeping the list and choosing the
		// binding in effect AT THE CALL is what makes both pairs visible; the single-binding form
		// reported the second pair in the RWKV decode scan and silently missed the first.
		maxOf := map[string][]maxBinding{}
		for _, st := range blk.List {
			as, ok := st.(*ast.AssignStmt)
			if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
				continue
			}
			call, ok := as.Rhs[0].(*ast.CallExpr)
			if !ok || !isMathCall(call, "Max") || len(call.Args) != 2 {
				continue
			}
			if nm := identName(as.Lhs[0]); nm != "" {
				maxOf[nm] = append(maxOf[nm], maxBinding{pos: as.Pos(), args: call.Args})
			}
		}
		if len(maxOf) == 0 {
			return true
		}
		// Calls already guarded by a test of the max against an argument are the APPLIED form —
		// that guard is exactly the fix — and reporting them files the fix as the defect.
		guarded := map[token.Pos]bool{}
		ast.Inspect(blk, func(g ast.Node) bool {
			ifs, ok := g.(*ast.IfStmt)
			if !ok {
				return true
			}
			cmp, ok := ifs.Cond.(*ast.BinaryExpr)
			if !ok || (cmp.Op != token.NEQ && cmp.Op != token.EQL &&
				cmp.Op != token.GEQ && cmp.Op != token.LEQ &&
				cmp.Op != token.GTR && cmp.Op != token.LSS) {
				return true
			}
			if _, isMax := maxOf[identName(cmp.X)]; !isMax {
				if _, isMax := maxOf[identName(cmp.Y)]; !isMax {
					return true
				}
			}
			ast.Inspect(ifs, func(h ast.Node) bool {
				if c, ok := h.(*ast.CallExpr); ok && isMathCall(c, "Exp") {
					guarded[c.Pos()] = true
				}
				return true
			})
			return true
		})
		ast.Inspect(blk, func(m ast.Node) bool {
			call, ok := m.(*ast.CallExpr)
			if !ok || !isMathCall(call, "Exp") || len(call.Args) != 1 || seen[call.Pos()] ||
				guarded[call.Pos()] {
				return true
			}
			sub, ok := unparen(call.Args[0]).(*ast.BinaryExpr)
			if !ok || sub.Op != token.SUB {
				return true
			}
			args := bindingAt(maxOf[identName(sub.Y)], call.Pos())
			if args == nil {
				return true
			}
			lhs := renderExpr(unparen(sub.X))
			for _, a := range args {
				if renderExpr(unparen(a)) != lhs {
					continue
				}
				seen[call.Pos()] = true
				out = append(out, finding{
					pos:      fset.Position(call.Pos()),
					category: "max-normalized-exp",
					msg: fmt.Sprintf("math.Exp(%s - %s) where %s is the max OF %s, so this call is"+
						" exp(0) whenever the max picked it — and exp(0) is exactly 1. Replace the"+
						" call with the literal when the argument equals the max. MEASURED TWICE: on"+
						" the RWKV WKV forward scan, where both stabilized pairs had this shape and"+
						" half of four calls per element were computing a constant, -12.42%% and"+
						" -11.93%% (p=0.000, n=8) on a kernel that was 75.6%% math.archExp; and on"+
						" the WKV backward, -13.59%% and -14.87%% (p=0.000, n=12), bit-identical"+
						" over 6540 gradients across five decay regimes including negative and zero"+
						" decay. CHECK FOR REPEATS AT THE SAME TIME: this check reports each CALL"+
						" SITE, and in the backward each of the two exponents appeared TWICE."+
						" math.Exp is not inlined, so a repeat is a second evaluation and not a"+
						" subexpression the compiler folds — binding it to a local was half of that"+
						" win."+
						" TEST THE MAX AGAINST THE ARGUMENT, do not branch on the original"+
						" comparison: with a NaN operand math.Max yields NaN, the equality fails and"+
						" both exponentials still evaluate exactly as before, whereas an if/else on"+
						" `a >= b` substitutes a 1 the original never produced. THE WIN SCALES WITH"+
						" 1/N: over a max of N terms this saves one call in N and is not worth the"+
						" branch; it paid here because N is two. Note also the saving came to a third"+
						" of what the exp share predicted, so rank by term count rather than by"+
						" profile share (THE-MAX-NORMALIZED-EXP-IS-EXACTLY-ONE-001)",
						lhs, identName(sub.Y), identName(sub.Y), renderExpr(args[0])+" and "+renderExpr(args[1])),
				})
				break
			}
			return true
		})
		return true
	})
	return out
}

// isMathCall reports whether a call is math.<name>.
func isMathCall(c *ast.CallExpr, name string) bool {
	sel, ok := c.Fun.(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != name {
		return false
	}
	pkg, ok := sel.X.(*ast.Ident)
	return ok && pkg.Name == "math"
}

// unrolledIndexNotWindowedFindings flags PS3019 — a manually unrolled loop bounded by `i+K <= len(x)`
// whose body indexes x with K distinct constant offsets and never cuts x to the K-wide window.
//
// `i+K <= len(x)` does NOT let the prove pass discharge x[i+K-1]: i+K can overflow, so the bound
// says nothing about i on its own and every one of the K element reads keeps a check. Cutting a
// K-wide window once per iteration replaces those K checks with ONE slice check, and the window's
// length is a constant the compiler folds, so the K reads off it are free.
func unrolledIndexNotWindowedFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		fl, ok := n.(*ast.ForStmt)
		if !ok || fl.Cond == nil || fl.Body == nil {
			return true
		}
		lv, k, ok := unrollBound(fl.Cond)
		if !ok {
			return true
		}
		// Offsets read off each base, and which bases are already cut to a window.
		offsets := map[string]map[int]bool{}
		// Report each base at its OWN first read rather than at the loop header: two bases in one
		// loop would otherwise carry the same position and be deduplicated into a single finding.
		firstRead := map[string]token.Pos{}
		ast.Inspect(fl.Body, func(m ast.Node) bool {
			ix, ok := m.(*ast.IndexExpr)
			if !ok {
				return true
			}
			base, ok := unparen(ix.X).(*ast.Ident)
			if !ok {
				return true
			}
			off, ok := constOffsetOf(unparen(ix.Index), lv)
			if !ok {
				return true
			}
			if offsets[base.Name] == nil {
				offsets[base.Name] = map[int]bool{}
			}
			offsets[base.Name][off] = true
			if _, seen := firstRead[base.Name]; !seen {
				firstRead[base.Name] = ix.Pos()
			}
			return true
		})
		for _, name := range sortedOffsetBases(offsets) {
			if len(offsets[name]) < 2 {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(firstRead[name]),
				category: "unrolled-index-not-windowed",
				msg: fmt.Sprintf("loop is unrolled by %d over %q but reads %s at %d separate"+
					" constant offsets and never cuts it to the window. `%s+%d <= len(...)` does NOT"+
					" discharge those reads: %s+%d can overflow, so the prove pass keeps a bounds"+
					" check on EVERY one of them. Cut a window once per iteration"+
					" (`w := %s[%s : %s+%d]`) and index the window — ONE slice check for the whole"+
					" step, and the window length is a constant the compiler folds. MEASURED on nlp"+
					" dotAndNorm, eight reads to two checks: -16.15%% and -18.55%% (p<=0.001, n=12),"+
					" geomean -17.36%%. BIT-IDENTICAL — operands and their order are untouched."+
					" BUT IT DID NOT PAY AT THE SITE IT WAS FOUND ON, and that is the important"+
					" half: the classic ballTree L2 leaf test went four checks to one and measured"+
					" -1.11%% against an UNTOUCHED L1 control that moved -1.06%% in the same run, so"+
					" nothing was attributable. That loop carries a data-dependent early exit whose"+
					" misprediction dominates, and the checks hide behind it. REQUIRE A BRANCHLESS"+
					" LOOP BODY with no loop-carried dependency before spending the edit, and run an"+
					" untouched sibling as a control — this class produces a plausible small win that"+
					" is really run-to-run drift. REORDERING IS NOT A SUBSTITUTE: reading the highest"+
					" offset first to establish the rest left all four checks in place. A second"+
					" operand clamped to the first outside the loop (b = b[:len(a)]) needs no check of"+
					" its own, so its window is free; without the clamp each operand costs one slice"+
					" check (PERF-BCE-PAYOFF-NEEDS-BRANCHLESS-001)",
					k, lv, name, len(offsets[name]), lv, k, lv, k, name, lv, lv, k),
			})
		}
		return true
	})
	return out
}

// unrollBound reports the loop variable and the unroll factor of a condition shaped `i+K <= len(x)`
// or `i+K < len(x)`. K must be a constant of at least 2: an unroll of one is an ordinary loop.
func unrollBound(cond ast.Expr) (string, int, bool) {
	cmp, ok := unparen(cond).(*ast.BinaryExpr)
	if !ok || (cmp.Op != token.LEQ && cmp.Op != token.LSS) {
		return "", 0, false
	}
	call, ok := unparen(cmp.Y).(*ast.CallExpr)
	if !ok || len(call.Args) != 1 {
		return "", 0, false
	}
	if fnID, ok := unparen(call.Fun).(*ast.Ident); !ok || fnID.Name != "len" || !isBuiltinName(fnID.Name) {
		return "", 0, false
	}
	add, ok := unparen(cmp.X).(*ast.BinaryExpr)
	if !ok || add.Op != token.ADD {
		return "", 0, false
	}
	id, ok := unparen(add.X).(*ast.Ident)
	if !ok {
		return "", 0, false
	}
	k, ok := intLit(unparen(add.Y))
	if !ok || k < 2 {
		return "", 0, false
	}
	return id.Name, k, true
}

// constOffsetOf reports the constant offset of an index expression off the loop variable: `i` is 0,
// `i+3` is 3. Anything else — a different variable, a product, a call — is not a fixed lane of an
// unrolled step and cannot be covered by one window.
func constOffsetOf(e ast.Expr, lv string) (int, bool) {
	if id, ok := e.(*ast.Ident); ok {
		return 0, id.Name == lv
	}
	add, ok := e.(*ast.BinaryExpr)
	if !ok || add.Op != token.ADD {
		return 0, false
	}
	id, ok := unparen(add.X).(*ast.Ident)
	if !ok || id.Name != lv {
		return 0, false
	}
	k, ok := intLit(unparen(add.Y))
	return k, ok
}

func sortedOffsetBases(m map[string]map[int]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// invariantBehindBoundsCheckFindings flags PS3020 — a counted loop that indexes slices with its
// loop variable AND recomputes a loop-invariant value on every iteration.
//
// The two halves are one defect. Each indexed read carries a bounds check, and a bounds check is a
// panic edge that splits the body into separate basic blocks; Go's SSA will not hoist across a
// block that can panic, so the invariant stays inside the loop even though nothing in it moves.
// Discharging the checks is what makes the hoist possible, which is why the remedy is both edits.
func invariantBehindBoundsCheckFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		fl, ok := n.(*ast.ForStmt)
		if !ok || fl.Body == nil || fl.Cond == nil {
			return true
		}
		lv := counterName(fl)
		if lv == "" {
			return true
		}
		assigned := assignedNames(fl.Body)
		assigned[lv] = true
		var reported bool
		ast.Inspect(fl.Body, func(m ast.Node) bool {
			be, ok := m.(*ast.BinaryExpr)
			if !ok || reported {
				return true
			}
			switch be.Op {
			case token.ADD, token.SUB, token.MUL, token.QUO:
			default:
				return true
			}
			// The invariant must be an OPERAND against an element indexed by the loop variable.
			// That is what makes it a VALUE the loop recomputes rather than addressing arithmetic,
			// which folds into an addressing mode and costs nothing.
			if !pairedWithIndexedRead(fl.Body, be, lv) || !allLoopInvariant(be, assigned) {
				return true
			}
			reported = true
			out = append(out, finding{
				pos:      fset.Position(be.Pos()),
				category: "invariant-behind-bounds-check",
				msg: fmt.Sprintf("%s does not change across this loop, yet it is evaluated on"+
					" every iteration. The compiler would normally hoist it — it does not here"+
					" because the loop indexes slices with %q, each read is a bounds check, and a"+
					" bounds check is a PANIC EDGE that splits the body into separate basic"+
					" blocks; Go's SSA will not hoist across a block that can panic. So the two"+
					" defects are one: the checks cost more than their own instructions because"+
					" they also trap the invariant. FIX BOTH — range over the destination slice"+
					" and cut every companion to its length, then lift the invariant to a local"+
					" above the loop. MEASURED on the rl Polyak soft update, where (1-tau) was"+
					" rematerialized per element (FMOVD of 1.0 then FSUBD, both visible in"+
					" -gcflags=-S) behind two bounds checks: the body went from 14 instructions to"+
					" 5 for -21.30%% on the F64 arm and -18.62%% on the F32 arm (p=0.000, n=12),"+
					" with an untouched sibling benchmark flat as the control. VERIFY THE FUSION"+
					" DID NOT MOVE: merging basic blocks can change which multiply the backend"+
					" contracts into an FMA, and this repo has already recorded a 1-ulp failure"+
					" from exactly that, so re-read the -S dump and confirm the same FMADDD in the"+
					" same position before claiming bit-identity",
					renderExpr(be), lv),
			})
			return true
		})
		return true
	})
	return out
}

// counterName reports the variable a counted loop advances, or "" when the post statement is not a
// simple increment of one identifier.
func counterName(fl *ast.ForStmt) string {
	switch post := fl.Post.(type) {
	case *ast.IncDecStmt:
		if id, ok := unparen(post.X).(*ast.Ident); ok {
			return id.Name
		}
	case *ast.AssignStmt:
		if len(post.Lhs) == 1 {
			if id, ok := unparen(post.Lhs[0]).(*ast.Ident); ok {
				return id.Name
			}
		}
	}
	return ""
}

// assignedNames collects every identifier written in a block, including the base of an indexed
// assignment, so that anything the loop mutates is excluded from being called invariant.
func assignedNames(b *ast.BlockStmt) map[string]bool {
	out := map[string]bool{}
	ast.Inspect(b, func(n ast.Node) bool {
		switch a := n.(type) {
		case *ast.AssignStmt:
			for _, l := range a.Lhs {
				switch t := l.(type) {
				case *ast.Ident:
					out[t.Name] = true
				case *ast.IndexExpr:
					if id, ok := unparen(t.X).(*ast.Ident); ok {
						out[id.Name] = true
					}
				}
			}
		case *ast.IncDecStmt:
			if id, ok := unparen(a.X).(*ast.Ident); ok {
				out[id.Name] = true
			}
		}
		return true
	})
	return out
}

// allLoopInvariant reports whether every leaf of e is a literal or an identifier the loop never
// writes. Calls, selectors and index expressions are rejected: without a type checker there is no
// way to know they are pure, and hoisting an impure expression changes behavior.
func allLoopInvariant(e ast.Expr, assigned map[string]bool) bool {
	idents, ok := 0, true
	ast.Inspect(e, func(n ast.Node) bool {
		if n == nil { // Inspect signals end-of-children with nil; it is not a leaf
			return false
		}
		switch t := n.(type) {
		case *ast.Ident:
			idents++
			if assigned[t.Name] {
				ok = false
			}
		case *ast.BasicLit, *ast.BinaryExpr, *ast.ParenExpr:
		default:
			ok = false
		}
		return ok
	})
	return ok && idents > 0
}

// pairedWithIndexedRead reports whether e is one side of a binary expression whose other side reads
// a slice indexed by the loop variable.
func pairedWithIndexedRead(body *ast.BlockStmt, e ast.Expr, lv string) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		be, ok := n.(*ast.BinaryExpr)
		if !ok {
			return true
		}
		var other ast.Expr
		switch {
		case unparen(be.X) == e:
			other = be.Y
		case unparen(be.Y) == e:
			other = be.X
		default:
			return true
		}
		ast.Inspect(other, func(q ast.Node) bool {
			if ix, ok := q.(*ast.IndexExpr); ok {
				if id, ok := unparen(ix.Index).(*ast.Ident); ok && id.Name == lv {
					found = true
				}
			}
			return true
		})
		return true
	})
	return found
}

// monotoneGuardInLoopFindings flags PS3021 — a counted loop whose body is wrapped in a single guard
// that moves MONOTONELY with the loop variable and is compared against something invariant.
//
// Such a guard does not vary independently: it is false for a run of iterations at one end and true
// for the rest, so it describes the loop's BOUNDS rather than a per-iteration decision. Computing
// the crossing point once and starting (or stopping) the loop there removes a branch from every
// iteration — and, because a branch splits the basic block and Go SSA will not hoist across a block
// that can panic or diverge, it also frees whatever invariant the guard was trapping.
func monotoneGuardInLoopFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		fl, ok := n.(*ast.ForStmt)
		if !ok || fl.Body == nil || len(fl.Body.List) == 0 {
			return true
		}
		lv := counterName(fl)
		if lv == "" {
			return true
		}
		last, ok := fl.Body.List[len(fl.Body.List)-1].(*ast.IfStmt)
		if !ok || last.Else != nil || last.Init != nil || last.Body == nil || len(last.Body.List) == 0 {
			return true
		}
		cmp, ok := unparen(last.Cond).(*ast.BinaryExpr)
		if !ok {
			return true
		}
		switch cmp.Op {
		case token.GEQ, token.GTR, token.LSS, token.LEQ:
		default:
			return true // equality is not monotone: it selects one iteration, not a run
		}
		// Names bound earlier in the body to an expression that moves with the loop variable.
		moving := map[string]bool{lv: true}
		for _, st := range fl.Body.List[:len(fl.Body.List)-1] {
			as, ok := st.(*ast.AssignStmt)
			if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
				continue
			}
			if id, ok := unparen(as.Lhs[0]).(*ast.Ident); ok && movesWithLoop(as.Rhs[0], moving) {
				moving[id.Name] = true
			}
		}
		xm, ym := movesWithLoop(cmp.X, moving), movesWithLoop(cmp.Y, moving)
		if xm == ym {
			return true // neither side moves, or both do — no single crossing point
		}
		invariant := cmp.Y
		if ym {
			invariant = cmp.X
		}
		if mentionsAnyName(invariant, moving) {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(last.Pos()),
			category: "monotone-guard-in-loop",
			msg: fmt.Sprintf("the whole body of this loop sits behind %q, which moves monotonely"+
				" with %q and is compared against something the loop does not change. A guard like"+
				" that is not a per-iteration decision — it is false for a RUN of iterations at one"+
				" end and true for the rest, so it belongs in the loop BOUNDS. Compute the crossing"+
				" point once outside and start (or stop) the loop there. TWO costs go, not one: the"+
				" branch per iteration, and whatever loop-invariant the branch was trapping, since"+
				" it splits the body into its own basic block and Go SSA will not hoist across it."+
				" MEASURED on the autograd conv1d backward, whose per-tap `j >= 0` was false for"+
				" only the first K-1 of L positions — 3 of 2048, so 99.85%% pure overhead — and"+
				" whose loop HEADER profiled as a larger share than either line the guard"+
				" protected: F32 -7.92%% (p=0.000, n=16). REPORTED HONESTLY: the F64 arm of the"+
				" same change was directionally -4.7%% but did NOT reach significance at n=16"+
				" (p=0.210), so expect this to pay when the skipped run is a small fraction and the"+
				" body is cheap, and to be unmeasurable otherwise. It is still worth applying when"+
				" bit-identical, because it strictly removes instructions and cannot regress."+
				" BIT-IDENTICAL by construction: only iterations whose body never executed are"+
				" skipped, so the surviving ones are the same values in the same order. CHECK THE"+
				" DIRECTION before rewriting — a guard that is true for a PREFIX bounds the loop"+
				" above, one true for a suffix bounds it below, and getting it backwards silently"+
				" drops real work", renderExpr(cmp), lv),
		})
		return true
	})
	return out
}

// movesWithLoop reports whether e is an additive expression over identifiers and literals that
// mentions at least one name already known to advance with the loop. Restricted to + and - because
// those are the forms whose crossing point is a single index; a product or a modulus can re-enter
// the guarded region and has no single bound.
func movesWithLoop(e ast.Expr, moving map[string]bool) bool {
	found, ok := false, true
	ast.Inspect(e, func(n ast.Node) bool {
		if n == nil {
			return false
		}
		switch t := n.(type) {
		case *ast.Ident:
			if moving[t.Name] {
				found = true
			}
		case *ast.BasicLit, *ast.ParenExpr:
		case *ast.BinaryExpr:
			if t.Op != token.ADD && t.Op != token.SUB {
				ok = false
			}
		default:
			ok = false
		}
		return ok
	})
	return ok && found
}

func mentionsAnyName(e ast.Expr, names map[string]bool) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && names[id.Name] {
			found = true
		}
		return !found
	})
	return found
}

// transposePassFindings flags PS3023 — a nested loop that materializes a TRANSPOSED COPY of a matrix
// this function itself built.
//
// This is deliberately the case PS1010 EXCLUDES. That check reports a column walk only when the
// inner loop assigns to something free of the inner variable, because then interchange is the
// remedy; a transpose writes the inner variable on the left and, as its comment says, strides
// whichever way it is run, so interchange buys nothing. The remedy here is not to reorder the copy
// but to DELETE it: when the source is built in this same function, the producer can write the
// layout the consumer wants and both the pass and the intermediate disappear.
func transposePassFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	built := locallyBuiltMatrices(fn)
	if len(built) == 0 {
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
			for _, st := range ibody.List {
				as, ok := st.(*ast.AssignStmt)
				if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
					continue
				}
				// A transpose WRITE: the destination moves with the inner variable — usually as an
				// INDEX, which is why mentionsAsValue is the wrong test here. That is what PS1010
				// refuses, because interchanging it only moves the stride.
				if !mentions(as.Lhs[0], inner) {
					continue
				}
				src, ok := transposedRead(as.Rhs[0], inner, outer)
				if !ok || !built[src] {
					continue
				}
				line := fset.Position(as.Pos()).Line
				if seen[line] {
					continue
				}
				seen[line] = true
				out = append(out, finding{
					pos:      fset.Position(as.Pos()),
					category: "transpose-pass-over-built-matrix",
					msg: fmt.Sprintf("this loop materializes a TRANSPOSED COPY of %s, which this"+
						" function built itself. PS1010 deliberately does not report it: a"+
						" transpose writes the inner variable on the left, so INTERCHANGE buys"+
						" nothing and only moves the stride. The remedy is to DELETE the pass"+
						" instead — have the producer of %s write the layout this consumer wants,"+
						" and the copy, the intermediate and its per-row allocations all go."+
						" MEASURED on the autograd logdet VJP, which solved its triangular inverse"+
						" row-major and then transposed it because the contraction needs columns:"+
						" solving straight into column-major went -10.37%% at n=512 and -6.93%% at"+
						" n=256 (p=0.000, n=12), allocs/op down about a third, with an untouched"+
						" sibling benchmark flat. Bit-identical there, and it should be wherever"+
						" the producer merely relabels which slot a value lands in — no operand"+
						" and no summation order changes."+
						" TWO COSTS BESIDES THE COPY, both invisible to a line profile. The"+
						" consumer's inner loop stops walking down a column of a slice-of-slices,"+
						" so it loses a row-pointer load and a bounds check per element. And if"+
						" the producer is PARALLEL over the transposed axis, each worker was"+
						" writing a column — adjacent workers storing into adjacent bytes of the"+
						" same rows, contending for a cache line on every write — and now owns a"+
						" contiguous row. CHECK FIRST that the source is not ALSO consumed in its"+
						" original layout somewhere else; if it is, flipping the producer only"+
						" moves the transpose rather than removing it", src, src),
				})
			}
			return true
		})
		return true
	})
	return out
}

// transposedRead reports the base name of a read shaped src[inner][outer] appearing in e.
func transposedRead(e ast.Expr, inner, outer string) (string, bool) {
	name, found := "", false
	ast.Inspect(e, func(n ast.Node) bool {
		elem, ok := n.(*ast.IndexExpr)
		if !ok || found {
			return true
		}
		row, ok := unparen(elem.X).(*ast.IndexExpr)
		if !ok {
			return true
		}
		rid, ok1 := unparen(row.Index).(*ast.Ident)
		cid, ok2 := unparen(elem.Index).(*ast.Ident)
		base, ok3 := unparen(row.X).(*ast.Ident)
		if ok1 && ok2 && ok3 && rid.Name == inner && cid.Name == outer {
			name, found = base.Name, true
		}
		return true
	})
	return name, found
}

// locallyBuiltMatrices collects names assigned from make of a slice type in this function, which is
// what makes the producer reachable.
func locallyBuiltMatrices(fn *ast.FuncDecl) map[string]bool {
	out := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		id, ok := unparen(as.Lhs[0]).(*ast.Ident)
		if !ok {
			return true
		}
		call, ok := unparen(as.Rhs[0]).(*ast.CallExpr)
		if !ok || len(call.Args) == 0 {
			return true
		}
		fnID, ok := unparen(call.Fun).(*ast.Ident)
		if !ok || fnID.Name != "make" {
			return true
		}
		if _, isArr := call.Args[0].(*ast.ArrayType); isArr {
			out[id.Name] = true
		}
		return true
	})
	return out
}

// fixedArityVariadicCallFindings flags PS3024 — a call to a VARIADIC dispatch wrapper that passes a
// fixed number of arguments. Go builds a fresh slice for the variadic pack at every such site, so a
// wrapper existing only to forward its pack into a dispatch costs one allocation per dispatch, at
// every caller, forever.
func fixedArityVariadicCallFindings(fset *token.FileSet, fn *ast.FuncDecl, ns nameSets) []finding {
	if len(ns.variadicWrappers) == 0 || fn.Body == nil {
		return nil
	}
	wrappers := ns.variadicWrappers
	// A pooled helper's own fallback call is CORRECT: exec2 and friends hand their fixed
	// arguments to the variadic form precisely when a recorder is attached, because the tape may
	// retain the slice. Reporting those would flag the fix as the defect.
	if functionBorrowsFromPool(fn) {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || call.Ellipsis.IsValid() {
			return true // a genuine spread cannot avoid the pack
		}
		name := ""
		switch f := unparen(call.Fun).(type) {
		case *ast.Ident:
			name = f.Name
		case *ast.SelectorExpr:
			name = f.Sel.Name
		}
		if !wrappers[name] {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(call.Pos()),
			category: "fixed-arity-variadic-call",
			msg: fmt.Sprintf("%q is a variadic dispatch wrapper and this call passes a FIXED"+
				" number of arguments, so Go allocates a fresh slice for the variadic pack right"+
				" here — one allocation per dispatch, at every caller, forever. Call a"+
				" fixed-arity sibling that borrows a pooled slice instead. MEASURED on nlp, where"+
				" an MHA method was a byte-for-byte clone of the package's own variadic helper,"+
				" never touched its receiver, and had thirteen call sites while recorder-guarded"+
				" pooled siblings for arities one through four sat unused beside it: routing each"+
				" site to its matching sibling took a 500-token generate from 235.3k to 225.3k"+
				" allocs/op, -4.26%% (p=0.000, n=10), with an untouched control identical."+
				" JUDGE THIS ON allocs/op, NOT ns/op — the same change did not separate on time"+
				" (p=0.143), because the benchmarks reaching it are dominated by backend"+
				" worker-pool park and wake. IF THE PACKAGE HAS NO POOLED SIBLING, adding one is"+
				" the work, and check first: nn already ships nnIns1Pool through nnIns3Pool that"+
				" 43 of its own wrappers bypass. THE POOLED FORM MUST DEFER to the variadic one"+
				" when a recorder is attached, or the tape retains a slice that is about to be"+
				" reused. Silent on a genuine spread", name),
		})
		return true
	})
	return out
}

// functionBorrowsFromPool reports whether fn takes a slice out of a sync.Pool, which marks it as
// one of the pooled dispatch helpers rather than a caller that should be using one.
func functionBorrowsFromPool(fn *ast.FuncDecl) bool {
	found := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		sel, ok := n.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "Get" {
			return true
		}
		if id, ok := unparen(sel.X).(*ast.Ident); ok && strings.Contains(strings.ToLower(id.Name), "pool") {
			found = true
		}
		return true
	})
	return found
}

// unroundedProductUnderExactnessClaimFindings flags PS3025 — a function whose doc comment claims
// bit-identity with a DIFFERENT implementation while its body contains a bare multiply-add.
//
// Go contracts `a*b + c` into a fused multiply-add on arm64 and does not on amd64. An FMA rounds
// once where a separate multiply and add round twice, so a hand-written path and the path it
// claims to equal can differ by an ulp — on one architecture only.
func unroundedProductUnderExactnessClaimFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || fn.Doc == nil {
		return nil
	}
	doc := strings.ToLower(fn.Doc.Text())
	if !strings.Contains(doc, "bit-identical") && !strings.Contains(doc, "bit-exact") &&
		!strings.Contains(doc, "no fma") {
		return nil
	}
	// The claim has to be about a SECOND implementation. "Bit-identical" said of a parallel split
	// of the same loop is safe under contraction, because both halves are the same instructions;
	// only two independently written paths can fuse in different places.
	peer := false
	for _, k := range []string{"dispatch", "generic path", "slow path", "fallback", "reference",
		"einsum", "scalar path", "the generic"} {
		if strings.Contains(doc, k) {
			peer = true
		}
	}
	if !peer {
		return nil
	}
	var out []finding
	seen := map[int]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok {
			return true
		}
		var bare ast.Expr
		switch as.Tok {
		case token.ADD_ASSIGN, token.SUB_ASSIGN:
			for _, r := range as.Rhs {
				if isUnroundedProduct(r) {
					bare = r
				}
			}
		case token.ASSIGN:
			for _, r := range as.Rhs {
				add, ok := unparen(r).(*ast.BinaryExpr)
				if !ok || add.Op != token.ADD {
					continue
				}
				if isUnroundedProduct(add.X) {
					bare = add.X
				} else if isUnroundedProduct(add.Y) {
					bare = add.Y
				}
			}
		default:
			return true
		}
		if bare == nil {
			return true
		}
		line := fset.Position(as.Pos()).Line
		if seen[line] {
			return true
		}
		seen[line] = true
		out = append(out, finding{
			pos:      fset.Position(as.Pos()),
			category: "unrounded-product-under-exactness-claim",
			msg: fmt.Sprintf("this function's doc claims bit-identity with another"+
				" implementation, and %s is a bare multiply feeding an add. Go CONTRACTS that"+
				" into a fused multiply-add on arm64 and does NOT on amd64, so the product"+
				" rounds once here and twice there — the two paths differ by an ulp on one"+
				" architecture only, and amd64 CI cannot see it. Wrap the product in an explicit"+
				" float64/float32 conversion, which is the only construct the Go spec guarantees"+
				" forces the intermediate rounding. AN INTERMEDIATE VARIABLE DOES NOT WORK:"+
				" assigning the product to a local first left all 32 FMADDD in place in the"+
				" measured case. FOUND EXACTLY THIS WAY: four fused inference paths (TPA, KAN,"+
				" MemorizingAttention, MTA) shipped green on amd64 CI and failed on every arm64"+
				" machine, two of them carrying doc comments that asserted 'no FMA' while the"+
				" code contracted. BOTH SIDES MUST BE ROUNDED: if the peer is a backend kernel"+
				" that also contracts — the cpu matmul emits 202 FMADDD — exact equality is not"+
				" reachable by editing this side alone, and the pin belongs at a tolerance"+
				" instead. SEPARATE STATEMENTS DO NOT HELP EITHER: the Go spec permits fusing"+
				" across statements, and `x *= s` followed by `x += b` on the same slice element"+
				" contracted anyway, diverging on 66 of 256 logits in the measured case. In"+
				" generic code the conversion is to the type parameter, T(x*y), and that does"+
				" force the rounding. A claim of bit-identity across a PARALLEL split of the same"+
				" loop is not affected and is not reported", renderExpr(bare)),
		})
		return true
	})
	out = append(out, unroundedScaleThenAddFindings(fset, fn, seen)...)
	return out
}

// unroundedScaleThenAddFindings is the ACROSS-STATEMENT half of PS3025: `x *= s` followed by
// `x += b` on the same target. Splitting the chain into separate statements reads like it forces a
// rounding between them, and it does not — the Go spec lets an implementation fuse floating-point
// operations across statements, and arm64 contracted exactly this pair into an FMADD, diverging
// from the dispatched peer on 66 of 256 logits before the conversions went in.
func unroundedScaleThenAddFindings(fset *token.FileSet, fn *ast.FuncDecl, seen map[int]bool) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		blk, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, st := range blk.List {
			mul, ok := st.(*ast.AssignStmt)
			if !ok || mul.Tok != token.MUL_ASSIGN || len(mul.Lhs) != 1 {
				continue
			}
			// Deliberately no "the scale factor is already a conversion" escape hatch. It was
			// written and deleted: `x *= float32(s)` rounds the FACTOR, and the product of x with
			// that factor still contracts into the following add. Only wrapping the whole product,
			// `x = float32(x * s)`, forces the rounding — and that is an ASSIGN, which this shape
			// does not match in the first place.
			target := renderExpr(mul.Lhs[0])
			if !addAssignsTo(blk.List[i+1:], target) {
				continue
			}
			line := fset.Position(mul.Pos()).Line
			if seen[line] {
				continue
			}
			seen[line] = true
			out = append(out, finding{
				pos:      fset.Position(mul.Pos()),
				category: "unrounded-product-under-exactness-claim",
				msg: fmt.Sprintf("this function's doc claims bit-identity with another"+
					" implementation, and %s is scaled here and added to below. Written as two"+
					" statements this LOOKS like it rounds in between, and it does not: the Go"+
					" spec permits fusing floating-point operations ACROSS statements, so arm64"+
					" contracts the pair into a fused multiply-add while amd64 does not."+
					" MEASURED: exactly this shape diverged from its dispatched peer on 66 of 256"+
					" logits until each step was wrapped in an explicit conversion. Wrap them:"+
					" x = float32(x * s), then x = float32(x + b). In generic code the conversion"+
					" is to the type parameter, T(x*s), which does force the rounding", target),
			})
		}
		return true
	})
	return out
}

// addAssignsTo reports whether any statement in the list adds to the named target, at this block
// level or inside a plain if that is part of the same chain.
func addAssignsTo(list []ast.Stmt, target string) bool {
	found := false
	for _, st := range list {
		ast.Inspect(st, func(n ast.Node) bool {
			as, ok := n.(*ast.AssignStmt)
			if !ok || (as.Tok != token.ADD_ASSIGN && as.Tok != token.SUB_ASSIGN) || len(as.Lhs) != 1 {
				return true
			}
			if renderExpr(as.Lhs[0]) == target {
				found = true
			}
			return !found
		})
		if found {
			return true
		}
	}
	return false
}

// fixedOffsetStoresNotWindowedFindings flags PS3030 — three or more accesses to one slice at
// CONSTANT offsets from an invariant base plus the loop variable, where one fixed-length window
// would replace a bounds check per access with a single slice check per group.
//
// Distinct from PS3019, which is about an UNROLLED loop whose lanes sit at i+0..i+K-1 under a len
// bound. Here the loop steps by one and the offsets are the strides of a packed group.
func fixedOffsetStoresNotWindowedFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		var body *ast.BlockStmt
		var idx string
		switch l := n.(type) {
		case *ast.RangeStmt:
			body = l.Body
			if k, ok := l.Key.(*ast.Ident); ok {
				idx = k.Name
			}
		case *ast.ForStmt:
			body, idx = l.Body, counterName(l)
		default:
			return true
		}
		if body == nil || idx == "" || idx == "_" {
			return true
		}
		written := assignedNames(body)
		// key: "slice|base" -> the distinct constant offsets seen on it
		groups := map[string]map[int]bool{}
		ast.Inspect(body, func(m ast.Node) bool {
			ix, ok := m.(*ast.IndexExpr)
			if !ok {
				return true
			}
			sl := identName(ix.X)
			if sl == "" {
				return true
			}
			base, off, ok := baseIdxOffset(ix.Index, idx)
			if !ok || base == "" || written[base] {
				return true // no invariant base to slice from
			}
			k := sl + "|" + base
			if groups[k] == nil {
				groups[k] = map[int]bool{}
			}
			groups[k][off] = true
			return true
		})
		for k, offs := range groups {
			if len(offs) < 3 {
				continue
			}
			span := 0
			for o := range offs {
				if o > span {
					span = o
				}
			}
			sl, base, _ := strings.Cut(k, "|")
			out = append(out, finding{
				pos:      fset.Position(n.Pos()),
				category: "fixed-offset-stores-not-windowed",
				msg: fmt.Sprintf("this loop touches %s at %d distinct constant offsets from"+
					" %s+%s, and each of those carries its own bounds check — one check per element"+
					" in a body that may be only a few operations wide. Cut ONE fixed-length window"+
					" above the loop, %s := %s[%s : %s+%d : %s+%d], and index it by the offsets"+
					" alone; that leaves a single slice check per group. MEASURED on a Q6_K"+
					" dequantizer with four stores per iteration: -16.5%%, with the compiler's BCE"+
					" diagnostic confirming the four per-store checks gone and one slice check left."+
					" PURE ADDRESSING — no value changes, so the existing goldens are the right"+
					" gate. Look for siblings before assuming: that site was the last of its family"+
					" to be cut, and the same file's dot-product twin had already done it",
					sl, len(offs), base, idx, "w", sl, base, base, span+1, base, span+1),
			})
		}
		return true
	})
	return out
}

// baseIdxOffset decomposes an index expression of the form base+idx, base+idx+K or idx+base+K into
// its invariant base name and constant offset. Anything else is not a windowable access.
func baseIdxOffset(e ast.Expr, idx string) (base string, off int, ok bool) {
	var walk func(ast.Expr, int) bool
	seenIdx := false
	walk = func(x ast.Expr, sign int) bool {
		switch t := unparen(x).(type) {
		case *ast.BinaryExpr:
			if t.Op != token.ADD {
				return false
			}
			return walk(t.X, sign) && walk(t.Y, sign)
		case *ast.Ident:
			if t.Name == idx {
				if seenIdx {
					return false
				}
				seenIdx = true
				return true
			}
			if base != "" {
				return false
			}
			base = t.Name
			return true
		case *ast.BasicLit:
			if t.Kind != token.INT {
				return false
			}
			v, err := strconv.Atoi(t.Value)
			if err != nil {
				return false
			}
			off += v
			return true
		}
		return false
	}
	if !walk(e, 1) || !seenIdx {
		return "", 0, false
	}
	return base, off, true
}

// closureAccessorInLoopFindings flags PS3032 — a function VALUE obtained from a factory call and
// then invoked inside a loop, so every element costs an indirect call.
func closureAccessorInLoopFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	// A name can only be CALLED if it holds a function, so no type information is needed: the tell
	// is a name bound from a call's results and later used in call position.
	from := map[string]string{} // closure name -> the factory that produced it
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Rhs) != 1 {
			return true
		}
		call, ok := unparen(as.Rhs[0]).(*ast.CallExpr)
		if !ok {
			return true
		}
		factory := calleeName(call.Fun)
		if factory == "" {
			return true
		}
		for _, l := range as.Lhs {
			if nm := identName(l); nm != "" && nm != "_" {
				from[nm] = factory
			}
		}
		return true
	})
	if len(from) == 0 {
		return nil
	}
	// A function that already exits early through a GUARDED FAST PATH — `if a, b, ok := f(x); ok {
	// … return … }` — has had this conversion done, and the closure loop that remains is the
	// fallback the fast path deliberately leaves in place for the dtypes it cannot serve. Reporting
	// it would file the fix as the defect: the applied form of this check KEEPS the closure loop.
	if hasGuardedFastPathReturn(fn.Body) {
		return nil
	}
	seen := map[string]bool{}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		var body *ast.BlockStmt
		switch l := n.(type) {
		case *ast.RangeStmt:
			body = l.Body
		case *ast.ForStmt:
			body = l.Body
		default:
			return true
		}
		if body == nil {
			return true
		}
		ast.Inspect(body, func(m ast.Node) bool {
			call, ok := m.(*ast.CallExpr)
			if !ok {
				return true
			}
			nm := identName(call.Fun)
			factory, ok := from[nm]
			if !ok || seen[factory] {
				return true
			}
			seen[factory] = true
			out = append(out, finding{
				pos:      fset.Position(n.Pos()),
				category: "closure-accessor-in-loop",
				msg: fmt.Sprintf("%s holds a function value from %s and is CALLED inside this"+
					" loop, so every element pays an indirect call that cannot be inlined. This is"+
					" the per-element dispatch anti-pattern one level shallower, and it hides"+
					" better: a helper that hands back readers and writers reads like setup, and"+
					" the cost is in the calls, not in the helper. Add typed arms that walk the raw"+
					" storage and keep the closure form as the fallback for the dtypes the typed"+
					" arms cannot serve. MEASURED on two pooling backward rules: -48.2%% to -53.2%%"+
					" across four benchmark cells. TWO TRAPS IN THE CONVERSION, both hit while"+
					" making that change. The closure boundary BLOCKS FMA CONTRACTION that a typed"+
					" arm allows — a scale product and an accumulating add in the same function"+
					" fuse where a call between them cannot — so wrap the product in an explicit"+
					" conversion or the arms drift an ulp. And the fixture must make an element"+
					" receive SEVERAL accumulations, or f32 narrowing differences cannot appear at"+
					" all and a wrong arm passes its test", nm, factory),
			})
			return true
		})
		return true
	})
	return out
}

// hasGuardedFastPathReturn reports whether the body contains an if whose INIT binds results from a
// call and whose block returns — the shape of a typed fast path guarded on an ok result.
func hasGuardedFastPathReturn(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		ifs, ok := n.(*ast.IfStmt)
		if !ok || ifs.Init == nil {
			return true
		}
		as, ok := ifs.Init.(*ast.AssignStmt)
		if !ok || len(as.Rhs) != 1 {
			return true
		}
		if _, isCall := unparen(as.Rhs[0]).(*ast.CallExpr); !isCall {
			return true
		}
		for _, st := range ifs.Body.List {
			if _, isRet := st.(*ast.ReturnStmt); isRet {
				found = true
			}
		}
		return !found
	})
	return found
}

// symmetricPairComputedTwiceFindings flags PS3031 — a full i,j nest that forms BOTH orientations of
// a pair and stores a symmetric combination, so every pair is computed twice.
func symmetricPairComputedTwiceFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		outer, ok := n.(*ast.RangeStmt)
		if !ok || outer.Body == nil {
			return true
		}
		oi := identName(outer.Key)
		if oi == "" || oi == "_" {
			return true
		}
		for _, st := range outer.Body.List {
			inner, ok := st.(*ast.RangeStmt)
			if !ok || inner.Body == nil {
				continue
			}
			ij := identName(inner.Key)
			if ij == "" || ij == "_" || ij == oi {
				continue
			}
			// Two scalar accumulators whose right-hand sides are each other's mirror: one written
			// in terms of (oi, ij), the other in (ij, oi).
			orig := map[string]string{} // accumulator name -> its rhs as written
			mirr := map[string]string{} // ... and with the two loop indexes exchanged
			ast.Inspect(inner.Body, func(m ast.Node) bool {
				as, ok := m.(*ast.AssignStmt)
				if !ok || as.Tok != token.ADD_ASSIGN || len(as.Rhs) != 1 || len(as.Lhs) != 1 {
					return true
				}
				name := identName(as.Lhs[0])
				if name == "" {
					return true // an indexed accumulator is a different shape
				}
				r := renderExpr(as.Rhs[0])
				orig[name], mirr[name] = r, swapIdents(r, oi, ij)
				return true
			})
			var a, b string
			for na, ma := range mirr {
				for nb, ob := range orig {
					if na != nb && ma == ob {
						a, b = na, nb
					}
				}
			}
			if a == "" {
				continue
			}
			// THE STORE IS WHAT MAKES THE MIRROR FREE. Without a symmetric combination of the two
			// accumulators the second sum is a different quantity, not a duplicate — a first
			// version of this check omitted this and reported 27 sites, of which the ones read were
			// unrolled GEMM bands and convolution taps that merely LOOK mirrored because renaming
			// two loop variables happens to collide.
			sym := false
			ast.Inspect(inner.Body, func(m ast.Node) bool {
				be, ok := m.(*ast.BinaryExpr)
				if !ok || be.Op != token.ADD {
					return true
				}
				x, y := identName(be.X), identName(be.Y)
				if (x == a && y == b) || (x == b && y == a) {
					sym = true
				}
				return !sym
			})
			if !sym {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(st.Pos()),
				category: "symmetric-pair-computed-twice",
				msg: fmt.Sprintf("this %s,%s nest accumulates %s and its MIRROR %s and then stores"+
					" their sum, so every pair is formed TWICE over the full range and the diagonal"+
					" forms the identical sum twice. Run the inner loop from the outer index and"+
					" write both (%s,%s) and (%s,%s). BIT-IDENTICAL because the store is symmetric:"+
					" the full loop stored f(b,a) at the mirrored position where the triangle"+
					" stores f(a,b), and IEEE addition is commutative — a+b and b+a have the same"+
					" bits for every non-NaN operand. Each sum keeps its own operands and ascending"+
					" order, so nothing is reassociated. MEASURED TWICE, both about a third: a"+
					" Cholesky VJP at -34.33%% and an eigh VJP at -33.9%%. The symmetric store is"+
					" REQUIRED, not incidental — without it the second sum is a different quantity"+
					" rather than a duplicate, and an earlier version of this check that omitted the"+
					" requirement reported unrolled GEMM bands and convolution taps that merely look"+
					" mirrored", oi, ij, a, b, oi, ij, ij, oi),
			})
		}
		return true
	})
	return dedupeByPos(out)
}

// swapIdents exchanges two identifier names in a rendered expression, so a term and its mirror
// render identically. Word boundaries matter: a substring replace would rewrite parts of longer
// names.
func swapIdents(src, a, b string) string {
	var sb strings.Builder
	word := func(r byte) bool {
		return r == '_' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9'
	}
	for i := 0; i < len(src); {
		if word(src[i]) {
			j := i
			for j < len(src) && word(src[j]) {
				j++
			}
			w := src[i:j]
			switch w {
			case a:
				w = b
			case b:
				w = a
			}
			sb.WriteString(w)
			i = j
			continue
		}
		sb.WriteByte(src[i])
		i++
	}
	return sb.String()
}

// unbufferedFileToParserFindings flags PS3029 — a file handle opened here and handed straight to a
// parser, with no buffering in between.
func unbufferedFileToParserFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	opened := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Rhs) != 1 || len(as.Lhs) == 0 {
			return true
		}
		call, ok := unparen(as.Rhs[0]).(*ast.CallExpr)
		if !ok {
			return true
		}
		switch renderExpr(call.Fun) {
		case "os.Open", "os.OpenFile", "os.Create":
		default:
			return true
		}
		if nm := identName(as.Lhs[0]); nm != "" {
			opened[nm] = true
		}
		return true
	})
	if len(opened) == 0 {
		return nil
	}
	// Bulk consumers read the whole thing in a few large reads; buffering them buys nothing and
	// costs a copy. Only a handle passed to something that will read it FIELD BY FIELD is a
	// finding, and this list is what tells the two apart.
	bulk := map[string]bool{
		"bufio.NewReader": true, "bufio.NewReaderSize": true, "bufio.NewScanner": true,
		"io.ReadAll": true, "io.Copy": true, "io.ReadFull": true, "io.CopyBuffer": true,
		"os.ReadFile": true,
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || bulk[renderExpr(call.Fun)] {
			return true
		}
		for _, a := range call.Args {
			nm := identName(a)
			if nm == "" || !opened[nm] {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(call.Pos()),
				category: "unbuffered-file-to-parser",
				msg: fmt.Sprintf("the file handle %s is opened here and passed straight to %s with"+
					" no buffering in between. If the callee reads FIELD BY FIELD — a length, then"+
					" the bytes, for every string — each of those is its own read syscall. Wrap it:"+
					" bufio.NewReaderSize(%s, 1<<20). MEASURED on a GGUF loader whose header is"+
					" dominated by tokenizer arrays: a 32k-token vocabulary cost on the order of"+
					" 160k syscalls before a single tensor was touched, and buffering took that load"+
					" from 66.0ms to 5.5ms, -91.7%%. The tensor-heavy shape of the same benchmark"+
					" moved only -18.9%%, which is the tell — THIS COST IS CONSTANT IN FILE SIZE, so"+
					" it is worst where the file is smallest and it hides completely behind a"+
					" benchmark that only loads big payloads. The buffer itself costs one allocation"+
					" of its size per open. Silent when the handle goes to a bulk consumer"+
					" (io.ReadAll, io.Copy, an existing bufio wrapper), where buffering buys nothing",
					nm, renderExpr(call.Fun), nm),
			})
			return true
		}
		return true
	})
	return out
}

// unpooledFullyOverwrittenScratchFindings flags PS3028 — a large per-call scratch buffer that is
// written with plain assignments only, so the runtime's zeroing of a fresh allocation is waste.
func unpooledFullyOverwrittenScratchFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || mentionsPool(fn.Body) {
		return nil // already recycling something; leave the judgment to whoever wrote it
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		call, ok := unparen(as.Rhs[0]).(*ast.CallExpr)
		if !ok || calleeName(call.Fun) != "make" || len(call.Args) != 2 {
			return true
		}
		if factorCount(call.Args[1]) < 3 {
			return true // a small or one-dimensional scratch: the zeroing is not the cost
		}
		name := identName(as.Lhs[0])
		if name == "" || name == "_" {
			return true
		}
		if !onlyPlainIndexedWrites(fn.Body, name) {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(as.Pos()),
			category: "unpooled-fully-overwritten-scratch",
			msg: fmt.Sprintf("%s is a per-call scratch buffer whose size is a product of three or"+
				" more dimensions, and every write to it in this function is a plain assignment"+
				" rather than an accumulate — so nothing reads a slot before writing it and the"+
				" runtime's zeroing of a fresh allocation buys nothing. Recycle it through a"+
				" sync.Pool. MEASURED: an attention backward's per-head buffer was 16.7 MB at 8"+
				" heads and 512x512, and pooling it cut allocation bytes 53%%, from 25.2 MB to 11.8"+
				" MB per call. EXPECT NO SPEEDUP: on that kernel ns/op did not move at all, because"+
				" a memset is a few percent of a body doing heads*sq*sk*dk MACs. This is a resource"+
				" win, and its value is that the buffer grows with the square of the sequence while"+
				" the compute grows with the square times the head dimension. BEFORE SHIPPING,"+
				" PROVE THE OVERWRITE rather than trusting this check: poison every borrowed buffer"+
				" with NaN and confirm the suite stays green, then delete one write and confirm it"+
				" reddens. This scanner sees the SHAPE of the writes, not their coverage", name),
		})
		return true
	})
	return out
}

// mentionsPool reports whether the function already talks to a pool, in which case whoever wrote
// it has made this call already.
func mentionsPool(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && strings.Contains(strings.ToLower(id.Name), "pool") {
			found = true
		}
		return !found
	})
	return found
}

// factorCount reports how many operands a multiplicative size expression has: make([]T, a*b*c) is
// three. A buffer sized by a single dimension is not what this check is about.
func factorCount(e ast.Expr) int {
	b, ok := unparen(e).(*ast.BinaryExpr)
	if !ok || b.Op != token.MUL {
		return 1
	}
	return factorCount(b.X) + factorCount(b.Y)
}

// onlyPlainIndexedWrites reports whether every indexed write to the named slice is a plain
// assignment. One compound assignment means the buffer accumulates, and a recycled accumulator
// would need clearing first — which is the cost this check exists to remove.
func onlyPlainIndexedWrites(body *ast.BlockStmt, name string) bool {
	wrote, clean := false, true
	ast.Inspect(body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok {
			return true
		}
		for _, l := range as.Lhs {
			ix, ok := unparen(l).(*ast.IndexExpr)
			if !ok || identName(ix.X) != name {
				continue
			}
			wrote = true
			if as.Tok != token.ASSIGN {
				clean = false
			}
		}
		return true
	})
	return wrote && clean
}

// inputViewOnOutputTensorFindings flags PS3027 — a READ-ONLY view helper applied to a tensor the
// function allocated as an output.
//
// These helpers exist so a kernel can walk typed storage instead of dispatching per element. The
// input form returns the live storage when the dtype already matches and a WIDENED COPY when it
// does not; the output form returns a buffer plus a flush that narrows it back. Reach for the
// input form on an output and the mismatched-dtype path writes into a copy nobody reads — silently,
// with the right shapes and no error.
func inputViewOnOutputTensorFindings(fset *token.FileSet, fn *ast.FuncDecl, ns nameSets) []finding {
	if fn.Body == nil || len(ns.inputViews) == 0 {
		return nil
	}
	// Names this function ALLOCATED. A parameter may be an input; a tensor made here and returned
	// is an output, and that is the whole distinction the check rests on.
	allocated := map[string]token.Pos{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		call, ok := unparen(as.Rhs[0]).(*ast.CallExpr)
		if !ok || !ns.allocators[calleeName(call.Fun)] {
			return true
		}
		if nm := identName(as.Lhs[0]); nm != "" {
			allocated[nm] = as.Pos()
		}
		return true
	})
	if len(allocated) == 0 { // early-out only: the per-call membership test below decides
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || !ns.inputViews[calleeName(call.Fun)] || len(call.Args) != 1 {
			return true
		}
		nm := identName(call.Args[0])
		if _, isOut := allocated[nm]; !isOut {
			return true
		}
		alt := "the output-view helper"
		for k := range ns.outputViews {
			alt = k
			break
		}
		out = append(out, finding{
			pos:      fset.Position(call.Pos()),
			category: "input-view-on-output-tensor",
			msg: fmt.Sprintf("%s is a READ-ONLY view helper and %s is a tensor this function"+
				" allocated as an output. The view returns live storage when the dtype already"+
				" matches its element type and a WIDENED COPY when it does not, so on the"+
				" mismatched dtype the kernel accumulates into a buffer nobody reads and the"+
				" output comes back untouched — right shape, no error, all zeros. FOUND EXACTLY"+
				" THIS WAY: an attention backward returned four all-zero gradient tensors on F32"+
				" while F64 was correct, so training a trainable attention bias in F32 propagated"+
				" nothing; it survived because every test touching that op built F64 tensors. Use"+
				" %s instead and call its flush before every return that took the fast path. Then"+
				" add a test in the OTHER dtype — the reason this class hides is that the fast"+
				" path is usually exercised in the dtype where the view happens to alias",
				calleeName(call.Fun), nm, alt),
		})
		return true
	})
	return out
}

// fullFanoutUnderTopKGateFindings flags PS3026 — a function that selects a SUBSET of branches with
// a top-k gate and then evaluates ALL of them anyway, relying on the unselected ones carrying a
// zero weight.
func fullFanoutUnderTopKGateFindings(fset *token.FileSet, fn *ast.FuncDecl, ns nameSets) []finding {
	if fn.Body == nil || len(ns.topKSelectors) == 0 {
		return nil
	}
	gate := token.NoPos
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if name := calleeName(call.Fun); ns.topKSelectors[name] && (gate == token.NoPos || call.Pos() < gate) {
			gate = call.Pos()
		}
		return true
	})
	if gate == token.NoPos {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		var body *ast.BlockStmt
		var idx string
		switch l := n.(type) {
		case *ast.RangeStmt:
			body = l.Body
			if k, ok := l.Key.(*ast.Ident); ok {
				idx = k.Name
			}
		case *ast.ForStmt:
			body = l.Body
			idx = counterName(l)
		default:
			return true
		}
		// The fan-out has to come AFTER the gate: a loop that ran first could not have skipped on
		// a selection that did not exist yet.
		if body == nil || idx == "" || idx == "_" || n.Pos() < gate {
			return true
		}
		if !indexedCallOn(body, idx) || hasContinue(body) {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(n.Pos()),
			category: "full-fanout-under-topk-gate",
			msg: fmt.Sprintf("this function picks a subset of branches with a top-k gate and then"+
				" runs EVERY branch over %s, leaving the unselected ones to be multiplied by a zero"+
				" weight. Skip them: mark the chosen indices in the selection loop that already"+
				" exists, and `continue` past the rest. The result is the same BITS, not merely"+
				" close — an unselected branch contributes output times exactly zero, and adding an"+
				" exact zero returns a finite accumulator unchanged, with the surviving addends"+
				" still in their original relative order. Two exceptions to state in the doc: a"+
				" negative-zero accumulator sign, and a NaN or Inf escaping a branch nobody routed"+
				" to. MEASURED on a mixture-of-experts decode step: -23.7%% ns/op, -20.8%% allocs"+
				" (8 samples per arm, interleaved in both orders). These fan-outs are usually GEMVs,"+
				" so the step is bound by the weight bytes it streams and skipping k of E branches"+
				" removes that fraction of the footprint directly. BENCHMARK PROTOCOL: if the"+
				" benchmark carries state across iterations — a growing KV cache is the common case"+
				" — pin -benchtime to a fixed iteration count and interleave the arms in BOTH"+
				" orders; a single-order run of this very change reported it as 239%% SLOWER",
				idx),
		})
		return true
	})
	return out
}

// indexedCallOn reports whether the block calls something reached through an index by the named
// loop variable — `things[i].Do()` or `f(things[i])` — which is what makes a loop a fan-out over
// branches rather than an ordinary counted loop.
func indexedCallOn(body *ast.BlockStmt, idx string) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		ast.Inspect(call, func(m ast.Node) bool {
			ix, ok := m.(*ast.IndexExpr)
			if !ok {
				return true
			}
			if id, ok := unparen(ix.Index).(*ast.Ident); ok && id.Name == idx {
				found = true
			}
			return !found
		})
		return !found
	})
	return found
}

// hasContinue reports whether the block skips iterations, which is the applied form.
func hasContinue(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if br, ok := n.(*ast.BranchStmt); ok && br.Tok == token.CONTINUE {
			found = true
		}
		return !found
	})
	return found
}

// isUnroundedProduct reports a multiplication not already wrapped in a conversion. The applied
// form — float64(x*y), the only construct the Go spec guarantees forces the intermediate rounding
// — is a call expression, and a call is not a *ast.BinaryExpr, so this assertion is what excludes
// it. An explicit CallExpr test was written first and then deleted: it could never change an
// outcome, and a clause that cannot be reddened by mutation is not a clause.
func isUnroundedProduct(e ast.Expr) bool {
	b, ok := unparen(e).(*ast.BinaryExpr)
	return ok && b.Op == token.MUL
}

// --- PS3033: a helper that allocates its result and is only ever called per item ---------------

// perItemCallSite records one in-file call of a package-local function: where it was called from,
// whether that position is per-item (inside a loop, or inside a function literal handed to
// another call — the per-row callback shape), and whether the result was stored somewhere that
// outlives the iteration.
type perItemCallSite struct {
	caller  string
	perItem bool
	escapes bool
}

// perQueryAllocHelperFindings flags PS3033 — a package-local helper that allocates its result with
// make and hands it back, whose every in-file call site is per item.
//
// The analysis is a fixed point rather than a single pass because the cost hides one level up:
// the measured case had the allocating helper called from another helper, which was called from
// the per-row callback. Only the innermost call is lexically inside anything loop-shaped, so a
// direct test sees neither. A function whose every call site is per-item IS per-item, and that
// closes transitively.
func perQueryAllocHelperFindings(fset *token.FileSet, f *ast.File) []finding {
	// Functions declared exactly once in this file. A duplicated name (a method on two receivers)
	// cannot be resolved without types, so it is dropped rather than guessed at.
	decls := map[string]*ast.FuncDecl{}
	dup := map[string]bool{}
	for _, d := range f.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		if _, seen := decls[fn.Name.Name]; seen {
			dup[fn.Name.Name] = true
		}
		decls[fn.Name.Name] = fn
	}
	sites := map[string][]perItemCallSite{}
	for name, fn := range decls {
		if dup[name] {
			continue
		}
		collectPerItemCallSites(fn, decls, dup, sites)
	}
	// A function is per-item when it has in-file call sites and every one of them is either
	// lexically per-item or inside a function that is itself per-item.
	perItem := map[string]bool{}
	for changed := true; changed; {
		changed = false
		for name, ss := range sites {
			if perItem[name] || len(ss) == 0 {
				continue
			}
			all := true
			for _, s := range ss {
				if !s.perItem && !perItem[s.caller] {
					all = false
					break
				}
			}
			if all {
				perItem[name] = true
				changed = true
			}
		}
	}
	var out []finding
	for _, d := range f.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		name := fn.Name.Name
		// An exported helper can be called from outside this file, where the result may well
		// outlive the call. Only a package-local one can be reasoned about from here.
		if dup[name] || !perItem[name] || fn.Name.IsExported() {
			continue
		}
		buf := allocatedAndReturnedBuffer(fn)
		if buf == "" {
			continue
		}
		esc := false
		for _, s := range sites[name] {
			esc = esc || s.escapes
		}
		if esc {
			continue
		}
		out = append(out, finding{
			pos:      fset.Position(fn.Pos()),
			category: "per-item-alloc-helper",
			msg: fmt.Sprintf("%s allocates %s with make and returns it, and every call site in this"+
				" file is PER ITEM — inside a loop, or inside a helper that is itself only called"+
				" per item. So this allocation happens once per element of every batch, on a path"+
				" whose whole job is to be called once per element. Nothing about the buffer"+
				" outlives the call, so it belongs to the WORKER, not to the item: take a scratch"+
				" parameter, grow it only when cap is short, and reslice it to the length the"+
				" caller needs. MEASURED on a k-nearest-neighbours predict, where the k-best heap"+
				" and its backing array, the neighbour weights and the per-class vote accumulator"+
				" were all per query: 36020 allocations per batch became 92, -99.7%%, bytes -97.7%%,"+
				" and ns/op -4.9%%. JUDGE THIS ON allocs/op AND B/op, NOT ns/op — the time win is"+
				" the collector no longer walking those objects, so it is small on a machine with"+
				" memory bandwidth to spare and larger on one without. TWO THINGS TO GET RIGHT:"+
				" every reused buffer must be fully overwritten before it is read (truncate the"+
				" heap, reslice the weights, CLEAR an accumulator), and a result the caller KEEPS"+
				" must still be copied out. PREFER SIZING A STAGING BUFFER BY WHAT THE CALLER WILL"+
				" WRITE rather than by the maximum: sized to the write, there is no tail left from"+
				" the previous item and no clearing pass is needed, and a caller whose count"+
				" disagrees fails a length check instead of silently reading stale values — returning the scratch itself gives every row of a"+
				" chunk one aliased array. Gate it by predicting a batch and comparing against"+
				" predicting each row alone: a batch of one gets a virgin scratch, so any carried"+
				" state disagrees. Silent when a call site stores the result into an index or a"+
				" field, which outlives the iteration, and when the helper is exported",
				name, buf),
		})
	}
	return out
}

// collectPerItemCallSites walks fn and records every call to a function declared in this file,
// tagging the context it was called from.
func collectPerItemCallSites(fn *ast.FuncDecl, decls map[string]*ast.FuncDecl, dup map[string]bool,
	sites map[string][]perItemCallSite) {
	// escaping holds the calls whose result is stored into something that outlives an iteration:
	// an element of an outer container, a field, or the target of a pointer. Collected first,
	// because the walk below sees the call without its parent.
	escaping := map[*ast.CallExpr]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok {
			return true
		}
		for i, rhs := range as.Rhs {
			call, ok := unparen(rhs).(*ast.CallExpr)
			if !ok || i >= len(as.Lhs) {
				continue
			}
			switch unparen(as.Lhs[i]).(type) {
			case *ast.IndexExpr, *ast.SelectorExpr, *ast.StarExpr:
				escaping[call] = true
			}
		}
		return true
	})
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok && renderExpr(call.Fun) == "append" {
			for _, a := range call.Args { // appending a result keeps it past the iteration
				if c, ok := unparen(a).(*ast.CallExpr); ok {
					escaping[c] = true
				}
			}
		}
		return true
	})
	var walk func(n ast.Node, loop bool)
	walk = func(n ast.Node, loop bool) {
		if n == nil {
			return
		}
		switch x := n.(type) {
		case *ast.ForStmt:
			walk(x.Init, loop)
			walk(x.Body, true)
			return
		case *ast.RangeStmt:
			walk(x.Body, true)
			return
		case *ast.CallExpr:
			// calleeName takes the selector's last segment, so a method call resolves to the
			// method name a file-scoped declaration map can match. A package-qualified call
			// resolves the same way and is filtered by decls: only a name this file declares
			// counts, which is also why an import shadowing a local function name would be
			// the one thing this misreads.
			if name := calleeName(x.Fun); name != "" && decls[name] != nil && !dup[name] {
				sites[name] = append(sites[name], perItemCallSite{
					caller: fn.Name.Name, perItem: loop, escapes: escaping[x],
				})
			}
			// A function literal passed as an argument is the per-row callback shape: whoever
			// receives it decides how often to run it, and the answer is once per element.
			for _, a := range x.Args {
				if lit, ok := unparen(a).(*ast.FuncLit); ok {
					walk(lit.Body, true)
					continue
				}
				walk(a, loop)
			}
			walk(x.Fun, loop)
			return
		}
		var kids []ast.Node
		ast.Inspect(n, func(c ast.Node) bool {
			if c == nil || c == n {
				return c == n
			}
			kids = append(kids, c)
			return false
		})
		for _, c := range kids {
			walk(c, loop)
		}
	}
	walk(fn.Body, false)
}

// allocatedAndReturnedBuffer returns the name of a local slice that fn allocates with make and
// hands back, or "" if there is none.
//
// The make must land on a plain local. That single condition is what makes the APPLIED form
// silent: once the buffer lives on a caller-supplied scratch, the make targets a field and the
// returned local is a reslice of it, so the helper stops looking like an allocator — which is
// exactly what it stopped being.
func allocatedAndReturnedBuffer(fn *ast.FuncDecl) string {
	made := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE {
			return true
		}
		for i, rhs := range as.Rhs {
			call, ok := unparen(rhs).(*ast.CallExpr)
			if !ok || renderExpr(call.Fun) != "make" || len(call.Args) < 2 || i >= len(as.Lhs) {
				continue // len(args) < 2 excludes maps and channels: only a sized slice
			}
			if _, isSlice := unparen(call.Args[0]).(*ast.ArrayType); !isSlice {
				continue
			}
			if nm := identName(as.Lhs[i]); nm != "" {
				made[nm] = true
			}
		}
		return true
	})
	if len(made) == 0 {
		return ""
	}
	// A buffer HANDED TO ANYTHING ELSE may be retained by it, and this scanner cannot see where.
	// The case that forced this: a memoizing helper allocated two rows, stored them in a sync.Map
	// through a composite literal, and returned them — reusing that buffer would corrupt the
	// cache for every later reader. len/cap/copy/clear are the exceptions: they cannot retain.
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.CallExpr:
			switch renderExpr(x.Fun) {
			case "len", "cap", "copy", "clear":
				return true
			}
			for _, a := range x.Args {
				if nm := identName(a); nm != "" {
					delete(made, nm)
				}
			}
		case *ast.CompositeLit:
			for _, e := range x.Elts {
				if nm := identName(e); nm != "" {
					delete(made, nm)
				}
				if kv, ok := e.(*ast.KeyValueExpr); ok {
					if nm := identName(kv.Value); nm != "" {
						delete(made, nm)
					}
				}
			}
		}
		return true
	})
	// A buffer stored into a field or an element outlives the call and cannot be scratch.
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok {
			return true
		}
		for i, rhs := range as.Rhs {
			if i >= len(as.Lhs) {
				continue
			}
			switch unparen(as.Lhs[i]).(type) {
			case *ast.IndexExpr, *ast.SelectorExpr, *ast.StarExpr:
				if nm := identName(rhs); nm != "" {
					delete(made, nm)
				}
			}
		}
		return true
	})
	var found string
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		ret, ok := n.(*ast.ReturnStmt)
		if !ok {
			return true
		}
		for _, r := range ret.Results {
			// The result may be the buffer itself or a reslice of it (cand[:k]).
			if nm, ok := rootIdentName(unparen(r)); ok && made[nm] && found == "" {
				found = nm
			}
		}
		return true
	})
	return found
}

// --- PS3034: an independent nest running serial where the package already has a fan-out --------

// fanoutReg maps a package name to the fan-out helpers it declares: functions whose LAST
// parameter is a callback over a work index or a half-open range. Package-level, because the
// helper and the loop that should use it are routinely in different files — nn declares
// parallelRows in bitnet.go and the serial matmul that wanted it lives in muon.go.
var fanoutReg = map[string]map[string]bool{}

// collectFanoutHelpers pre-scans every package for those helpers.
func collectFanoutHelpers(files []*ast.File) {
	for _, f := range files {
		if f.Name == nil {
			continue
		}
		pkg := f.Name.Name
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv != nil || fn.Type.Params == nil {
				continue
			}
			ps := fn.Type.Params.List
			if len(ps) == 0 {
				continue
			}
			ft, ok := ps[len(ps)-1].Type.(*ast.FuncType)
			if !ok || ft.Params == nil {
				continue
			}
			// A callback MAY return an error. Requiring no results at all missed
			// classic.parallelBuild, whose callback is func(t int) error, and with it every
			// fan-out-keyed check across that package — which is how the serial below-gate arm
			// of an already-parallel predictor got reported as unused parallelism.
			if ft.Results != nil {
				if len(ft.Results.List) > 1 {
					continue
				}
				if len(ft.Results.List) == 1 {
					if id, ok := ft.Results.List[0].Type.(*ast.Ident); !ok || id.Name != "error" {
						continue
					}
				}
			}
			// One index (per item) or two (a half-open range). Anything else is a callback
			// over values, not over work.
			nInt := 0
			for _, p := range ft.Params.List {
				if id, ok := p.Type.(*ast.Ident); !ok || id.Name != "int" {
					nInt = -99
					break
				}
				nInt += max(len(p.Names), 1)
			}
			if nInt != 1 && nInt != 2 {
				continue
			}
			if fanoutReg[pkg] == nil {
				fanoutReg[pkg] = map[string]bool{}
			}
			fanoutReg[pkg][fn.Name.Name] = true
		}
	}
}

// serialNestWithIdleFanoutFindings flags PS3034 — a three-deep nest that fills a destination this
// function allocated, indexed by the OUTERMOST loop variable, in a package that already declares
// a fan-out helper the function never calls.
func serialNestWithIdleFanoutFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || len(fanoutReg[f.Name.Name]) == 0 {
		return nil
	}
	// Which NESTS are already parallel, not which functions. A whole-function test silenced any
	// function that fans out ANYWHERE, and the rules in this package routinely convert one loop
	// and leave three — a Cholesky VJP with three parallel products hid a serial triangular
	// inverse worth 44.7%% behind exactly that.
	//
	// A nest counts as already handled when it is lexically inside a fan-out callback, or when the
	// fan-out call sits in its own body.
	inFanout := map[ast.Node]bool{}
	var markFanout func(n ast.Node, inside bool)
	markFanout = func(n ast.Node, inside bool) {
		ast.Inspect(n, func(m ast.Node) bool {
			if m == nil {
				return false
			}
			if inside {
				inFanout[m] = true
			}
			call, ok := m.(*ast.CallExpr)
			if !ok {
				return true
			}
			isFan := fanoutReg[f.Name.Name][calleeName(call.Fun)]
			for _, a := range call.Args {
				markFanout(a, inside || isFan)
			}
			return false
		})
	}
	markFanout(fn.Body, false)
	dst := allocatedSliceLocals(fn)
	if len(dst) == 0 {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		// Both loop forms. A three-index `for i := 0; i < m; i++` is the same nest as a range
		// over the bound, and restricting this to RangeStmt silently skipped every one of them
		// — including the APPLIED form of the measured case, whose band loop is three-index,
		// which made the already-fans-out condition look untestable when it was merely unreached.
		outer, iv := outerLoop(n)
		if outer == nil || iv == "" || iv == "_" || loopDepthOf(outer) < 2 || inFanout[n] {
			return true
		}
		already := false
		ast.Inspect(outer, func(m ast.Node) bool {
			if call, ok := m.(*ast.CallExpr); ok && fanoutReg[f.Name.Name][calleeName(call.Fun)] {
				already = true
			}
			return true
		})
		if already {
			return true
		}
		// Independence: EVERY indexed write inside the nest must be indexed by an expression
		// naming the outer variable, or go through a window cut with it, or land on a buffer
		// this iteration created for itself. Anything else is written by one iteration and read
		// by another, and splitting the outer loop across workers would race on it.
		//
		// Checking only writes to the ALLOCATED DESTINATION is not enough, and the tree sweep
		// proved it: a SparseGPT pruning loop writes its mask at [r][c] — indexed by the outer
		// variable, so the destination test passes — while the same body eliminates INTO COLUMNS
		// AHEAD of c on a matrix it did not allocate, which the next iteration of c reads. That
		// is a sequential elimination reported as parallelizable, the worst answer this check
		// can give.
		perIter := localBuffersMadeIn(outer)
		target, independent := "", true
		ast.Inspect(outer, func(m ast.Node) bool {
			as, ok := m.(*ast.AssignStmt)
			if !ok {
				return true
			}
			for _, lhs := range as.Lhs {
				ix, ok := unparen(lhs).(*ast.IndexExpr)
				if !ok {
					continue
				}
				root, _ := rootIdentName(ix.X)
				if perIter[root] {
					continue // this iteration made it; no other iteration can see it
				}
				// The whole index CHAIN counts, not just the innermost one. A write to inner[i][j]
				// is indexed by the outer variable at its first position, and checking only ix.Index
				// saw the j and called the nest dependent — which is why this check missed the two
				// largest wins it was built for, an eigh product and a Cholesky inverse.
				if !mentionsIdent(ix.X, iv) && !mentionsIdent(ix.Index, iv) &&
					!aliasOfOuter(outer, ix.X, iv) {
					independent = false
					return false
				}
				if target == "" && (dst[root] || dstAliasedIn(fn, root, dst)) {
					target = root
				}
			}
			return true
		})
		if !independent || target == "" {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(n.Pos()),
			category: "serial-nest-with-idle-fanout",
			msg: fmt.Sprintf("this %d-deep nest fills %s — which %s allocated — indexed by the"+
				" outermost variable %s, so the iterations are independent, and yet it runs on one"+
				" core while this package already declares a fan-out helper. Split the OUTER loop"+
				" into bands: each band owns whole rows of the destination, reads whatever it needs"+
				" of the operands, and every element still accumulates in the same order, so the"+
				" result is BIT-IDENTICAL and the gate can assert exact equality rather than a"+
				" tolerance. MEASURED on a Muon optimizer step, where a flat matmul was 48.8%% of"+
				" the profile and serial: 195.8ms to 77.4ms, -60.5%%, confirmed in both benchmark"+
				" orders. GATE ON THE PRODUCT of the loop bounds, not the outer count — a"+
				" Newton-Schulz iteration drives a few rows through a great deal of work each, and"+
				" a row-count gate leaves exactly that shape serial. Expect allocations to RISE (46"+
				" to 568 per step here, one closure per band) and bytes to stay flat; that trade is"+
				" the point, not a regression. Do NOT also move the first band onto the calling"+
				" goroutine — measured here at no change in time for 41 fewer allocations. Silent"+
				" when any write is indexed without the outer variable, which is a cross-iteration"+
				" accumulator that would race",
				loopDepthOf(outer)+1, target, fn.Name.Name, iv),
		})
		return true
	})
	return out
}

// outerLoop returns the body of a loop statement and the name of the variable it counts, for
// either loop form: `for i := range m` and `for i := 0; i < m; i++` describe the same nest.
func outerLoop(n ast.Node) (*ast.BlockStmt, string) {
	switch x := n.(type) {
	case *ast.RangeStmt:
		if x.Body == nil {
			return nil, ""
		}
		return x.Body, identName(x.Key)
	case *ast.ForStmt:
		if x.Body == nil || x.Init == nil {
			return nil, ""
		}
		as, ok := x.Init.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE || len(as.Lhs) != 1 {
			return nil, ""
		}
		return x.Body, identName(as.Lhs[0])
	}
	return nil, ""
}

// localBuffersMadeIn returns the names inside body that are bound to a buffer the iteration
// CREATES — a make or a literal. No other iteration can see one, so writing through it says
// nothing about whether the loop can be split.
//
// Being defined inside the body is not enough on its own: `ci := c[i*n : i*n+n]` is also defined
// there, and writing through it is a write to c. Excusing every local defined in the body made
// the measured positive silent, since its inner loop writes only through exactly such a window.
func localBuffersMadeIn(body *ast.BlockStmt) map[string]bool {
	out := map[string]bool{}
	ast.Inspect(body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE {
			return true
		}
		for i, lhs := range as.Lhs {
			nm := identName(lhs)
			if nm == "" || i >= len(as.Rhs) {
				continue
			}
			switch r := unparen(as.Rhs[i]).(type) {
			case *ast.CallExpr:
				if renderExpr(r.Fun) == "make" {
					out[nm] = true
				}
			case *ast.CompositeLit:
				out[nm] = true
			}
		}
		return true
	})
	return out
}

// allocatedSliceLocals returns the locals fn allocates with make and a slice type.
func allocatedSliceLocals(fn *ast.FuncDecl) map[string]bool {
	out := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE {
			return true
		}
		for i, rhs := range as.Rhs {
			call, ok := unparen(rhs).(*ast.CallExpr)
			if !ok || renderExpr(call.Fun) != "make" || len(call.Args) < 2 || i >= len(as.Lhs) {
				continue
			}
			if _, isSlice := unparen(call.Args[0]).(*ast.ArrayType); !isSlice {
				continue
			}
			if nm := identName(as.Lhs[i]); nm != "" {
				out[nm] = true
			}
		}
		return true
	})
	return out
}

// dstAliasedIn reports whether name was defined as a window into an allocated destination —
// `ci := c[i*n : i*n+n]` for a flat buffer, or `cj := linvT[j]` for a slice of slices. Only the
// slice form was recognized at first, and the index form is the more common one in this tree: it
// is how every row of a [][]float64 is taken.
func dstAliasedIn(fn *ast.FuncDecl, name string, dst map[string]bool) bool {
	found := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE {
			return true
		}
		for i, lhs := range as.Lhs {
			if identName(lhs) != name || i >= len(as.Rhs) {
				continue
			}
			switch r := unparen(as.Rhs[i]).(type) {
			case *ast.SliceExpr:
				if root, _ := rootIdentName(r.X); dst[root] {
					found = true
				}
			case *ast.IndexExpr:
				if root, _ := rootIdentName(r.X); dst[root] {
					found = true
				}
			}
		}
		return true
	})
	return found
}

// aliasOfOuter reports whether x is a window cut from the destination using the outer variable —
// the axpy form, where the row is selected once above the inner loop and the inner index alone
// never mentions the outer variable.
func aliasOfOuter(body *ast.BlockStmt, x ast.Expr, iv string) bool {
	name, ok := rootIdentName(x)
	if !ok {
		return false
	}
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE {
			return true
		}
		for i, lhs := range as.Lhs {
			if identName(lhs) != name || i >= len(as.Rhs) {
				continue
			}
			switch r := unparen(as.Rhs[i]).(type) {
			case *ast.SliceExpr:
				if r.Low != nil && mentionsIdent(r.Low, iv) || r.High != nil && mentionsIdent(r.High, iv) {
					found = true
				}
			case *ast.IndexExpr: // cj := linvT[j] — a row selected by the outer variable
				if mentionsIdent(r.Index, iv) {
					found = true
				}
			}
		}
		return true
	})
	return found
}

// loopDepthOf returns how many loops are nested below body.
func loopDepthOf(body *ast.BlockStmt) int {
	best := 0
	for _, st := range body.List {
		var inner *ast.BlockStmt
		switch x := st.(type) {
		case *ast.ForStmt:
			inner = x.Body
		case *ast.RangeStmt:
			inner = x.Body
		case *ast.IfStmt:
			if d := loopDepthOf(x.Body); d > best {
				best = d
			}
			continue
		default:
			continue
		}
		if d := 1 + loopDepthOf(inner); d > best {
			best = d
		}
	}
	return best
}

// --- PS3035: a loop-invariant scratch buffer allocated once per iteration ----------------------

// loopHoistableScratchFindings flags PS3035 — a slice allocated with make at the top of a loop
// body, sized by something the loop does not vary, that never leaves the iteration.
//
// PS2001 does not cover this: it fires only on the configured TENSOR allocators, so a plain
// make([]float64, n) inside a loop is invisible to every check in this table. That is how a
// Cholesky solve came to allocate its forward-substitution buffer once per right-hand side.
func loopHoistableScratchFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		body, iv := outerLoop(n)
		if body == nil {
			return true
		}
		for _, st := range body.List {
			as, ok := st.(*ast.AssignStmt)
			if !ok || as.Tok != token.DEFINE {
				continue
			}
			for i, rhs := range as.Rhs {
				call, ok := unparen(rhs).(*ast.CallExpr)
				if !ok || renderExpr(call.Fun) != "make" || len(call.Args) < 2 || i >= len(as.Lhs) {
					continue
				}
				if _, isSlice := unparen(call.Args[0]).(*ast.ArrayType); !isSlice {
					continue
				}
				name := identName(as.Lhs[i])
				if name == "" {
					continue
				}
				// A size that varies with the loop cannot be hoisted without deciding a maximum,
				// which is a different edit with a different cost.
				varying := false
				for _, a := range call.Args[1:] {
					if iv != "" && mentionsIdent(a, iv) {
						varying = true
					}
				}
				if varying || escapesIteration(fn, name) {
					continue
				}
				out = append(out, finding{
					pos:      fset.Position(as.Pos()),
					category: "loop-hoistable-scratch",
					msg: fmt.Sprintf("%q is allocated at the top of this loop, sized by something the"+
						" loop does not vary, and never leaves the iteration — so the loop makes one"+
						" of it per pass and throws it away. Hoist it above the loop. MEASURED on a"+
						" Cholesky solve whose forward-substitution buffer was allocated per"+
						" right-hand side: at 128 columns, 133 allocations became 43 and bytes fell"+
						" 18.1%%. PS2001 does NOT cover this — it fires only on the configured tensor"+
						" allocators, so a plain make of a slice inside a loop is invisible to every"+
						" other check here. PROVE THE OVERWRITE BEFORE HOISTING: a fresh make is"+
						" zeroed and a reused buffer is not, so an iteration that READS a slot before"+
						" writing it would silently start seeing the previous pass's value. Poison the"+
						" buffer with NaN between iterations and confirm the suite stays green, then"+
						" delete one write and confirm it reddens. JUDGE ON allocs/op AND B/op — the"+
						" time win is usually nil, and was here at both shapes measured. If the loop"+
						" is a WORKER BAND rather than an item loop, this is already the right place"+
						" (see PS6008): one buffer per worker is bounded by GOMAXPROCS", name),
				})
			}
		}
		return true
	})
	return out
}

// escapesIteration reports whether name can outlive one pass of the loop that defines it: it is
// returned, stored into an index, a field or through a pointer, appended somewhere, placed in a
// composite literal, or handed to a call that might keep it. len, cap, copy and clear cannot keep
// anything and are exempt.
func escapesIteration(fn *ast.FuncDecl, name string) bool {
	escapes := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if escapes {
			return false
		}
		switch x := n.(type) {
		case *ast.ReturnStmt:
			for _, r := range x.Results {
				if root, ok := rootIdentName(unparen(r)); ok && root == name {
					escapes = true
				}
			}
		case *ast.AssignStmt:
			for i, rhs := range x.Rhs {
				if i >= len(x.Lhs) || identName(rhs) != name {
					continue
				}
				switch unparen(x.Lhs[i]).(type) {
				case *ast.IndexExpr, *ast.SelectorExpr, *ast.StarExpr:
					escapes = true
				}
			}
		case *ast.CallExpr:
			switch renderExpr(x.Fun) {
			case "len", "cap", "copy", "clear":
				return true
			}
			for _, a := range x.Args {
				if identName(a) == name {
					escapes = true
				}
			}
		case *ast.CompositeLit:
			for _, e := range x.Elts {
				if identName(e) == name {
					escapes = true
				}
				if kv, ok := e.(*ast.KeyValueExpr); ok && identName(kv.Value) == name {
					escapes = true
				}
			}
		}
		return true
	})
	return escapes
}

// --- PS3036: a gate whose expected value comes from the function under test --------------------

// sameInputs reports whether two calls were handed the SAME inputs: equal arity, and every
// argument either written identically or bound from an identical defining expression.
//
// This is what separates a vacuous gate from a real differential test. Two runs of one entry point
// with a cpu context and a reference context, or with an F32 and an F64 tensor, are comparing
// DIFFERENT implementations through one door and are a genuine oracle — their differing arguments
// are built by different expressions. Two runs over copies of one input made the same way are
// comparing the code against itself, and nothing inside it can fail.
func sameInputs(fset *token.FileSet, a, b *ast.CallExpr, defText map[string]string) bool {
	if a == nil || b == nil || len(a.Args) != len(b.Args) {
		return false
	}
	for i := range a.Args {
		ta, tb := typeText(fset, a.Args[i]), typeText(fset, b.Args[i])
		if ta == tb && ta != "" {
			continue
		}
		da, aok := defText[ta]
		db, bok := defText[tb]
		if !aok || !bok || da != db || da == "" {
			return false
		}
	}
	return true
}

// isAssertionCall reports whether a call looks like an equality assertion — a testing failure
// method, or a local helper whose name says it compares. Restricting the argument scan to these
// keeps an ordinary call that merely USES both values from reading as a comparison of them.
func isAssertionCall(call *ast.CallExpr) bool {
	name := renderExpr(call.Fun)
	if i := strings.LastIndex(name, "."); i >= 0 {
		name = name[i+1:]
	}
	switch {
	case strings.HasPrefix(name, "Fatal"), strings.HasPrefix(name, "Error"):
		return true
	case strings.HasPrefix(name, "assert"), strings.HasPrefix(name, "require"),
		strings.HasPrefix(name, "expect"):
		return true
	case strings.Contains(name, "Equal"), strings.Contains(name, "Match"),
		strings.Contains(name, "Same"), strings.Contains(name, "Close"):
		return true
	}
	return false
}

// selfComparisonOracleFindings flags PS3036 — a test that computes both sides of its comparison
// with the SAME function and then asserts they agree.
//
// This is not a style complaint. Every optimization in this repository is defended by a
// bit-identity gate, and a gate built this way can only see state carried BETWEEN calls: any
// mistake INSIDE the computation changes both sides identically and the test stays green. That is
// not hypothetical — a Newton-Schulz orthogonalization test written this way passed with two of
// its intermediate buffers wired to the same slice.
//
// Only runs with -tests, since it reads test functions.
func selfComparisonOracleFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || !strings.HasPrefix(fn.Name.Name, "Test") {
		return nil
	}
	// name -> the callee that produced it, for locals bound to a single call, plus the call node
	// itself and the source text of every local's defining expression. The texts are what decide
	// whether the two sides were handed the SAME inputs.
	from := map[string]string{}
	site := map[string]*ast.CallExpr{}
	defText := map[string]string{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE {
			return true
		}
		for i, rhs := range as.Rhs {
			call, ok := unparen(rhs).(*ast.CallExpr)
			if !ok || i >= len(as.Lhs) {
				continue
			}
			// Only the FIRST result binds: a (value, err) pair still names the value first.
			if nm := identName(as.Lhs[0]); nm != "" && nm != "_" && len(as.Rhs) == 1 {
				if callee := renderExpr(call.Fun); callee != "" {
					from[nm] = callee
					site[nm] = call
				}
			}
		}
		return true
	})
	if len(from) < 2 {
		return nil
	}
	// Drop anything that is a test INPUT rather than a test RESULT. Two matrices built by the same
	// constructor and then fed to the function under test are not a self-oracle, and without this
	// the check is 893 findings of exactly that. A name passed as an ARGUMENT anywhere in the test
	// is an input; a name that only ever appears in the comparison is the thing being judged.
	//
	// The comparison itself is the exception: an assertion helper takes both sides as arguments,
	// which is the very shape being looked for.
	inArg := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || isAssertionCall(call) {
			return true
		}
		for _, a := range call.Args {
			if nm, ok := rootIdentName(unparen(a)); ok {
				inArg[nm] = true
			}
		}
		return true
	})
	for nm := range from {
		if inArg[nm] {
			delete(from, nm)
		}
	}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE || len(as.Rhs) != 1 {
			return true
		}
		if nm := identName(as.Lhs[0]); nm != "" {
			defText[nm] = typeText(fset, as.Rhs[0])
		}
		return true
	})
	if len(from) < 2 {
		return nil
	}
	// A test that ALSO compares against a value from a different producer has a real oracle, and
	// the self-comparison beside it is then a deliberate second check rather than the whole gate.
	hasIndependent := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		chk := func(x, y ast.Expr) {
			a, aok := rootIdentName(unparen(x))
			b, bok := rootIdentName(unparen(y))
			if !aok || !bok || a == b {
				return
			}
			ca, aok := from[a]
			cb, bok := from[b]
			if aok && bok && ca != cb {
				hasIndependent = true
			}
		}
		switch x := n.(type) {
		case *ast.BinaryExpr:
			if x.Op == token.EQL || x.Op == token.NEQ {
				chk(x.X, x.Y)
			}
		case *ast.CallExpr:
			if isAssertionCall(x) {
				for i := range x.Args {
					for j := i + 1; j < len(x.Args); j++ {
						chk(x.Args[i], x.Args[j])
					}
				}
			}
		}
		return true
	})
	if hasIndependent {
		return nil
	}
	reported := map[string]bool{}
	var out []finding
	report := func(pos token.Pos, a, b, callee string) {
		key := a + "|" + b
		if a > b {
			key = b + "|" + a
		}
		if reported[key] {
			return
		}
		reported[key] = true
		out = append(out, finding{
			pos:      fset.Position(pos),
			category: "self-comparison-oracle",
			msg: fmt.Sprintf("%s and %s both come from %s, and this test compares them — so the"+
				" EXPECTED value is produced by the code under test. A gate built this way can only"+
				" see state carried BETWEEN calls: any mistake INSIDE the computation changes both"+
				" sides identically and the test stays green. FOUND EXACTLY THAT WAY: a"+
				" Newton-Schulz orthogonalization gate of this shape passed with two intermediate"+
				" buffers wired to the same slice, and only an independently written reference"+
				" caught it. Add one — a slow, obvious implementation with its own buffers — and"+
				" compare to a tolerance if the summation orders differ. THIS IS SOMETIMES"+
				" DELIBERATE: comparing the same function across a CONFIG difference (GOMAXPROCS 1"+
				" against many, a fast path against a fallback) is a real differential test. It"+
				" still gates only that difference, so say so in the test's doc and keep a separate"+
				" reference for the arithmetic. Matters here because every optimization in this"+
				" repository is defended by a bit-identity gate, and a gate that cannot fail is"+
				" worse than none: it reports coverage that does not exist. THE SILENCE IS PER"+
				" TEST FUNCTION, not per file: a reference living in a sibling test covers the"+
				" code but does not make THIS gate able to fail, so the finding stands and the"+
				" right response may be one sentence in the doc comment saying what it does and"+
				" does not catch",
				a, b, callee),
		})
	}
	pair := func(pos token.Pos, x, y ast.Expr) {
		a, aok := rootIdentName(unparen(x))
		b, bok := rootIdentName(unparen(y))
		if !aok || !bok || a == b {
			return
		}
		if ca, ok := from[a]; ok && from[b] == ca && sameInputs(fset, site[a], site[b], defText) {
			report(pos, a, b, ca)
		}
	}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.BinaryExpr:
			if x.Op == token.EQL || x.Op == token.NEQ {
				pair(x.Pos(), x.X, x.Y)
			}
		case *ast.CallExpr:
			// An assertion helper: assertBitsEqual(t, got, want, …). Any two arguments tracing
			// back to one producer are the same defect as a direct comparison.
			if !isAssertionCall(x) {
				return true
			}
			for i := range x.Args {
				for j := i + 1; j < len(x.Args); j++ {
					pair(x.Pos(), x.Args[i], x.Args[j])
				}
			}
		}
		return true
	})
	return out
}

// --- PS3037: a per-iteration append buffer whose capacity hint is a factor too small ------------

// misSizedAppendBufferFindings flags PS3037 — a slice made with an explicit capacity inside a loop
// body and then appended to from a NESTED loop, so the hint is sized per outer pass while the
// appends run outer times inner.
//
// The hint then guarantees the opposite of what it looks like: the slice doubles its way up from
// the hint to its true size on EVERY outer pass, copying everything it holds each time. In the
// measured case the hint was 8 per live beam and the true size was one candidate per beam per
// VOCABULARY ENTRY.
//
// PS3035 does not cover this. It requires a size expression that the loop does not vary and only
// recognizes loops with a range clause or an init statement, and the measured site is
// `for len(live) > 0` with a hint mentioning live — so it saw nothing.
func misSizedAppendBufferFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		body := loopBodyOf(n)
		if body == nil {
			return true
		}
		for _, st := range body.List {
			as, ok := st.(*ast.AssignStmt)
			if !ok || as.Tok != token.DEFINE || len(as.Rhs) != 1 {
				continue
			}
			call, ok := unparen(as.Rhs[0]).(*ast.CallExpr)
			// Three arguments is the whole point: a capacity was stated deliberately.
			if !ok || renderExpr(call.Fun) != "make" || len(call.Args) != 3 {
				continue
			}
			if _, isSlice := unparen(call.Args[0]).(*ast.ArrayType); !isSlice {
				continue
			}
			name := identName(as.Lhs[0])
			// Deliberately NOT escapesIteration: that treats any call argument as retention, and
			// the measured site passes its buffer to slices.SortFunc, which keeps nothing. What
			// matters here is whether the SLICE ITSELF outlives the pass — returned, or stored
			// somewhere durable — since the recommendation is to hoist it and truncate.
			if name == "" || retainedBeyondLoop(fn, name) {
				continue
			}
			inner := innerAppendHeaderIdents(body, name)
			if len(inner) == 0 {
				continue
			}
			// The hint is only WRONG if it fails to account for the inner loop. A buffer hinted at
			// len(r.items) and filled by ranging over r.items is correctly sized however deeply it
			// nests, and so is one hinted at k+1 and filled by a counted loop to k. The measured
			// defect is a hint that names the OUTER collection and nothing the inner loop uses.
			//
			// Comparing against the range subject alone was not enough: a bare counted inner loop
			// has no subject, so every correctly hinted one — a JLens position list sized
			// seq-1-skipFirst and filled by a loop over exactly that span, a speculative decoder
			// sized k+1 and filled by a loop to k — reported. The whole loop HEADER is what a
			// correct hint can draw on.
			// EVERY append site is checked, not the first one found. A diverse beam search appends
			// to its buffer twice: once per beam to carry a finished hypothesis — correctly
			// covered by a len(beams) hint — and once per beam PER VOCABULARY ENTRY. Stopping at
			// the first site let the correctly sized one vouch for the other, and the check lost
			// its own second motivating site.
			mis := false
			for _, hdr := range inner {
				sized := false
				for id := range hdr {
					if mentionsIdent(call.Args[2], id) {
						sized = true
						break
					}
				}
				if !sized {
					mis = true
					break
				}
			}
			if !mis {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(as.Pos()),
				category: "mis-sized-append-buffer",
				msg: fmt.Sprintf("%q is made with a stated capacity inside this loop and then"+
					" appended to from a NESTED loop, so the hint is sized per outer pass while the"+
					" appends run outer times inner. The hint then guarantees the opposite of what"+
					" it looks like: the slice doubles its way up from the hint to its true size on"+
					" EVERY outer pass, copying everything it holds each time. MEASURED on a beam"+
					" search whose hint was 8 per live beam against a true size of one candidate"+
					" per beam per VOCABULARY ENTRY: that single append line was 2.45 GB of the"+
					" benchmark's 2.90 GB, and hoisting the buffer above the loop with a truncation"+
					" per pass took bytes -85.0%% on beam search and -98.0%% on its diverse variant."+
					" Hoist it and reset with [:0] — it then reaches its true size once and stays"+
					" there. JUDGE ON B/op FIRST; the time win was -8.8%% and -2.9%%, real but much"+
					" smaller than the byte win. PROVE NOTHING SURVIVES THE RESET: the buffer must"+
					" be truncated before it is refilled, and dropping that truncation must redden"+
					" the suite — on the measured site it did. PS3035 does not cover this: it wants"+
					" a size the loop does not vary and only sees loops with a range clause or an"+
					" init statement, and the measured site is a bare condition loop with a hint"+
					" that mentions the collection it iterates", name),
			})
		}
		return true
	})
	return out
}

// retainedBeyondLoop reports whether name is returned or stored into an index, a field or through
// a pointer — the ways a slice survives the iteration that built it.
func retainedBeyondLoop(fn *ast.FuncDecl, name string) bool {
	found := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if found {
			return false
		}
		switch x := n.(type) {
		case *ast.ReturnStmt:
			for _, r := range x.Results {
				if root, ok := rootIdentName(unparen(r)); ok && root == name {
					found = true
				}
			}
		case *ast.AssignStmt:
			for i, rhs := range x.Rhs {
				if i >= len(x.Lhs) || identName(rhs) != name {
					continue
				}
				switch unparen(x.Lhs[i]).(type) {
				case *ast.IndexExpr, *ast.SelectorExpr, *ast.StarExpr:
					found = true
				}
			}
		case *ast.CallExpr:
			// Appending the buffer INTO something else keeps it: the collection holds the slice
			// header, and the next pass would rewrite what was collected. A sort or any other
			// call that merely reads it does not, which is why only append counts here.
			if renderExpr(x.Fun) == "append" {
				for _, a := range x.Args[min(1, len(x.Args)):] {
					if identName(a) == name {
						found = true
					}
				}
			}
		}
		return true
	})
	return found
}

// innerAppendHeaderIdents returns, for EVERY append to name inside a loop nested within body, the
// identifiers in that loop's header — what it ranges over, or the init, condition and post of a
// counted loop. A correctly sized capacity hint has to be expressible in those terms, and a buffer
// is mis-sized when ANY of its append sites is left unaccounted for.
func innerAppendHeaderIdents(body *ast.BlockStmt, name string) []map[string]bool {
	var found []map[string]bool
	var walk func(n ast.Node, hdr map[string]bool, depth int)
	collect := func(dst map[string]bool, es ...ast.Node) map[string]bool {
		out := map[string]bool{}
		for k := range dst {
			out[k] = true
		}
		for _, e := range es {
			if e == nil {
				continue
			}
			ast.Inspect(e, func(n ast.Node) bool {
				if id, ok := n.(*ast.Ident); ok {
					out[id.Name] = true
				}
				return true
			})
		}
		return out
	}
	walk = func(n ast.Node, hdr map[string]bool, depth int) {
		ast.Inspect(n, func(m ast.Node) bool {
			switch x := m.(type) {
			case *ast.RangeStmt:
				walk(x.Body, collect(nil, x.X), depth+1)
				return false
			case *ast.ForStmt:
				walk(x.Body, collect(nil, x.Init, x.Cond, x.Post), depth+1)
				return false
			case *ast.AssignStmt:
				if depth == 0 || len(x.Lhs) != 1 || len(x.Rhs) != 1 || identName(x.Lhs[0]) != name {
					return true
				}
				c, ok := unparen(x.Rhs[0]).(*ast.CallExpr)
				if ok && renderExpr(c.Fun) == "append" && len(c.Args) > 0 &&
					identName(c.Args[0]) == name {
					if hdr == nil {
						hdr = map[string]bool{}
					}
					found = append(found, hdr)
				}
			}
			return true
		})
	}
	walk(body, nil, 0)
	return found
}

// --- PS3038: a dispatch call building its input slice inline, beside a pooled helper -----------

// execPoolReg maps a package name to the arities its pooled dispatch helpers cover. A helper is a
// function that borrows its input slice from a sync.Pool and hands it to backend.Execute — the
// shape nn's execPool1..3, nlp's exec1a/exec2/exec3 and vision's swinExec1a/swinExec2 all have.
var execPoolReg = map[string]map[int]string{}

// collectExecPoolHelpers pre-scans every package for those helpers, recording the number of
// tensor parameters each covers.
func collectExecPoolHelpers(files []*ast.File) {
	for _, f := range files {
		if f.Name == nil {
			continue
		}
		pkg := f.Name.Name
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv != nil || fn.Body == nil || fn.Type.Params == nil {
				continue
			}
			pooled, dispatches := false, false
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				switch name := renderExpr(call.Fun); {
				case strings.HasSuffix(name, "Pool.Get"):
					pooled = true
				case name == "backend.Execute":
					dispatches = true
				}
				return true
			})
			if !pooled || !dispatches {
				continue
			}
			// Count the trailing *tensor.Tensor parameters: the arity this helper covers.
			n := 0
			for _, p := range fn.Type.Params.List {
				if typeText(token.NewFileSet(), p.Type) == "" {
					continue
				}
				if star, ok := p.Type.(*ast.StarExpr); ok {
					if renderExpr(star.X) == "tensor.Tensor" {
						n += max(len(p.Names), 1)
					}
				}
			}
			if n == 0 {
				continue
			}
			if execPoolReg[pkg] == nil {
				execPoolReg[pkg] = map[int]string{}
			}
			if _, seen := execPoolReg[pkg][n]; !seen {
				execPoolReg[pkg][n] = fn.Name.Name
			}
		}
	}
}

// dispatchLiteralSliceFindings flags PS3038 — a direct backend.Execute whose inputs argument is an
// inline slice literal, in a package that already has a pooled helper of that arity.
func dispatchLiteralSliceFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil {
		return nil
	}
	byArity := execPoolReg[f.Name.Name]
	if len(byArity) == 0 {
		return nil
	}
	// The helpers themselves build a literal on purpose: their recorder fallback must hand the
	// tape a slice of its own. Reporting them would be reporting the applied form.
	for _, h := range byArity {
		if h == fn.Name.Name {
			return nil
		}
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || renderExpr(call.Fun) != "backend.Execute" || len(call.Args) < 3 {
			return true
		}
		cl, ok := unparen(call.Args[2]).(*ast.CompositeLit)
		if !ok {
			return true
		}
		at, ok := cl.Type.(*ast.ArrayType)
		if !ok || at.Len != nil {
			return true
		}
		helper, covered := byArity[len(cl.Elts)]
		if !covered || len(cl.Elts) == 0 {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(call.Pos()),
			category: "dispatch-literal-slice",
			msg: fmt.Sprintf("this backend.Execute builds its %d-input slice inline, in a package"+
				" that already has %s for exactly that arity. The literal is one allocation per"+
				" dispatch, and Execute drops the slice the instant it returns unless a recorder"+
				" is attached — which is what the pooled helper checks before borrowing. MEASURED"+
				" on nn.Linear.Forward, the most-called forward in the package since every MLP"+
				" block routes through it: two literals became two pooled borrows and a"+
				" per-image MLP-Mixer forward went 3944 to 3687 allocs/op, -6.5%%, with a ViT"+
				" forward at -2.0%%. JUDGE ON allocs/op — the time was flat on every cell, because"+
				" these forwards are dominated by the kernels the slice merely names. THE"+
				" RECORDER GUARD IS THE WHOLE CONTRACT: Execute's tape node stores that exact"+
				" slice, so a pooled one would be overwritten by the next op and a training run"+
				" would silently get wrong gradients. Use the helper, which defers to a fresh"+
				" literal when ctx.Recorder is set; never inline the borrow. 214 of these sit in"+
				" nn alone, so rank by call frequency and convert where a benchmark can see it",
				len(cl.Elts), helper),
		})
		return true
	})
	return out
}

// --- PS3039: a recursive split allocating both sides instead of partitioning in place ----------

// recursiveSplitAllocFindings flags PS3039 — a self-recursive function that allocates TWO slices
// sized by its input, fills them by appending each element to one or the other, and passes them to
// its own recursive calls.
//
// That is a divide-and-conquer partition written the allocating way. The cost is per NODE of the
// recursion, so it scales with the tree rather than with the data, and one reused buffer plus an
// in-place compaction replaces it exactly.
func recursiveSplitAllocFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	self := fn.Name.Name
	// Two sized allocations bound with := at the same level.
	made := map[string]*ast.AssignStmt{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE || len(as.Rhs) != 1 {
			return true
		}
		call, ok := unparen(as.Rhs[0]).(*ast.CallExpr)
		if !ok || renderExpr(call.Fun) != "make" || len(call.Args) < 2 {
			return true
		}
		if _, isSlice := unparen(call.Args[0]).(*ast.ArrayType); !isSlice {
			return true
		}
		if nm := identName(as.Lhs[0]); nm != "" {
			made[nm] = as
		}
		return true
	})
	// Both must be appended to from the two arms of one branch — the partition itself.
	appended := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		nm := identName(as.Lhs[0])
		if nm == "" || made[nm] == nil {
			return true
		}
		c, ok := unparen(as.Rhs[0]).(*ast.CallExpr)
		if ok && renderExpr(c.Fun) == "append" && len(c.Args) > 0 && identName(c.Args[0]) == nm {
			appended[nm] = true
		}
		return true
	})
	// …and both must be handed to a RECURSIVE call, which is what makes the cost per node. That
	// last count is the only one worth testing: recursed is a subset of appended, which is a
	// subset of made, so earlier count guards on those two could not redden any fixture the
	// final one does not already silence. They were removed rather than left as untestable
	// early-outs.
	recursed := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || calleeName(call.Fun) != self {
			return true
		}
		for _, a := range call.Args {
			if nm := identName(a); appended[nm] {
				recursed[nm] = true
			}
		}
		return true
	})
	if len(recursed) < 2 {
		return nil
	}
	names := make([]string, 0, len(recursed))
	for nm := range recursed {
		names = append(names, nm)
	}
	sort.Strings(names)
	return []finding{{
		pos:      fset.Position(made[names[0]].Pos()),
		category: "recursive-split-alloc",
		msg: fmt.Sprintf("%s and %s are allocated here, filled by appending each element to one or"+
			" the other, and handed to %s's own recursive calls — a divide-and-conquer partition"+
			" written the allocating way. The cost is per NODE of the recursion, so it scales with"+
			" the TREE rather than with the data. Partition the input IN PLACE against one reused"+
			" buffer: compact one side forward over the input while collecting the other into the"+
			" buffer, copy that back after the loop, and recurse on the two subslices. MEASURED on"+
			" a CART builder's subsampled path: 352029 to 192021 allocs/op (-45.5%%), bytes"+
			" -63.9%%, ns/op -6.8%% to -9.2%% against a control drifting under 2%%. WHY IT IS SAFE:"+
			" writing dst[mid] while ranging over dst cannot clobber an unread element, because mid"+
			" advances only on a write and every write consumes a value already read; copying the"+
			" second side back in order preserves the relative order both appends produced. GATE IT"+
			" WITH AN EXACT GOLDEN GENERATED FROM THE OLD CODE — the property tests that usually"+
			" cover a tree builder (it beats a single tree, variance falls) stay green for a"+
			" DIFFERENT tree, and on the measured site they stayed green with the copy-back deleted",
			names[0], names[1], self),
	}}
}

// --- PS3040: an independent MIDDLE loop under a sequential outer one --------------------------

// innerIndependentUnderSequentialOuterFindings flags PS3040 — a three-deep nest whose outer loop
// carries a real dependence (it is read, not written, by the body) while the MIDDLE loop is
// independent: every write is indexed by the middle variable and none by the outer.
//
// PS3034 cannot see this and should not: it asks whether the OUTER loop can be split, and here it
// cannot. The split belongs one level in. That is the shape of every classical factorization —
// pivot sequentially, update the remaining rows in parallel — and the LU rank-1 update it was
// built from went -40.8% at 512 wide.
// callsFanoutHelper reports whether body calls one of the package's fan-out helpers anywhere
// inside it.
func callsFanoutHelper(body *ast.BlockStmt, reg map[string]bool) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if c, ok := n.(*ast.CallExpr); ok && reg[calleeName(c.Fun)] {
			found = true
		}
		return !found
	})
	return found
}

func innerIndependentUnderSequentialOuterFindings(fset *token.FileSet, f *ast.File,
	fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || len(fanoutReg[f.Name.Name]) == 0 {
		return nil
	}
	// A nest already sitting inside a fan-out callback is parallel one level OUT, and splitting it
	// again is not the advice. Without this the triangular solves in this very package report:
	// their column loop is already the parallel one, and the row loop under it is what this check
	// would otherwise point at.
	inFanout := map[ast.Node]bool{}
	var markFanout func(n ast.Node, inside bool)
	markFanout = func(n ast.Node, inside bool) {
		ast.Inspect(n, func(m ast.Node) bool {
			if m == nil {
				return false
			}
			if inside {
				inFanout[m] = true
			}
			call, ok := m.(*ast.CallExpr)
			if !ok {
				return true
			}
			isFan := fanoutReg[f.Name.Name][calleeName(call.Fun)]
			for _, a := range call.Args {
				markFanout(a, inside || isFan)
			}
			return false
		})
	}
	markFanout(fn.Body, false)
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		outer, ov := outerLoop(n)
		if outer == nil || ov == "" || ov == "_" || loopDepthOf(outer) < 2 || inFanout[n] {
			return true
		}
		// THE SPLIT MAY ALREADY BE HERE. Applying this check leaves a below-gate arm that is a
		// PLAIN DUPLICATED LOOP — the check's own advice, because routing small inputs through
		// the callback costs a few percent — and that duplicate is this exact shape. It sits as
		// a sibling of the gated dispatch inside the same outer body, so neither the depth test
		// nor inFanout sees it, and the check reported the very site it had just been used to
		// fix. An earlier note here said a third guard could redden no fixture the other two do
		// not; that was true until the transform shipped and produced this one.
		if callsFanoutHelper(outer, fanoutReg[f.Name.Name]) {
			return true
		}
		for _, st := range outer.List {
			mid, mv := outerLoop(st)
			if mid == nil || mv == "" || mv == "_" || mv == ov || loopDepthOf(mid) < 1 {
				continue
			}
			// Every indexed write must name the MIDDLE variable and none may name the outer: that
			// is what makes the middle loop splittable while the outer one is not.
			perIter := localBuffersMadeIn(mid)
			ok, target := true, ""
			ast.Inspect(mid, func(m ast.Node) bool {
				as, aok := m.(*ast.AssignStmt)
				if !aok {
					return true
				}
				for _, lhs := range as.Lhs {
					ix, iok := unparen(lhs).(*ast.IndexExpr)
					if !iok {
						continue
					}
					root, _ := rootIdentName(ix.X)
					if perIter[root] {
						continue
					}
					// Ownership by the MIDDLE variable is the whole test. The outer variable may
					// appear too — an LU update writes mi[k], the multiplier, at the pivot column
					// of row i — and rejecting that lost the very site this check was built from.
					// What matters is that the location belongs to this middle iteration.
					if !mentionsIdent(ix, mv) && !aliasOfOuter(mid, ix.X, mv) {
						ok = false
						return false
					}
					if target == "" {
						target = root
					}
				}
				return true
			})
			if !ok || target == "" {
				continue
			}
			// The outer variable must actually be READ in the body, or it is not a dependence and
			// PS3034's question — can the OUTER loop be split — is the right one instead.
			//
			// Only READS count. Counting every index expression made a write to out[i][j] look
			// like a dependence on i and reported a nest whose OUTER loop is perfectly splittable,
			// which is the other check's finding and the opposite advice.
			// The WHOLE left-hand subtree is a write, not just its top node: out[i][j] contains
			// the nested index out[i], and marking only the outer one left that inner expression
			// looking like a read of i.
			written := map[ast.Node]bool{}
			ast.Inspect(mid, func(m ast.Node) bool {
				as, aok := m.(*ast.AssignStmt)
				if !aok {
					return true
				}
				for _, lhs := range as.Lhs {
					ast.Inspect(unparen(lhs), func(w ast.Node) bool {
						if w != nil {
							written[w] = true
						}
						return true
					})
				}
				return true
			})
			reads := false
			ast.Inspect(mid, func(m ast.Node) bool {
				ix, iok := m.(*ast.IndexExpr)
				if iok && !written[ast.Node(ix)] && mentionsIdent(ix, ov) {
					reads = true
				}
				return true
			})
			if !reads {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(st.Pos()),
				category: "inner-independent-under-sequential-outer",
				msg: fmt.Sprintf("this loop over %s sits under a SEQUENTIAL loop over %s and is"+
					" itself independent: every write here is indexed by %s and none by %s, while"+
					" %s is only read. So the outer loop cannot be split and this one can — split"+
					" HERE, one level in. That is the shape of every classical factorization:"+
					" choose a pivot in order, then update the remaining rows in parallel."+
					" MEASURED on an LU rank-1 update, which was 92%% of its own benchmark on ONE"+
					" line: -40.8%% at 512 wide and -11.1%% at 256, with 128 unchanged because it"+
					" sits below the work gate. PS3034 does NOT cover this and should not — it asks"+
					" whether the OUTER loop can be split, and the answer here is no. TWO THINGS"+
					" THE CONVERSION NEEDS. Gate on the work at THIS step (rows times columns, not"+
					" the row count) or mid-sized inputs stay serial while large ones improve, which"+
					" reads as a size effect. And keep the below-gate path a PLAIN DUPLICATED LOOP:"+
					" routing a 128-wide factorization through the callback cost 3 to 4%%, and"+
					" hoisting the gate above the call did not recover it, because the cost is the"+
					" closure rather than the dispatch. Gate it with an oracle that knows nothing"+
					" of the internals — solving and checking the residual caught a dropped row"+
					" that every existing test in the package missed."+
					" EXPECT LESS THAN THE SERIAL SHARE PROMISES, and by a lot when the outer"+
					" loop is long: a Gauss-Jordan solve that was 40%% of an AQLM encode's wall"+
					" clock returned only 6.6%% end to end, because EVERY step of the outer loop"+
					" pays its own fork and there are n of them. Widening or narrowing the gate"+
					" did not recover it — 1<<12, 1<<14, 1<<16 and 1<<18 were all measured, best"+
					" at 1<<14. Divide the serial share by the fork count before promising"+
					" anything. A SECOND APPLICATION WENT FURTHER AND LOST OUTRIGHT: a Cholesky"+
					" whose row update was fanned out this way ran SLOWER than the serial"+
					" reference, 24.1 ms against 22.0 at n=512 with allocations going from 6 to"+
					" 878, because there are n columns and the row update shrinks as j grows."+
					" Taking four rows per pass instead — no fan-out at all — went 21.1 to 7.0 ms"+
					" on the same shape. When the outer loop is long and the inner work shrinks,"+
					" try the arithmetic before the parallelism."+
					" AND TEST THE BELOW-GATE ARM AGAINST THE WORK PER WORKER, not against the"+
					" on/off threshold. A transpose kernel sitting EXACTLY on that threshold"+
					" routed through the callback, got one worker back, ran inline anyway and"+
					" paid the escaping closure for nothing — 10%% slower than the reference it"+
					" replaced, and one extra allocation per call. Gating on 2x the per-worker"+
					" floor instead brought it back to parity at that size while keeping the"+
					" 2.2x at a large one",
					mv, ov, mv, ov, ov),
			})
		}
		return true
	})
	return out
}

// --- PS3041: a per-item loop that re-streams a shared collection -------------------------------

// recvNameAndType returns the receiver's variable name and its base type name.
func recvNameAndType(fn *ast.FuncDecl) (name, typ string) {
	if fn.Recv == nil || len(fn.Recv.List) != 1 {
		return "", ""
	}
	f := fn.Recv.List[0]
	if len(f.Names) == 1 {
		name = f.Names[0].Name
	}
	t := f.Type
	if st, ok := t.(*ast.StarExpr); ok {
		t = st.X
	}
	return name, identName(t)
}

// indexedByLoopVar reports whether anything in body is indexed by an expression mentioning iv.
// This is what separates PER-ITEM work — a loop whose iteration reaches its own slot — from a
// loop that merely counts.
func indexedByLoopVar(body ast.Node, iv string) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.IndexExpr:
			if mentionsIdent(x.Index, iv) {
				found = true
			}
		case *ast.SliceExpr:
			// A per-item WINDOW counts. Leaving this out made the check silent on the very site
			// it was built from: that loop reaches its query row as qs[ti*headDim:...] and writes
			// its output block through an offset computed once, so not one IndexExpr in the body
			// mentions the loop variable.
			if (x.Low != nil && mentionsIdent(x.Low, iv)) || (x.High != nil && mentionsIdent(x.High, iv)) {
				found = true
			}
		}
		return !found
	})
	return found
}

// alreadyTiled reports whether n is a loop that advances by something other than one — the
// APPLIED form of this check, a loop over blocks of items. Without this the check would go on
// reporting the site after it had been fixed, since the block loop still calls the same scan.
func alreadyTiled(n ast.Node) bool {
	fs, ok := n.(*ast.ForStmt)
	if !ok {
		return false
	}
	as, ok := fs.Post.(*ast.AssignStmt)
	if !ok || as.Tok != token.ADD_ASSIGN || len(as.Rhs) != 1 {
		return false
	}
	lit, ok := unparen(as.Rhs[0]).(*ast.BasicLit)
	return !ok || lit.Value != "1" // a named or computed step is a tile width
}

// indexesCollection reports whether body reads coll[key] — the access that distinguishes a walk
// over a COLLECTION from a range over an integer field that happens to index something else.
func indexesCollection(body ast.Node, coll, key string) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		ix, ok := n.(*ast.IndexExpr)
		if ok && renderExpr(ix.X) == coll && mentionsIdent(ix.Index, key) {
			found = true
		}
		return !found
	})
	return found
}

// sharedFieldScan returns the rendered name of a receiver-held collection that body walks in
// full WITHOUT reference to iv: a loop whose range expression is recv.field, whose element is
// then indexed inside the loop. Every pass over the enclosing loop reads exactly the same
// memory, which is the signature of a re-stream. iv may be empty, which is the case when the
// scan sits in a callee.
func sharedFieldScan(body ast.Node, recv, iv string) string {
	if recv == "" {
		return ""
	}
	hit := ""
	ast.Inspect(body, func(n ast.Node) bool {
		if hit != "" {
			return false
		}
		rs, ok := n.(*ast.RangeStmt)
		if !ok || rs.Body == nil {
			return true
		}
		sel, ok := unparen(rs.X).(*ast.SelectorExpr)
		if !ok || identName(sel.X) != recv {
			return true
		}
		if iv != "" && mentionsIdent(rs.X, iv) {
			return true
		}
		// The scan must READ THE COLLECTION'S OWN ELEMENTS. Testing only that the body indexes
		// something by the scan key is not enough and made the first version useless: a
		// range-over-INT field, `for d := range m.dim`, satisfies it by indexing unrelated
		// slices, and eight of the nine findings on the first tree-wide run were that.
		key := identName(rs.Key)
		if key == "" || !(rs.Value != nil || indexesCollection(rs.Body, renderExpr(rs.X), key)) {
			return true
		}
		hit = renderExpr(rs.X)
		return false
	})
	return hit
}

// perItemRescanFindings flags PS3041 — an outer loop over items whose body walks a large
// collection held on the receiver, directly or one same-type method call deep, without that walk
// depending on the item. Each item re-reads the whole collection, so the pass moves items ×
// collection bytes through the caches and reuses none of it.
//
// MEASURED on the memorizing-attention neighbour search, where each query token scanned the whole
// key bank on its own. Batching the token loop into tiles of 16 and dotting each loaded key row
// against all 16 queries cut the traffic by that factor: BenchmarkMemForward_512 −24%,
// BenchmarkMemGatherLarge −30%, bit-identically, since each dot keeps its own accumulator and
// summation order and each item keeps its own result heap.
func perItemRescanFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	recv, typ := recvNameAndType(fn)
	if fn.Body == nil || recv == "" {
		return nil
	}
	// Same-receiver-type methods declared in this file, so a scan one call deep is visible.
	sib := map[string]*ast.FuncDecl{}
	for _, d := range f.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Body == nil || fd == fn {
			continue
		}
		if r, tp := recvNameAndType(fd); tp == typ && r != "" && fd.Name != nil {
			sib[fd.Name.Name] = fd
		}
	}
	var out []finding
	seen := map[int]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		body, iv := outerLoop(n)
		if body == nil || iv == "" || alreadyTiled(n) || !indexedByLoopVar(body, iv) {
			return true
		}
		coll, depth := sharedFieldScan(body, recv, iv), "in this loop"
		if coll == "" { // …or one call deep, on a sibling method of the same receiver type
			ast.Inspect(body, func(m ast.Node) bool {
				if coll != "" {
					return false
				}
				call, ok := m.(*ast.CallExpr)
				if !ok {
					return true
				}
				sel, ok := unparen(call.Fun).(*ast.SelectorExpr)
				if !ok || identName(sel.X) != recv || sel.Sel == nil {
					return true
				}
				callee := sib[sel.Sel.Name]
				if callee == nil {
					return true
				}
				cr, _ := recvNameAndType(callee)
				if c := sharedFieldScan(callee.Body, cr, ""); c != "" {
					coll, depth = strings.Replace(c, cr+".", recv+".", 1), "in "+sel.Sel.Name+", which it calls per item"
				}
				return true
			})
		}
		if coll == "" || seen[int(n.Pos())] {
			return true
		}
		seen[int(n.Pos())] = true
		out = append(out, finding{
			pos:      fset.Position(n.Pos()),
			category: "per-item-rescan-of-shared-collection",
			msg: fmt.Sprintf("this loop does per-item work, and %s it walks %s in full without"+
				" reference to %q — so every item re-reads the same memory and reuses none of it."+
				" The pass moves items × collection bytes through the caches, which makes it"+
				" BANDWIDTH-bound however cheap the arithmetic is. Batch the item loop into TILES:"+
				" load each element of %s once and do all B items' work on it while it is in cache."+
				" MEASURED on the memorizing-attention neighbour search, where each query token"+
				" scanned the whole key bank alone — tiles of 16 cut BenchmarkMemForward_512 by 24%%"+
				" and BenchmarkMemGatherLarge by 30%%. CONFIRM THE DIAGNOSIS FIRST: if the collection"+
				" already fits in L2 the traffic was never the cost and tiling buys nothing, so"+
				" profile before rewriting. THE TILE MUST NOT REASSOCIATE: give each item its own"+
				" accumulator and its own result state and visit the collection in the same order,"+
				" and the output stays bit-identical — that is what lets the existing goldens gate"+
				" the rewrite. Sweep the tile width; the curve flattens once the tile's own working"+
				" set competes with the collection for L1",
				depth, coll, iv, coll),
		})
		return true
	})
	return out
}

// --- PS3042: a whole-tensor staging buffer between fused parallel stages -----------------------

// countArgIdent returns the identifier a fan-out call passes as its ITEM COUNT, together with
// the callback body, when the call has the (n, …, func(lo, hi int)) shape.
func countArgIdent(call *ast.CallExpr) (string, *ast.BlockStmt) {
	if len(call.Args) < 2 {
		return "", nil
	}
	lit, ok := unparen(call.Args[len(call.Args)-1]).(*ast.FuncLit)
	if !ok || lit.Body == nil {
		return "", nil
	}
	n := identName(unparen(call.Args[0]))
	if n == "" {
		return "", nil
	}
	return n, lit.Body
}

// productMentions reports whether e is a MULTIPLICATION that mentions name. A buffer sized by
// the item count ALONE is one slot per item, which is the right size for a per-item result; a
// buffer sized by the count TIMES a width is a staging area for rows of work.
func productMentions(e ast.Expr, name string) bool {
	b, ok := unparen(e).(*ast.BinaryExpr)
	if !ok || b.Op != token.MUL {
		return false
	}
	return mentionsIdent(b, name)
}

// wholeTensorStagingFindings flags PS3042 — a scratch buffer allocated before a fan-out call,
// sized by the fan-out's ITEM COUNT times a width, and used only inside the callback. Each band
// touches only its own rows, so the buffer's size is set by the whole tensor while its working
// set is one band's worth: every element written by the producing stage goes out to memory and
// comes back for the consuming stage.
//
// MEASURED on conv2d, where the im2col column matrix was sized rows x k. im2col replicates each
// input element kh*kw times, so one 512x512 head convolution of the multi-token-attention
// benchmark materialized 138 MB in order to multiply it by a 66-element weight vector. Sized to
// an L2-resident chunk instead: the largest torch conv shape -11%, the attention forward -12%,
// and B/op down 62-88% across the conv benchmarks.
func wholeTensorStagingFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || len(fanoutReg[f.Name.Name]) == 0 {
		return nil
	}
	// Names this function RETURNS are outputs, not staging: their size is the caller's contract.
	returned := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		rs, ok := n.(*ast.ReturnStmt)
		if !ok {
			return true
		}
		for _, r := range rs.Results {
			for _, nm := range identNamesIn(r) {
				returned[nm] = true
			}
		}
		return true
	})
	// Every buffer this function binds from an allocation, with the size expression it used.
	sized := map[string]ast.Expr{}
	pos := map[string]token.Pos{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE {
			return true
		}
		for i, rhs := range as.Rhs {
			call, ok := unparen(rhs).(*ast.CallExpr)
			if !ok || i >= len(as.Lhs) {
				continue
			}
			var size ast.Expr
			switch {
			case renderExpr(call.Fun) == "make" && len(call.Args) >= 2:
				if _, isSlice := unparen(call.Args[0]).(*ast.ArrayType); isSlice {
					size = call.Args[1]
				}
			case len(call.Args) == 1: // a pooled getter: getF64(rows*k) and friends
				size = call.Args[0]
			}
			if size == nil {
				continue
			}
			if nm := identName(as.Lhs[i]); nm != "" {
				sized[nm] = size
				pos[nm] = as.Pos()
			}
		}
		return true
	})
	if len(sized) == 0 {
		return nil
	}
	// A pooled buffer is bound TWICE — the pointer from the getter and its dereference — and it
	// is the dereference the callback uses. Without following that alias the check was silent on
	// the very site it was built from.
	alias := map[string][]string{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE {
			return true
		}
		for i, rhs := range as.Rhs {
			st, ok := unparen(rhs).(*ast.StarExpr)
			if !ok || i >= len(as.Lhs) {
				continue
			}
			if base, nm := identName(st.X), identName(as.Lhs[i]); base != "" && nm != "" {
				alias[base] = append(alias[base], nm)
			}
		}
		return true
	})
	var out []finding
	seen := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || !fanoutReg[f.Name.Name][calleeName(call.Fun)] {
			return true
		}
		count, body := countArgIdent(call)
		if body == nil {
			return true
		}
		for nm, size := range sized {
			if seen[nm] || returned[nm] || !productMentions(size, count) {
				continue
			}
			used := mentionsIdent(body, nm)
			for _, a := range alias[nm] {
				used = used || mentionsIdent(body, a)
			}
			if !used {
				continue
			}
			seen[nm] = true
			out = append(out, finding{
				pos:      fset.Position(pos[nm]),
				category: "whole-tensor-staging-buffer",
				msg: fmt.Sprintf("%q is sized by %q — the fan-out's ITEM COUNT — times a width, and"+
					" is only touched inside the callback. Each band writes and reads only its own"+
					" rows, so the buffer's SIZE is set by the whole tensor while its WORKING SET is"+
					" one band: every element the producing stage writes goes out to memory and comes"+
					" back for the consuming stage. Size it per band or per chunk instead, hand each"+
					" band its own window, and the intermediate stays in cache. MEASURED on conv2d,"+
					" whose im2col column matrix was rows x k: im2col replicates each input element"+
					" kh*kw times, so one 512x512 head convolution materialized 138 MB to multiply it"+
					" by a 66-element weight vector — chunked to an L2-resident window the largest"+
					" torch conv shape went -11%%, the attention forward -12%%, and B/op fell 62-88%%."+
					" CHECK WHETHER THE CONSUMER ACCUMULATES: a whole-tensor buffer writes each row's"+
					" slot exactly once, so an accumulating consumer (C[i] += …) reads as a store and"+
					" the pool's zeroing is invisible; a REUSED window must be cleared between chunks"+
					" or every chunk after the first adds onto the last one's result. CAP THE CHUNK AT"+
					" ONE BAND: sized purely by a cache target, the total becomes workers x chunk,"+
					" which is MORE memory than before on inputs too small to have had a problem",
					nm, count),
			})
		}
		return true
	})
	return out
}

// --- PS3043: a source rebound once per output ------------------------------------------------

// boundIn returns the names defined by := directly in body's own statement list, skipping the
// statement to exclude (the nested loop whose body is examined separately).
func boundIn(body *ast.BlockStmt, skip ast.Stmt) map[string]bool {
	out := map[string]bool{}
	for _, st := range body.List {
		if st == skip {
			continue
		}
		as, ok := st.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE {
			continue
		}
		for _, lhs := range as.Lhs {
			if nm := identName(lhs); nm != "" {
				out[nm] = true
			}
		}
	}
	return out
}

// writtenElementOf reports whether body assigns to an element of one of names, which is what
// makes that name the destination this loop level owns.
func writtenElementOf(body ast.Node, names map[string]bool) string {
	hit := ""
	ast.Inspect(body, func(n ast.Node) bool {
		if hit != "" {
			return false
		}
		as, ok := n.(*ast.AssignStmt)
		if !ok {
			return true
		}
		for _, lhs := range as.Lhs {
			ix, ok := unparen(lhs).(*ast.IndexExpr)
			if !ok {
				continue
			}
			if nm := identName(ix.X); names[nm] {
				hit = nm
				return false
			}
		}
		return true
	})
	return hit
}

// sourceReboundBy returns the name of a value bound inside body from an expression that INDEXES a
// collection with the inner variable and never mentions the outer one. Bound there, it is
// recomputed in full for every outer iteration.
func sourceReboundBy(body *ast.BlockStmt, inner, outer string) (name, src string) {
	for _, st := range body.List {
		as, ok := st.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			continue
		}
		if mentionsIdent(as.Rhs[0], outer) {
			continue
		}
		var ix *ast.IndexExpr
		ast.Inspect(as.Rhs[0], func(n ast.Node) bool {
			if e, ok := n.(*ast.IndexExpr); ok && ix == nil && mentionsIdent(e.Index, inner) {
				ix = e
			}
			return ix == nil
		})
		if ix == nil {
			continue
		}
		if nm := identName(as.Lhs[0]); nm != "" {
			return nm, renderExpr(ix)
		}
	}
	return "", ""
}

// sourceReboundPerOutputFindings flags PS3043 — a nest whose OUTER loop owns a destination and
// whose INNER loop rebinds a source element from a collection. The set of sources is re-read once
// per outer iteration although it does not depend on the outer variable, so the pass moves
// outer-count times the source volume. Interchanging the loops reads each source once.
//
// MEASURED on the multi-token-attention head mix, out[o] = sum_p w[o,p]*maps[p], written as a loop
// over outputs re-reading every map: 16 passes over 33 MB per group. With the source loop outside
// and the element range split across workers, BenchmarkMTAForward_ch16 fell 11.7%.
func sourceReboundPerOutputFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	seen := map[int]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		obody, outer := outerLoop(n)
		if obody == nil || outer == "" {
			return true
		}
		for _, st := range obody.List {
			pbody, inner := outerLoop(st)
			if pbody == nil || inner == "" || inner == outer {
				continue
			}
			// The interchange is only worth naming when the innermost work is itself a loop:
			// otherwise the source is one element and rebinding it costs nothing.
			if loopDepthOf(pbody) < 1 {
				continue
			}
			dst := writtenElementOf(pbody, boundIn(obody, st))
			if dst == "" {
				continue
			}
			src, from := sourceReboundBy(pbody, inner, outer)
			if src == "" || seen[int(n.Pos())] {
				continue
			}
			seen[int(n.Pos())] = true
			out = append(out, finding{
				pos:      fset.Position(n.Pos()),
				category: "source-rebound-per-output",
				msg: fmt.Sprintf("this loop over %q owns the destination %q, and the %q loop inside"+
					" it rebinds %q from %s — a source that does not depend on %q. So the whole set of"+
					" sources is re-read once per %q, and the pass moves (count of %q) times the"+
					" source volume through the caches. INTERCHANGE THE LOOPS: put %q outside, bind"+
					" each source once, and update every destination while it is loaded. The"+
					" accumulation order per destination element is unchanged by the interchange, so"+
					" the result stays BIT-IDENTICAL and an existing exact-equality gate still"+
					" applies — that is what makes this cheap to verify. MEASURED on the"+
					" multi-token-attention head mix, out[o] = sum_p w[o,p]*maps[p] written as a loop"+
					" over outputs re-reading every map: 16 passes over 33 MB per group, and"+
					" BenchmarkMTAForward_ch16 fell 11.7%% once the source loop moved out and the"+
					" element range was split across workers. THE INTERCHANGE ALONE IS RARELY THE"+
					" WIN: it makes the destinations live simultaneously, so hold them for a BAND of"+
					" elements rather than the whole array, and check whether the loop was serial"+
					" while the rest of the path was parallel — a serial stretch costs its full CPU"+
					" time in wall clock however small its profile share looks",
					outer, dst, inner, src, from, outer, outer, outer, inner),
			})
			return true
		}
		return true
	})
	return out
}

// --- PS3044: a serial reduction holding an expensive map serial --------------------------------

// reductionTargetIn returns the name of a collection this body accumulates into at an index that
// does NOT mention the loop variable — cnt[b]++ or sums[b][t] += x. That is the write that makes
// the whole loop serial: two items can land on the same slot.
func reductionTargetIn(body ast.Node, iv string) string {
	hit := ""
	ast.Inspect(body, func(n ast.Node) bool {
		if hit != "" {
			return false
		}
		var lhs ast.Expr
		switch x := n.(type) {
		case *ast.AssignStmt:
			if len(x.Lhs) != 1 || (x.Tok != token.ADD_ASSIGN && x.Tok != token.SUB_ASSIGN) {
				return true
			}
			lhs = x.Lhs[0]
		case *ast.IncDecStmt:
			lhs = x.X
		default:
			return true
		}
		// Peel index expressions down to the base collection, checking every index on the way.
		e := unparen(lhs)
		depth := 0
		for {
			ix, ok := e.(*ast.IndexExpr)
			if !ok {
				break
			}
			if mentionsIdent(ix.Index, iv) {
				return true // indexed by the item: a per-item write, not a reduction
			}
			e = unparen(ix.X)
			depth++
		}
		if depth == 0 {
			return true
		}
		if nm := identName(e); nm != "" {
			hit = nm
		}
		return hit == ""
	})
	return hit
}

// pureValueCallIn returns the name a loop body binds from a CALL that takes the item and nothing
// the loop mutates — the expensive per-item work sitting in front of the reduction.
func pureValueCallIn(body *ast.BlockStmt, iv string) (name, callee string) {
	for _, st := range body.List {
		as, ok := st.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			continue
		}
		call, ok := unparen(as.Rhs[0]).(*ast.CallExpr)
		if !ok || len(call.Args) == 0 {
			continue
		}
		if nm := identName(as.Lhs[0]); nm != "" {
			return nm, calleeName(call.Fun)
		}
	}
	return "", ""
}

// mapBlockedByReductionFindings flags PS3044 — a loop whose body computes a per-item value with a call
// and then folds it into shared state at an index the item does not determine. The fold is what
// keeps the loop serial, and the map in front of it is usually where all the time is.
//
// MEASURED on the AQLM encoder's k-means assignment: finding a point's nearest centroid costs
// k*dim and is pure, folding it into that centroid's running sum costs dim and collides. Computing
// the assignments into an array in a parallel pass and folding them afterwards in ascending item
// order left every sum accumulating exactly the same terms in the same order — bit-identical —
// and BenchmarkEncodeAQLM fell 37.2%.
func mapBlockedByReductionFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || len(fanoutReg[f.Name.Name]) == 0 {
		return nil
	}
	var out []finding
	seen := map[int]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		body, iv := outerLoop(n)
		if body == nil || iv == "" || seen[int(n.Pos())] {
			return true
		}
		// Only the innermost item loop: an outer loop whose body is another loop is a different
		// shape, and reporting both buries the one that matters.
		if loopDepthOf(body) > 1 {
			return true
		}
		// A loop that already hands its work to a fan-out helper is the APPLIED form. Without this
		// the check reported the ENCLOSING loop of a site it had just been used to fix: moving the
		// inner loop into a closure took it out of the nesting count while its accumulating write
		// stayed visible.
		fanned := false
		ast.Inspect(body, func(m ast.Node) bool {
			if c, ok := m.(*ast.CallExpr); ok && fanoutReg[f.Name.Name][calleeName(c.Fun)] {
				fanned = true
			}
			return !fanned
		})
		if fanned {
			return true
		}
		val, callee := pureValueCallIn(body, iv)
		if val == "" {
			return true
		}
		tgt := reductionTargetIn(body, iv)
		if tgt == "" || tgt == val {
			return true
		}
		seen[int(n.Pos())] = true
		out = append(out, finding{
			pos:      fset.Position(n.Pos()),
			category: "serial-reduction-blocks-parallel-map",
			msg: fmt.Sprintf("this loop computes %q with a call to %s and then folds it into %q at an"+
				" index %q does not determine. That fold is a REDUCTION — two items can land on the"+
				" same slot — so the loop cannot fan out as written, and it is usually the call in"+
				" front of it that holds all the time. SPLIT THE MAP FROM THE REDUCE: run the call"+
				" over the items in a parallel pass that writes a per-item array, then fold that"+
				" array in ascending item order exactly as now. No partial sums are merged and every"+
				" accumulator takes the same terms in the same order, so the result is BIT-IDENTICAL"+
				" and a golden from the previous implementation gates it. MEASURED on the AQLM"+
				" encoder's k-means assignment, where the nearest-centroid search costs k*dim and the"+
				" fold costs dim: BenchmarkEncodeAQLM fell 37.2%%. RANK IT AGAINST WALL CLOCK — a"+
				" serial stretch pays its full CPU time while the rest of the path is parallel, so a"+
				" CPU-profile share understates it by the parallelism factor. If the map is CHEAP"+
				" relative to the fold there is nothing here: the extra array and the second pass are"+
				" then the whole cost",
				val, callee, tgt, iv),
		})
		return true
	})
	return out
}

// --- PS3045: a colliding scatter whose destination a loop dimension partitions -----------------

// stridedNamesIn returns the names a loop body advances by a loop-invariant step — the
// strength-reduced form of an index that is affine in the loop variable. `base += nb` inside the
// feature loop is the same addressing as `f*nb`, and a check that only looked for the loop
// variable in the index would miss every hand-optimized site.
func stridedNamesIn(body *ast.BlockStmt, iv string) map[string]bool {
	out := map[string]bool{}
	ast.Inspect(body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.ADD_ASSIGN || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		if mentionsIdent(as.Rhs[0], iv) {
			return true // a step that varies with the loop is not a constant stride
		}
		if nm := identName(as.Lhs[0]); nm != "" {
			out[nm] = true
		}
		return true
	})
	return out
}

// dataDependentSum reports whether e is a SUM of a term affine in the loop dimension (the loop
// variable itself or a constant-stride accumulator) and a term read out of a slice — the
// signature of dst[f*stride + bin[i]], a scatter whose column is chosen by the loop and whose
// offset within it is chosen by the data.
func dataDependentSum(e ast.Expr, iv string, strided map[string]bool) bool {
	b, ok := unparen(e).(*ast.BinaryExpr)
	if !ok || b.Op != token.ADD {
		return false
	}
	affine := func(x ast.Expr) bool {
		if mentionsIdent(x, iv) {
			return true
		}
		for nm := range strided {
			if mentionsIdent(x, nm) {
				return true
			}
		}
		return false
	}
	dataDep := func(x ast.Expr) bool {
		found := false
		ast.Inspect(x, func(n ast.Node) bool {
			if _, ok := n.(*ast.IndexExpr); ok {
				found = true
			}
			return !found
		})
		return found
	}
	return (affine(b.X) && dataDep(b.Y)) || (affine(b.Y) && dataDep(b.X))
}

// collidingScatterFindings flags PS3045 — an item loop whose inner loop over a second dimension
// accumulates into a destination indexed by that dimension PLUS a data-dependent offset. Two items
// can land on the same slot, so the item loop cannot fan out; but the second dimension partitions
// the destination into disjoint windows, and splitting THERE is safe and exact.
//
// MEASURED on the histogram gradient boosting builder, where every sample updates one bin of every
// feature. Splitting the feature range gave each worker a private window while each bin still
// accumulated its samples in ascending sample order — bit-identical — and BenchmarkGBMHist_hist_80k
// fell 19.0%, the 20k cell 11.3%.
func collidingScatterFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || len(fanoutReg[f.Name.Name]) == 0 {
		return nil
	}
	var out []finding
	seen := map[int]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		obody, item := outerLoop(n)
		if obody == nil || item == "" || seen[int(n.Pos())] {
			return true
		}
		fanned := false
		ast.Inspect(obody, func(m ast.Node) bool {
			if c, ok := m.(*ast.CallExpr); ok && fanoutReg[f.Name.Name][calleeName(c.Fun)] {
				fanned = true
			}
			return !fanned
		})
		if fanned {
			return true
		}
		for _, st := range obody.List {
			ibody, dim := outerLoop(st)
			if ibody == nil || dim == "" || dim == item {
				continue
			}
			// A dimension loop whose bounds come from this function's PARAMETERS is the APPLIED
			// form: the caller already hands each worker a band of it. Without this the check went
			// on reporting the band function it had just been used to produce, because that
			// function is reached through a raw goroutine rather than a registered fan-out helper.
			if boundedByParams(fn, st) {
				continue
			}
			strided := stridedNamesIn(ibody, dim)
			// The index may be written inline or bound to a local first.
			idxOf := map[string]ast.Expr{}
			for _, s := range ibody.List {
				if as, ok := s.(*ast.AssignStmt); ok && as.Tok == token.DEFINE && len(as.Lhs) == 1 && len(as.Rhs) == 1 {
					if nm := identName(as.Lhs[0]); nm != "" {
						idxOf[nm] = as.Rhs[0]
					}
				}
			}
			dst := ""
			ast.Inspect(ibody, func(m ast.Node) bool {
				if dst != "" {
					return false
				}
				var lhs ast.Expr
				switch x := m.(type) {
				case *ast.AssignStmt:
					if len(x.Lhs) != 1 || (x.Tok != token.ADD_ASSIGN && x.Tok != token.SUB_ASSIGN) {
						return true
					}
					lhs = x.Lhs[0]
				case *ast.IncDecStmt:
					lhs = x.X
				default:
					return true
				}
				e := unparen(lhs)
				if sel, ok := e.(*ast.SelectorExpr); ok { // h[c].sum += y
					e = unparen(sel.X)
				}
				ix, ok := e.(*ast.IndexExpr)
				if !ok {
					return true
				}
				idx := ix.Index
				if nm := identName(idx); nm != "" && idxOf[nm] != nil {
					idx = idxOf[nm]
				}
				if dataDependentSum(idx, dim, strided) {
					dst = renderExpr(ix.X)
				}
				return dst == ""
			})
			if dst == "" {
				continue
			}
			seen[int(n.Pos())] = true
			out = append(out, finding{
				pos:      fset.Position(n.Pos()),
				category: "colliding-scatter-with-partitionable-destination",
				msg: fmt.Sprintf("this loop over %q accumulates into %q at an index made of the %q"+
					" dimension PLUS a data-dependent offset. Two items can land on the same slot, so"+
					" the ITEM loop cannot fan out — but %q partitions %q into disjoint windows, and"+
					" splitting THERE is safe. SPLIT THE INNER DIMENSION, NOT THE ITEMS: each worker"+
					" owns whole windows, every slot still accumulates its items in ascending item"+
					" order, and the result is BIT-IDENTICAL. Per-worker partial copies merged"+
					" afterwards would reassociate every slot's sum and are not. MEASURED on the"+
					" histogram gradient-boosting builder, where every sample updates one bin of every"+
					" feature: BenchmarkGBMHist_hist_80k fell 19.0%%, the 20k cell 11.3%%. FLOOR THE"+
					" WINDOWS PER WORKER — every worker re-walks the WHOLE item list, so the per-item"+
					" cost of that walk is paid once per worker instead of once in total; swept at 20"+
					" features on 12 cores, 4 per worker was best and 8 was 10%% worse because it left"+
					" only two workers. GATE ON items times windows, so small calls stay serial",
					item, dst, dim, dim, dst),
			})
			return true
		}
		return true
	})
	return out
}

// boundedByParams reports whether the loop st takes its start or its limit from a parameter of fn
// or of any function literal inside it — the signature of a range the caller has already
// partitioned. The literal half is load-bearing: a fan-out callback's (lo, hi) band is exactly
// this, and a loop inside such a callback is the APPLIED form of the transform this check
// recommends.
func boundedByParams(fn *ast.FuncDecl, st ast.Stmt) bool {
	fs, ok := st.(*ast.ForStmt)
	if !ok {
		return false
	}
	params := map[string]bool{}
	if fn.Type.Params != nil {
		for _, p := range fn.Type.Params.List {
			for _, nm := range p.Names {
				params[nm.Name] = true
			}
		}
	}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		lit, ok := n.(*ast.FuncLit)
		if !ok || lit.Type.Params == nil {
			return true
		}
		for _, p := range lit.Type.Params.List {
			for _, nm := range p.Names {
				params[nm.Name] = true
			}
		}
		return true
	})
	// Only the START matters. A loop whose LIMIT is a parameter is just being told how big the
	// dimension is — `for f := 0; f < d; f++` with d passed in is the unsplit form, and treating
	// that as banded silenced the check on the shape it was built for. A caller-supplied OFFSET
	// is the actual banding signal.
	as, ok := fs.Init.(*ast.AssignStmt)
	if !ok {
		return false
	}
	for nm := range params {
		for _, r := range as.Rhs {
			if mentionsIdent(r, nm) {
				return true
			}
		}
	}
	return false
}

// --- PS3046: an item reduction into windows the inner loops already partition -----------------

// windowedAccumIn returns the name of a slice WINDOW cut inside body from a base that does not
// depend on the item, and accumulated into. `g := grams[q*mm+aoff : q*mm+aend]` followed by
// `g[j] += …` is a reduction over items whose destination the inner loops have already carved
// into disjoint pieces.
func windowedAccumIn(body ast.Node, item string) (win, base string) {
	// Names bound in the body from an expression that mentions the item — the strength-reduced
	// offsets a hand-optimized loop uses. `aoff := a*mAug + a` makes `grams[q*mm+aoff : …]` look
	// independent of a to a purely syntactic test, which is exactly backwards.
	derived := map[string]bool{item: true}
	for range 3 { // a short fixed point: offsets are rarely more than a couple of steps deep
		ast.Inspect(body, func(n ast.Node) bool {
			as, ok := n.(*ast.AssignStmt)
			if !ok || as.Tok != token.DEFINE || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
				return true
			}
			for nm := range derived {
				if mentionsIdent(as.Rhs[0], nm) {
					if lhs := identName(as.Lhs[0]); lhs != "" {
						derived[lhs] = true
					}
				}
			}
			return true
		})
	}
	dependsOnItem := func(e ast.Expr) bool {
		if e == nil {
			return false
		}
		for nm := range derived {
			if mentionsIdent(e, nm) {
				return true
			}
		}
		return false
	}
	cuts := map[string]string{}
	ast.Inspect(body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		se, ok := unparen(as.Rhs[0]).(*ast.SliceExpr)
		if !ok || se.Low == nil || dependsOnItem(se.Low) || dependsOnItem(se.High) {
			return true
		}
		if nm := identName(as.Lhs[0]); nm != "" {
			cuts[nm] = renderExpr(se.X)
		}
		return true
	})
	if len(cuts) == 0 {
		return "", ""
	}
	ast.Inspect(body, func(n ast.Node) bool {
		if win != "" {
			return false
		}
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.ADD_ASSIGN || len(as.Lhs) != 1 {
			return true
		}
		ix, ok := unparen(as.Lhs[0]).(*ast.IndexExpr)
		if !ok || dependsOnItem(ix.Index) {
			return true
		}
		if nm := identName(ix.X); cuts[nm] != "" {
			win, base = nm, cuts[nm]
		}
		return win == ""
	})
	return win, base
}

// itemReductionWindowFindings flags PS3046 — a loop over items whose inner loops accumulate into a
// WINDOW of a shared destination, cut at an offset the item does not appear in. The item loop is a
// reduction and cannot fan out; the dimensions that choose the window can, and splitting there
// keeps every window summing its items in ascending order.
//
// MEASURED on the softmax regression Hessian, where every sample contributes to every
// (class pair, feature) window: BenchmarkSoftmaxRegressionFit fell 43.9% and the two smaller
// softmax cells 24.5% and 26.6%.
func itemReductionWindowFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil {
		return nil
	}
	var out []finding
	seen := map[int]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		body, item := outerLoop(n)
		if body == nil || item == "" || seen[int(n.Pos())] || loopDepthOf(body) < 2 {
			return true
		}
		fanned := false
		ast.Inspect(body, func(m ast.Node) bool {
			c, ok := m.(*ast.CallExpr)
			if ok && (fanoutReg[f.Name.Name][calleeName(c.Fun)] || calleeName(c.Fun) == "Wait") {
				fanned = true
			}
			return !fanned
		})
		if fanned {
			return true
		}
		// If any loop under this one already STARTS at a caller-supplied offset, the dimension that
		// chooses the window is banded and this is the APPLIED form. Reporting it anyway is the
		// failure mode that teaches readers to ignore a check: the band function is reached through
		// a raw goroutine, so nothing else about it looks parallel.
		banded := false
		ast.Inspect(body, func(m ast.Node) bool {
			if fs, ok := m.(*ast.ForStmt); ok && boundedByParams(fn, fs) {
				banded = true
			}
			return !banded
		})
		if banded {
			return true
		}
		win, base := windowedAccumIn(body, item)
		if win == "" {
			return true
		}
		seen[int(n.Pos())] = true
		out = append(out, finding{
			pos:      fset.Position(n.Pos()),
			category: "item-reduction-into-partitioned-windows",
			msg: fmt.Sprintf("this loop over %q accumulates into %q, a window of %q cut at an offset"+
				" %q does not appear in. The item loop is a REDUCTION — every item touches every"+
				" window — so it cannot fan out; but the loops that CHOOSE the window already"+
				" partition %q into disjoint pieces, and splitting one of those is safe. Each worker"+
				" then owns whole windows and every window still sums its items in ascending order,"+
				" so the result is BIT-IDENTICAL and a golden from the previous implementation gates"+
				" it. MEASURED on the softmax regression Hessian, where every sample contributes to"+
				" every (class pair, feature) window: BenchmarkSoftmaxRegressionFit fell 43.9%%, and"+
				" the two smaller softmax cells 24.5%% and 26.6%%. CUT THE BANDS ON CUMULATIVE WORK"+
				" WHEN THE INNER RANGE IS TRIANGULAR — a loop whose iteration a writes m-a columns"+
				" gives its first band about 2m/workers times the last band's work under an"+
				" equal-count split, and the makespan is the first band's. CHECK THE GATE AGAINST THE"+
				" REAL SHAPE: this one first measured as no change at all because the work estimate"+
				" fell 4%% short of the threshold and the split never ran",
				item, win, base, item, base),
		})
		return true
	})
	return out
}

// --- PS3047: one shared accumulator away from a split -----------------------------------------

// accumTargets splits a body's accumulating writes into those whose index mentions the loop
// dimension — private to that iteration — and those whose does not, which every iteration shares.
// Locals the body itself creates are excluded: they are per-iteration scratch either way.
func accumTargets(body *ast.BlockStmt, dim string) (private, shared []string) {
	local := localBuffersMadeIn(body)
	// Names the body derives FROM the dimension. `hc := h*dh` and then `dvcs[j*cols+hc+d]` is a
	// head-private write, and a purely syntactic test for h calls it shared — which inverts the
	// finding. Without this the check was silent on the site it was written for.
	derived := map[string]bool{dim: true}
	for range 3 {
		ast.Inspect(body, func(n ast.Node) bool {
			as, ok := n.(*ast.AssignStmt)
			if !ok || as.Tok != token.DEFINE || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
				return true
			}
			for nm := range derived {
				if mentionsIdent(as.Rhs[0], nm) {
					if lhs := identName(as.Lhs[0]); lhs != "" {
						derived[lhs] = true
					}
				}
			}
			return true
		})
	}
	mentionsDim := func(e ast.Expr) bool {
		for nm := range derived {
			if mentionsIdent(e, nm) {
				return true
			}
		}
		return false
	}
	seenP, seenS := map[string]bool{}, map[string]bool{}
	ast.Inspect(body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 {
			return true
		}
		// A read-modify-write spelled as an explicit rounded store is how an f32 accumulator has
		// to be written when each add must round through float32; it accumulates just as much as
		// += does, and skipping it made this check silent on the f32 arm of the very function it
		// was written for. The test has to be narrow: `a[j] = math.Exp(a[j] - m)` also reads its
		// own cell, and admitting it once suppressed the finding entirely and once produced a
		// second, wrong finding on the nested loop.
		roundStore := as.Tok == token.ASSIGN && len(as.Rhs) == 1 &&
			isRoundedAccumulation(as.Lhs[0], as.Rhs[0])
		if as.Tok != token.ADD_ASSIGN && as.Tok != token.SUB_ASSIGN && !roundStore {
			return true
		}
		e := unparen(as.Lhs[0])
		if sel, ok := e.(*ast.SelectorExpr); ok {
			e = unparen(sel.X)
		}
		ix, ok := e.(*ast.IndexExpr)
		if !ok {
			return true
		}
		nm := identName(ix.X)
		if nm == "" || local[nm] {
			return true
		}
		if mentionsDim(ix.Index) {
			if !seenP[nm] {
				seenP[nm] = true
				private = append(private, nm)
			}
			return true
		}
		if !seenS[nm] {
			seenS[nm] = true
			shared = append(shared, nm)
		}
		return true
	})
	return private, shared
}

// oneSharedAccumulatorFindings flags PS3047 — a loop over a dimension whose accumulating writes
// are ALL indexed by that dimension except one. The exception is the only thing keeping the loop
// serial, and it does not have to: record the shared accumulator's factor during the parallel pass
// and fold it afterwards in the original order.
//
// MEASURED on the MLA attention backward, where four of five gradients are written at columns the
// head chooses and the fifth, the shared decoupled-key gradient, is accumulated by every head at
// the same address. Recording its factors and folding them in ascending (head, query, key) order
// took BenchmarkMLAVJPSeq256 down 67.9% and Seq128 66.7%, bit-identically.
func oneSharedAccumulatorFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	seen := map[int]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		body, dim := outerLoop(n)
		if body == nil || dim == "" || seen[int(n.Pos())] || loopDepthOf(body) < 2 {
			return true
		}
		fanned := false
		ast.Inspect(body, func(m ast.Node) bool {
			if fs, ok := m.(*ast.ForStmt); ok && boundedByParams(fn, fs) {
				fanned = true
			}
			if c, ok := m.(*ast.CallExpr); ok && f.Name != nil && fanoutReg[f.Name.Name][calleeName(c.Fun)] {
				fanned = true
			}
			return !fanned
		})
		if fanned {
			return true
		}
		private, shared := accumTargets(body, dim)
		// The point is a MAJORITY that already partitions. One private destination against three
		// shared ones is an ordinary reduction, not a loop held back by an exception.
		if len(private) < 2 || len(shared) != 1 {
			return true
		}
		seen[int(n.Pos())] = true
		out = append(out, finding{
			pos:      fset.Position(n.Pos()),
			category: "one-shared-accumulator-blocks-split",
			msg: fmt.Sprintf("this loop over %q accumulates into %d destinations that its own"+
				" iteration owns (%s — every index mentions %q) and exactly ONE it shares with every"+
				" other iteration: %q. That single exception is the only thing keeping the loop"+
				" serial. RECORD AND FOLD: run the iterations in parallel, storing the shared"+
				" accumulator's per-item FACTOR into a buffer instead of adding it, then fold that"+
				" buffer afterwards in the original iteration order. Every add then happens in the"+
				" sequence the serial loop used, so the result is BIT-IDENTICAL — per-iteration"+
				" partial sums merged at the end are NOT, because the serial form is one running sum"+
				" and a partial restarts from zero. MEASURED on the MLA attention backward, where"+
				" four of five gradients are written at head-chosen columns and the fifth is the"+
				" shared decoupled-key gradient: BenchmarkMLAVJPSeq256 fell 67.9%% and Seq128 66.7%%."+
				" SIZE THE RECORDING BUFFER FIRST — it holds one value per (iteration, inner index)"+
				" pair, so process the iterations in GROUPS that keep it under a cache-or-memory"+
				" budget; a group of one degrades to the serial form and stays correct. THE FOLD"+
				" MUST REPRODUCE THE ORIGINAL BOUNDS: a triangular inner range folded with the"+
				" rectangular bound agrees with the serial form on the rectangular case and nowhere"+
				" else",
				dim, len(private), strings.Join(private, ", "), dim, shared[0]),
		})
		return true
	})
	return out
}

// baseOfIndex peels index and selector layers off e and returns the underlying expression.
func baseOfIndex(e ast.Expr) ast.Expr {
	for {
		switch x := unparen(e).(type) {
		case *ast.IndexExpr:
			e = x.X
		case *ast.SelectorExpr:
			e = x.X
		default:
			return unparen(e)
		}
	}
}

// peelConversions strips type conversions — calls whose callee is a bare identifier — so a value
// wrapped in float64(...) can be compared with the cell it came from. A qualified call such as
// math.Exp is NOT a conversion and stops the peel, which is what separates an f32 accumulator
// from ordinary scratch that happens to read its own cell.
func peelConversions(e ast.Expr) ast.Expr {
	for {
		call, ok := unparen(e).(*ast.CallExpr)
		if !ok || len(call.Args) != 1 {
			return unparen(e)
		}
		if _, ok := call.Fun.(*ast.Ident); !ok {
			return unparen(e)
		}
		e = call.Args[0]
	}
}

// isRoundedAccumulation reports whether rhs is lhs plus (or minus) something, up to conversions:
// the explicit-store spelling of an accumulation.
func isRoundedAccumulation(lhs, rhs ast.Expr) bool {
	b, ok := peelConversions(rhs).(*ast.BinaryExpr)
	if !ok || (b.Op != token.ADD && b.Op != token.SUB) {
		return false
	}
	want := renderExpr(lhs)
	return renderExpr(peelConversions(b.X)) == want || renderExpr(peelConversions(b.Y)) == want
}

// --- PS3048: a fan-out helper with no floor on the work each worker gets ----------------------

// dividesBy reports whether body contains a division whose dividend mentions one of names — the
// shape of "how many workers does this much work justify", as opposed to "how many cores are
// there".
func dividesBy(body ast.Node, names, procs map[string]bool) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		b, ok := n.(*ast.BinaryExpr)
		if !ok || b.Op != token.QUO || found {
			return !found
		}
		// A division BY the worker count is the chunk size, not a decision about how many
		// workers to wake. Counting it hid the finding on the pool this check was written for,
		// whose every version computes chunk := (n + workers - 1) / workers.
		for nm := range procs {
			if mentionsIdent(b.Y, nm) {
				return true
			}
		}
		for nm := range names {
			if mentionsIdent(b.X, nm) {
				found = true
			}
		}
		return !found
	})
	return found
}

// fanoutWorkFloorFindings flags PS3048 — a fan-out helper that takes its worker count from
// GOMAXPROCS and gates only on the TOTAL work, with nothing bounding the work each worker
// receives. An op just over the total threshold is then split every way the machine allows, and
// each band carries a fraction of the amount that justified splitting at all.
//
// MEASURED on this repository's CPU worker pool. A per-token decode step issues a long run of ops
// sitting just above the threshold; the profile of BenchmarkLlamaPromptStepwise was 42%
// runtime.usleep, 22% cond_wait and 9% cond_signal against 14% arithmetic, and the benchmark ran
// 2.3x SLOWER at twelve cores than at one. Deriving the worker count as total/floor, capped at
// GOMAXPROCS, took Llama down 36%, Mixtral 44.7% and Mamba prefill 27.6%, with the large-op cells
// unchanged.
func fanoutWorkFloorFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || fn.Name == nil || !fanoutReg[f.Name.Name][fn.Name.Name] {
		return nil
	}
	// The integer parameters are the candidate work quantities, plus anything derived from them.
	work := map[string]bool{}
	if fn.Type.Params != nil {
		for _, p := range fn.Type.Params.List {
			if id, ok := p.Type.(*ast.Ident); !ok || id.Name != "int" {
				continue
			}
			for _, nm := range p.Names {
				work[nm.Name] = true
			}
		}
	}
	if len(work) == 0 {
		return nil
	}
	for range 3 {
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			as, ok := n.(*ast.AssignStmt)
			if !ok || as.Tok != token.DEFINE || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
				return true
			}
			for nm := range work {
				if mentionsIdent(as.Rhs[0], nm) {
					if lhs := identName(as.Lhs[0]); lhs != "" {
						work[lhs] = true
					}
				}
			}
			return true
		})
	}
	// The names holding the core count, so a division BY one of them can be told apart from a
	// division that sizes the fan-out.
	procs := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		call, isCall := unparen(as.Rhs[0]).(*ast.CallExpr)
		derived := false
		for nm := range procs {
			if mentionsIdent(as.Rhs[0], nm) {
				derived = true
			}
		}
		if (isCall && renderExpr(call.Fun) == "runtime.GOMAXPROCS") || derived {
			if nm := identName(as.Lhs[0]); nm != "" {
				procs[nm] = true
			}
		}
		return true
	})
	if len(procs) == 0 || dividesBy(fn.Body, work, procs) {
		return nil
	}
	return []finding{{
		pos:      fset.Position(fn.Pos()),
		category: "fanout-without-a-work-floor",
		msg: fmt.Sprintf("%q hands work to as many workers as GOMAXPROCS reports and never divides"+
			" the work to decide how many are worth waking. A total-work threshold only answers"+
			" whether to fan out AT ALL; without a second gate on the share each worker receives, an"+
			" op just over that threshold is split every way the machine allows and every band"+
			" carries a fraction of the amount that justified splitting once. DERIVE THE COUNT FROM"+
			" THE WORK: workers = min(GOMAXPROCS, total/floor), and fall back to the serial body when"+
			" that comes out at one. MEASURED on this repository's CPU pool, where a per-token decode"+
			" step issues a long run of ops sitting just above the threshold: the profile was 42%%"+
			" runtime.usleep, 22%% cond_wait and 9%% cond_signal against 14%% arithmetic, and"+
			" BenchmarkLlamaPromptStepwise ran 2.3x SLOWER at twelve cores than at one. A floor equal"+
			" to the fan-out threshold itself took Llama down 36%%, Mixtral 44.7%% and Mamba prefill"+
			" 27.6%% with the large-op cells unchanged. THE CURVE IS NOT MONOTONE — swept at 2^14,"+
			" 2^15, 2^16 and 2^17 the times were 153, 85, 110 and 127 ms, so pick the floor by"+
			" measurement and not by argument. CHANGING THE BAND COUNT MUST NOT CHANGE A VALUE: this"+
			" is only safe where every caller's split is already band-count independent, which the"+
			" GOMAXPROCS parity tests are what prove", fn.Name.Name),
	}}
}

// --- PS3049: an axpy that re-reads its destination once per outer iteration -------------------

// derivedNames returns the names body binds from expressions mentioning seed, seed included.
func derivedNames(body ast.Node, seed string) map[string]bool {
	out := map[string]bool{seed: true}
	for range 3 {
		ast.Inspect(body, func(n ast.Node) bool {
			as, ok := n.(*ast.AssignStmt)
			if !ok || as.Tok != token.DEFINE || len(as.Lhs) != len(as.Rhs) {
				return true
			}
			for i, rhs := range as.Rhs {
				for nm := range out {
					if mentionsIdent(rhs, nm) {
						if lhs := identName(as.Lhs[i]); lhs != "" {
							out[lhs] = true
						}
					}
				}
			}
			return true
		})
	}
	return out
}

// axpyReloadFindings flags PS3049 — a nest whose inner loop accumulates into a destination the
// OUTER loop does not choose, from a source the outer loop does. Every outer iteration reads and
// writes the whole destination again, so each element carries a load-modify-store chain through
// memory once per outer step.
//
// MEASURED on the decode matrix-vector kernel, the single hottest function of a generate loop at
// 46.6% of its serial profile. Taking four source rows per pass over the destination, with each
// element's contributions still added ONE AT A TIME in ascending order, left the result
// bit-identical and took the generate benchmarks down 10.8% to 26.7%.
func axpyReloadFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	seen := map[int]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		obody, p := outerLoop(n)
		if obody == nil || p == "" || seen[int(n.Pos())] || alreadyTiled(n) {
			return true
		}
		fromP := derivedNames(obody, p)
		for _, st := range obody.List {
			ibody, j := outerLoop(st)
			if ibody == nil || j == "" || j == p || loopDepthOf(ibody) > 0 {
				continue
			}
			// The inner loop must RANGE OVER the outer loop's own row. That is what makes this an
			// axpy — a full pass over a source slice chosen by the outer step — rather than any
			// nest that happens to accumulate; without it the check reported 95 sites tree-wide,
			// most of them a couple of elements wide and with nothing to unroll.
			rs, ok := st.(*ast.RangeStmt)
			if !ok || !mentionsAnyOf(rs.X, fromP) {
				continue
			}
			var dst string
			ast.Inspect(ibody, func(m ast.Node) bool {
				if dst != "" {
					return false
				}
				as, ok := m.(*ast.AssignStmt)
				if !ok || as.Tok != token.ADD_ASSIGN || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
					return true
				}
				ix, ok := unparen(as.Lhs[0]).(*ast.IndexExpr)
				if !ok || !mentionsIdent(ix.Index, j) {
					return true
				}
				base := identName(ix.X)
				if base == "" || fromP[base] {
					return true // a destination the outer loop chooses: nothing is re-read
				}
				for nm := range fromP { // the source must move with the outer loop
					if nm != p && mentionsIdent(as.Rhs[0], nm) {
						dst = base
						return false
					}
				}
				return true
			})
			if dst == "" {
				continue
			}
			seen[int(n.Pos())] = true
			out = append(out, finding{
				pos:      fset.Position(n.Pos()),
				category: "axpy-reloads-its-destination",
				msg: fmt.Sprintf("the %q loop accumulates into %q, which %q does not choose, from"+
					" operands that move with %q. Every %q iteration therefore reads and writes the"+
					" WHOLE of %q again, and each element carries a load-modify-store chain through"+
					" memory once per %q. UNROLL THE OUTER LOOP AND HOLD THE RUNNING VALUE: take four"+
					" %q steps per pass over %q, load the element once, add each contribution to it,"+
					" store once. FOUR, MEASURED, NOT AS MANY AS FIT: on the decode kernel 1, 4 and 8"+
					" steps per pass came to 823 ms, 682 ms and 937 ms, so past four the scalars and"+
					" row bases stop fitting in registers and the spills cost more than the saved"+
					" trips. THE INNER PASS MUST BE LONG — the extra scalars and row bases are set up"+
					" once per pass and amortized over its length, so a pass of a few dozen elements"+
					" LOSES: the identical four-way transform on an attention value accumulation whose"+
					" inner pass is one head width cost 6 to 9%% on three attention cells and 1.6%% on"+
					" the decode benchmark, and was reverted. ADD THEM ONE AT A TIME, NOT AS A SUM — v += a0*x0;"+
					" v += a1*x1 keeps every element's ascending accumulation order and is"+
					" BIT-IDENTICAL, while summing the products first reassociates and moves the last"+
					" bits, which is the difference between reusing the existing parity surface and"+
					" needing a new tolerance argument. MEASURED on the decode matrix-vector kernel,"+
					" 46.6%% of a generate loop's serial profile: four rows per pass took"+
					" BenchmarkGPTGenerate500RowBuf down 12.6%%, CLA decode 10.8%%, T5 decode 13.5%%"+
					" and the Llama prompt 26.7%%, with the blocked matmul cells untouched. TEST THE"+
					" REMAINDER AND A NON-ZERO WINDOW: an unrolled body that rebuilds its source"+
					" offsets from the loop variable alone agrees with the original on a full-width"+
					" call and on nothing else",
					p, dst, p, p, p, dst, p, p, dst),
			})
			return true
		}
		return true
	})
	return out
}

// --- PS3050: a whole-output pass left serial after the fan-out --------------------------------

// elementwisePassOver returns the name of the slice a statement walks end to end, writing each
// element from the same index of something else — the shape of a narrowing, a cast, or a scale
// applied to a whole buffer.
func elementwisePassOver(st ast.Stmt) (dst, src string) {
	body, iv := outerLoop(st)
	if body == nil || iv == "" || loopDepthOf(body) > 0 {
		return "", ""
	}
	if rs, ok := st.(*ast.RangeStmt); ok && rs.Value != nil {
		return "", "" // ranging the values, not an index pass
	}
	hit, from := "", ""
	ast.Inspect(body, func(n ast.Node) bool {
		if hit != "" {
			return false
		}
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		ix, ok := unparen(as.Lhs[0]).(*ast.IndexExpr)
		if !ok || identName(ix.Index) != iv {
			return true
		}
		// The source has to be indexed by the same variable: an elementwise map, not a reduction
		// or a scatter. Its NAME matters too — the bands usually own the source (an accumulator
		// they filled) rather than the destination, and a check that only looked at the
		// destination was silent on the site it was written for.
		ast.Inspect(as.Rhs[0], func(m ast.Node) bool {
			if e, ok := m.(*ast.IndexExpr); ok && identName(e.Index) == iv && from == "" {
				from = identName(e.X)
			}
			return from == ""
		})
		if from != "" {
			hit = identName(ix.X)
		}
		return hit == ""
	})
	return hit, from
}

// serialTailAfterFanoutFindings flags PS3050 — a fan-out call followed, in the same function, by a
// serial elementwise pass over a whole buffer the bands already own. The pass is an Amdahl term
// bolted onto an otherwise parallel op: every worker finishes, then one goroutine walks the entire
// output again.
//
// MEASURED on the portable f32 matmul, which accumulates into an f64 scratch and then narrowed the
// whole result to f32 after the parallel section. Each band already owns those rows, so converting
// them inside the band is disjoint and element-wise — bit-identical — and took three batched
// vision forwards down 4.7% to 7.6%.
func serialTailAfterFanoutFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || len(fanoutReg[f.Name.Name]) == 0 {
		return nil
	}
	var out []finding
	for i, st := range fn.Body.List {
		es, ok := st.(*ast.ExprStmt)
		if !ok {
			continue
		}
		call, ok := unparen(es.X).(*ast.CallExpr)
		if !ok || !fanoutReg[f.Name.Name][calleeName(call.Fun)] || len(call.Args) == 0 {
			continue
		}
		lit, ok := unparen(call.Args[len(call.Args)-1]).(*ast.FuncLit)
		if !ok || lit.Body == nil {
			continue
		}
		for _, after := range fn.Body.List[i+1:] {
			dst, src := elementwisePassOver(after)
			if dst == "" || (!mentionsIdent(lit.Body, dst) && (src == "" || !mentionsIdent(lit.Body, src))) {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(after.Pos()),
				category: "serial-tail-after-fanout",
				msg: fmt.Sprintf("this loop walks the whole of %q element by element AFTER the"+
					" fan-out above, and the bands already touch %q. Every worker finishes and then"+
					" one goroutine walks the entire buffer again — an Amdahl term bolted onto an"+
					" otherwise parallel op, and one that grows with the output rather than with the"+
					" work. FOLD IT INTO THE BAND: each band owns rows [lo,hi) of the buffer, so"+
					" doing its own slice there is disjoint, elementwise and BIT-IDENTICAL — no"+
					" accumulation order changes because nothing accumulates. MEASURED on the"+
					" portable f32 matmul, which accumulates into an f64 scratch and narrowed the"+
					" whole result afterwards: the tail was 6.4%% of a batched vision forward at one"+
					" worker, and folding it took three of those forwards down 4.7%% to 7.6%% with the"+
					" f64 matmul beside them flat. GATE IT ON A SENTINEL: pre-fill the output with a"+
					" value the correct result cannot produce, so a band that narrows the wrong range"+
					" shows as an untouched cell rather than as a plausible number", dst, dst),
			})
			break
		}
	}
	return out
}

// --- PS3051: a blocked kernel with no guard for its degenerate shape --------------------------

// innermostRangeLen returns the name of the parameter whose value bounds the INNERMOST loop of a
// nest — the dimension the kernel iterates one element at a time.
func innermostRangeLen(body *ast.BlockStmt, params map[string]bool) string {
	hit := ""
	ast.Inspect(body, func(n ast.Node) bool {
		if hit != "" {
			return false
		}
		ib, iv := outerLoop(n)
		if ib == nil || iv == "" || loopDepthOf(ib) > 0 {
			return true
		}
		rs, ok := n.(*ast.RangeStmt)
		if !ok {
			return true
		}
		// The ranged expression is a window whose LENGTH comes from a parameter: B[p*n : p*n+n].
		// It is usually bound to a name first — `bp := B[p*n : (p+1)*n]` and then `range bp` — so
		// a name is resolved back to the slice expression it came from. Without that the check saw
		// no windows at all in the kernel it was written for.
		x := unparen(rs.X)
		if nm := identName(x); nm != "" {
			ast.Inspect(body, func(m ast.Node) bool {
				as, ok := m.(*ast.AssignStmt)
				if !ok || as.Tok != token.DEFINE || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
					return true
				}
				if identName(as.Lhs[0]) == nm {
					x = unparen(as.Rhs[0])
				}
				return true
			})
		}
		se, ok := x.(*ast.SliceExpr)
		if !ok || se.High == nil {
			return true
		}
		for nm := range params {
			if mentionsIdent(se.High, nm) {
				hit = nm
			}
		}
		return hit == ""
	})
	return hit
}

// degenerateShapeGuardFindings flags PS3051 — a kernel that BLOCKS one dimension (an outer loop
// advancing by more than one) while iterating another innermost, with nothing special-cased for
// that inner dimension being 1. At width one the innermost loop runs a single iteration, so the
// block pays all of its slicing and loop machinery to move one element per pass, and the
// accumulators stay in memory when they would fit in registers.
//
// MEASURED on the CPU band GEMM. A conv2d with one output filter reaches it as n == 1, and the
// multi-token-attention head convolution issues thirty-two of those per forward. Holding the four
// blocked accumulators in registers across the whole reduction took a 2048-square matrix-vector
// product down 23.4% in f64 and 36.7% in f32, the conv-shaped 262144x66 case 17.7%, and the
// attention forward 6.3%.
func degenerateShapeGuardFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || fn.Type.Params == nil {
		return nil
	}
	params := map[string]bool{}
	for _, p := range fn.Type.Params.List {
		if id, ok := p.Type.(*ast.Ident); !ok || id.Name != "int" {
			continue
		}
		for _, nm := range p.Names {
			params[nm.Name] = true
		}
	}
	if len(params) == 0 {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if len(out) > 0 || !alreadyTiled(n) { // the blocking loop: it advances by more than one
			return len(out) == 0
		}
		// Take the body straight off the for statement. The shared loop helper wants a DEFINING
		// init, and a blocking loop is routinely written without one — i declared above, then
		// `for ; i+3 < hiRow; i += 4` — which made this check silent on the kernel it was written
		// for.
		fs, _ := n.(*ast.ForStmt)
		if fs == nil || fs.Body == nil {
			return true
		}
		dim := innermostRangeLen(fs.Body, params)
		if dim == "" {
			return true
		}
		// ANY int parameter compared against a literal counts as the function already having a
		// degenerate path. Keying on the reported dimension alone was wrong: the applied form
		// branches on the OTHER dimension (n == 1) and then blocks over rows with the reduction
		// length innermost, so the check reported its own fix. The test is coarse on purpose —
		// this is a "look here" check, and a kernel with no shape branch at all is the signal.
		guarded := false
		ast.Inspect(fn.Body, func(m ast.Node) bool {
			b, ok := m.(*ast.BinaryExpr)
			if !ok || (b.Op != token.EQL && b.Op != token.LSS && b.Op != token.LEQ) {
				return true
			}
			if _, isLit := unparen(b.Y).(*ast.BasicLit); !isLit {
				return true
			}
			for nm := range params {
				if mentionsIdent(b.X, nm) {
					guarded = true
				}
			}
			return !guarded
		})
		if guarded {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(n.Pos()),
			category: "blocked-kernel-without-a-degenerate-shape-guard",
			msg: fmt.Sprintf("this loop BLOCKS its dimension — it advances by more than one — while"+
				" the innermost loop runs over %q, and nothing in the function special-cases %q == 1."+
				" At width one that innermost loop is a single iteration, so the block pays all of"+
				" its slicing and loop machinery to move one element per pass, and the accumulators"+
				" sit in memory when they would fit in registers. ADD THE DEGENERATE PATH: keep the"+
				" same blocking over the other dimension, hold the block's accumulators as SCALARS"+
				" across the whole reduction, and store them once. Bit-identical, since each"+
				" accumulator still takes its terms in the same order into the value it already"+
				" held. MEASURED on the CPU band GEMM, which a conv2d with one output filter reaches"+
				" as n == 1: a 2048-square matrix-vector product fell 23.4%% in f64 and 36.7%% in f32,"+
				" the conv-shaped 262144x66 case 17.7%%, and the attention forward that issues"+
				" thirty-two such convolutions 6.3%%. MEASURE THE OBVIOUS FORM TOO, AND EXPECT IT TO"+
				" LOSE: a plain per-row dot for the degenerate case was tried FIRST and was slightly"+
				" WORSE than the block it replaced, because the block amortizes its loop over four"+
				" rows at once — the win comes from the registers, not from the simpler loop",
				dim, dim),
		})
		return false
	})
	return out
}

// --- PS3052: a staged matrix reduced against a single column ----------------------------------

// stagedThenReduced returns the buffer two consecutive calls share — the first filling it, the
// second consuming it — together with the identifier the second passes as its output width.
func stagedThenReduced(body *ast.BlockStmt) (buf, width string, pos token.Pos) {
	for i := 0; i+1 < len(body.List); i++ {
		first, ok := body.List[i].(*ast.ExprStmt)
		if !ok {
			continue
		}
		fc, ok := unparen(first.X).(*ast.CallExpr)
		if !ok || len(fc.Args) == 0 {
			continue
		}
		fill := identName(fc.Args[0])
		if fill == "" {
			continue
		}
		second, ok := body.List[i+1].(*ast.ExprStmt)
		if !ok {
			continue
		}
		sc, ok := unparen(second.X).(*ast.CallExpr)
		if !ok || len(sc.Args) < 2 || identName(sc.Args[0]) != fill {
			continue
		}
		// The consumer's LAST integer-looking argument is taken as the output width: these
		// kernels are written (A, B, C, lo, hi, k, n).
		w := identName(sc.Args[len(sc.Args)-1])
		if w == "" {
			continue
		}
		return fill, w, fc.Pos()
	}
	return "", "", token.NoPos
}

// stagedSingleColumnFindings flags PS3052 — a buffer filled by one call and immediately consumed
// by another, where the consumer's output width is a variable the function never branches on. At
// width one the staged buffer exists only to be reduced against a single vector: it is written
// once and read once, and the values could have come straight from the source.
//
// MEASURED on conv2d. Its im2col matrix writes c*kh*kw values per output pixel so the GEMM can
// read them back, and with one output filter the fill was 12.7% of the multi-token-attention
// forward against 6.7% for the GEMM it fed. Computing the dot directly, when there is no padding,
// took a 256x256 3x3 single-filter convolution down 22.1% and that forward 3.6%.
func stagedSingleColumnFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if len(out) > 0 {
			return false
		}
		blk, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		buf, width, pos := stagedThenReduced(blk)
		if buf == "" {
			return true
		}
		branched := false
		ast.Inspect(fn.Body, func(m ast.Node) bool {
			b, ok := m.(*ast.BinaryExpr)
			if !ok || (b.Op != token.EQL && b.Op != token.LSS && b.Op != token.LEQ && b.Op != token.GTR) {
				return true
			}
			if _, isLit := unparen(b.Y).(*ast.BasicLit); isLit && mentionsIdent(b.X, width) {
				branched = true
			}
			return !branched
		})
		if branched {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(pos),
			category: "staged-matrix-reduced-against-one-column",
			msg: fmt.Sprintf("%q is filled here and consumed by the very next call, whose output"+
				" width is %q — and nothing in this function branches on %q being 1. At width one"+
				" the staged buffer exists only to be reduced against a SINGLE vector: every value"+
				" is written once and read once, and it could have come straight from the source."+
				" ADD A FUSED PATH FOR WIDTH ONE that computes the reduction directly, visiting the"+
				" same elements in the same order into one accumulator. MEASURED on conv2d, whose"+
				" im2col matrix writes c*kh*kw values per output pixel so the GEMM can read them"+
				" back: with one output filter the fill was 12.7%% of a multi-token-attention forward"+
				" against 6.7%% for the GEMM it fed, and fusing took a 256x256 3x3 single-filter"+
				" convolution down 22.1%% and the forward 3.6%%. MIND THE ZEROS THE STAGING HOLDS:"+
				" conv2d's columns carry zeros for padded taps and the GEMM adds them, so the fused"+
				" path is only exact where there is no padding — skipping an addition of zero is not"+
				" the same operation when the accumulator is negative zero. Gate the fusion on that"+
				" and test BOTH sides of the gate, or a fusion that ignores it passes the unpadded"+
				" case and nothing else. THE GAIN IS IN THE SHORT KERNELS: the same fusion was worth"+
				" 22.1%% at a 3x3 kernel and 4.3%% at 6x11, where the staged read was already"+
				" contiguous and long enough to amortize", buf, width, width),
		})
		return false
	})
	return out
}

// --- PS3053: independent reductions computed one at a time ------------------------------------

// scalarReductionOverShared reports whether body computes a scalar accumulator by looping over a
// source that does NOT depend on the enclosing item variable — one dot product per item, over the
// same data.
func scalarReductionOverShared(body *ast.BlockStmt, item string) (acc, shared string) {
	declared := map[string]bool{}
	for _, st := range body.List {
		if ds, ok := st.(*ast.DeclStmt); ok {
			if gd, ok := ds.Decl.(*ast.GenDecl); ok && gd.Tok == token.VAR {
				for _, sp := range gd.Specs {
					if vs, ok := sp.(*ast.ValueSpec); ok && len(vs.Values) == 0 {
						for _, nm := range vs.Names {
							declared[nm.Name] = true
						}
					}
				}
			}
		}
	}
	if len(declared) == 0 {
		return "", ""
	}
	for _, st := range body.List {
		ib, iv := outerLoop(st)
		if ib == nil || iv == "" || loopDepthOf(ib) > 0 {
			continue
		}
		rs, ok := st.(*ast.RangeStmt)
		if !ok || mentionsIdent(rs.X, item) {
			continue // the source must be shared across items, not selected by one
		}
		src := identName(rs.X)
		if src == "" {
			continue
		}
		hit := ""
		ast.Inspect(ib, func(n ast.Node) bool {
			as, ok := n.(*ast.AssignStmt)
			if !ok || as.Tok != token.ADD_ASSIGN || len(as.Lhs) != 1 {
				return true
			}
			if nm := identName(as.Lhs[0]); declared[nm] {
				hit = nm
			}
			return hit == ""
		})
		if hit != "" {
			return hit, src
		}
	}
	return "", ""
}

// independentReductionsFindings flags PS3053 — a loop over items where each item computes its own
// SCALAR reduction over the same shared source. Every one of those is a dependent add chain, so
// the loop is bound by add LATENCY rather than throughput, and running the items one at a time
// leaves the chains end to end when they could interleave.
//
// This is not PS3010. That one wants to split a SINGLE sum into partials, which reassociates and
// moves the last bits. Here the accumulators belong to DIFFERENT results: interleaving four items
// leaves every sum with its own accumulator and its own ascending order, so it is bit-identical.
//
// MEASURED on the memorizing-attention k-nearest-neighbour scan, whose per-query dot over the head
// width was 44.5% of the benchmark: four queries per pass over the key row took
// BenchmarkMemForward_512 down 34.6%, the 128 cell 22.6% and BenchmarkMemGatherLarge 44.6%.
func independentReductionsFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if len(out) > 0 {
			return false
		}
		// The item loop must sit INSIDE another loop whose iteration supplies the shared source.
		// That nesting is the whole shape: the outer loop streams the data once, the inner loop
		// reduces it separately for each item, and interleaving the items is what stops the data
		// being re-walked one dependent chain at a time. Without this the check reported 111 sites,
		// nearly all of them a single reduction with no sibling to interleave with.
		ob, ov := outerLoop(n)
		if ob == nil || ov == "" {
			return true
		}
		fromOuter := derivedNames(ob, ov)
		found := false
		ast.Inspect(ob, func(m ast.Node) bool {
			if found || m == n {
				return !found
			}
			body, item := outerLoop(m)
			if body == nil || item == "" || alreadyTiled(m) {
				return true
			}
			acc, shared := scalarReductionOverShared(body, item)
			if acc == "" || !fromOuter[shared] {
				return true
			}
			found = true
			out = append(out, mkIndependentReductionFinding(fset, m, item, acc, shared))
			return false
		})
		return !found
	})
	return out
}

// mkIndependentReductionFinding builds the PS3053 report.
func mkIndependentReductionFinding(fset *token.FileSet, n ast.Node, item, acc, shared string) finding {
	{
		return finding{
			pos:      fset.Position(n.Pos()),
			category: "independent-reductions-one-at-a-time",
			msg: fmt.Sprintf("each %q computes the scalar %q by reducing over %q, which every item"+
				" shares. A single-accumulator reduction is a DEPENDENT add chain, so this loop is"+
				" bound by add LATENCY and not by throughput — the multiplies beside it are free —"+
				" and taking the items one at a time leaves those chains end to end when they could"+
				" interleave. TAKE FOUR ITEMS PER PASS over %q with four separate accumulators, and"+
				" load each element of %q once for all four. BIT-IDENTICAL, and that is what"+
				" separates this from PS3010: there the fix splits ONE sum into partials and"+
				" reassociates it, here the accumulators belong to DIFFERENT results and each keeps"+
				" its own terms in its own ascending order. MEASURED on the memorizing-attention"+
				" k-nearest-neighbour scan, whose per-query dot over the head width was 44.5%% of the"+
				" benchmark: BenchmarkMemForward_512 fell 34.6%%, the 128 cell 22.6%% and"+
				" BenchmarkMemGatherLarge 44.6%%. DESIGN THE MUTATION THAT GATES IT WITH CARE: the observable is"+
				" which items the scan SELECTS, so a perturbation that scales or shifts one item's"+
				" scores UNIFORMLY changes no ranking and leaves the oracle green — a 1%% scale and a"+
				" constant offset both did. It has to depend on the shared source's index and be"+
				" large enough to reorder the top of the list; at that point the oracle reddens"+
				" immediately. A last-bit change is genuinely unobservable here, which is a fact"+
				" about the contract rather than a hole in the test"+
				" THE CONVERSE DOES NOT HOLD. This transform wins by removing PASSES, not by"+
				" adding arithmetic, and taking chains AWAY from a reduction that already has"+
				" several buys nothing: independent chains interleave into the latency of one,"+
				" so three cost what one costs. Measured twice on the Jacobi SVD, both slower,"+
				" the second time with no extra pass at all and bit-identical output —"+
				" SVD_192x192 90.4 to 109.3 ms", item, acc, shared, shared, shared),
		}
	}
}

// --- PS3054: one arm of a dtype branch optimized, its twin left behind ------------------------

// typeAssertedBools returns the names a function binds from a TYPE ASSERTION — `_, isF32 :=
// any(q).([]float32)` — the flag a kernel branches on to pick a dtype-specific path.
func typeAssertedBools(fn *ast.FuncDecl) map[string]bool {
	out := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE || len(as.Rhs) != 1 || len(as.Lhs) != 2 {
			return true
		}
		if _, ok := unparen(as.Rhs[0]).(*ast.TypeAssertExpr); !ok {
			return true
		}
		if nm := identName(as.Lhs[1]); nm != "" {
			out[nm] = true
		}
		return true
	})
	return out
}

// hasAccumulationLoop reports whether a block contains a loop that accumulates into a scalar —
// the hand-written reduction an optimized arm replaces with a call.
func hasAccumulationLoop(b *ast.BlockStmt) bool {
	found := false
	ast.Inspect(b, func(n ast.Node) bool {
		if found {
			return false
		}
		body, iv := outerLoop(n)
		if body == nil || iv == "" {
			return true
		}
		ast.Inspect(body, func(m ast.Node) bool {
			as, ok := m.(*ast.AssignStmt)
			if ok && as.Tok == token.ADD_ASSIGN && len(as.Lhs) == 1 {
				if ix, isIx := unparen(as.Lhs[0]).(*ast.IndexExpr); !isIx || !mentionsIdent(ix.Index, iv) {
					found = true
				}
			}
			return !found
		})
		return !found
	})
	return found
}

// asymmetricDtypeArmFindings flags PS3054 — an if/else on a type-assertion flag where ONE arm
// delegates its reduction to a helper and the other spells it out as a scalar loop. That is what a
// half-finished optimization looks like: somebody vectorized or unrolled one dtype and left the
// twin, and because the benchmark suite usually has a cell for only the optimized dtype, nothing
// reported it.
//
// MEASURED twice. The flash-attention f32 scores were unroll-and-jammed over four keys and the f64
// ones were not; adding an f64 cell and matching the arms took it down 35.7%. The retention
// backward had the same split in two places — f32 reducing through dot4T, f64 serial — and
// matching them took BenchmarkRetentionBwdF64 down 25.3% while the f32 cell beside it did not
// move. In both cases the change read as NOISE against the existing f32 benchmark, because a
// change to one arm cannot be seen from a cell that enters the other.
func asymmetricDtypeArmFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	flags := typeAssertedBools(fn)
	if len(flags) == 0 {
		return nil
	}
	// Statement lists in which a GROUPED loop (one advancing by more than one) already precedes
	// something. A dtype branch that survives only in the remainder after such a loop is the
	// APPLIED form: the bulk of the work now runs through a symmetric grouped path, and the last
	// few iterations keep the old split because levelling them would buy nothing.
	inRemainder := map[ast.Stmt]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		blk, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		grouped := false
		for _, st := range blk.List {
			if grouped {
				inRemainder[st] = true
				continue
			}
			if alreadyTiled(st) {
				grouped = true
			}
			if ifs, ok := st.(*ast.IfStmt); ok && ifs.Body != nil {
				for _, in := range ifs.Body.List {
					if alreadyTiled(in) {
						grouped = true
					}
				}
			}
		}
		return true
	})
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		is, ok := n.(*ast.IfStmt)
		if !ok || is.Else == nil || is.Body == nil {
			return true
		}
		remainder := false
		ast.Inspect(fn.Body, func(m ast.Node) bool {
			fs, ok := m.(*ast.ForStmt)
			if ok && inRemainder[fs] && mentionsIdent(fs, "") {
				return true
			}
			if ok && inRemainder[fs] {
				ast.Inspect(fs, func(k ast.Node) bool {
					if k == n {
						remainder = true
					}
					return !remainder
				})
			}
			return !remainder
		})
		if remainder {
			return true
		}
		els, ok := is.Else.(*ast.BlockStmt)
		if !ok || !mentionsAnyOf(is.Cond, flags) {
			return true
		}
		thenLoop, elseLoop := hasAccumulationLoop(is.Body), hasAccumulationLoop(els)
		if thenLoop == elseLoop {
			return true // both spelled out, or both delegated: symmetric either way
		}
		plain, helper := "else", "then"
		if thenLoop {
			plain, helper = "then", "else"
		}
		out = append(out, finding{
			pos:      fset.Position(is.Pos()),
			category: "asymmetric-dtype-arm",
			msg: fmt.Sprintf("this branch turns on a TYPE-ASSERTION flag and its two arms are not"+
				" the same shape: the %s arm spells its reduction out as a scalar loop while the %s"+
				" arm hands the same work to a helper. That is what a half-finished optimization"+
				" looks like — one dtype was unrolled or vectorized and the twin was left — and it"+
				" survives because the benchmark suite usually has a cell for the optimized dtype"+
				" only. BRING THE ARMS LEVEL, and ADD THE MISSING CELL FIRST: a change to one arm"+
				" reads as NOISE against a benchmark that enters the other, which is not a small"+
				" measurement error but a measurement of nothing. MEASURED TWICE. The flash"+
				" attention f32 scores were unroll-and-jammed over four keys and the f64 ones were"+
				" not: against the f32 cell the f64 fix read 7.38 / 7.22 / 7.63 ms, and against an"+
				" f64 cell added for it, -35.7%%. The retention backward had the same split in two"+
				" places and matching them took BenchmarkRetentionBwdF64 down 25.3%% with the f32"+
				" cell beside it unmoved. CHECK WHICH ARM IS BEHIND before assuming: the helper arm"+
				" is usually the faster one, but a helper that reassociates may be gated to a"+
				" tolerance the other dtype does not have, in which case the scalar arm is correct"+
				" and needs an EXACT grouping rather than the same call", plain, helper),
		})
		return true
	})
	return out
}

// --- PS3055: a full sort whose result is immediately truncated --------------------------------

// sortThenTruncate returns the slice a block sorts and then cuts down to a prefix, together with
// the bound it cuts to.
func sortThenTruncate(blk *ast.BlockStmt) (name, bound string, pos token.Pos) {
	for i := 0; i+1 < len(blk.List); i++ {
		es, ok := blk.List[i].(*ast.ExprStmt)
		if !ok {
			continue
		}
		call, ok := unparen(es.X).(*ast.CallExpr)
		if !ok || len(call.Args) == 0 {
			continue
		}
		switch calleeName(call.Fun) {
		case "SortFunc", "SortStableFunc", "Sort", "Slice", "SliceStable":
		default:
			continue
		}
		sorted := identName(call.Args[0])
		if sorted == "" {
			continue
		}
		// The very next statement — possibly guarded by a length test — reslices it to a prefix.
		next := blk.List[i+1]
		if ifs, ok := next.(*ast.IfStmt); ok && ifs.Body != nil && len(ifs.Body.List) == 1 {
			next = ifs.Body.List[0]
		}
		as, ok := next.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 || identName(as.Lhs[0]) != sorted {
			continue
		}
		se, ok := unparen(as.Rhs[0]).(*ast.SliceExpr)
		if !ok || se.Low != nil || se.High == nil || identName(se.X) != sorted {
			continue
		}
		if b := renderExpr(se.High); b != "" && b != "len("+sorted+")" {
			return sorted, b, call.Pos()
		}
	}
	return "", "", token.NoPos
}

// sortThenTruncateFindings flags PS3055 — a slice sorted in full and then cut to a small prefix.
// Everything below the cut was ordered for nothing.
//
// MEASURED on diverse beam search, which sorted every beam's whole vocabulary expansion to keep a
// handful: BenchmarkDiverseBeamSearch/cheap fell 90.5% and /realistic 42.7%.
func sortThenTruncateFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		blk, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		// Every block, not the first: a function can sort-and-cut in a nested loop AND again at
		// the end, and stopping at the first hit reported the cheap outer one while hiding the
		// per-step selection inside the loop, which is the one that costs.
		name, bound, pos := sortThenTruncate(blk)
		if name == "" {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(pos),
			category: "sort-then-truncate",
			msg: fmt.Sprintf("%q is sorted in FULL and then cut to %s. Everything past the cut was"+
				" ordered for nothing: the sort is O(n log n) in the candidates and only %s of them"+
				" survive. SELECT INSTEAD OF SORTING: a bounded worst-at-root heap keeps the best %s"+
				" in O(n log %s), and sorting just those at the end reproduces the prefix exactly."+
				" BIT-IDENTICAL WHEN THE COMPARATOR IS A STRICT TOTAL ORDER — then the kept set and"+
				" its order are uniquely determined, so any method that finds them yields the same"+
				" slice. Check that first: with genuine ties a heap and a sort can disagree about"+
				" WHICH equal element is kept, and the fix needs the tie-break made explicit before"+
				" it is exact. MEASURED on diverse beam search, which sorted every beam's whole"+
				" vocabulary expansion to keep a handful: BenchmarkDiverseBeamSearch/cheap fell"+
				" 90.5%% and /realistic 42.7%%, with plain beam search beside them unmoved. GATE THE"+
				" ORDER, NOT JUST THE SET: leaving the survivors in heap order rather than sorted"+
				" passed every existing test of the measured site, because the final result is"+
				" re-sorted before return and the permutation only shows up through which"+
				" candidates survive the NEXT step. BUILD THE HEAP FROM A COPY if it reuses the"+
				" input's array, or the pushes overwrite entries not yet read", name, bound, bound,
				bound, bound),
		})
		return true
	})
	return out
}

// --- PS3057: a column read through a jagged row-major matrix ----------------------------------

// columnIndexPair returns the row and column index expressions of a nested index x[row][col],
// or nil when e is not one. A double index implies the base is at least [][]T; a single-level
// slice cannot be indexed twice.
func columnIndexPair(e ast.Expr) (row, col ast.Expr) {
	outer, ok := e.(*ast.IndexExpr)
	if !ok {
		return nil, nil
	}
	inner, ok := outer.X.(*ast.IndexExpr)
	if !ok {
		return nil, nil
	}
	return inner.Index, outer.Index
}

// columnReadFindings flags PS3057 — a loop reading one COLUMN of a [][]T, where the row index
// varies with the loop variable and the column index does not.
//
// Restricted to READS on purpose. A column WRITE has the same locality, but a mirror does not
// fix it: the write would have to land in both copies, which costs more than it saves. The
// transform is mirror-once-read-many, so only the many-reads side is reported.
func columnReadFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	seen := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		body, iv := outerLoop(n)
		if body == nil || iv == "" {
			return true
		}
		ast.Inspect(body, func(m ast.Node) bool {
			// Skip the left-hand side of an assignment: a column write is not this shape.
			if as, ok := m.(*ast.AssignStmt); ok {
				for _, r := range as.Rhs {
					ast.Inspect(r, func(k ast.Node) bool { return collectColumnRead(fset, fn, k, iv, seen, &out) })
				}
				return false
			}
			return collectColumnRead(fset, fn, m, iv, seen, &out)
		})
		return true
	})
	return out
}

// containsIndexExpr reports whether e contains an index expression anywhere inside it.
func containsIndexExpr(e ast.Expr) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if _, ok := n.(*ast.IndexExpr); ok {
			found = true
		}
		return !found
	})
	return found
}

// collectColumnRead appends a finding when m is a nested index whose row varies with iv and
// whose column does not. Reported once per (matrix, column) text so a loop body that reads the
// same column twice does not report twice.
func collectColumnRead(fset *token.FileSet, fn *ast.FuncDecl, m ast.Node, iv string,
	seen map[string]bool, out *[]finding) bool {
	e, ok := m.(ast.Expr)
	if !ok {
		return true
	}
	row, col := columnIndexPair(e)
	if row == nil {
		return true
	}
	if !mentionsIdent(row, iv) || mentionsIdent(col, iv) {
		return true
	}
	// The row must itself be a GATHER — an index into a permutation array — not the loop
	// variable directly. Both forms read a column, but they are not the same problem. With
	// x[i][f] the rows are visited in order and a prefetcher can follow the fixed stride;
	// with x[idx[k]][f] the row is data-dependent and every read is an independent miss.
	// Without this condition the check reported 118 sites tree-wide instead of 8, and the
	// extra 110 were overwhelmingly the strided form.
	if !containsIndexExpr(row) {
		return true
	}
	key := exprText(e)
	if seen[key] {
		return false
	}
	seen[key] = true
	*out = append(*out, finding{
		pos:      fset.Position(e.Pos()),
		category: "column-read-through-a-jagged-matrix",
		msg: fmt.Sprintf("%q reads ONE COLUMN of a jagged row-major matrix: the row index varies"+
			" with the enclosing loop over %q and the column index does not. Every read is a"+
			" pointer chase into a separate row, so n of them touch n cache lines spread across"+
			" the whole matrix and use one element of each. Mirror the matrix FEATURE-MAJOR once"+
			" — xt[col*n+row] — where it is already being walked, and read the mirror here: the"+
			" same access becomes a scattered read inside ONE contiguous length-n array."+
			" IT IS A PURE COPY, so nothing downstream moves by a bit. That cuts both ways."+
			" Only a bit-exact digest can gate it — an accuracy or tolerance comparison passes"+
			" whatever the layout does — and the mirror COSTS n*d elements of memory, which is a"+
			" trade to state rather than hide. RANK BY THE PRODUCT of the loop bound and how many"+
			" times the same column is re-read; a column read once does not repay a mirror."+
			" MEASURED on the GBM exact-split builder, where this gather was 51%% of scanFeatures"+
			" and 16%% of the package: BenchmarkGBMHist_exact_80k -19.5%%, _20k -13.6%% for"+
			" +15.6%% bytes, with ForestFit flat as a control", key, iv),
	})
	return false
}

// wideStrideOperands returns the base names indexed inside any loop of fn whose stride is more
// than one — the operands an existing jam already reads once for several items.
func wideStrideOperands(fn *ast.FuncDecl) map[string]bool {
	out := map[string]bool{}
	if fn.Body == nil {
		return out
	}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		f, ok := n.(*ast.ForStmt)
		if !ok || f.Body == nil {
			return true
		}
		as, ok := f.Post.(*ast.AssignStmt)
		if !ok || as.Tok != token.ADD_ASSIGN || len(as.Rhs) != 1 {
			return true
		}
		if lit, ok := as.Rhs[0].(*ast.BasicLit); !ok || lit.Value == "1" {
			return true
		}
		ast.Inspect(f.Body, func(m ast.Node) bool {
			if ix, ok := m.(*ast.IndexExpr); ok {
				if nm := identName(ix.X); nm != "" {
					out[nm] = true
				}
			}
			return true
		})
		return true
	})
	return out
}

// sharedOperandIsJammed reports whether every shared operand this accumulator reads is already
// amortized by a wide-stride loop elsewhere in the function.
func sharedOperandIsJammed(outBody *ast.BlockStmt, acc string, derived map[string]bool,
	outVar string, jammed map[string]bool) bool {
	if len(jammed) == 0 {
		return false
	}
	all := true
	seen := false
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
			if base == "" || derived[base] || mentions(ix.Index, outVar) {
				continue
			}
			seen = true
			if !jammed[base] {
				all = false
			}
		}
		return true
	})
	return seen && all
}

// --- PS3081: an operand streamed once per output unit -----------------------------------------

// perUnitStreamFindings flags PS3081 — a per-output-unit function whose loop reads a slice
// indexed WITHOUT the unit variable, so that operand is streamed once per unit.
func perUnitStreamFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	// APPLYING THIS CHECK LEAVES THE PER-UNIT PATH IN PLACE as the tail of the blocked loop, and
	// that path is the reported shape exactly. A function that already steps its unit by more
	// than one has made the change; without this it reports its own fix, and reports it twice.
	blocked := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		f, ok := n.(*ast.ForStmt)
		if !ok || blocked {
			return !blocked
		}
		as, ok := f.Post.(*ast.AssignStmt)
		if !ok || as.Tok != token.ADD_ASSIGN || len(as.Rhs) != 1 {
			return true
		}
		stride := identName(as.Lhs[0])
		if lit, ok := as.Rhs[0].(*ast.BasicLit); ok && lit.Value != "1" && stride != "" {
			// AND THE LOOP MUST HAND THAT COUNTER TO SOMETHING. A wide stride anywhere in the
			// function is not evidence; a wide stride whose body PASSES the stepped variable as
			// an argument is the blocked unit loop this check asks for. Written loosely first,
			// it suppressed an unrelated fan-out in the mixture-model fitter.
			ast.Inspect(f.Body, func(m ast.Node) bool {
				c, ok := m.(*ast.CallExpr)
				if !ok || len(c.Args) < 2 {
					return true
				}
				for _, a := range c.Args {
					if mentionsIdent(a, stride) {
						blocked = true
						return false
					}
				}
				return true
			})
		}
		return !blocked
	})
	if blocked {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		lit, ok := n.(*ast.FuncLit)
		if !ok || lit.Body == nil || lit.Type.Params == nil {
			return true
		}
		// THE UNIT IS THE LAST PARAMETER OF A PER-UNIT CLOSURE. A fan-out hands one of these an
		// index and calls it once per output; everything it reads that does not depend on that
		// index is read again for the next one.
		var unit string
		for _, p := range lit.Type.Params.List {
			if _, ok := p.Type.(*ast.Ident); ok && len(p.Names) > 0 {
				unit = p.Names[len(p.Names)-1].Name
			}
		}
		if unit == "" || unit == "_" {
			return true
		}
		// The closure must WRITE something indexed by the unit — that is what makes it the
		// output unit rather than an ordinary callback.
		writesUnit := false
		ast.Inspect(lit.Body, func(m ast.Node) bool {
			as, ok := m.(*ast.AssignStmt)
			if !ok || len(as.Lhs) != 1 {
				return true
			}
			if ix, ok := as.Lhs[0].(*ast.IndexExpr); ok && mentionsIdent(ix.Index, unit) {
				writesUnit = true
			}
			return true
		})
		if !writesUnit {
			return true
		}
		found := false
		ast.Inspect(lit.Body, func(m ast.Node) bool {
			if found {
				return false
			}
			iv, body, ok := loopVarBody(m)
			if !ok {
				// A JAMMED LOOP DECLARES ITS COUNTER OUTSIDE ITSELF — `mi := 0` then
				// `for ; mi+8 <= m; mi += 8` — so the Init is empty and loopVarBody cannot
				// read it. That is the exact shape this check exists for, and testing only
				// the Init made it miss the site it was built from.
				f, isFor := m.(*ast.ForStmt)
				if !isFor || f.Body == nil {
					return true
				}
				as, isAssign := f.Post.(*ast.AssignStmt)
				if !isAssign || as.Tok != token.ADD_ASSIGN || len(as.Lhs) != 1 {
					return true
				}
				if iv = identName(as.Lhs[0]); iv == "" {
					return true
				}
				body = f.Body
			}
			// EVERYTHING THE CLOSURE ITSELF DECLARES IS PER-UNIT, not just what the loop
			// declares. A component accumulator allocated at the top of the closure is cut by
			// an inner index and looks exactly like a shared operand from inside the loop —
			// which is how the mixture-model fitter read as a finding.
			local := declaredIn(lit.Body)
			ast.Inspect(body, func(w ast.Node) bool {
				if found {
					return false
				}
				se, ok := w.(*ast.SliceExpr)
				if !ok {
					return true
				}
				base := identName(se.X)
				if base == "" || local[base] || mentionsIdent(se, unit) {
					return true
				}
				// A slice cut with the LOOP variable and not the unit is the per-item operand
				// re-cut on every unit — the whole matrix, once per output.
				if !mentionsIdent(se, iv) {
					return true
				}
				found = true
				out = append(out, finding{
					pos:      fset.Position(se.Pos()),
					category: "operand-streamed-once-per-output-unit",
					msg: fmt.Sprintf("this closure runs once per output unit %q and cuts %q by"+
						" %q, an index that has nothing to do with the unit — so the whole of"+
						" %q is streamed again for every output. BLOCK THE UNIT: take three at a"+
						" time, derive three per-unit operands, and compute three outputs from"+
						" ONE pass over the shared one. MEASURED on the quantized matmul, which"+
						" read eight activation elements and one weight element to do eight"+
						" FMAs once per output column: BenchmarkQuantMamba2Prefill_512 276.2 to"+
						" 216.4 ms, -21.6%%, with decode flat because a single row takes the"+
						" tail. THREE, AND SWEPT — at 2, 3 and 4 the cell read 237.2, 220.3 and"+
						" 261.3 ms against 270.8, so four already spills. PASS THE PER-UNIT"+
						" SCRATCH AS NAMED PARAMETERS, NOT A SLICE OF SLICES: identical"+
						" arithmetic measured 237.2 against 222.5 ms at two columns, so a sweep"+
						" run through an indexed harness understates every arm and its winner"+
						" must be re-measured in the shipped form. BIT-IDENTICAL when each"+
						" output keeps its OWN accumulator over the same ascending index",
						unit, base, iv, base),
				})
				return false
			})
			return true
		})
		return true
	})
	return out
}

// --- PS3080: a one-dimensional per-element accessor walk -------------------------------------

// typedExposerReg keys package -> function name for the helpers that EXPOSE a tensor as a typed
// slice: their body reads Storage().F64/F32 and they return a slice. A fast path often goes
// through one of these rather than touching storage inline.
var typedExposerReg = map[string]map[string]bool{}

// bodyTouchesTypedStorage reports whether n contains a Storage().F64/F32 read.
func bodyTouchesTypedStorage(n ast.Node) bool {
	found := false
	ast.Inspect(n, func(m ast.Node) bool {
		sel, ok := m.(*ast.SelectorExpr)
		if !ok || found {
			return !found
		}
		if sel.Sel.Name != "F64" && sel.Sel.Name != "F32" {
			return true
		}
		if c, ok := sel.X.(*ast.CallExpr); ok && calleeName(c.Fun) == "Storage" {
			found = true
		}
		return !found
	})
	return found
}

// collectTypedExposers pre-scans every package for those helpers.
func collectTypedExposers(files []*ast.File) {
	for _, f := range files {
		if f.Name == nil {
			continue
		}
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil || fn.Name == nil || fn.Type.Results == nil {
				continue
			}
			slice := false
			for _, r := range fn.Type.Results.List {
				if _, ok := r.Type.(*ast.ArrayType); ok {
					slice = true
				}
			}
			if !slice || !bodyTouchesTypedStorage(fn.Body) {
				continue
			}
			if typedExposerReg[f.Name.Name] == nil {
				typedExposerReg[f.Name.Name] = map[string]bool{}
			}
			typedExposerReg[f.Name.Name][fn.Name.Name] = true
		}
	}
}

// hasTypedStorageAccess reports whether fn already has a typed fast path — the presence of which
// means the accessor loop beside it is a deliberate fallback.
//
// A LITERAL Storage().F64() IS NOT THE ONLY FORM, and testing only for it made this check report
// five reference kernels whose fast path goes through a package helper (f64Data) instead. Their
// accessor loops sit in an else branch labelled as the fallback for dtypes that helper cannot
// expose — already-converted code, which is the worst thing a check can send a reader to.
func hasTypedStorageAccess(f *ast.File, fn *ast.FuncDecl) bool {
	if bodyTouchesTypedStorage(fn.Body) {
		return true
	}
	if f.Name == nil {
		return false
	}
	reg := typedExposerReg[f.Name.Name]
	if len(reg) == 0 {
		return false
	}
	found := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if c, ok := n.(*ast.CallExpr); ok && reg[calleeName(c.Fun)] {
			found = true
		}
		return !found
	})
	return found
}

// accessorWalk1DFindings flags PS3080 — a loop calling AtF64 or SetF64 with a SINGLE index that
// is the loop variable, in a function that never touches typed storage.
func accessorWalk1DFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || hasTypedStorageAccess(f, fn) {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		iv, body, ok := loopVarBody(n)
		if !ok {
			return true
		}
		calls := 0
		var pos token.Pos
		ast.Inspect(body, func(m ast.Node) bool {
			c, ok := m.(*ast.CallExpr)
			if !ok {
				return true
			}
			name := calleeName(c.Fun)
			if name != "AtF64" && name != "SetF64" {
				return true
			}
			// EXACTLY ONE INDEX, AND IT IS THE LOOP VARIABLE. Two or more is the multi-dim walk
			// PS1005 already reports; ONE is a flat tensor read element by element, which that
			// check declines by construction and which is why this one exists.
			idx := c.Args
			if name == "SetF64" && len(idx) > 0 {
				idx = idx[1:] // SetF64(value, indices...)
			}
			if len(idx) != 1 || identName(idx[0]) != iv {
				return true
			}
			calls++
			if pos == token.NoPos {
				pos = c.Pos()
			}
			return true
		})
		// ONE ACCESSOR IN A LOOP IS ORDINARY. The finding is a loop built OUT of them, where the
		// dispatch is most of the body rather than an incidental read.
		if calls < 3 {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(pos),
			category: "one-dimensional-accessor-walk",
			msg: fmt.Sprintf("this loop makes %d AtF64/SetF64 calls per element, each indexed by"+
				" %q alone. Every one walks the shape to a flat offset and switches on the"+
				" storage type, and on a FLAT tensor that is the whole cost of the read."+
				" PS1005 REPORTS THE MULTI-DIMENSIONAL VERSION AND DECLINES THIS ONE, which is"+
				" why it exists: PS1005 requires two or more index arguments, so a rank-1 walk"+
				" is invisible to it. MEASURED on the PPO clipped-surrogate backward, which made"+
				" four such calls per element and had no benchmark at all until one was written"+
				" for it: BenchmarkPPOVJP_65536 went 2000 to 680 microseconds and the 4096 cell"+
				" 124 to 42, both -66%% (2.9x). TAKE THE TYPED SLICE ONCE when every operand is"+
				" already the right dtype and KEEP THE ACCESSOR ARM for the rest — the output"+
				" dtype follows the input here, so the fallback is not optional. THE TWO ARMS"+
				" CANNOT BE COMPARED AS EQUAL BITS: the accessor arm stores float32 where the"+
				" typed arm stores float64, so what must hold is that the accessor result equals"+
				" the typed one rounded ONCE. WHEN SEVERAL RULES IN A PACKAGE SHARE THE SHAPE,"+
				" GIVE THEM ONE WALKER rather than one typed arm each, and hand it CONTIGUOUS"+
				" SLICES: the preference-optimization backwards went that way at -49.8%% to"+
				" -86.6%% each, and a first version calling the rule once per element cost half"+
				" of it — CPO 1888.6 to 1208.8 microseconds against 1888.6 to 817.2 with slices."+
				" One walker also means ONE arms-agreement test instead of one per rule", calls, iv),
		})
		return true
	})
	return out
}

// --- PS3079: a whole-input allocation once per fan-out job -----------------------------------

// allocGatherReg keys package -> function name for the functions that ALLOCATE and RETURN
// slices sized by one of their arguments — a gather that materializes a whole input per call.
var allocGatherReg = map[string]map[string]bool{}

// collectAllocGathers pre-scans every package for those functions.
func collectAllocGathers(files []*ast.File) {
	for _, f := range files {
		if f.Name == nil {
			continue
		}
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil || fn.Name == nil || fn.Type.Results == nil {
				continue
			}
			if len(fn.Type.Results.List) < 2 {
				continue // one returned slice is a result; two or more is a materialized view
			}
			makes := 0
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				c, ok := n.(*ast.CallExpr)
				if !ok || identName(c.Fun) != "make" || len(c.Args) != 2 {
					return true
				}
				if _, ok := c.Args[0].(*ast.ArrayType); ok {
					makes++
				}
				return true
			})
			if makes < 2 {
				continue
			}
			if allocGatherReg[f.Name.Name] == nil {
				allocGatherReg[f.Name.Name] = map[string]bool{}
			}
			allocGatherReg[f.Name.Name][fn.Name.Name] = true
		}
	}
}

// perJobGatherFindings flags PS3079 — a fan-out body calling a function that allocates a whole
// input's worth of slices, so every job pays that allocation.
func perJobGatherFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || len(allocGatherReg[f.Name.Name]) == 0 {
		return nil
	}
	reg := allocGatherReg[f.Name.Name]
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok || !fanoutReg[f.Name.Name][calleeName(call.Fun)] {
			return true
		}
		for _, a := range call.Args {
			lit, ok := a.(*ast.FuncLit)
			if !ok || lit.Body == nil {
				continue
			}
			// ALREADY RECYCLED IS THE FIX, NOT THE FINDING.
			if mentionsPool(lit.Body) {
				continue
			}
			found := false
			ast.Inspect(lit.Body, func(m ast.Node) bool {
				if found {
					return false
				}
				c, ok := m.(*ast.CallExpr)
				if !ok || !reg[calleeName(c.Fun)] {
					return true
				}
				found = true
				out = append(out, finding{
					pos:      fset.Position(c.Pos()),
					category: "per-job-whole-input-allocation",
					msg: fmt.Sprintf("%q allocates and returns slices sized by its input, and"+
						" this fan-out body calls it once per JOB — every job pays a whole"+
						" input's worth of allocation. RECYCLE THE BUFFERS through a sync.Pool"+
						" taken at the top of the job and returned at its end, resizing only"+
						" when the job needs more than the recycled one holds. MEASURED on the"+
						" random forest, where each tree materialized its own row-pointer slice"+
						" and label copy: BenchmarkForestFit went from 33.70 to 20.14 MB per"+
						" operation, -40.2%%, and 1883 to 1666 allocations, with the wall clock"+
						" FLAT at 78.4 vs 77.6 ms. EXPECT BYTES, NOT TIME, and say so when"+
						" reporting it. THE SAFETY QUESTION IS RETENTION, AND IT IS ANSWERABLE:"+
						" read what the callee stores. The tree fitter keeps only its fitted"+
						" root, the class set and the feature count, and the builder holding the"+
						" rows dies with the call, so nothing outlives the job — and both"+
						" buffers are fully overwritten before being read, so a recycled one"+
						" carries nothing forward. If either of those is false the pool is a"+
						" correctness bug, not an optimization", calleeName(c.Fun)),
				})
				return false
			})
		}
		return true
	})
	return out
}

// --- PS3078: a radix pass that cannot be skipped ---------------------------------------------

// hasTwoDimArray reports whether fn declares a [N][M] array — the all-passes histogram that
// makes a uniform-pass skip possible.
func hasTwoDimArray(fn *ast.FuncDecl) bool {
	found := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		at, ok := n.(*ast.ArrayType)
		if !ok || at.Len == nil {
			return true
		}
		if inner, ok := at.Elt.(*ast.ArrayType); ok && inner.Len != nil {
			found = true
		}
		return !found
	})
	return found
}

// radixPassFindings flags PS3078 — a byte-wise radix loop that builds its histogram inside the
// pass, so it cannot know that a pass is the identity and skip it.
func radixPassFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || hasTwoDimArray(fn) {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		f, ok := n.(*ast.ForStmt)
		if !ok || f.Body == nil {
			return true
		}
		as, ok := f.Post.(*ast.AssignStmt)
		if !ok || as.Tok != token.ADD_ASSIGN || len(as.Rhs) != 1 {
			return true
		}
		// A STRIDE OF EIGHT OVER A SHIFT is what makes this a byte-wise radix rather than any
		// other blocked loop; the same stride on an index is PS3076's business.
		if lit, ok := as.Rhs[0].(*ast.BasicLit); !ok || lit.Value != "8" {
			return true
		}
		shift := identName(as.Lhs[0])
		if shift == "" {
			return true
		}
		found := false
		ast.Inspect(f.Body, func(m ast.Node) bool {
			if found {
				return false
			}
			inc, ok := m.(*ast.IncDecStmt)
			if !ok || inc.Tok != token.INC {
				return true
			}
			ix, ok := inc.X.(*ast.IndexExpr)
			if !ok || !mentionsIdent(ix.Index, shift) {
				return true
			}
			found = true
			out = append(out, finding{
				pos:      fset.Position(f.Pos()),
				category: "radix-pass-cannot-be-skipped",
				msg: fmt.Sprintf("this byte-wise radix builds its histogram INSIDE the pass over"+
					" %q, so it can never learn that a pass is the identity. A counting pass in"+
					" which every key lands in one bucket emits them in the order it read them,"+
					" so it can be SKIPPED and the sorted order is the same permutation — build"+
					" all eight histograms in ONE traversal, skip any pass whose bucket holds"+
					" every key, and copy home when the surviving pass count comes out ODD,"+
					" which the fixed eight-pass form never had to do. MEASURED on the CART"+
					" builder, where the per-feature radix was 24%% of the profile:"+
					" BenchmarkForestFit 88.6 to 79.4 ms, -10.4%%, winning every paired round,"+
					" with GBMFit and the SVC cells flat. IT PAYS ON THE DATA, NOT ON THE CODE —"+
					" the keys there are float64 bit patterns of one feature column, whose sign"+
					" and exponent bytes barely move within a node and whose high mantissa bytes"+
					" stop moving deeper in the tree. A column of full-entropy keys skips"+
					" nothing and the single traversal is then the only gain. RANK BY CALL"+
					" COUNT, WHICH IS WHAT ACTUALLY DECIDES IT. The two halves were separated by"+
					" measurement: 88.6 to 83.14 ms from the single traversal alone and 83.14 to"+
					" 79.49 from adding the skip, so both are real and the skip is worth more"+
					" than its 12.5%% of passes suggests, because a skipped pass drops a random"+
					" SCATTER and a buffer swap while the counting read had already been"+
					" amortized. Neither half can reach a sort that runs a few times: the same"+
					" transform applied to the GBM presort beside it measured FLAT (129.2 vs"+
					" 130.0 ms), and a counter says why — the CART radix runs 3515136 passes"+
					" across a fit and the GBM one runs 640. Count the calls before converting"+
					" the site", shift),
			})
			return false
		})
		return true
	})
	return out
}

// --- PS3077: a math.Min/math.Max clamp inside a loop -----------------------------------------

// mathMinMaxCall reports whether e is a call to math.Min or math.Max, and which.
func mathMinMaxCall(e ast.Expr) (string, *ast.CallExpr, bool) {
	c, ok := unparen(e).(*ast.CallExpr)
	if !ok {
		return "", nil, false
	}
	sel, ok := c.Fun.(*ast.SelectorExpr)
	if !ok || identName(sel.X) != "math" {
		return "", nil, false
	}
	if sel.Sel.Name != "Min" && sel.Sel.Name != "Max" {
		return "", nil, false
	}
	return sel.Sel.Name, c, true
}

// clampInLoopFindings flags PS3077 — a math.Min wrapped around a math.Max (or the reverse)
// inside a loop: a two-bound clamp written as two function calls.
func clampInLoopFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		_, body, ok := loopVarBody(n)
		if !ok {
			return true
		}
		found := false
		ast.Inspect(body, func(m ast.Node) bool {
			if found {
				return false
			}
			me, ok := m.(ast.Expr)
			if !ok {
				return true
			}
			outer, oc, ok := mathMinMaxCall(me)
			if !ok || len(oc.Args) != 2 {
				return true
			}
			// THE NESTED FORM IS THE ONE THAT IS A CLAMP. A lone math.Min is often a genuine
			// two-value choice; Min around Max is a value being held between two bounds, which
			// is what a comparison chain replaces exactly.
			for _, a := range oc.Args {
				inner, _, ok := mathMinMaxCall(a)
				if !ok || inner == outer {
					continue
				}
				found = true
				out = append(out, finding{
					pos:      fset.Position(oc.Pos()),
					category: "minmax-clamp-in-a-loop",
					msg: fmt.Sprintf("math.%s around math.%s inside a loop is a two-bound CLAMP"+
						" written as two function calls, and those calls carry the whole NaN and"+
						" signed-zero contract at every iteration. MEASURED on the HQQ"+
						" quantizer, whose profile was 29%% archMin and archMax against 35%% of"+
						" its own arithmetic: replacing the clamp with a comparison chain took"+
						" BenchmarkHQQuantize from 77.14 to 37.79 ms, -51.0%% (2.04x), with an"+
						" optimizer benchmark flat as a control. TRY internal/fmath FIRST (see PS3082): it keeps the exact math.Min/math.Max contract, is a smaller edit than a chain, and measured FASTER than one — 40.5 us against the chain's 63.6 on the reference PPO surrogate. The chain below is what you fall back to when the call is already gone: converting an EXISTING chain to fmath measured +13%% and was reverted."+
						" WRITE THE CHAIN THAT IS ACTUALLY"+
						" EQUIVALENT, not the obvious one: use `if r <= lo { r = lo }`, because"+
						" `<` lets a NEGATIVE ZERO through where math.Max(0, -0) returns +0, and"+
						" leave NaN to fall through both bounds untouched, which is what math.Min"+
						" and math.Max also do when either operand is NaN. GATE IT TWICE — a"+
						" table comparing the chain to the two calls bit-for-bit over -0, +0,"+
						" NaN, both infinities and each boundary, AND a digest of the caller,"+
						" because ordinary data never produces the two cases that make the naive"+
						" rewrite wrong: the digest stayed GREEN under a `<` for `<=` mutation"+
						" that the table caught", outer, inner),
				})
				return false
			}
			return true
		})
		return true
	})
	return out
}

// --- PS3082: a math.Min/math.Max CALL inside a loop -------------------------------------------

// builtinMinMaxArgs renders the argument pair of every builtin min/max call in fn. A function
// that already guards a builtin with a math call — take the instruction, fall back on NaN —
// contains both spellings of the same pair, and the math one is the recovery, not the waste.
func builtinMinMaxArgs(fn *ast.FuncDecl) map[string]bool {
	out := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		c, ok := n.(*ast.CallExpr)
		if !ok || len(c.Args) != 2 {
			return true
		}
		if name := identName(c.Fun); name != "min" && name != "max" {
			return true
		}
		out[exprText(c.Args[0])+","+exprText(c.Args[1])] = true
		return true
	})
	return out
}

// minMaxCallInLoopFindings flags PS3082 — math.Min or math.Max called inside a loop, where the
// builtin lowers to one instruction and the math function does not lower at all.
func minMaxCallInLoopFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	guarded := builtinMinMaxArgs(fn)
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		_, body, ok := loopVarBody(n)
		if !ok {
			return true
		}
		ast.Inspect(body, func(m ast.Node) bool {
			me, ok := m.(ast.Expr)
			if !ok {
				return true
			}
			which, c, ok := mathMinMaxCall(me)
			if !ok || len(c.Args) != 2 {
				return true
			}
			if guarded[exprText(c.Args[0])+","+exprText(c.Args[1])] {
				// Already the guarded form: the builtin runs, and this call is the NaN recovery.
				return true
			}
			out = append(out, finding{
				pos:      fset.Position(c.Pos()),
				category: "minmax-call-in-a-loop",
				msg: fmt.Sprintf("math.%s inside a loop is a CALL, not an instruction. On arm64"+
					" it compiles to CALL math.arch%s inside a 48-byte frame with a"+
					" stack-growth check, while the %s builtin compiles to a single F%sD in a"+
					" leaf with no frame at all. THE TWO ARE NOT THE SAME FUNCTION, so"+
					" substituting the builtin raw is a real bug and not a style change:"+
					" math.Max documents +Inf as beating NaN and math.Min documents -Inf as"+
					" beating NaN, while the builtins propagate NaN unconditionally, and they"+
					" disagree on exactly four ordered pairs. THE DIVERGENCE IS REACHABLE, not"+
					" theoretical — a log-probability of -Inf makes a ratio exactly +0, and +0"+
					" times an infinite advantage is the NaN that pairs with the -Inf the other"+
					" branch produces. USE internal/fmath: it takes the instruction and consults"+
					" math only when the instruction returns NaN, which is the only result the"+
					" two can disagree on, so it is bit-identical. MEASURED on the reference RL"+
					" surrogates: PPOClip at batch 4096 100.5 to 41.6 us (-58.7%%), GRPO 126.7"+
					" to 79.8 (-37.0%%), GSPO 21.6 to 19.8 (-8.4%%, one clamp per sequence"+
					" against a 256-token inner loop). IT DOES NOT BEAT AN EXISTING COMPARISON"+
					" CHAIN: converting the PPO VJP's chain to fmath went 51.8 to 58.6 us, +13%%,"+
					" and was reverted — this replaces CALLS, not branchless code, so rank a"+
					" site by whether the call is still there. CHECK WHICH ARM THE BENCHMARK"+
					" ACTUALLY TAKES: the cpu WKV kernel's F64 path dispatches into SIMD"+
					" assembly, so its Go-level max sits in a dead exotic-dtype fallback and the"+
					" F64 benchmark measured flat while the F32 arm, which is Go, measured"+
					" -17.9%%; the same recurrence in nn measured -13.7%% against an archMax"+
					" profile share of 13.1%%. AND A SITE IN A PARALLEL SOLVER MAY BE INVISIBLE:"+
					" the SVM fit was converted at eight sites and measured 6.11 to 6.27 ms,"+
					" slightly WORSE, because its profile is dominated by scheduler wait — that"+
					" one was reverted. GATE IT ON ONE PLANTED VALUE PER"+
					" CALL: a kernel that reduces to a scalar CANNOT see the divergence, because"+
					" one NaN poisons the sum and both formulations then agree on NaN — a"+
					" whole-grid batch was green under the naive rewrite this check exists to"+
					" reject",
					which, which, strings.ToLower(which), strings.ToUpper(which)),
			})
			return true
		})
		return true
	})
	return out
}

// --- PS3076: an unroll factor fixed at two ---------------------------------------------------

// narrowUnrollFindings flags PS3076 — a jammed loop whose stride is 2 while its body already
// holds several accumulators in locals, which is the shape of an unroll factor that was chosen
// once and never swept.
func narrowUnrollFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		f, ok := n.(*ast.ForStmt)
		if !ok || f.Body == nil {
			return true
		}
		as, ok := f.Post.(*ast.AssignStmt)
		if !ok || as.Tok != token.ADD_ASSIGN || len(as.Rhs) != 1 {
			return true
		}
		lit, ok := as.Rhs[0].(*ast.BasicLit)
		if !ok || (lit.Value != "2" && lit.Value != "4") {
			return true
		}
		// THE BODY MUST ALREADY BE REGISTER-BLOCKED. A stride of two on an ordinary loop says
		// nothing; the finding is about a kernel that ALREADY holds its accumulators in locals
		// and stores them back, because that is where the factor is a tuning choice.
		held := map[string]bool{}
		ast.Inspect(f.Body, func(m ast.Node) bool {
			a, ok := m.(*ast.AssignStmt)
			if !ok || a.Tok != token.ASSIGN || len(a.Lhs) != 1 || len(a.Rhs) != 1 {
				return true
			}
			ix, ok := a.Lhs[0].(*ast.IndexExpr)
			if !ok {
				return true
			}
			if nm := identName(a.Rhs[0]); nm != "" {
				if b := identName(ix.X); b != "" {
					held[b] = true
				}
			}
			return true
		})
		// THE TWO SHAPES ARE GATED BY DIFFERENT STRIDES, ON PURPOSE.
		//
		// A jam holding INDEXED accumulators in locals is reported only at stride two. At four it
		// is the gemm C-row block, and that axis was swept and closed: two, six and eight rows
		// were all worse, and the one pairing that won a cell lost another
		// (PERF-GEMM-CROW-BLOCK-001). Reporting it again would be reporting a finished
		// experiment.
		//
		// A jam whose accumulators are SCALAR locals — var s0, s1, s2, s3 — feeding a call
		// rather than an indexed store is reported at both, because four is simply what a first
		// jam produces and none of those has been swept. The memory-retrieval tile was exactly
		// that shape at four, and eight measured 9.3% faster.
		scalars := scalarAccumulators(f.Body)
		switch {
		case len(scalars) >= 3:
			held = map[string]bool{}
			for _, nm := range scalars {
				held[nm] = true
			}
		case len(held) >= 3 && lit.Value == "2":
		default:
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(f.Pos()),
			category: "unroll-factor-fixed-at-two",
			msg: fmt.Sprintf("this loop is register-blocked over %d accumulators and takes TWO"+
				" steps per pass. A factor of two is what an argument about register pressure"+
				" produces; the optimum is what a SWEEP produces, and they were not the same"+
				" here. MEASURED on the two gemm band kernels, both of which carried a comment"+
				" reasoning that four would spill: sweeping 3, 4, 6 and 8 found SIX best and"+
				" eight back at the baseline, and the eight-step arm is what shows the spill"+
				" boundary is real but was placed two steps too early. BenchmarkMTAForward_ch16"+
				" 277.2 to 250.3 ms (-9.7%%) and ch8 -9.5%% on the f32 band; GemmDirF64_1024"+
				" 19.10 to 15.74 ms (-17.6%%) and the 512x2048x2048 cell -11.7%% on the f64"+
				" band. BIT-IDENTICAL AT ANY FACTOR, which is what makes the sweep cheap: each"+
				" accumulator still takes one rounding per step in ascending order, never a"+
				" summed pair — so the existing bit-exact oracle gates every arm of the sweep"+
				" and the whole experiment costs one measurement each. SWEEP, DO NOT ARGUE: a"+
				" register-pressure argument cannot see the scheduler, and the only cost of"+
				" being wrong is one benchmark run. SECOND SWEEP, ON A JAM ALREADY AT FOUR:"+
				" the memory-retrieval tile scored four queries per pass over each key row, and"+
				" swept at 6, 8 and 10 it read 103.1, 85.9 and 101.9 ms against 94.7 at four —"+
				" EIGHT wins by 9.3%% while six and ten are both WORSE THAN FOUR. The curve is"+
				" not monotone and no argument predicts it; only the measurement does. The"+
				" oracle that gates it is an independent per-row implementation, and it took a"+
				" tile length of exactly 7 to make an off-by-one in the jam bound red: with"+
				" lengths 16 and 9 present but none at 7 mod 8, reading past the last query"+
				" stayed GREEN. THIRD SWEEP, SAME ANSWER: the masked-attention score jam"+
				" went 92.58 to 79.92 ms at eight, -13.7%%, with six at 82.59 — eight has now"+
				" won twice and six has lost twice. NOT EVERY JAM IN A KERNEL IS WORTH"+
				" SWEEPING: the weighted sum beside that score loop was swept too and six and"+
				" eight both landed inside the run-to-run spread, because that loop is bound by"+
				" streaming its value rows rather than by the accumulator round trip the jam to"+
				" four already removed. Sweep the loop the PROFILE names, not every loop the"+
				" check reports", len(held)),
		})
		return true
	})
	return out
}

// --- PS3075: an inner loop accumulating into a shared buffer ---------------------------------

// indexIsInnerVar reports whether an index expression is the inner loop variable, alone or as a
// term of a sum — d, ob+d, base+d+off. Anything multiplying it is a stride and addresses a
// different element per item, which is not the shape.
func indexIsInnerVar(e ast.Expr, dv string) bool {
	switch x := unparen(e).(type) {
	case *ast.Ident:
		return x.Name == dv
	case *ast.BinaryExpr:
		if x.Op != token.ADD {
			return false
		}
		return indexIsInnerVar(x.X, dv) || indexIsInnerVar(x.Y, dv)
	}
	return false
}

// sharedAccumulatorFindings flags PS3075 — an item loop whose inner loop accumulates into a
// buffer that does not vary with the item, so that buffer is loaded and stored once per item.
func sharedAccumulatorFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	// The accumulators any wide-stride loop already folds several items into. Applying this check
	// leaves a by-one tail whose body is the reported shape exactly, so without this the check
	// reports its own fix forever — the same tail PS3074 has to ignore.
	jammed := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		f, ok := n.(*ast.ForStmt)
		if !ok || f.Body == nil {
			return true
		}
		as, ok := f.Post.(*ast.AssignStmt)
		if !ok || as.Tok != token.ADD_ASSIGN || len(as.Rhs) != 1 {
			return true
		}
		if lit, ok := as.Rhs[0].(*ast.BasicLit); !ok || lit.Value == "1" {
			return true
		}
		ast.Inspect(f.Body, func(m ast.Node) bool {
			if ix, ok := m.(*ast.IndexExpr); ok {
				if nm := identName(ix.X); nm != "" {
					jammed[nm] = true
				}
			}
			return true
		})
		return true
	})
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		jv, body, ok := loopVarBody(n)
		if !ok {
			return true
		}
		perItem := declaredIn(body)
		found := false
		ast.Inspect(body, func(m ast.Node) bool {
			if found {
				return false
			}
			rs, ok := m.(*ast.RangeStmt)
			if !ok || rs.Body == nil {
				return true
			}
			dv := identName(rs.Key)
			if dv == "" {
				return true
			}
			ast.Inspect(rs.Body, func(w ast.Node) bool {
				if found {
					return false
				}
				as, ok := w.(*ast.AssignStmt)
				if !ok || as.Tok != token.ADD_ASSIGN || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
					return true
				}
				ix, ok := as.Lhs[0].(*ast.IndexExpr)
				// THE INDEX IS OFTEN A BASE PLUS THE INNER VARIABLE. A kernel that addresses its
				// output as os[ob+d] instead of slicing a row first is the same accumulator, and
				// a bare-identifier test found only the arm of one kernel that happened to use a
				// scratch slice while missing the arm beside it.
				if !ok || !indexIsInnerVar(ix.Index, dv) {
					return true
				}
				acc := identName(ix.X)
				// THE ACCUMULATOR MUST OUTLIVE THE ITEM. One rebound per item is the item's own
				// output and is already written once; the finding is about a buffer every item
				// reads and writes in turn.
				if acc == "" || perItem[acc] || mentionsIdent(ix.X, jv) || jammed[acc] {
					return true
				}
				// AND THE CONTRIBUTION MUST BE THE ITEM'S. A term that does not vary with the
				// item makes the whole accumulation loop-invariant, which is a hoist, not a jam.
				if !mentionsAnyOf(as.Rhs[0], perItem) && !mentionsIdent(as.Rhs[0], jv) {
					return true
				}
				found = true
				out = append(out, finding{
					pos:      fset.Position(as.Pos()),
					category: "inner-loop-accumulates-into-a-shared-buffer",
					msg: fmt.Sprintf("%q does not vary with %q, so every item LOADS and STORES"+
						" the whole of it in turn — one round trip through that buffer per item"+
						" for a single addition each. JAM THE ITEM LOOP — take 4 items per pass"+
						" and hold %s[%s] in a LOCAL across their four additions, storing once."+
						" BIT-IDENTICAL when the four additions keep the same ascending item"+
						" order: the accumulator sees the same sequence of adds, only in a"+
						" register instead of memory. MEASURED on the cpu selective-attention"+
						" kernel, where this weighted-sum loop was 26%% of the profile and a"+
						" score loop in the same function was 46%%: jamming both took"+
						" MHASelectF32CPU_1024x1024x64x16 from 191.3 to 100.9 ms, -47.3%%"+
						" (1.90x), the 512 cell -42.7%%, and the F64 arm -40.1%%, with a masked"+
						" attention cell flat as a control. THE MIRROR OF PS3074: there the"+
						" SUBJECT of the inner loop is shared and the outputs are per item; here"+
						" the outputs are shared and the subject is per item. Neither is"+
						" reachable from PS6010, which needs the shared operand to appear as an"+
						" index into a per-output accumulator. CHECK THE GATE MEASURES BIT"+
						" IDENTITY, not just agreement: swapping two of the four additions must"+
						" turn the parity test red, or the test cannot see the property this"+
						" transform preserves. SECOND MEASUREMENT, ON THE SIBLING KERNEL: the"+
						" masked-attention kernel carried the identical loop — its SCORE loop had"+
						" been jammed and this one had not — and jamming it took"+
						" MHAMaskedF32CPU_1024x1024x64x16 from 118.0 to 89.7 ms, -24.0%%, the 512"+
						" cell -15.2%% and the F64 arm -7.0%%, with a selective-attention cell flat"+
						" as a control. WHEN THIS CHECK FIRES IN ONE KERNEL, READ ITS SIBLINGS: a"+
						" half-applied jam is the normal state, because the score loop is the one"+
						" people look at. THIRD MEASUREMENT, AND A SECOND KIND OF HALF-APPLIED:"+
						" the MLA kernel had this loop AND a single-accumulator score dot re-reading"+
						" the query row, and jamming both took BenchmarkMLA_cpu_512 from 27.80 to"+
						" 8.97 ms, -67.7%% (3.1x) — far more than the 76%% profile share of the two"+
						" loops predicts, because four independent chains fix a latency-bound dot"+
						" rather than merely removing loads. ONE OF ITS TWO ARMS WAS INVISIBLE TO AN"+
						" EARLIER VERSION OF THIS CHECK: it addressed the accumulator as os[ob+d]"+
						" instead of slicing a row first, and the index test only recognized a bare"+
						" variable. Widening it to a SUM found 11 more sites in the tree",
						acc, jv, acc, dv),
				})
				return false
			})
			return true
		})
		return true
	})
	return out
}

// --- PS3074: an inner loop ranging over a shared operand -------------------------------------

// sharedRangeSubjectFindings flags PS3074 — an item loop whose inner loop RANGES over an operand
// that does not vary with the item, so that operand is re-streamed once per item.
func sharedRangeSubjectFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	// APPLYING THIS CHECK LEAVES A BY-ONE TAIL BEHIND THE JAMMED LOOP, and that tail is the
	// reported shape exactly: stride one, same body, same shared operand. Collect the subjects
	// any wide-stride loop in this function already traverses and stay silent about them, or the
	// check reports its own fix forever.
	jammed := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		f, ok := n.(*ast.ForStmt)
		if !ok || f.Body == nil {
			return true
		}
		as, ok := f.Post.(*ast.AssignStmt)
		if !ok || as.Tok != token.ADD_ASSIGN || len(as.Rhs) != 1 {
			return true
		}
		if lit, ok := as.Rhs[0].(*ast.BasicLit); !ok || lit.Value == "1" {
			return true
		}
		ast.Inspect(f.Body, func(m ast.Node) bool {
			if rs, ok := m.(*ast.RangeStmt); ok {
				if nm := identName(rs.X); nm != "" {
					jammed[nm] = true
				}
			}
			return true
		})
		return true
	})
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		jv, body, ok := loopVarBody(n)
		if !ok {
			return true
		}
		// A STRIDE TEST ON THIS LOOP WAS TRIED AND REMOVED. It duplicated the jammed-subject
		// suppression below — a wide-stride loop puts its own range subject into that set — and
		// no fixture could isolate the two, which is the same reason PS3073 lost its parameter
		// count.
		perItem := declaredIn(body) // names rebound per item: the outputs and the item's own rows
		found := false
		ast.Inspect(body, func(m ast.Node) bool {
			if found {
				return false
			}
			rs, ok := m.(*ast.RangeStmt)
			if !ok || rs.Body == nil {
				return true
			}
			sub := identName(rs.X)
			if sub == "" || perItem[sub] || mentionsIdent(rs.X, jv) {
				return true
			}
			// A RANGE WITHOUT A VALUE IS NOT STREAMING AN OPERAND. `for d := range dk` counts
			// an integer and `for cat := range set` walks keys; neither reads a buffer that
			// jamming could read once. Requiring the value form took the tree-wide count from
			// 126 to a population whose members are all the measured shape.
			if rs.Value == nil || identName(rs.Value) == "" {
				return true
			}
			if jammed[sub] {
				return true
			}
			// THE INNER LOOP MUST PRODUCE SOMETHING PER ITEM. A loop that ranges over a shared
			// slice and writes nothing item-specific is loop-invariant outright, which is a
			// hoist and not a jam.
			writesPerItem := false
			bases := map[string]bool{}
			ast.Inspect(rs.Body, func(w ast.Node) bool {
				if ix, ok := w.(*ast.IndexExpr); ok {
					if b := identName(ix.X); b != "" && (perItem[b] || mentionsIdent(ix.X, jv)) {
						bases[b] = true
					}
				}
				as, ok := w.(*ast.AssignStmt)
				if !ok || len(as.Lhs) != 1 {
					return true
				}
				ix, ok := as.Lhs[0].(*ast.IndexExpr)
				if !ok {
					return true
				}
				if perItem[identName(ix.X)] || mentionsIdent(ix.X, jv) {
					writesPerItem = true
				}
				return true
			})
			// TWO DISTINCT PER-ITEM OPERANDS, not one. A body touching a single per-item buffer
			// is a copy or a normalization pass — real per-item work reads one thing and writes
			// another, which is also what makes four items worth carrying at once.
			if !writesPerItem || len(bases) < 2 {
				return true
			}
			found = true
			out = append(out, finding{
				pos:      fset.Position(rs.Pos()),
				category: "inner-loop-ranges-over-a-shared-operand",
				msg: fmt.Sprintf("this inner loop ranges over %q, which does not vary with %q, so"+
					" the whole of %q is re-streamed once per item while the body writes"+
					" per-item output. JAM THE ITEM LOOP — take 4 items per pass and run their"+
					" bodies together over ONE traversal of %q, giving each item its own"+
					" accumulators. Bit-identical when every per-item element keeps the same"+
					" additions in the same order: the jammed dimension is the free one, and a"+
					" shared accumulator can stay bit-identical too by holding it in a local"+
					" across the 4 additions in the SAME ascending item order. MEASURED on the"+
					" cpu attention backward, whose two key loops both re-streamed the gradient"+
					" row and the query row: BenchmarkMHA512/bwd/cpu 24.73 to 13.40 ms, -45.8%%"+
					" (1.85x), with the forward and masked cells flat as controls. THIS IS THE"+
					" SHAPE PS6010 CANNOT SEE: it requires the shared operand to appear as an"+
					" INDEX expression, and an operand delivered as the RANGE SUBJECT never"+
					" does, which is why the hottest line of that function — 39%% of the"+
					" benchmark — was reported by nothing", sub, jv, sub, sub),
			})
			return false
		})
		return true
	})
	return out
}

// --- PS3073: a reduction helper reloading a loop-invariant operand ---------------------------

// reductionHelperReg keys package -> function name for the functions that ARE a reduction over
// their slice parameters: a loop that accumulates with += into locals and returns them. The dot
// helpers of a factorization are the shape; a check that only recognizes an INLINE accumulator
// loop cannot see them, which is exactly how the site this check was built from stayed hidden.
// The value is one flag per PARAMETER POSITION, true where that parameter is a slice. A scalar
// argument is not an operand — it streams no memory — and reading every argument alike reported
// the trailing length of a dot as the shared operand.
var reductionHelperReg = map[string]map[string][]bool{}

// collectReductionHelpers pre-scans every package for those functions.
func collectReductionHelpers(files []*ast.File) {
	for _, f := range files {
		if f.Name == nil {
			continue
		}
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil || fn.Name == nil || fn.Type.Results == nil {
				continue
			}
			// A COUNT OF SLICE PARAMETERS IS NOT A GATE. Requiring two of them was tried and
			// removed: the report already needs one varying and one invariant SLICE ARGUMENT,
			// so a helper with fewer can never reach it, and no fixture could isolate the
			// count from the argument test it duplicated.
			var isSlice []bool
			for _, p := range fn.Type.Params.List {
				_, sl := p.Type.(*ast.ArrayType)
				for range max(len(p.Names), 1) {
					isSlice = append(isSlice, sl)
				}
			}
			acc := false
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				as, ok := n.(*ast.AssignStmt)
				if !ok || as.Tok != token.ADD_ASSIGN {
					return true
				}
				if _, ok := as.Lhs[0].(*ast.Ident); ok {
					acc = true
				}
				return true
			})
			if !acc {
				continue
			}
			if reductionHelperReg[f.Name.Name] == nil {
				reductionHelperReg[f.Name.Name] = map[string][]bool{}
			}
			reductionHelperReg[f.Name.Name][fn.Name.Name] = isSlice
		}
	}
}

// invariantOperandReloadFindings flags PS3073 — a loop calling a reduction helper with one
// operand that does not vary with the loop variable, so that operand is re-streamed once per
// iteration.
func invariantOperandReloadFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || len(reductionHelperReg[f.Name.Name]) == 0 {
		return nil
	}
	reg := reductionHelperReg[f.Name.Name]
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		iv, body, ok := loopVarBody(n)
		if !ok {
			return true
		}
		// AN ARGUMENT THAT DOES NOT SPELL THE LOOP VARIABLE STILL VARIES WITH IT. The per-item
		// operand of a row update is re-sliced at the top of the body — li := lf[i*n:...] — and
		// reaches the call as a bare identifier, so testing the argument alone finds NO varying
		// operand and reports nothing at all. Names bound inside the body are the varying ones;
		// this set is what identifies them, not a suppression.
		rebound := declaredIn(body)
		found := false // one finding per loop, so a second nest in the same function still reports
		ast.Inspect(body, func(m ast.Node) bool {
			if found {
				return false
			}
			c, ok := m.(*ast.CallExpr)
			if !ok || len(c.Args) < 2 {
				return true
			}
			isSlice := reg[calleeName(c.Fun)]
			if len(isSlice) != len(c.Args) {
				return true
			}
			// BOTH HALVES ARE REQUIRED. An argument that varies with the loop makes the call
			// per-iteration work; one that does not is the operand being re-read for nothing.
			// A call whose arguments ALL vary reloads nothing, and one where none varies is
			// loop-invariant outright, which is a different (and larger) finding.
			varying, invariant := "", ""
			for ai, a := range c.Args {
				nm := baseIdentName(a)
				if nm == "" || !isSlice[ai] {
					continue
				}
				if mentionsIdent(a, iv) || rebound[nm] {
					varying = nm
				} else if invariant == "" {
					invariant = nm
				}
			}
			if varying == "" || invariant == "" {
				return true
			}
			out = append(out, finding{
				pos:      fset.Position(c.Pos()),
				category: "invariant-operand-reloaded-per-iteration",
				msg: fmt.Sprintf("%q is a reduction over its slice arguments, and this loop calls"+
					" it once per %q with %q re-read every time: %q does not vary with the loop,"+
					" so the same memory is streamed on every iteration. JAM THE LOOP — take 4"+
					" iterations per pass and compute their reductions in ONE pass over the"+
					" shared operand, each keeping its own accumulators. That is bit-identical"+
					" when every result keeps the helper's own accumulator count, order and"+
					" combination: the jammed dimension is the free one. MEASURED on the linalg"+
					" Cholesky factorization, whose row update called a 4-accumulator dot with"+
					" the pivot row as the shared operand: Cholesky512 8.99 to 6.25 ms, -30.5%%,"+
					" Cholesky256 -24.7%%, CholSolve256x128 -13.7%%, bit-identical in both dtype"+
					" arms with an SVD cell flat as a control. THE SHARING IS THE SINGLE PASS,"+
					" NOT THE UNROLLING — writing the same 4 rows as two calls to a 2-row helper"+
					" reloaded the operand twice and measured NOTHING (6.99 ms, the 2-row"+
					" number). AND THE JAM MUST REACH EVERY ARM: this factorization has a second"+
					" dtype path with the same loop, which a digest over one dtype would never"+
					" have covered. SECOND MEASUREMENT, AND THE RANKING LESSON IN IT: the F32"+
					" arm of the cpu attention forward scored one key per pass while its F64"+
					" sibling had jammed four for a long time, because the F32 arm's reduction"+
					" sits behind dot4. Jamming it took BenchmarkMHA512/fwd/cpu from 8.42 to"+
					" 6.73 ms, -20.1%%, with the masked and backward cells flat as controls —"+
					" but the FIRST measurement of it was flat, because the five F32 attention"+
					" benchmarks that look like the obvious cells route to a GEMM kernel and"+
					" never reach this arm at all. COUNT THE EXECUTIONS BEFORE BELIEVING A FLAT"+
					" RESULT: an atomic counter at the site, read from a TestMain AFTER m.Run"+
					" (tests run before benchmarks, so a plain test reads zero and proves"+
					" nothing), said 12288 and pointed at the one cell missing from the sweep."+
					" AND WATCH FOR A CODEGEN SHIFT IN THE NEIGHBOR: written inline, the jam"+
					" changed the code generated for the untouched F64 arm beside it and moved"+
					" one ALiBi element of the F64 parity test by a single ULP; putting the jam"+
					" behind its own function restored that arm exactly. PS6010 IS THE SAME"+
					" TRANSFORM FOR AN INLINE LOOP — this check exists because a reduction"+
					" behind a CALL is invisible to it",
					calleeName(c.Fun), iv, invariant, invariant),
			})
			found = true
			return false
		})
		return true
	})
	return out
}

// --- PS3072: a serial loop that reseeds its own state ----------------------------------------

// reseedNames are the calls that put a carried object back into a state determined ENTIRELY by
// their arguments. A generator reseeded from the item is a pure function of that item, whatever
// it computed for the previous one.
var reseedNames = map[string]bool{"Seed": true, "Reset": true, "Reinit": true, "Rewind": true}

// reseededSerialLoopFindings flags PS3072 — a loop that reseeds, from the current item, the
// state it appears to carry, in a function that never fans out.
func reseededSerialLoopFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || len(fanoutReg[f.Name.Name]) == 0 {
		return nil
	}
	if callsFanoutHelper(fn.Body, fanoutReg[f.Name.Name]) ||
		callsFanoutHelper(fn.Body, fansOutReg[f.Name.Name]) {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		iv, body, ok := loopVarBody(n)
		if !ok {
			return true
		}
		// PER-ITEM WORK, NOT A COPY. A reseed in a loop whose body is straight-line is a few
		// nanoseconds either way; the shape only pays when each item runs its own inner loop.
		if !containsLoop(body) {
			return true
		}
		// State DECLARED IN THE LOOP is already private to the iteration and says nothing.
		inner := declaredIn(body)
		ast.Inspect(body, func(m ast.Node) bool {
			if len(out) > 0 {
				return false
			}
			c, ok := m.(*ast.CallExpr)
			if !ok {
				return true
			}
			sel, ok := c.Fun.(*ast.SelectorExpr)
			if !ok || !reseedNames[sel.Sel.Name] {
				return true
			}
			recv, ok := sel.X.(*ast.Ident)
			if !ok || inner[recv.Name] {
				return true
			}
			// THE ARGUMENT IS THE WHOLE ARGUMENT. A reseed from a constant genuinely carries the
			// generator forward and the iterations are a chain; a reseed that mentions the loop
			// variable makes every iteration reproducible from the item alone.
			fromItem := false
			for _, a := range c.Args {
				if mentionsIdent(a, iv) {
					fromItem = true
				}
			}
			if !fromItem {
				return true
			}
			out = append(out, finding{
				pos:      fset.Position(c.Pos()),
				category: "serial-loop-reseeded-from-the-item",
				msg: fmt.Sprintf("this loop reseeds %q from the current item — %q takes an"+
					" argument built from %q — so the state it appears to carry across"+
					" iterations is DEAD: whatever the previous item computed is overwritten"+
					" before it is read, and the iteration is a pure function of its own item."+
					" A loop like this READS as sequential and is not, which is why it survives"+
					" a scaling probe unnoticed. BAND IT, giving every worker its own copy of"+
					" the reseeded object and of any scratch the body reuses, and check what"+
					" ELSE crosses iterations: a running COUNT or SUM crosses, but integer"+
					" addition is order-free, so accumulate per worker and fold afterwards."+
					" MEASURED on the watermark detector, whose per-token partial Fisher-Yates"+
					" reseeds a PCG from (key, previous token) and restores its permutation"+
					" before the next: BenchmarkWatermarkDetect went from 39.17 to 7.17 ms,"+
					" -81.7%%, bit-identical, with a scaling ratio of 0.99x before the change."+
					" CHECK THE RESTORE — the win depends on the body leaving its scratch as it"+
					" found it; if it does not, the copy per worker is what makes that true",
					recv.Name, sel.Sel.Name, iv),
			})
			return false
		})
		return true
	})
	return out
}

// --- PS3071: a local buffer that escapes on every call ---------------------------------------

// escapingLocalBufferFindings flags PS3071 — a method declaring a local fixed-size byte array
// and passing a slice of it to a call.
func escapingLocalBufferFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || fn.Recv == nil {
		return nil
	}
	// Locals declared as a fixed-size array of a byte-ish element.
	arrays := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		ds, ok := n.(*ast.DeclStmt)
		if !ok {
			return true
		}
		gd, ok := ds.Decl.(*ast.GenDecl)
		if !ok || gd.Tok != token.VAR {
			return true
		}
		for _, sp := range gd.Specs {
			vs, ok := sp.(*ast.ValueSpec)
			if !ok || vs.Type == nil || len(vs.Values) > 0 {
				continue
			}
			at, ok := vs.Type.(*ast.ArrayType)
			if !ok || at.Len == nil {
				continue
			}
			if id, ok := at.Elt.(*ast.Ident); !ok || id.Name != "byte" {
				continue
			}
			for _, nm := range vs.Names {
				arrays[nm.Name] = true
			}
		}
		return true
	})
	if len(arrays) == 0 {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if len(out) > 0 {
			return false
		}
		c, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		for _, a := range c.Args {
			se, ok := unparen(a).(*ast.SliceExpr)
			if !ok {
				continue
			}
			id, ok := se.X.(*ast.Ident)
			if !ok || !arrays[id.Name] {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(id.Pos()),
				category: "local-buffer-escapes-per-call",
				msg: fmt.Sprintf("%q is a LOCAL fixed-size array and a slice of it is handed to"+
					" %q. If that call takes an interface — io.ReadFull and its relatives — the"+
					" slice escapes and the array is HEAP-allocated on every invocation, which a"+
					" decode primitive pays once per scalar it reads. Hang the buffer on the"+
					" receiver, which is already on the heap, and the per-call allocation"+
					" disappears. MEASURED on the GGUF header reader, where u32 and u64 were"+
					" 1.34M objects of a 4.0M allocation profile: moving their arrays onto the"+
					" reader, plus reusing one scratch for string bodies, took"+
					" BenchmarkReadFileSynth/header-heavy from 223892 to 95804 allocations,"+
					" -57.2%%, and 5.45 to 4.50 ms, -17.3%%. SAFE ONLY IF NOTHING KEEPS THE"+
					" BUFFER — the string case works because string(b) COPIES. Check every"+
					" caller before sharing one scratch, and check the type is not used"+
					" concurrently", id.Name, calleeName(c.Fun)),
			})
			return false
		}
		return true
	})
	return out
}

// --- PS3070: one threshold serving two regimes -----------------------------------------------

// thresholdUses maps "pkg.CONST" to the distinct texts it is compared against, and to the
// functions those comparisons live in.
var thresholdUses = map[string]map[string]string{}

// collectThresholdComparisons pre-scans for comparisons against an untyped package constant.
func collectThresholdComparisons(files []*ast.File) {
	for _, f := range files {
		if f.Name == nil {
			continue
		}
		consts := map[string]bool{}
		for _, decl := range f.Decls {
			gd, ok := decl.(*ast.GenDecl)
			if !ok || gd.Tok != token.CONST {
				continue
			}
			for _, sp := range gd.Specs {
				vs, ok := sp.(*ast.ValueSpec)
				if !ok {
					continue
				}
				for _, nm := range vs.Names {
					consts[nm.Name] = true
				}
			}
		}
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				be, ok := n.(*ast.BinaryExpr)
				if !ok {
					return true
				}
				switch be.Op {
				case token.LSS, token.GTR, token.LEQ, token.GEQ:
				default:
					return true
				}
				for _, pair := range [][2]ast.Expr{{be.X, be.Y}, {be.Y, be.X}} {
					id, ok := pair[1].(*ast.Ident)
					if !ok || !consts[id.Name] {
						continue
					}
					key := f.Name.Name + "." + id.Name
					if thresholdUses[key] == nil {
						thresholdUses[key] = map[string]string{}
					}
					thresholdUses[key][exprText(pair[0])] = fn.Name.Name
				}
				return true
			})
		}
	}
}

// sharedThresholdFindings flags PS3070 — a constant compared against two or more different
// quantities, in two or more different functions.
func sharedThresholdFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil {
		return nil
	}
	var out []finding
	seen := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		id, ok := n.(*ast.Ident)
		if !ok || seen[id.Name] {
			return true
		}
		uses := thresholdUses[f.Name.Name+"."+id.Name]
		if len(uses) < 2 {
			return true
		}
		// TWO REGIMES, NOT TWO CALL SITES: the same quantity tested in two places is one
		// regime. Require the comparisons to live in different functions as well.
		fns := map[string]bool{}
		for _, where := range uses {
			fns[where] = true
		}
		if len(fns) < 2 || !fns[fn.Name.Name] {
			return true
		}
		seen[id.Name] = true
		var quantities []string
		for q := range uses {
			quantities = append(quantities, q)
		}
		sort.Strings(quantities)
		out = append(out, finding{
			pos:      fset.Position(id.Pos()),
			category: "one-threshold-two-regimes",
			msg: fmt.Sprintf("%q is compared against %d DIFFERENT quantities (%s) in %d"+
				" functions — one threshold serving callers whose costs do not move together."+
				" Whatever value suits one is wrong for the other, and tuning it looks like a"+
				" tradeoff when it is really a missing second constant. MEASURED: this"+
				" repository's tree builders shared one radix cutoff between a PER-NODE sort of"+
				" a shrinking range and a ONE-TIME presort of every row. Lowering it for the"+
				" first took BenchmarkForestFit 121.5 to 101.3 ms, -16.6%%, and simultaneously"+
				" cost GBMFit about 25%% and moved its bit-exact digest — which read as a"+
				" model-behavior change until the constants were split. Split, both are free:"+
				" ForestFit -16.6%%, GBMFit flat, BOTH digests unchanged. SPLIT FIRST, THEN"+
				" SWEEP — a sweep of a shared constant measures the sum of two answers and finds"+
				" neither."+
				" TRIAGE BEFORE SPLITTING, because two false-positive classes account for every"+
				" candidate this check found in its own repository. FIRST, IS IT A KNOB AT ALL:"+
				" a numeric guard — an underflow floor reused on two different quantities — is a"+
				" CORRECTNESS bound, and splitting it buys nothing and risks the invariant."+
				" SECOND, IS THE CROSSOVER BENCHMARKED: a size gate at 16 whose only benchmarks"+
				" run at 512 and 1024 cannot be swept at all, and a value nothing measures is"+
				" not a tuning opportunity but an unfalsifiable constant. Add a cell in the"+
				" crossover region or leave it alone; do not split on the shape alone",
				id.Name, len(uses), strings.Join(quantities, ", "), len(fns)),
		})
		return true
	})
	return out
}

// --- PS3069: a fan-out helper that queues work it need not queue ------------------------------

// queueingFanoutFindings flags PS3069 — a registered fan-out helper that dispatches through an
// unbuffered channel and has no direct path for the case where the jobs already fit the
// workers.
func queueingFanoutFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || !fanoutReg[f.Name.Name][fn.Name.Name] {
		return nil
	}
	// An UNBUFFERED channel of ints, made with no capacity argument, is the job queue.
	queues := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		c, ok := n.(*ast.CallExpr)
		if !ok || calleeName(c.Fun) != "make" || len(c.Args) != 1 {
			return true
		}
		if _, ok := c.Args[0].(*ast.ChanType); ok {
			queues = true
		}
		return true
	})
	if !queues {
		return nil
	}
	// A DIRECT PATH shows up as a `go` statement that is NOT inside a range over the channel —
	// the helper spawning one goroutine per job rather than one per worker. Approximated by
	// counting GoStmts: the queueing form has exactly one.
	gos := 0
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if _, ok := n.(*ast.GoStmt); ok {
			gos++
		}
		return true
	})
	if gos > 1 {
		return nil
	}
	return []finding{{
		pos:      fset.Position(fn.Pos()),
		category: "fanout-queues-jobs-it-does-not-need-to",
		msg: fmt.Sprintf("%q hands work out through an UNBUFFERED channel and has no path that"+
			" skips the queue when the jobs already fit the workers. The queue exists so more"+
			" jobs than workers can be served one at a time; when there are no more, every job"+
			" pays a rendezvous on top of its goroutine. MEASURED on the classic tree builders,"+
			" which call this helper with exactly one job per chunk once per NODE and thousands"+
			" of times per fit: a direct fan-out for n <= workers took BenchmarkGBMFit 72.06 to"+
			" 68.31 ms, -5.2%%, with system CPU 0.43 to 0.37 s and the forest and single-tree"+
			" cells flat. EXPECT THE SMALLER NUMBER, NOT THE PROFILE'S — 95.6%% of that"+
			" benchmark's samples sat in pthread_cond_wait, pthread_cond_signal and usleep"+
			" against 1.75%% in the split scan, and the fix was worth 5%%. Parked threads are"+
			" sampled, and sampled is not the same as costing wall clock."+
			" AND RE-SWEEP THE WORK GATES AFTERWARDS. Every gate that decides whether to fan out"+
			" was calibrated against the OLD cost of forking; making the helper cheaper leaves"+
			" them stale and too conservative. The GBM split gate moved from 1<<15 to 1<<13 on"+
			" that account and bought another 7.7%% — BenchmarkGBMFit 67.02 to 61.86 ms, every"+
			" run at the new gate below every run at the old — which is more than the helper"+
			" change itself was worth", fn.Name.Name),
	}}
}

// --- PS3068: a serial best-of scan over independent items ------------------------------------

// declaredIn returns the names declared by := anywhere inside n.
func declaredIn(n ast.Node) map[string]bool {
	out := map[string]bool{}
	ast.Inspect(n, func(m ast.Node) bool {
		as, ok := m.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE {
			return true
		}
		for _, lhs := range as.Lhs {
			if id, ok := lhs.(*ast.Ident); ok {
				out[id.Name] = true
			}
		}
		return true
	})
	return out
}

// serialBestOfScanFindings flags PS3068 — a loop that keeps a running best under a strict
// comparison, in a function that never fans out.
func serialBestOfScanFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || len(fanoutReg[f.Name.Name]) == 0 {
		return nil
	}
	reg := fanoutReg[f.Name.Name]
	// THE FAN-OUT MAY BE ONE CALL AWAY. Applying this check produces a function that dispatches
	// to a helper which fans out, and the caller itself then contains no helper call at all —
	// the converted CART feature scan was still reported for exactly that reason. A function
	// that reaches a fan-out through one of its own package's functions has made the choice.
	if callsFanoutHelper(fn.Body, reg) || callsFanoutHelper(fn.Body, fansOutReg[f.Name.Name]) {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		body, iv := outerLoop(n)
		if body == nil {
			return true
		}
		// A RANGE OVER VALUES has "_" for its index, and outerLoop reports that. The scan this
		// check is written from is written that way — for _, f := range feats — so taking the
		// index alone made the check silent on its own motivating site.
		if iv == "" || iv == "_" {
			rs, ok := n.(*ast.RangeStmt)
			if !ok || rs.Value == nil {
				return true
			}
			id, ok := rs.Value.(*ast.Ident)
			if !ok || id.Name == "_" {
				return true
			}
			iv = id.Name
		}
		if loopSpansAParameterRange(n, declaredParamNames(fn)) {
			return false
		}
		local := declaredIn(body)
		found := ""
		ast.Inspect(body, func(m ast.Node) bool {
			ifs, ok := m.(*ast.IfStmt)
			if !ok || found != "" {
				return found == ""
			}
			be, ok := ifs.Cond.(*ast.BinaryExpr)
			if !ok {
				return true
			}
			// The comparison must be STRICT — a first-wins rule — and one side must be a name
			// the loop does not declare, which is the running accumulator.
			if be.Op != token.LSS && be.Op != token.GTR {
				return true
			}
			acc := ""
			for _, e := range []ast.Expr{be.X, be.Y} {
				if id, ok := e.(*ast.Ident); ok && !local[id.Name] && id.Name != iv {
					acc = id.Name
				}
			}
			if acc == "" {
				return true
			}
			// The body must write BACK to names the loop does not declare — the winner's
			// fields. One assignment is an ordinary running total; two or more is a record.
			n := 0
			ast.Inspect(ifs.Body, func(k ast.Node) bool {
				as, ok := k.(*ast.AssignStmt)
				if !ok {
					return true
				}
				for _, lhs := range as.Lhs {
					if id, ok := lhs.(*ast.Ident); ok && !local[id.Name] {
						n++
					}
				}
				return true
			})
			if n >= 2 {
				found = acc
			}
			return found == ""
		})
		if found == "" {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(n.Pos()),
			category: "serial-best-of-scan",
			msg: fmt.Sprintf("this loop over %q keeps a running best in %q and records the"+
				" winner's fields under a STRICT comparison, and the function never calls the"+
				" fan-out helper its package declares. Chunk the items, let every chunk apply the"+
				" same rule within itself, then FOLD THE CHUNKS IN ASCENDING ORDER WITH THE SAME"+
				" STRICT COMPARISON — that reproduces the serial winner exactly, ties included,"+
				" because first-wins survives both levels. MEASURED on the CART feature scan,"+
				" which ran at 1.01x on twelve cores: BenchmarkTreeFit 10.20 to 9.69 ms with the"+
				" forest fit flat as a control. Modest, because a tree's work sits in a few large"+
				" nodes and the rest is below the fork gate — expect the gate to matter more than"+
				" the core count. A DIGEST WILL NOT GATE THE TIE RULE: mutations flipping either"+
				" strict comparison to <= left a bit-exact prediction digest green, because"+
				" ordinary data produces no exact tie between two items. MANUFACTURE ONE —"+
				" duplicate an item so two are exactly equal — and assert the LOWER index wins,"+
				" in BOTH placements: adjacent items share a chunk and exercise only the inner"+
				" comparison, distant ones land in different chunks and exercise only the fold."+
				" A KNOWN FALSE POSITIVE REMAINS: a scan inside a function whose CALLER runs it"+
				" in a fan-out band is already parallel and does not look it. This check sees"+
				" one call in the other direction — a function that dispatches to a helper which"+
				" fans out — but not the caller's. Three of its reports in the CART builder are"+
				" the per-feature cut scan, which runs inside the feature band",
				iv, found),
		})
		return false
	})
	return out
}

// --- PS3067: a sequential outer loop with an independent inner one ---------------------------

// interchangeBeforeBandFindings flags PS3067 — a nest whose outer loop carries a dependence and
// whose inner loop is independent, where the fix is to INTERCHANGE and band the outer position
// rather than to split the inner loop where it stands.
//
// This is PS3040's nest seen from the other side. That check is right for a factorization,
// whose inner extent shrinks as the outer index advances so the rows really must be split per
// pivot; it is wrong for a sweep whose inner extent is CONSTANT, where interchanging turns one
// fork per outer step into one fork per call. The two are told apart here by whether the inner
// loop's bound mentions the outer variable.
func interchangeBeforeBandFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || len(fanoutReg[f.Name.Name]) == 0 {
		return nil
	}
	reg := fanoutReg[f.Name.Name]
	if callsFanoutHelper(fn.Body, reg) {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		outer, ov := outerLoop(n)
		if outer == nil || ov == "" || ov == "_" || loopDepthOf(outer) < 1 {
			return true
		}
		if loopSpansAParameterRange(n, declaredParamNames(fn)) {
			return false
		}
		for _, st := range outer.List {
			mid, mv := outerLoop(st)
			if mid == nil || mv == "" || mv == "_" || mv == ov {
				continue
			}
			// THE INDEPENDENT LOOP MUST CARRY REAL WORK — a loop of its own inside it. Without
			// this the check reported 210 sites tree-wide, nearly all of them ordinary
			// elementwise inner loops where an interchange buys nothing and the fork gate would
			// keep them serial anyway. Both measured cases have a compensation loop inside.
			if loopDepthOf(mid) < 1 {
				continue
			}
			// CONSTANT INNER EXTENT is what separates this from PS3040. The shrink lives in
			// the INIT as often as in the bound — for i := k+1; i < n; i++ has a bound of n and
			// is still a triangle — so both are tested.
			if fs, ok := st.(*ast.ForStmt); ok {
				if (fs.Init != nil && stmtMentions(fs.Init, ov)) ||
					(fs.Cond != nil && exprMentions(fs.Cond, ov)) {
					continue
				}
			}
			ok, dst := true, ""
			ast.Inspect(mid, func(m ast.Node) bool {
				as, aok := m.(*ast.AssignStmt)
				if !aok {
					return true
				}
				for _, lhs := range as.Lhs {
					ix, iok := unparen(lhs).(*ast.IndexExpr)
					if !iok {
						continue
					}
					// A JAGGED WRITE HAS TWO INDICES and the owning one is the INNER one:
					// wm[r][j] parses as (wm[r])[j], so testing only the outer index asks
					// whether j names the row variable and always answers no. Walk the chain.
					base, idxs := ix.X, []ast.Expr{ix.Index}
					for {
						inner, more := unparen(base).(*ast.IndexExpr)
						if !more {
							break
						}
						idxs = append(idxs, inner.Index)
						base = inner.X
					}
					owned := false
					for _, e := range idxs {
						if mentionsIdent(e, ov) {
							ok = false
						}
						if mentionsIdent(e, mv) {
							owned = true
						}
					}
					if !owned {
						ok = false
						continue
					}
					if dst == "" {
						dst = exprText(base)
					}
				}
				return true
			})
			if !ok || dst == "" {
				continue
			}
			out = append(out, finding{
				pos:      fset.Position(st.Pos()),
				category: "sequential-outer-with-an-independent-inner-loop",
				msg: fmt.Sprintf("the loop over %q is independent — every indexed write in it,"+
					" including into %q, names %q and none names the enclosing %q — while the"+
					" loop over %q carries a dependence. Banding %q WHERE IT STANDS pays one"+
					" fork per %q step, which has failed on fork count four times in this"+
					" repository. INTERCHANGE FIRST so %q is outermost, then band it once per"+
					" call. PS3040 describes the same nest and recommends the in-place split;"+
					" that is right for a factorization, whose inner extent SHRINKS as the outer"+
					" index advances, and wrong here, where the inner extent is constant."+
					" MEASURED TWICE: the SparseGPT pruner went 52.27 to 41.83 ms (-20.0%%) and"+
					" the GPTQ quantizer 55.18 to 44.34 ms (-19.6%%), both bit-identical with an"+
					" untouched sibling flat as a control. TWO THINGS TO CHECK: any scratch the"+
					" body reuses must become PER WORKER, since a shared sort buffer reddened"+
					" both the digest and -race; and a CALLER-SUPPLIED CALLBACK inside the nest"+
					" is now called concurrently, which had to become a documented requirement"+
					" on GPTQ's exported API rather than an implementation detail",
					mv, dst, mv, ov, ov, mv, ov, mv),
			})
			return false
		}
		return true
	})
	return out
}

// --- PS3066: consecutive loops that each stream one buffer -----------------------------------

// loopBoundText returns a stable text for a loop's range, or "" when the loop is not a simple
// counted or range loop.
func loopBoundText(n ast.Node) string {
	switch t := n.(type) {
	case *ast.RangeStmt:
		return exprText(t.X)
	case *ast.ForStmt:
		be, ok := t.Cond.(*ast.BinaryExpr)
		if !ok || be.Op != token.LSS {
			return ""
		}
		return exprText(be.Y)
	}
	return ""
}

// indexedBases returns the names indexed or SLICED anywhere inside n. Slicing counts: a stage
// that hands one row to a helper writes S[r*dk : r*dk+dk], which is the same traffic over the
// same buffer as indexing it, and counting only index expressions made the check silent on the
// very step it was written from.
func indexedBases(n ast.Node) map[string]bool {
	out := map[string]bool{}
	add := func(e ast.Expr) {
		if id, ok := e.(*ast.Ident); ok {
			out[id.Name] = true
		}
	}
	ast.Inspect(n, func(m ast.Node) bool {
		switch t := m.(type) {
		case *ast.IndexExpr:
			add(t.X)
		case *ast.SliceExpr:
			add(t.X)
		}
		return true
	})
	return out
}

// consecutiveLoopFindings flags PS3066 — three or more sibling loops over the same bound that
// all index the same buffer.
func consecutiveLoopFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if len(out) > 0 {
			return false
		}
		blk, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		run, bound := 0, ""
		var shared map[string]bool
		var first ast.Node
		flush := func() {
			// TWO IS ENOUGH, and that was measured rather than assumed. This check shipped
			// requiring three because two passes looked like an ordinary fill-then-use; the
			// DeltaNet recurrence has exactly two over its state and merging them measured
			// -15.3%%, against -26.4%% for the three-pass gated variant beside it.
			if run >= 2 && len(shared) > 0 && len(out) == 0 {
				var buf string
				for nm := range shared {
					if buf == "" || nm < buf {
						buf = nm
					}
				}
				out = append(out, finding{
					pos:      fset.Position(first.Pos()),
					category: "consecutive-loops-over-one-buffer",
					msg: fmt.Sprintf("%d sibling loops here all run over %q and all index %q, so"+
						" each streams the whole buffer and evicts it before the next one starts."+
						" If every index is INDEPENDENT ACROSS THE LOOPS, merge them — the buffer"+
						" is then touched once and stays in cache through all the stages."+
						" MEASURED on the Kimi delta-attention step, whose four per-step passes"+
						" over the state (scale, S·k, the rank-1 delta write, S·q) became one:"+
						" BenchmarkKDA_F64_256x128 8.71 to 7.33 ms, -15.9%%. BIT-IDENTICAL,"+
						" because merging changes only WHEN a row is visited and never how —"+
						" each row's stages already ran in that order relative to each other."+
						" THE WIN IS THE BUFFER SIZE, NOT THE LOOP COUNT: the same merge on a"+
						" 64x64 state, 32 KB and already L1-resident, measured -1.8%%, and TWO"+
						" passes are worth merging as readily as four — the DeltaNet recurrence"+
						" has exactly two over its state and merging them measured -15.3%%,"+
						" against -26.4%% for the three-pass gated variant beside it."+
						" MOST CANDIDATES ARE NOT MERGEABLE and the count is not a work list:"+
						" of the four this check found at its old threshold, three were correct"+
						" in shape and had to be rejected — the SOAP and Shampoo preconditioners"+
						" accumulate a whole intermediate ACROSS the shared index before the next"+
						" loop reads it, and the Titans scan builds its backward vector from"+
						" every row of a matrix it then overwrites, with a comment saying the"+
						" old values are required. Check for"+
						" a cross-index dependency first — a later loop that needs ALL of an"+
						" earlier loop's output cannot merge", run, bound, buf),
				})
			}
			run, bound, shared, first = 0, "", nil, nil
		}
		for _, st := range blk.List {
			b := loopBoundText(st)
			if b == "" {
				flush()
				continue
			}
			bases := indexedBases(st)
			if run > 0 && b == bound {
				next := map[string]bool{}
				for nm := range shared {
					if bases[nm] {
						next[nm] = true
					}
				}
				if len(next) > 0 {
					run++
					shared = next
					continue
				}
			}
			flush()
			run, bound, shared, first = 1, b, bases, st
		}
		flush()
		return true
	})
	return out
}

// --- PS3065: a serial loop whose per-item work lives in a callee ------------------------------

// loopyFuncReg maps a package name to the functions and methods in it whose body contains a
// loop — the ones whose per-call cost is O(something) rather than O(1).
var loopyFuncReg = map[string]map[string]bool{}

func collectLoopyFuncs(files []*ast.File) {
	for _, f := range files {
		if f.Name == nil {
			continue
		}
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			loops := false
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				switch n.(type) {
				case *ast.ForStmt, *ast.RangeStmt:
					loops = true
				}
				return !loops
			})
			if !loops {
				continue
			}
			if loopyFuncReg[f.Name.Name] == nil {
				loopyFuncReg[f.Name.Name] = map[string]bool{}
			}
			loopyFuncReg[f.Name.Name][fn.Name.Name] = true
		}
	}
}

// serialLoopOverCallFindings flags PS3065 — a single loop writing an indexed destination from a
// call to a function that itself loops.
func serialLoopOverCallFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || len(fanoutReg[f.Name.Name]) == 0 {
		return nil
	}
	reg := fanoutReg[f.Name.Name]
	if callsFanoutHelper(fn.Body, reg) {
		return nil
	}
	loopy := loopyFuncReg[f.Name.Name]
	var out []finding
	// EVERY loop in the function, not only the first — the same correction PS3059 needed, and
	// for the same reason: which one you hear about was an accident of source order.
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		body, iv := outerLoop(n)
		// EXACTLY ONE LOOP. With an inner loop the nest checks own the shape; the whole point
		// here is the case they cannot see, where the per-item work is behind a call.
		if body == nil || iv == "" || iv == "_" || loopDepthOf(body) != 0 {
			return true
		}
		if loopSpansAParameterRange(n, declaredParamNames(fn)) {
			return false
		}
		callee, dst := "", ""
		ast.Inspect(body, func(m ast.Node) bool {
			as, ok := m.(*ast.AssignStmt)
			if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 || callee != "" {
				return callee == ""
			}
			ix, ok := unparen(as.Lhs[0]).(*ast.IndexExpr)
			if !ok || !mentionsIdent(ix.Index, iv) {
				return true
			}
			c, ok := as.Rhs[0].(*ast.CallExpr)
			if !ok || !loopy[calleeName(c.Fun)] {
				return true
			}
			callee, dst = calleeName(c.Fun), exprText(ix.X)
			return false
		})
		if callee == "" {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(n.Pos()),
			category: "serial-loop-over-an-expensive-call",
			msg: fmt.Sprintf("this loop fills %q one entry per iteration from %q, which itself"+
				" loops, and the function never calls the fan-out helper its package declares."+
				" EVERY NEST-BASED CHECK MISSES THIS: PS3034, PS3059 and PS3063 all require a"+
				" depth of 2 or more, and a loop whose per-item work lives inside a CALLEE has"+
				" depth 1 however expensive that callee is. MEASURED on the RBF kernel cache,"+
				" where a column is n independent kernel evaluations: BenchmarkSVCFit/n4000_rbf"+
				" 6.99 to 5.48 ms, -21.5%%, with the GBM fit flat as a control, on a fit that had"+
				" scaled at 1.02x on twelve cores. Bit-identical — each entry performs exactly the"+
				" arithmetic it did before and only which goroutine performs it moves."+
				" EXPECT AMDAHL, NOT THE CORE COUNT: the parallelized part was 40.6%% of a serial"+
				" profile, so 6x on it is 1.27x overall, and the solver it feeds is sequential by"+
				" construction", dst, callee),
		})
		return false
	})
	return out
}

// --- PS3064: a jagged matrix allocated one row at a time -------------------------------------

// jaggedMatrixFindings flags PS3064 — an outer make([][]T, r) whose rows are then filled in by a
// loop of per-row make() calls.
func jaggedMatrixFindings(fset *token.FileSet, fn *ast.FuncDecl) []finding {
	if fn.Body == nil {
		return nil
	}
	// Names assigned an outer make([][]T, ...), and where they were assigned.
	outer := map[string]ast.Node{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok {
			return true
		}
		for i, lhs := range as.Lhs {
			id, ok := lhs.(*ast.Ident)
			if !ok || i >= len(as.Rhs) {
				continue
			}
			c, ok := as.Rhs[i].(*ast.CallExpr)
			if !ok || calleeName(c.Fun) != "make" || len(c.Args) == 0 {
				continue
			}
			at, ok := c.Args[0].(*ast.ArrayType)
			if !ok || at.Len != nil {
				continue
			}
			if inner, ok := at.Elt.(*ast.ArrayType); !ok || inner.Len != nil {
				continue
			}
			outer[id.Name] = as
		}
		return true
	})
	if len(outer) == 0 {
		return nil
	}
	var out []finding
	seen := map[string]bool{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		body, iv := outerLoop(n)
		if body == nil || iv == "" {
			return true
		}
		ast.Inspect(body, func(m ast.Node) bool {
			as, ok := m.(*ast.AssignStmt)
			if !ok || len(as.Lhs) != len(as.Rhs) {
				return true
			}
			for i, lhs := range as.Lhs {
				ix, ok := unparen(lhs).(*ast.IndexExpr)
				if !ok {
					continue
				}
				id, ok := ix.X.(*ast.Ident)
				if !ok || outer[id.Name] == nil || !mentionsIdent(ix.Index, iv) {
					continue
				}
				c, ok := as.Rhs[i].(*ast.CallExpr)
				if !ok || calleeName(c.Fun) != "make" || seen[id.Name] {
					continue
				}
				seen[id.Name] = true
				out = append(out, finding{
					pos:      fset.Position(as.Pos()),
					category: "jagged-matrix-allocated-row-by-row",
					msg: fmt.Sprintf("%q is a [][]T whose rows are allocated ONE AT A TIME, so an"+
						" r-by-c matrix costs r+1 allocations and its rows land wherever the heap"+
						" puts them. Back them with ONE block and window into it —"+
						" d[i] = base[i*c:(i+1)*c:(i+1)*c] — and NO CALL SITE CHANGES, because the"+
						" type does not. MEASURED across the three autograd factorization"+
						" adjoints: allocations per call fell 1886 to 356 on the"+
						" eigendecomposition, 1762 to 491 on the SVD and 1447 to 429 on the QR, a"+
						" 70 to 81%% cut. THE CLOCK IS THE SMALLER HALF AND IS SHAPE-DEPENDENT —"+
						" SVD -8.6%%, eigh -5.4%%, QR flat, with an untouched sibling adjoint flat"+
						" as a control — so report it as a resource win and let the time be a"+
						" bonus. CAP THE ROW WINDOW at its own length so an append copies instead"+
						" of reaching into the next row", id.Name),
				})
			}
			return true
		})
		return true
	})
	return out
}

// --- PS3063: one loop left serial in a function that fans out elsewhere ----------------------

// indexedWriteTargets returns the index expressions of every write in body, counting BOTH
// dst[E] = v and a setter call used as a statement, recv.SetF64(v, i, j). The second form is
// what the checks that came before this one could not see: the eigendecomposition adjoint
// writes its result entirely through SetF64, so a walk that only looked for index expressions
// found no writes at all and concluded nothing.
func indexedWriteTargets(body ast.Node, accessors map[string]bool) []ast.Expr {
	var out []ast.Expr
	ast.Inspect(body, func(n ast.Node) bool {
		switch t := n.(type) {
		case *ast.AssignStmt:
			for _, lhs := range t.Lhs {
				if ix, ok := unparen(lhs).(*ast.IndexExpr); ok {
					out = append(out, ix.Index)
				}
			}
		case *ast.ExprStmt:
			// A call whose value is discarded is a write; the same name read for its value is
			// not. args[0] is the value, the rest are the indices.
			c, ok := t.X.(*ast.CallExpr)
			if !ok || !accessors[calleeName(c.Fun)] || len(c.Args) < 2 {
				return true
			}
			out = append(out, c.Args[1:]...)
		}
		return true
	})
	return out
}

// serialLoopInFanningFuncFindings flags PS3063 — a nest left serial in a function that fans out
// somewhere else.
func serialLoopInFanningFuncFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl,
	ns nameSets) []finding {
	if fn.Body == nil || f.Name == nil || len(fanoutReg[f.Name.Name]) == 0 {
		return nil
	}
	reg := fanoutReg[f.Name.Name]
	if !callsFanoutHelper(fn.Body, reg) {
		return nil // PS3034 and PS3059 own the function that never fans out at all
	}
	// A CLOSURE CALLED FROM INSIDE A CALLBACK IS ALREADY PARALLEL, and it does not look it:
	// the literal is assigned to a name at the top of the function and only invoked from
	// within the fan-out, so nothing about its syntax says so. The conv2d backward kernel
	// declares col2im that way and was reported for it.
	fanned := map[*ast.FuncLit]bool{}
	assigned := map[string]*ast.FuncLit{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		id, iok := as.Lhs[0].(*ast.Ident)
		fl, fok := as.Rhs[0].(*ast.FuncLit)
		if iok && fok {
			assigned[id.Name] = fl
		}
		return true
	})
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		c, ok := n.(*ast.CallExpr)
		if !ok || !reg[calleeName(c.Fun)] {
			return true
		}
		ast.Inspect(c, func(m ast.Node) bool {
			ic, ok := m.(*ast.CallExpr)
			if !ok {
				return true
			}
			if fl := assigned[calleeName(ic.Fun)]; fl != nil {
				fanned[fl] = true
			}
			return true
		})
		return true
	})
	var out []finding
	// EVERY loop in the function, not only the first — the same correction PS3059 needed, and
	// for the same reason: which one you hear about was an accident of source order.
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		// Anything inside a fan-out callback is already parallel one level out.
		if c, ok := n.(*ast.CallExpr); ok && reg[calleeName(c.Fun)] {
			return false
		}
		if fl, ok := n.(*ast.FuncLit); ok && fanned[fl] {
			return false
		}
		body, ov := outerLoop(n)
		if body == nil || ov == "" || ov == "_" || loopDepthOf(body) < 2 {
			return true
		}
		if loopSpansAParameterRange(n, declaredParamNames(fn)) {
			return false
		}
		// THE NEST ITSELF MUST NOT CONTAIN A DISPATCH. Two shapes hide here and one test
		// covers both. A converted loop keeps a plain duplicated arm beside its gated
		// dispatch — the advice PS3040 gives, since routing small inputs through the callback
		// costs a few percent — so the two sit in one block; and a sequential outer loop that
		// fans out its inner one, which is PS3040's own shape, contains the dispatch directly.
		// The LU factorization and the AQLM Gauss-Jordan are both already converted and both
		// were reported by their own leftovers. This is the THIRD check to need this guard.
		if callsFanoutHelper(body, reg) {
			return false
		}
		names := namesDerivedFromLoopVar(body, ov)
		idxs := indexedWriteTargets(body, ns.accessors)
		if len(idxs) == 0 {
			return true
		}
		for _, e := range idxs {
			owned := false
			for nm := range names {
				if mentionsIdent(e, nm) {
					owned = true
					break
				}
			}
			if !owned {
				return true // a write escapes this iteration; the loop is not independent
			}
		}
		out = append(out, finding{
			pos:      fset.Position(n.Pos()),
			category: "one-loop-left-serial-in-a-fanning-function",
			msg: fmt.Sprintf("this nest over %q is SERIAL, and the function it sits in already"+
				" fans out somewhere else — so the transform is available here and was simply not"+
				" applied. PS3034 and PS3059 both suppress this case, requiring the function to"+
				" never call a fan-out helper, and that suppression hides the most valuable shape"+
				" there is: a function that fans out has already proven it knows how."+
				" TWICE MEASURED, BOTH LARGE. The KAN forward fanned out buildBasis and not"+
				" fusedSpline, which was 84%% of the layer — banding it went 67.28 to 10.90 ms, a"+
				" 6.2x. The eigendecomposition adjoint fanned out its first two n^3 products and"+
				" not the triangular third — banding that, plus mirroring the intermediate it read"+
				" down its columns, went 18.63 to 8.17 ms at n=256, of which the banding was about"+
				" 2x and the mirror about 15%%. CHECK THE WRITES OWN DISJOINT OUTPUT: a triangular"+
				" loop writing (i,j) and (j,i) does, because a clash would force the two indices"+
				" equal. Gate with BOTH a bit-exact digest and -race — a band that skips work"+
				" fails the first and an overlapping one can fail only the second", ov),
		})
		return false
	})
	return out
}

// --- PS3062: an op with no optimized kernel --------------------------------------------------

// kernelReg maps a backend package name to the operations it registers a kernel for.
var kernelReg = map[string]map[string]bool{}

// collectKernelRegistrations pre-scans every package for std.add(backend.OpX, ...) calls.
func collectKernelRegistrations(files []*ast.File, ns nameSets) {
	for _, f := range files {
		if f.Name == nil {
			continue
		}
		ast.Inspect(f, func(n ast.Node) bool {
			c, ok := n.(*ast.CallExpr)
			if !ok || !ns.kernelRegisterFuncs[calleeName(c.Fun)] || len(c.Args) == 0 {
				return true
			}
			sel, ok := c.Args[0].(*ast.SelectorExpr)
			if !ok {
				return true
			}
			if kernelReg[f.Name.Name] == nil {
				kernelReg[f.Name.Name] = map[string]bool{}
			}
			kernelReg[f.Name.Name][sel.Sel.Name] = true
			return true
		})
	}
}

// refOnlyKernelFindings flags PS3062 — an op the reference backend implements and the optimized
// cpu backend does not, reported once at the reference's registration site.
func refOnlyKernelFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl, ns nameSets) []finding {
	if fn.Body == nil || f.Name == nil || ns.refBackendPkg == "" || f.Name.Name != ns.refBackendPkg {
		return nil
	}
	// An op is covered when ANY configured optimized backend registers it.
	covered := func(op string) bool {
		for pkg := range ns.optBackendPkgs {
			if kernelReg[pkg][op] {
				return true
			}
		}
		return false
	}
	if len(ns.optBackendPkgs) == 0 {
		return nil
	}
	seen := map[string]bool{}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		c, ok := n.(*ast.CallExpr)
		if !ok || !ns.kernelRegisterFuncs[calleeName(c.Fun)] || len(c.Args) == 0 {
			return true
		}
		sel, ok := c.Args[0].(*ast.SelectorExpr)
		if !ok {
			return true
		}
		op := sel.Sel.Name
		if covered(op) || seen[op] {
			return true
		}
		seen[op] = true
		out = append(out, finding{
			pos:      fset.Position(c.Pos()),
			category: "op-with-no-optimized-kernel",
			msg: fmt.Sprintf("%q is registered here in the REFERENCE backend and in no optimized"+
				" one, so every caller on the default backend runs the correctness implementation."+
				" The reference is written to be obviously right, not fast, and the gap is"+
				" routinely large: OpCholesky was in exactly this position, and giving it a cpu"+
				" kernel — ref's arithmetic line for line, with FOUR ROWS TAKEN PER PASS so the"+
				" pivot row is loaded once and four independent accumulator chains run instead of"+
				" one — went 21.1 ms to 7.0 ms at n=512, a 3.0x, bit-identical to ref in both"+
				" dtypes. THE FIRST ATTEMPT AT THAT KERNEL WAS SLOWER THAN REF: fanning out the"+
				" row update is the classical-factorization shape PS3040 describes, and it lost —"+
				" 24.1 ms against 22.0, with allocations going from 6 to 878 — because there are n"+
				" columns and the fork is paid once per column for a row update that shrinks as"+
				" the factorization proceeds. Arithmetic, not parallelism, was the lever."+
				" GATE THE NEW KERNEL BIT-FOR-BIT AGAINST REF, not to a tolerance: a tolerance"+
				" test passes a blocked or reassociated version too, and once one kernel disagrees"+
				" every cross-backend golden silently becomes a tolerance test. RANK BY CALLER"+
				" HOTNESS — an op nothing calls on a hot path is a gap on paper only", op),
		})
		return true
	})
	return out
}

// --- PS3061: a fan-out helper that never sizes itself to the work ----------------------------

// unsizedFanoutFindings flags PS3061 — a fan-out helper whose worker count is GOMAXPROCS,
// reduced only by the item count and by an on/off work gate, with nothing scaling the number of
// workers to the amount of work.
func unsizedFanoutFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || !fanoutReg[f.Name.Name][fn.Name.Name] {
		return nil
	}
	// The name the worker count lives in — whatever GOMAXPROCS was assigned to.
	procs := ""
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 || procs != "" {
			return procs == ""
		}
		if !mentionsCall(as.Rhs[0], "GOMAXPROCS") {
			return true
		}
		if id, ok := as.Lhs[0].(*ast.Ident); ok {
			procs = id.Name
		}
		return true
	})
	if procs == "" {
		return nil
	}
	// SIZED means a division flows into that worker count. The test is per STATEMENT — the
	// statement must both divide and assign the worker count — because a helper that divides
	// the ITEM count to get a chunk size, csz := (n + nw - 1) / nw, is not sizing anything:
	// it partitions a range whose worker count was already decided. Testing for a division
	// anywhere in the body reported 3 helpers instead of 9, having suppressed every one of
	// them on their chunk arithmetic.
	sized := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if sized {
			return false
		}
		// Only an assignment or an if — NOT any statement. A block is a statement too, and the
		// function body is a block, so testing every ast.Stmt made the outermost node match
		// whenever the helper divided anywhere and assigned the worker count anywhere. Every
		// helper in the tree does both, so the check reported 2 sites and all of them were the
		// ones that happened not to divide at all.
		var st ast.Stmt
		switch t := n.(type) {
		case *ast.AssignStmt:
			st = t
		case *ast.IfStmt:
			st = t
		default:
			return true
		}
		divides, assigns := false, false
		ast.Inspect(st, func(m ast.Node) bool {
			if be, ok := m.(*ast.BinaryExpr); ok && be.Op == token.QUO {
				divides = true
			}
			if as, ok := m.(*ast.AssignStmt); ok {
				for _, lhs := range as.Lhs {
					if id, ok := lhs.(*ast.Ident); ok && id.Name == procs {
						assigns = true
					}
				}
			}
			return true
		})
		if divides && assigns {
			sized = true
		}
		return !sized
	})
	if sized {
		return nil
	}
	return []finding{{
		pos:      fset.Position(fn.Pos()),
		category: "fanout-not-sized-to-the-work",
		msg: fmt.Sprintf("%q always splits into GOMAXPROCS workers: nothing here scales the WORKER"+
			" COUNT to the work, only an on/off gate decides whether to split at all. Above that"+
			" gate a small item still fans twelve ways and the wakeups cost more than the split"+
			" saves. MEASURED on the quantized decode matmul, called thousands of times per"+
			" generation with one activation row: a 500-token quantized Llama generation spent"+
			" 88%% of its profile samples in pthread_cond_signal and pthread_cond_wait and 1.5%%"+
			" in the kernel. One worker per 1<<15 of element-work, capped by GOMAXPROCS, beat BOTH"+
			" the fixed twelve and no fan-out at all — QuantLlamaGenerate500 549.7 to 527.4 ms and"+
			" system CPU -27%%, with the prefill cell flat as a control. THE ON/OFF GATE CANNOT"+
			" EXPRESS THIS: forcing the same call serial cost 8%% of the clock, and leaving it at"+
			" twelve burned 42%% more user CPU to buy that 8%% back. PICK THE GRAIN BY MEASURING"+
			" BOTH THE CLOCK AND THE CPU — a coarser grain traded 4%% of the clock for another"+
			" 44%% of the system time here, and which side of that is right depends on whether the"+
			" machine serves one request or many."+
			" DO NOT APPLY THIS BLINDLY TO EVERY HELPER — IT IS NEUTRAL OUTSIDE THE DECODE REGIME."+
			" The same transform on the nn helpers, which serve large ops called a few times"+
			" rather than small ops called thousands of times, measured FLAT on three benchmarks"+
			" and flat on CPU: HQQuantize 91.2 to 90.3 ms, KAN 11.99 to 11.91, TPA 55.9 to 54.2,"+
			" user CPU 25.7-26.1 s on both arms. It was not shipped, because neutral is not a"+
			" reason to add a knob. What made the decode case different is CALL FREQUENCY AT"+
			" NEAR-GATE WORK: thousands of calls per generation, each just above the threshold."+
			" A first, short run there read as a 6%% REGRESSION and a longer one showed noise —"+
			" measure this one with enough iterations to separate the arms before believing"+
			" either sign. And check whether the helper already sizes itself: the cpu backend's"+
			" pool does, which is why it is absent from this check's candidates, and raising its"+
			" per-worker floor made decode monotonically worse (522, 529, 542 ms at 1<<15, 1<<16,"+
			" 1<<17) — its constants are already at their optimum. THAT POOL IS NOW CLOSED AS A"+
			" TUNING TARGET: its dense-regime ceiling and its dense-regime time gap were swept"+
			" too, against a vision workload and an elementwise control, and neither moved"+
			" anything worth having (ViT -6%% at best from a 16x wider ceiling; a wider gap cost"+
			" Swin 79%% and the multi-token-attention forward 16%%). Forcing every op into the"+
			" dense regime looked like -12%% on Swin and then produced a 2.5x outlier on ViT."+
			" WHAT THE PROFILE ACTUALLY SAYS is not a tuning problem: a batched ViT forward"+
			" burns 11.1x CPU to deliver 3.86x of speedup, with 82%% of its samples in"+
			" pthread_cond_wait and pthread_cond_signal and 7.8%% in the GEMM — the same shape as"+
			" quantized decode. The lever there is FEWER AND LARGER OPERATIONS, which is kernel"+
			" fusion, and no constant in the pool reaches it", fn.Name.Name),
	}}
}

// mentionsCall reports whether e contains a call to a function with the given name.
func mentionsCall(e ast.Expr, name string) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if c, ok := n.(*ast.CallExpr); ok && calleeName(c.Fun) == name {
			found = true
		}
		return !found
	})
	return found
}

// --- PS3060: a serial loop over work that is itself parallel ---------------------------------

// fansOutReg maps a package name to the functions and methods in it whose body calls one of the
// package's fan-out helpers. Package-level, because the caller that loops over them is
// routinely in another file.
var fansOutReg = map[string]map[string]bool{}

// collectFanningFuncs pre-scans every package for those functions. It runs after
// collectFanoutHelpers, whose registry it reads.
func collectFanningFuncs(files []*ast.File) {
	for _, f := range files {
		if f.Name == nil || len(fanoutReg[f.Name.Name]) == 0 {
			continue
		}
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			if !callsFanoutHelper(fn.Body, fanoutReg[f.Name.Name]) {
				continue
			}
			if fansOutReg[f.Name.Name] == nil {
				fansOutReg[f.Name.Name] = map[string]bool{}
			}
			fansOutReg[f.Name.Name][fn.Name.Name] = true
		}
	}
}

// serialLoopOverParallelWorkFindings flags PS3060 — a serial loop whose body calls a function
// that fans out, in a function that does not.
func serialLoopOverParallelWorkFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || len(fansOutReg[f.Name.Name]) == 0 {
		return nil
	}
	reg := fansOutReg[f.Name.Name]
	// A function that already fans out somewhere has made the choice; this check is about the
	// level that is missing entirely. Self-recursion would otherwise report every fanning
	// function that loops.
	if callsFanoutHelper(fn.Body, fanoutReg[f.Name.Name]) || reg[fn.Name.Name] {
		return nil
	}
	var out []finding
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if len(out) > 0 {
			return false
		}
		// THE LOOP MUST ITERATE OVER ITEMS, NOT REPEAT A REFINEMENT. A loop with no variable —
		// "for range steps" — is a sequential refinement almost every time, and the Newton-Schulz
		// iteration this check was written from is exactly that: step i+1 reads what step i
		// wrote. Requiring a variable that the body actually uses separates the two, and it is
		// what stops the check reporting the inside of the function it was just used to fix.
		var body *ast.BlockStmt
		iv := ""
		switch t := n.(type) {
		case *ast.ForStmt:
			body = t.Body
			if as, ok := t.Init.(*ast.AssignStmt); ok && len(as.Lhs) > 0 {
				if id, ok := as.Lhs[0].(*ast.Ident); ok {
					iv = id.Name
				}
			}
		case *ast.RangeStmt:
			body = t.Body
			if id, ok := t.Key.(*ast.Ident); ok {
				iv = id.Name
			}
		default:
			return true
		}
		if iv == "" || iv == "_" || !mentionsIdent(body, iv) {
			return true
		}
		callee := ""
		ast.Inspect(body, func(m ast.Node) bool {
			c, ok := m.(*ast.CallExpr)
			if !ok || callee != "" {
				return callee == ""
			}
			if nm := calleeName(c.Fun); reg[nm] {
				callee = nm
			}
			return callee == ""
		})
		if callee == "" {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(n.Pos()),
			category: "serial-loop-over-parallel-work",
			msg: fmt.Sprintf("this loop calls %q, which ITSELF fans out, from a function that does"+
				" not fan out at all. The inner level is parallel and the outer one is not, so the"+
				" items run strictly one after another and each pays its own fork and join."+
				" THE WIN IS NOT ONLY THE WORK, IT IS THE STALLS: a Muon optimizer step spent 62%%"+
				" of its profile samples in pthread_cond_wait and pthread_cond_signal and only"+
				" 32%% in the matmul, because each parameter's five Newton-Schulz iterations fork"+
				" three times each and nothing overlaps them. Banding the parameter loop returned"+
				" -34.1%% with only TWO parameters, far more than two-way overlap of the work"+
				" alone can explain. CALL ANY CALLER-SUPPLIED CALLBACK SERIALLY FIRST — a gradient"+
				" or loss function belongs to the caller and is not documented as safe to call"+
				" from several goroutines, so hoist it into a serial pass and fan out only what"+
				" follows. And SIZE THE TEST ABOVE THE FAN-OUT HELPER'S WORK GATE, or both arms"+
				" run serially and an overlapping-band mutation passes — which is exactly what"+
				" happened on the first version of the test for this very change", callee),
		})
		return false
	})
	return out
}

// --- PS3059: a serial nest writing through a base derived from the outer variable ------------

// namesDerivedFromLoopVar returns the local names in body that are computed, directly or transitively, from
// ov: "obase := b * out" and then anything computed from obase.
func namesDerivedFromLoopVar(body *ast.BlockStmt, ov string) map[string]bool {
	out := map[string]bool{ov: true}
	for range 4 { // a fixpoint; nests this deep do not chain further in practice
		grew := false
		ast.Inspect(body, func(n ast.Node) bool {
			as, ok := n.(*ast.AssignStmt)
			if !ok || len(as.Lhs) != len(as.Rhs) {
				return true
			}
			for i, lhs := range as.Lhs {
				id, ok := lhs.(*ast.Ident)
				if !ok || out[id.Name] {
					continue
				}
				for nm := range out {
					if mentionsIdent(as.Rhs[i], nm) {
						out[id.Name] = true
						grew = true
						break
					}
				}
			}
			return true
		})
		if !grew {
			break
		}
	}
	return out
}

// derivedBaseNestFindings flags PS3059 — a serial nest whose indexed writes all land through a
// base derived from the outermost loop variable, with at least one not naming it directly.
func derivedBaseNestFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || len(fanoutReg[f.Name.Name]) == 0 {
		return nil
	}
	if callsFanoutHelper(fn.Body, fanoutReg[f.Name.Name]) {
		return nil
	}
	var out []finding
	// EVERY NEST IN THE FUNCTION, not only the first. A function large enough to hold two is
	// exactly the kind that holds a hot nest behind a cold one, and stopping at one finding
	// makes which one you hear about an accident of source order.
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		// A loop running from one PARAMETER to another has already been handed its band; the
		// fan-out call is in its caller, where this check cannot see it. Applying this very
		// check produces such a function — the band body is split out so the two arms share
		// one copy of the nest — so without this the check reports the site it just fixed.
		if loopSpansAParameterRange(n, declaredParamNames(fn)) {
			return false
		}
		body, ov := outerLoop(n)
		if body == nil || ov == "" || ov == "_" || loopDepthOf(body) < 2 {
			return true
		}
		names := namesDerivedFromLoopVar(body, ov)
		nWrites, viaDerived, dst := 0, false, ""
		allOwned := true
		ast.Inspect(body, func(m ast.Node) bool {
			as, ok := m.(*ast.AssignStmt)
			if !ok {
				return true
			}
			for _, lhs := range as.Lhs {
				ix, ok := unparen(lhs).(*ast.IndexExpr)
				if !ok {
					continue
				}
				nWrites++
				owned, direct := false, mentionsIdent(ix.Index, ov)
				for nm := range names {
					if mentionsIdent(ix.Index, nm) {
						owned = true
						break
					}
				}
				if !owned {
					allOwned = false
					continue
				}
				if !direct {
					viaDerived = true
					if dst == "" {
						dst = exprText(ix.X)
					}
				}
			}
			return true
		})
		if nWrites == 0 || !allOwned || !viaDerived {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(n.Pos()),
			category: "serial-nest-writing-through-a-derived-base",
			msg: fmt.Sprintf("every indexed write in this nest lands in %q through a base DERIVED"+
				" from the outer loop variable %q rather than through %q itself, and the function"+
				" never calls the fan-out helper its package declares. The iterations own disjoint"+
				" output, so the outer loop bands. THIS IS PS3034'S BLIND SPOT and it has cost two"+
				" finds: that check asks whether each write NAMES the outermost variable, and a"+
				" nest that hoists its row offset into a local does not. Both misses were large —"+
				" the GGUF weight transpose was 84%% of its own benchmark at one core and went"+
				" -66.3%% once banded, and the KAN fused spline was 84%% of its layer's and went"+
				" -83.8%% at 256x256, a 6.2x, in a file whose OTHER hot loop had fanned out since"+
				" it was written. WIDENING PS3034 WAS TRIED AND REVERTED: following derived names"+
				" there broke three of its own fixtures and took its count from 23 to 33 without"+
				" flagging the transpose. IT CANNOT SEE WRITES BEHIND A CALL: the ownership test"+
				" reads the nest's own statements, so a helper invoked from the body — one that"+
				" fills shared trig scratch per position, say — is invisible to it, and banding"+
				" the loop naively would race. Three MLA rotary nests this check surfaces are"+
				" exactly that shape; read the callees before converting."+
				" GATE WITH BOTH A BIT-EXACT DIGEST AND -race, and know"+
				" which one earns its keep: a destination that ACCUMULATES (+=) double-counts on"+
				" an overlapping band so both fire, while a pure permutation writes the same value"+
				" twice and only -race sees it", dst, ov, ov),
		})
		return false
	})
	return out
}

// --- PS3058: a per-iteration scratch allocation --------------------------------------------

// scratchTypeReg maps a package name to the types in it that declare a SCRATCH INITIALIZER: a
// method assigning 3 or more make() results to fields of its own receiver. Package-level for
// the same reason fanoutReg is — the initializer and the loop that builds one per iteration are
// routinely in different files.
var scratchTypeReg = map[string]map[string]bool{}

// scratchInitializers keys "pkg.Method" for the methods that qualified their type, so the
// finding lands on the initializer rather than on every method of the type.
var scratchInitializers = map[string]bool{}

// collectScratchTypes pre-scans every package for those types.
func collectScratchTypes(files []*ast.File) {
	for _, f := range files {
		if f.Name == nil {
			continue
		}
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv == nil || len(fn.Recv.List) != 1 || fn.Body == nil {
				continue
			}
			rt := receiverTypeName(fn.Recv.List[0].Type)
			if rt == "" || len(fn.Recv.List[0].Names) == 0 {
				continue
			}
			rn := fn.Recv.List[0].Names[0].Name
			n := 0
			ast.Inspect(fn.Body, func(m ast.Node) bool {
				as, ok := m.(*ast.AssignStmt)
				if !ok {
					return true
				}
				for i, lhs := range as.Lhs {
					sel, ok := lhs.(*ast.SelectorExpr)
					if !ok {
						continue
					}
					if id, ok := sel.X.(*ast.Ident); !ok || id.Name != rn {
						continue
					}
					if i < len(as.Rhs) && calleeName(peelCall(as.Rhs[i])) == "make" {
						n++
					}
				}
				return true
			})
			if n < 3 {
				continue
			}
			if scratchTypeReg[f.Name.Name] == nil {
				scratchTypeReg[f.Name.Name] = map[string]bool{}
			}
			scratchTypeReg[f.Name.Name][rt] = true
			scratchInitializers[f.Name.Name+"."+fn.Name.Name] = true
		}
	}
}

// peelCall returns e when it is a call expression, else e itself.
func peelCall(e ast.Expr) ast.Expr {
	if c, ok := e.(*ast.CallExpr); ok {
		return c.Fun
	}
	return e
}

// receiverTypeName returns the bare type name of a method receiver, pointer or not.
func receiverTypeName(e ast.Expr) string {
	if st, ok := e.(*ast.StarExpr); ok {
		e = st.X
	}
	if ix, ok := e.(*ast.IndexExpr); ok { // generic receiver
		e = ix.X
	}
	if id, ok := e.(*ast.Ident); ok {
		return id.Name
	}
	return ""
}

// callsPoolGet reports whether body calls Get on something, which is how a sync.Pool is read.
func callsPoolGet(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		c, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if sel, ok := c.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "Get" {
			found = true
		}
		return !found
	})
	return found
}

// perIterationScratchFindings flags PS3058 — a scratch initializer in a package that declares
// no sync.Pool, so nothing it allocates is ever recycled.
//
// REPORTED AT THE INITIALIZER, not at a construction site, and that is a deliberate limit. The
// value is usually built inside a function CALLED per iteration rather than inline in the loop
// — the CART builder that motivated this check is constructed in the tree's fit method, which
// the forest calls once per tree — and following that would need a call graph this tool does
// not build. The reader ranks it: an initializer whose type is built once per program is
// nothing, and one built once per work item is the whole allocation profile.
func perIterationScratchFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || fn.Recv == nil || len(fn.Recv.List) != 1 {
		return nil
	}
	// Suppressed only when THIS initializer already recycles. A package-level test was tried
	// and is useless: one sync.Pool anywhere in nlp or autograd silenced every initializer in
	// them, taking the tree-wide count from 8 to 0.
	if callsPoolGet(fn.Body) {
		return nil
	}
	rt := receiverTypeName(fn.Recv.List[0].Type)
	if rt == "" || !scratchTypeReg[f.Name.Name][rt] || !scratchInitializers[f.Name.Name+"."+fn.Name.Name] {
		return nil
	}
	return []finding{{
		pos:      fset.Position(fn.Pos()),
		category: "per-iteration-scratch-allocation",
		msg: fmt.Sprintf("%q assigns 3 or more make() results to fields of its %q receiver, and"+
			" nothing in this package is ever recycled — it declares no sync.Pool. If the type is"+
			" built once per work item, every item allocates the whole working set and throws it"+
			" away. RANK IT BY WHERE THE TYPE IS CONSTRUCTED: once per program is nothing, once"+
			" per work item is the entire allocation profile. This check cannot tell you which,"+
			" because the construction usually sits in a function CALLED per iteration rather"+
			" than inline in the loop. THE SAFETY ARGUMENT IS USUALLY ALREADY MADE — a buffer"+
			" reused across the thousands of inner steps of ONE work item is, by that fact,"+
			" written before it is read, and nothing distinguishes the first step of a new item"+
			" from the hundredth step of the old one. Check for the exception: a buffer the"+
			" finished product still points into cannot be recycled. Grow each pooled buffer"+
			" only when it is too small, and note that a table whose entries do not depend on"+
			" the size (c*ln c for c in [0,n]) can be EXTENDED rather than rebuilt."+
			" EXPECT MEMORY, NOT NECESSARILY TIME. MEASURED on the CART builder inside a random"+
			" forest fit: ForestFit allocated bytes -42.4%% and allocations -34.4%%, with the"+
			" wall clock FLAT (111.2 to 108.0 ms, inside the run-to-run spread) because the fit"+
			" is already parallel and was not allocation-bound. Report it as a resource win or"+
			" not at all", fn.Name.Name, rt),
	}}
}

// --- PS3056: a serial permutation left off the fan-out ----------------------------------------

// pureCopyNest reports whether every indexed write in body is a plain copy — dst[..] = src[..],
// with no accumulation and no arithmetic folding the destination back in — and returns the
// destination's name. A permutation has no dependence between its elements at all, so it splits
// bit-identically whatever the band count, which is what makes it worth naming separately from
// the checks that must reason about accumulation order.
func pureCopyNest(body *ast.BlockStmt) string {
	dst, ok := "", true
	ast.Inspect(body, func(n ast.Node) bool {
		as, isAs := n.(*ast.AssignStmt)
		if !isAs || !ok {
			return ok
		}
		if as.Tok != token.ASSIGN || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			if as.Tok == token.ADD_ASSIGN || as.Tok == token.SUB_ASSIGN || as.Tok == token.MUL_ASSIGN {
				if _, isIx := unparen(as.Lhs[0]).(*ast.IndexExpr); isIx {
					ok = false // an accumulation: not a permutation
				}
			}
			return ok
		}
		ix, isIx := unparen(as.Lhs[0]).(*ast.IndexExpr)
		if !isIx {
			return true
		}
		nm := identName(ix.X)
		if nm == "" {
			return true
		}
		// The right-hand side must be a read, possibly converted — not an expression folding the
		// destination back in.
		if mentionsIdent(as.Rhs[0], nm) {
			ok = false
			return false
		}
		// The right-hand side must BE an element read, up to conversions. Testing for arithmetic
		// anywhere inside it was wrong and made the check silent on the transpose it was written
		// for: src[row+j] carries index arithmetic in its SUBSCRIPT, which is addressing, not
		// computation on the value.
		if _, isRead := peelConversions(as.Rhs[0]).(*ast.IndexExpr); !isRead {
			ok = false
			return false
		}
		if dst == "" {
			dst = nm
		}
		return ok
	})
	if !ok {
		return ""
	}
	return dst
}

// serialPermutationFindings flags PS3056 — a multi-level nest that only COPIES elements from one
// buffer to another, in a package that declares a fan-out helper the function never calls.
//
// A permutation is the easiest thing in a tree to parallelize and the easiest to overlook, because
// it carries no arithmetic to show up in a profile as a kernel: it appears as one line of index
// arithmetic. MEASURED on the GGUF weight transpose, which is already cache-blocked and was still
// 84% of its own benchmark at one core: banding the source rows took
// BenchmarkTiedHeadTransposePerCall from 154.2 ms to 49.8, a 66.3% cut, on the model LOAD path.
// loopSpansAParameterRange reports whether a loop runs from one PARAMETER to another, as in
// "for r := lo; r < hi; r++". Such a function is not a serial nest that forgot to fan out —
// it IS the body of somebody else's fan-out, handed the band it is meant to cover, and the
// call to the helper lives in the caller where this check cannot see it. Two conv im2col
// fill functions were reported for exactly this reason, and both were already running inside
// a parallel band pass.
func loopSpansAParameterRange(n ast.Node, params map[string]bool) bool {
	fs, ok := n.(*ast.ForStmt)
	if !ok || fs.Init == nil || fs.Cond == nil {
		return false
	}
	as, ok := fs.Init.(*ast.AssignStmt)
	if !ok || len(as.Rhs) != 1 {
		return false
	}
	lo, ok := as.Rhs[0].(*ast.Ident)
	if !ok || !params[lo.Name] {
		return false
	}
	be, ok := fs.Cond.(*ast.BinaryExpr)
	if !ok {
		return false
	}
	hi, ok := be.Y.(*ast.Ident)
	return ok && params[hi.Name]
}

// declaredParamNames collects the parameter identifiers of fn.
func declaredParamNames(fn *ast.FuncDecl) map[string]bool {
	out := map[string]bool{}
	if fn.Type == nil || fn.Type.Params == nil {
		return out
	}
	for _, fl := range fn.Type.Params.List {
		for _, nm := range fl.Names {
			out[nm.Name] = true
		}
	}
	return out
}

func serialPermutationFindings(fset *token.FileSet, f *ast.File, fn *ast.FuncDecl) []finding {
	if fn.Body == nil || f.Name == nil || len(fanoutReg[f.Name.Name]) == 0 {
		return nil
	}
	fanned := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if c, ok := n.(*ast.CallExpr); ok && fanoutReg[f.Name.Name][calleeName(c.Fun)] {
			fanned = true
		}
		return !fanned
	})
	if fanned {
		return nil
	}
	var out []finding
	params := declaredParamNames(fn)
	// EVERY NEST IN THE FUNCTION, not only the first. A function large enough to hold two is
	// exactly the kind that holds a hot nest behind a cold one, and stopping at one finding
	// makes which one you hear about an accident of source order. The count is unchanged on
	// this tree (21), so this buys coverage rather than noise.
	//
	// A KNOWN LIMITATION, MEASURED AND NOT YET EXPLAINED. The LLM.int8 kernel held two nests
	// of this shape and this check reported only the first — the earlier, colder one — while
	// the second was 90.9%% of that benchmark and 5.0x once banded. Removing the
	// one-finding-per-function limit did NOT surface it. The second nest fires when lifted
	// into a fixture verbatim, with a fan-out helper declared beside it and every condition
	// logged as satisfied (depth 2, one write, owned, via a derived base), so the shape is
	// right and something about the surrounding file suppresses it. Three rounds of
	// bisection did not isolate what. Do not assume this check has enumerated a function's
	// nests; read the whole function when it reports one.
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if loopSpansAParameterRange(n, params) {
			return false // already a band body; its fan-out is in the caller
		}
		body, iv := outerLoop(n)
		if body == nil || iv == "" || loopDepthOf(body) < 2 {
			return true
		}
		dst := pureCopyNest(body)
		if dst == "" {
			return true
		}
		out = append(out, finding{
			pos:      fset.Position(n.Pos()),
			category: "serial-permutation",
			msg: fmt.Sprintf("this nest only COPIES elements into %q — every write is a read from"+
				" somewhere else, with no accumulation and no arithmetic folding the destination"+
				" back in — and the function never calls the fan-out helper its package declares."+
				" A permutation has NO dependence between its elements, so splitting the outer loop"+
				" is race-free and BIT-IDENTICAL at any band count: there is no summation order to"+
				" preserve because nothing is summed. That makes it the cheapest parallelization in"+
				" a tree to justify and the easiest to overlook, because it carries no arithmetic to"+
				" appear in a profile as a kernel — it shows up as one line of index arithmetic."+
				" CHECK THE BAND OWNS DISJOINT OUTPUT: a transpose writes COLUMNS of its"+
				" destination for a band of source rows, which is disjoint but not obvious, and a"+
				" nest that scatters by a data-dependent index is not this shape at all."+
				" GATE IT WITH TWO TESTS, NOT ONE: the property that makes a permutation safe to"+
				" split — every cell depending on nothing but its source — is also what makes a band"+
				" OVERLAP invisible to a value comparison, because the intruding band writes exactly"+
				" what the owner would have written. A measured overlap mutation passed the value"+
				" check and was caught only by -race, while a band that SKIPS rows or reads the wrong"+
				" cell is the reverse: caught by the values, invisible to -race. Neither subsumes the"+
				" other. Size the fixtures ABOVE the fan-out helper's work gate, or both arms run"+
				" serially and no banding mistake can fail at all. MEASURED on"+
				" the GGUF weight transpose, which is already cache-blocked and was still 84%% of"+
				" its own benchmark at one core: BenchmarkTiedHeadTransposePerCall went 154.2 ms to"+
				" 49.8, a 66.3%% cut, on the model LOAD path; and on the transpose VJP, the same shape"+
				" in another package, BenchmarkTransposeVJP -45.7%% and its F32 twin -50.8%%", dst),
		})
		return false
	})
	return out
}
