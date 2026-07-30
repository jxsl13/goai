# perfscan pattern catalog

Generic, benchmark-verified performance patterns found repeatedly in GoAI hot
paths. Each entry: the anti-pattern, the fix, why it's safe, and the measured
wins that justify it. **Every scanner hit is a CANDIDATE** — confirm with a
pre/post benchmark before changing, and skip cold paths (one-time init,
eval-only) where the fix isn't worth the code.

When you find a NEW generic pattern worth codifying, add it here AND teach the
scanner (extend a callee map or add a detector in `perfscan.go`, with a positive
+ negative fixture test in `perfscan_test.go`) — SPEC §C29.

This is the SINGLE perfscan for the repo (`internal/perfscan`, run via
`make perfscan` / `go run ./internal/perfscan`); each static check has a
PS-prefixed 4-digit ID (`perfscan -list` prints them all). The sections below
catalog a subset with detailed wins — their IDs head each section. The `P3`/`P4`/
`P5` sections are profile/benchmark heuristics with **no** static detector.
`PS4003` generalizes `PS4002` to a transcendental hidden one call deep in a helper.

## Repo-agnostic engine + config

perfscan detects the problems **independent of any one repo**. Eighteen of the twenty-three
checks are pure language/stdlib shapes and run on any Go module with no configuration
(PS1003, PS2002–PS2007, PS3001–PS3003, PS4001, PS4003–PS4006, PS4008, PS5001, PS5002). The five
**domain** checks — `PS1001` per-element-dispatch, `PS1002` per-element-closure, `PS2001`
alloc-in-loop, `PS4002` scalar-transcendental-vectorizable, `PS6004` unverified-dual-path
— key on a project's own vocabulary (its element accessors, allocators, fast-path helpers
and vectorized kernels), which lives in a **JSON config, not the engine**. With no config
those five stay silent — and say so: each one whose vocabulary is empty is named in a
loud stderr warning, because a silent zero from a starved check reads as "no instances"
and is the one failure mode that costs whole investigations. Supply a config
with `-config file.json` or a discovered `perfscan.json` / `.perfscan.json`:

```jsonc
{
  "elementAccessors":       ["AtF64", "SetF64"],     // PS1001/PS1002
  "fastPathHelpers":        ["flatF64", "flatF32"],   // PS1001 — presence silences a fallback loop
                                                     //          KEEP THIS COMPLETE: a comma-ok
                                                     //          helper missing from the list makes
                                                     //          PS1001 report the very fallback the
                                                     //          fast path exists to guard
  "elementCountMethods":    ["Numel"],               // PS1001 — a loop bound over this reads as per-element
  "shapeMethods":           ["Shape"],               // PS1001 — `d := t.Shape()[1]` then `for j := range d`
                                                     //          walks d elements exactly as Numel() does
  "indexDecomposeFuncs":    ["Unravel"],             // PS1001 — flat→multi-index marks a per-element loop
  "allocatorFuncs":         ["New", "Zeros", "Cast"], // PS2001 — allocation in a per-element loop
  "perElementVisitors":     ["readGen", "fillGen"],   // PS1002 — helper fed a per-element closure
  "bulkCopyHelpers":        ["rawCopyLE"],            // PS4001 — presence silences a genuine-decode path
  "vectorizedSiblingFuncs": ["vexpF32", "vsiluF32"]   // PS4002 — a SIMD kernel beside a scalar math.X
}
```

GoAI ships its own vocabulary in `internal/perfscan/perfscan.json`, which `make
perfscan` loads. This catalog's win figures are GoAI's measured results; the
patterns and the engine are generic.

## Check IDs, auto-fix, editor integration

Every check has a stable **PS-prefixed 4-digit ID** (`PS1001`…), grouped by the
thousands digit: `PS1xxx` per-element access, `PS2xxx` allocation, `PS3xxx`
indirection/reflection, `PS4xxx` vectorization, `PS5xxx` arithmetic, `PS6xxx`
verification gaps. IDs are **stable and never reused** — they are the handle an
`//perfscan:ignore` directive names, so a new check must claim an ID free on main
*and* in every open PR that touches the registry, not merely in its own branch
(§PERF-ID-COLLISION-001). `perfscan
-list` prints the table (ID, category, whether `-fix` can rewrite it, title).

- `-fix` applies the **safe mechanical fixes** in place. Only checks with a
  deterministic, bit-identical rewrite carry one (today `PS2005`
  regexp-compile-in-loop hoists a literal-pattern compile out of the loop);
  everything else is advisory — the transform needs an A/B + bit-identity proof
  (§C3/§V22) a static tool cannot give. Review the diff even for applied fixes.
- `-json` emits every finding, with any fix's text edits (line:col ranges + byte
  offsets + replacement text), for a VS Code task/extension to offer as a quick-fix.

**VS Code integration.** The text output is a standard `file:line:col: message (PSid)`,
so a task with a `problemMatcher` surfaces findings in the Problems panel — no
extension needed:

```jsonc
// .vscode/tasks.json
{
  "label": "perfscan",
  "type": "shell",
  "command": "go run ./internal/perfscan ./...",
  "problemMatcher": {
    "owner": "perfscan",
    "fileLocation": ["relative", "${workspaceFolder}"],
    "pattern": { "regexp": "^(.*):(\\d+):(\\d+): (.*) \\((PS\\d{4})[^)]*\\)$",
                 "file": 1, "line": 2, "column": 3, "message": 4, "code": 5 }
  }
}
```

For quick-fixes, a thin extension runs `perfscan -json ${file}` and maps each
`fix.edits` entry to a `vscode.WorkspaceEdit` (the `line`/`col`/`endLine`/`endCol`
range → `newText`). `perfscan -fix ./...` applies them from the CLI.

## Suppressing findings (per-check, staticcheck-style)

Silencing one check never hides another — accepting a site for `PS1001` still
surfaces a new, unrelated `PS2002` there. Name a check by its **ID** (precise —
the "ignore only this explicit detection" path) or its **category** alias:

```go
//perfscan:ignore PS4002 reason         // silence ONLY PS4002 on the next (or same) line
//perfscan:ignore PS4002,PS1002 reason   // several checks at once (IDs or categories)
//perfscan:ignore                       // bare: silence ALL checks at that site
```

A directive applies to its **entire enclosing comment block** and to the statement
directly below that block, so the explanation may wrap freely and the directive may sit
above or below the prose. It does NOT leak past its own block — a directive attached to
one declaration will not silence a finding further down. (Anchoring used to be the
directive's own line plus one, which made any wrapped explanation *silently inert*: the
comment reads as if it took effect while the finding is still reported. Two directives
in this repo were dead that way.)

Repo-wide, pass `-exclude=PS4002,per-element-closure`, or `-checks=PS1001,PS2002`
to run an allow-list. Example: the f64 exp/log/tanh/sigmoid/gelu kernels are
flagged by `PS4002` but are exact-locked (`TestCPUCrossReferenceExact`) — mark each
`//perfscan:ignore PS4002 exact-locked ref` so the one genuinely-open member of
the family still reports.

---

## PS1002 — per-element closure over a contiguous buffer  *(scanner: static)*

**Smell.** A helper that invokes a `func(...)` argument once per element
(`readGen`, `fillGen`, `forEach`, …). Even on the helper's contiguous fast
branch this is an indirect closure call per element — ~250k calls for a
512×512 tensor, on loops that run every training step.

**Fix.** Add a raw-slice tight loop for contiguous, offset-0 F32/F64 tensors
(`t.IsContiguous() && t.Offset() == 0`), doing the *identical* arithmetic (and
identical `float32(...)` rounding). Keep the closure only as the
non-contiguous / F16 / BF16 fallback. Result is **bit-identical**.

**Wins (4×[512,512] f32):**
- `nn.SWA.Update` — **4.21×** (4.72→1.12 ms)
- `nn.EMA.Update` — **3.09×** (2.43→0.79 ms)
- `nn.GradAccumulator.Add` — **3.75×** (2.16→0.58 ms)
- `nn.GradAccumulator.GradFn` — **1.62×** (2.56→1.58 ms)

---

## PS2006 — quadratic cache append in a per-token step  *(scanner: static)*

A cache slot reassigned to a concatenation of **itself** and a new row, inside a loop,
in a per-token step function: `c.K[l] = concatRows(c.K[l], k)`. The concat allocates a
fresh `[t+1, width]` buffer and recopies all `t` existing rows **every token**, so a
T-token decode moves O(T²) bytes and allocates O(T²).

**Fix:** an amortized row buffer — write row `t` in place into a doubling backing store
and hand back a zero-copy contiguous prefix view, which makes the whole sequence O(T).
`nlp/rowbuf.go` is the in-repo implementation.

**Shipped:** the GPT decode cache. End to end on a 500-token generate, **2.21 GB → 159 MB
(13.9×)** and 833 → 630 ms (**1.32×**); the append microbenchmark alone is 101× time and
104× bytes at width 2048, T=512. Note the memory win is the headline and the wall-clock
win is much smaller — the brief predicted 2–3× time and the measured figure was 1.32×.

**The risk is ALIASING, not numerics.** The concat hands back a fresh buffer each step;
the row buffer hands back a **view** into a growing one. Values are identical and
previously returned views stay valid (appends only ever write row `t` and beyond), so the
only real hazard is a caller that retains an earlier view **and mutates it in place**.
Audit for that before switching, and check whatever replaces cache entries wholesale
(eviction, truncation) — a good row buffer detects a foreign view and resynchronizes.

**A hit does NOT imply a worthwhile win — measure the site, never extrapolate from a
sibling.** The identical transformation returned **13.9×** (GPT), **8.3×** (CLA) and
**1.15×** (T5 decoder) on three call sites, decided entirely by whether the quadratic
append was that site's dominant cost. T5's decode is dominated instead by a per-token
rebuild of the full relative-position matrix, so removing its O(T²) cache append moved
12.8% of bytes and nothing measurable in wall clock.

**Deliberately silent** outside a per-token step function (the same statement in a
one-shot builder is an ordinary concatenation), outside a loop, and when the concat's
first operand is a *different* slot — that is not accumulate-into-itself and does not
grow quadratically. It is also silent when the concat is **nested inside another call**
(`keep(concatRows(c.K[l], k), …)`), which is how a bounded/evicting cache is usually
written; those are O(window) rather than O(T) and are a weaker target.

## PS2007 — build an N×N object to consume one row  *(scanner: static)*

A call given the **same size expression twice**, whose square result is then read at
exactly the position that expression is built from:

```go
full, _ := d.relBias.Bias(ctx, pos+1, pos+1)  // builds [pos+1, pos+1, heads]
… fs[(pos*kk+k)*heads+h] …                    // reads row pos, discards the rest
```

The callee's cost grows with the argument in **both** dimensions while the consumer
needs one row, so the work is an order higher than required — and on a per-token path,
one order higher again over the whole run (O(T³) where O(T²) suffices).

**Fix:** compute the row from whatever the callee derives it from — usually by exposing
the per-element rule (here a relative-position bucket lookup) instead of its
materialized matrix.

**Shipped:** the T5 decoder's relative-position bias. Per call **130× at pos=32, 261× at
pos=128, 713× at pos=512**, allocation 86.8 MB → 19.7 KB; end to end the 500-token
decode went **2,679 → 556 ms (4.82×)** and **13.87 GB → 125 MB (111×)**. The ns/op curve
against `pos` is the real evidence: quadratic before, linear after.

**Bit-identity is claimable but has genuine exceptions.** The old value arrived through
a one-hot matmul — a sum of `numBuckets` products of which all but one are `0·Table[j][h]`.
For finite tables that sum *is* `Table[b][h]`, so the gather is bit-identical. It differs
for ±Inf/NaN entries (`0·Inf = NaN`) and for a stored `−0` (`0 + −0 = +0`). Both mean a
broken checkpoint rather than normal operation. Gate with a tolerance-0 cross-reference
**and** a token-for-token identical greedy decode, not merely close logits.

**Precision took three passes and the two rejected cuts are worth knowing.** Asking only
"is this identifier used in some index in this function" flagged `a.AtF64(j, j)` — a
diagonal element read, not a builder — and a square attention mask consumed whole. The
rule now requires the result *or a value derived from it* to be indexed by the driving
position, and requires that position **not** to be a loop bound: when it bounds a loop
the object is being walked in full and the identifier appears in those indices only as a
stride. Tree-wide that went 6 findings → 2, both genuine.

## PS5003 — an inner-loop expression the outer loop rebuilds  *(scanner: static)*

An expression in the innermost loop that varies with the **inner** index but not the
outer one, so the outer loop recomputes the same value on every pass:

```go
for i := range n {
	for j := range m {
		h[i*m+j] = a*h[i*m+j] + b*(x[off+j]*delta) // x[off+j]*delta has no i
	}
}
```

Precompute it once into an m-sized scratch before the outer loop. **Bit-identical** — it
is the same product, so it rounds the same way; this is not a reassociation. Go will not
do it for you: the operands are index expressions it cannot prove unaliased across the
outer iteration, so both the load and the multiply stay put.

**Shipped:** the Mamba2 SSM scan, where `x[t][hOff+j]*delta` was rebuilt N times per `j` —
**1.08×–1.10×** on prefill end to end, and notably *larger* than fixing that same
function's pathological access pattern (1.05×). PS4006 pointed at the file; the bigger win
inside it was this, which PS4006 cannot see.

**The first cut was UNSOUND, not merely noisy, and that is the thing to remember.** An
expression can mention no outer variable and still change every outer iteration, because
the outer loop **rewrites what it reads** — a per-row softmax `p[j]*inv` is the canonical
case, and hoisting it would be *wrong*, not just useless. Every one of the first sampled
findings was of exactly that kind. The check now requires that no operand be assigned
anywhere in the outer body. Findings went **445 → 48 → 7** across the two tightenings
(tight-kernel, then the write guard), and the surviving set includes the sibling of the
shipped win plus a softmax normalization recomputed per output dimension.

**Deliberately silent** on calls (hoisting changes how often one runs, which is observable
if it is not pure and the rule cannot know that it is), on inner bodies longer than one
statement, and on expressions that are not an operand of a larger arithmetic expression —
a bare `v := x[j]*d` is already as hoisted as a reader will make it.

## PS3005 — a sort whose comparator dereferences the sorted index  *(scanner: static)*

Sorting an INDEX slice with a comparator that reaches two levels deep through the sorted
element:

```go
sort.Slice(idx, func(a, b int) bool { return m[idx[a]][f] < m[idx[b]][f] })
```

Every comparison pays a row-pointer load plus an index — O(n log n) times — to read a
value that depends only on the element. Fill a flat id-indexed key column once (O(n)) and
compare that.

**Safe by construction, not by argument:** the rewrite keeps the *same predicate*
(`key[id] == m[id][f]`), so the sort returns the same permutation, ties included. That is
stronger than the usual "tie order is unspecified but irrelevant" reasoning.

**Shipped, three times, all the same shape.** Figures below are INTERLEAVED (change
toggled in and out within one session, 3-4 alternations) — the separate-run numbers first
reported were 5-10% optimistic because the host drifts between sessions:
GBM presort + radix **1.050×** on `BenchmarkGBMFit` and **1.120×** on
`BenchmarkGBMHist_exact_20k` (first reported as 1.104× and 1.176×); ball-tree median
split **1.060×** on KNN fit, ranges overlapping (first reported 1.088×); Expert Choice
routing **1.21×**, the one case where the control moved *against* the candidate. The CART builder had
already solved it locally, which is what made the siblings findable by eye; this rule is
so the next one does not need luck.

**A hit is a candidate, not a win — and this one has a decline on the board already.**
SparseGPT's mask selection has the richest-looking instance of the shape (two lookups
*plus* a `sgSaliency` call per comparison) and hoisting it measured **1.0046×** in an
interleaved same-session A/B: the sort is a small share of a routine dominated by the
Hessian inverse and the OBS compensation loops. A first, NON-interleaved reading of the
same change suggested 1.048× — that was machine drift between two runs, not the change.
Interleave, or you will ship drift as a result.

**Deliberately silent** on the fixed form (`key[idx[a]]` — applying the advice clears the
finding), on a direct value sort (`cand[a].dist`, no indirection to hoist), on a single
dereference, and when the two-level lookup goes through a *different* slice than the one
being sorted — that is not "read this element's key" and a hoisted column would not be
well-defined.

**The first cut of this rule was silent on all three sites it was written from**, because
the nesting was inverted: in `m[idx[a]][f]` the sorted slice sits in the *outer*
IndexExpr's own Index, not in its X. Validated by scanning the pre-fix revisions and
confirming all three are found.

## PS3002 — closure-comparator sort on a large keyed slice  *(scanner: static)*

**Smell.** `slices.SortFunc` / `sort.Slice` / `sort.Sort` sorting a large slice
by a float/int key looked up through the comparator. The per-comparison indirect
call dominates (profiled at ~50% of the op), and it's O(n log n) where the
consumer often needs only a threshold.

**Fix.** If the key is **monotonic** (for non-negative `float64`,
`math.Float64bits` is monotonic in the value; invert the bits for descending),
replace with an 8-pass **LSD radix sort** on the key bits — closure-free, O(n).
The value order is identical to the comparison sort for distinct keys. Gate on
`n >= ~2048` so small slices keep the lower-constant comparison sort.

**Cheaper sub-fix — drop cosmetic stability (`SliceStable` → `Slice`).** When the
callee is `sort.SliceStable` and stability is used ONLY for a deterministic tie
order (not a semantic requirement), a full radix rewrite is overkill: just switch to
`sort.Slice`. `SliceStable`'s `symMerge` has a far higher constant than `Slice`'s
pdqsort, and the tie order is recovered for free when the comparator is already a
**total order** (a unique field like a token id or index breaks every tie → output is
byte-identical) or when the append order is reconstructable (index-argsort → append
`idx[a] < idx[b]`; beam frontier → `(score, parent, tok)`). Use this when the key
isn't a bare monotonic scalar (so radix doesn't apply) but n is still large.

**Wins (~50k-vocab flat distribution):**
- `nlp` top-p nucleus (`sortIdxDescByProb`) — **2.25×** (7.38→3.27 ms)
- `nlp` locally-typical (`sortIdxAscByScore`) — **1.89×** (7.14→3.77 ms)

**Wins (SliceStable → Slice sub-fix):**
- `nlp` logit-lens full-vocab readout (`newJLensReadout`) — **5.44×** (129→23.7 ms)
- `nlp` `CosineRerank` (20k×64) — **3.89×** (15.0→3.86 ms)
- `nlp` beam-search frontier (`BeamSearch`) — **2.4–3.1×**
- `nlp` diverse-beam per-group frontier — **1.7×**

---

## P3 — serial independent-per-item loop that dominates  *(heuristic: profile)*

**Smell.** A hot loop whose iterations are independent (each writes a disjoint
output slot, reductions excluded) but which runs serially — often *hidden*
because the heavy sub-op it calls is already parallelized, so a CPU-flat profile
shows the sub-op, not the serial driver. Look at WALL time / serial fraction.

**Fix.** Distribute the loop across `GOMAXPROCS` goroutines over disjoint index
ranges (or by cluster/feature ownership so per-group accumulation order is
preserved). Keep any reduction serial (or partition it so summation order is
unchanged) → **bit-identical**. Gate on a minimum size to avoid spawn overhead.

**Wins:**
- `classic` DBSCAN neighbour search — **6.6×** (now beats scikit-learn)
- `classic` KMeans assignment+means — **1.60×**
- `classic` SoftmaxRegression per-sample — **1.30×**
- `classic` histogram-GBM per-feature — **1.13×**
- `nlp` BPE Encode per-pre-token group — **1.61×** (~2.9× tiktoken)

Not scanner-detectable reliably (needs profiling); use the benchmark-ratio
check (P4) and CPU profiles to surface candidates.

---

## P4 — "optimized" path barely beats its naive/slow twin  *(heuristic: benchmark)*

**Smell.** A `BenchmarkXFast` / `BenchmarkXSlow` (or `…Naive`) pair where Fast is
only ~1.0–1.6× the slow reference. The "fast" path exists but still carries an
avoidable cost (a closure, a full sort, a fallback that hits the slow branch on
the common input) — this is how P1 (SWA was 1.53× over slow) and P2 (top-p was
1.03× over naive) were found.

**Check.** `perfscan-bench.sh` runs every `*Fast`/`*Slow`/`*Naive` benchmark pair
in a package and flags ratios below a threshold (default 2.0×). Investigate the
flagged ones; a genuinely-optimal path clears the bar easily (SAM 19.6×,
GradAccum-vs-AtF64 3.0×).

---

## PS4002 — scalar transcendental where a vectorized sibling exists  *(scanner: static)*

**Smell.** A numeric kernel switches on dtype: one branch runs a scalar libm
transcendental (`math.Exp`/`Tanh`/`Erf`/`Log`/…) in a loop, while the same kernel
calls a hand-vectorized `v*F32/F64` sibling for another dtype. The scalar branch
pays per-element libm cost the sibling already avoids.

**Fix.** Vectorize the scalar branch on a SIMD transcendental primitive (e.g. an
AVX2 f64 `exp`), keeping a **bit-identical scalar tail** (a `math.FMA` scalar twin
of the SIMD lane) so a value yields the same result in the body or the tail —
preserving any absorbed-decode == batched-forward byte-exactness.

**Caveat — check the invariant FIRST.** Some f64 ops are deliberately locked to
the scalar reference: `TestCPUCrossReferenceExact` asserts CPU==Ref **bit-exact**
for f64 `Exp`/`Log`/`Tanh`/`Sigmoid`/`GELU`, so a ~1-ulp SIMD poly would fail it.
The scanner flags all of these; only vectorize the ones the exact test does NOT
lock. `OpSiLU/F64` is the one it *skips* (cpu `x/(1+e⁻ˣ)` vs ref `x·σ(x)` are
already ulp-split) — which is why the **F64 SwiGLU SiLU** was the member of this
family that could ship.

**Win.** `siluKernelCPU` f64 branch was scalar `math.Exp` beside `vsiluF32`; an
AVX2 f64 exp (`expF64x4`, 1 ulp) made it **1.52× Llama prefill** (1.89× kernel),
goldens green (T667). The sibling exp/log/tanh/sigmoid/gelu f64 kernels are
flagged too but are exact-locked.

---

## PS4003 — a loop calls a local helper that WRAPS a transcendental  *(scanner: static)*

**Smell.** A hot per-element loop calls a package-local elementwise helper
(`softplus`, `mish`, `swish`, `silu`, a `gaussianQuantile`, …) whose body hides a
scalar libm transcendental one call deep. PS4002 only sees a **direct** `math.X`
in the loop, so a loop over such a wrapper reads as scalar-clean and slips past it.
The detector collects the file's scalar-`float→float` funcs that call a
transcendental, then flags any loop that calls one.

**Fix.** Same as K: give the op a vectorized SIMD kernel (compute the whole slice
4/8-wide on a SIMD transcendental primitive, bit-identical scalar tail), or route
it through a **batched tensor op** that already has a vectorized CPU kernel instead
of a scalar helper per element. Verify hotness (profile — a wrapper in a loop is
not automatically hot) and the CPU==Ref invariant first, exactly as K.

**Win.** `OpSoftplus` (Mamba/Jamba Δ) had **no CPU kernel** and fell through to the
scalar single-threaded ref backend — 32% of `math.archExp` in Mamba f64 prefill. A
4-wide AVX2 f64 softplus kernel (`vsoftplusF64` = `expF64x4` + the Cephes double-log
rational, ~1 ulp) made it **1.62× Mamba prefill**, goldens green (PR #249). The
detector still flags the Mamba2 **mixer** `softplus`/`silu` (`mamba2.go`) — a
scalar per-`(t,h)` loop that the kernel does not cover — as the leftover sub-win.

---

## P5 — a transcendental op registered ONLY on the ref backend  *(heuristic: registration diff)*

**Smell.** A tensor `Op` whose only kernel is on the **ref** backend (a scalar
`math.Exp/Log/Tanh/…` loop) with **no CPU kernel** — so on the CPU backend it falls
through to that scalar, single-threaded reference. This is the shape both `OpSoftplus`
(Mamba/Jamba Δ, 1.62× — PR #249) and `OpSoftCap` (Gemma-2 `cap·tanh(x/cap)`, 6.1× at
attention-score shape) had. PS4002/PS4003 never see it: the transcendental lives in
`backend/ref`, not in a hot first-party loop with a vector sibling.

**Finder (generic).** Diff the ref-registered ops against the CPU-registered ops:

```sh
grep -rhoE 'std\.add\(backend\.(Op[A-Za-z0-9]+)' backend/ref/*.go   | grep -oE 'Op[A-Za-z0-9]+' | sort -u > /tmp/ref_ops
grep -rhoE '(reg|std\.add)\(backend\.(Op[A-Za-z0-9]+)' backend/cpu/*.go | grep -oE 'Op[A-Za-z0-9]+' | sort -u > /tmp/cpu_ops
comm -23 /tmp/ref_ops /tmp/cpu_ops   # ops with NO CPU kernel — the candidates
```

Then keep the ones whose ref kernel is elementwise + transcendental (a SIMD kernel
pays) and that sit on a hot path. Live candidates from this diff (2026-07-21):
`OpConv1D` (Mamba causal conv), `OpSSM`/`OpWKV` (recurrent scans — sequential, harder),
`OpZLoss` (MoE router logsumexp), the RL preference losses (`OpDPO`/`OpKTO`/… — niche).

**Fix.** Add a CPU kernel: vectorize the transcendental on a SIMD primitive
(`expF64x4`, `vtanhF32`, …) with a bit-identical scalar tail, `parallel()` the outer
range, register `std.add(OpX, F64, kernelCPU)`. Verify the op is not under the
CPU==Ref exact invariant first, and prove the model golden (measure hotness — a
ref-only op is not automatically hot). *(Could graduate to a static detector once
perfscan grows a module-level cross-package registration pass.)*

---

## PS3003 — integer-keyed map read in a loop  *(scanner: static)*

**Smell.** A `map[int]…` / `map[rune]…` / `map[int32]…` (sets — `map[int]bool`,
`map[int]struct{}` — excluded) is READ inside a loop: `t.decoder[id]`, `u2b[r]`,
`votes[cls]`. Each lookup hashes the integer key and probes the table; the profile
shows it as `mapaccess1/2_fast64` / `_fast32`.

**Fix.** When the keys are DENSE over `[0,N)` — token/rune/class/vocab ids almost
always are — flatten the map into a `[]T` indexed by the key, built once at
construction. A `uint(k) < uint(len(s))` bounds check makes gaps and out-of-range
keys resolve to the zero value exactly as the map's `!ok` did, so the result is
byte-identical. **Verify density**: a sparse key space wastes memory and isn't a
candidate (which is why the detector skips set-typed maps and the report says
"verify key density").

**Wins (map→slice):**
- `nlp` BPE `Tokenizer.Decode` (`decoder[id]`) — **2.85×** (310→885 MB/s)
- `nlp` GGUF `BPETokenizer.Decode` (`decoder[id]` + `u2b[r]`) — **3.67×**
- `classic` RandomForest `Predict` vote map → precomputed slice — **1.13×** (T312)

---

## PS3008 — monotone accumulator tested on every iteration  *(scanner: static)*

**Smell.** A loop adds a provably non-negative term to a scalar and tests it against a
threshold every single iteration, bailing out when it is exceeded:

```go
for i := range a {
    d := a[i] - b[i]
    s += d * d
    if s > eps2 { return false }   // one unpredictable branch per element
}
```

**Fix.** The accumulator never decreases, so once it passes the threshold it stays past
it. Test every 4th iteration instead: a run that would have bailed at element k still
bails at the next checkpoint, and at the end. Keep ONE accumulator in the SAME order so
the sum stays bit-identical, and handle the leftovers in a scalar tail.

**Measured** (`classic.ballTree.within`, the leaf test DBSCAN runs per candidate pair):
- The branch was **450ms** of profile against **30ms** for the subtraction and square it
  guarded. The branch *was* the function — it is data-dependent and the predictor cannot
  learn it.
- `BenchmarkDBSCANFit` **−17.41%** at eps=2 (p=0.000) and **−8.51%** at eps=4 (p=0.010),
  geomean −13.07%, allocations unchanged, exact-label goldens green.
- The bigger win is at the *smaller* eps, where most pairs bail: the rewrite does a little
  more arithmetic before bailing and still wins, because it trades three branches for one.

**Non-negativity is the correctness condition, not a heuristic.** It is required
syntactically — `x*x` with identical operands, `math.Abs`, `math.Hypot`, or a sum of
those. A signed term lets the accumulator dip back under the threshold, and then moving
the test changes the *answer*.

**Mind the NaN tail.** With a NaN term the accumulator is NaN, every `acc > thr` is false,
and the original fell out of the loop and returned its not-exceeded answer. End the tail
with `!(acc > thr)`, **not** `acc <= thr`, which flips it. Gated by a test in `classic`.

**Silent** once the loop strides by more than 1 (the applied form), when the test does not
exit the loop, and when the term is not provably non-negative. Hotness is not visible to
the AST: benchmark the enclosing operation before restructuring a cold bail-out.

---

## PS3007 — membership set built from a slice, then probed in a loop  *(scanner: static)*

**Smell.** A set — `map[K]bool` or `map[K]struct{}` — is filled by ranging a slice the
caller already owns, then probed once per iteration of a loop:

```go
breaker := map[int]bool{}
for _, b := range s.DRYBreakers { breaker[b] = true }
for j, t := range window { brk[j] = breaker[t] }   // one hash per window position
```

PS3003 deliberately skips set-shaped maps, because a sparse set is not the dense `[0,N)`
lookup a slice would replace. That is right about *densification* and misses this: the
question the map answers is already answered by the slice it was built from.

**Fix.** Don't build the map. Scan the source slice — for a handful of elements the
compiler keeps it in registers, and a few predictable comparisons beat one hash. Keep the
map behind a size threshold so a large set doesn't fall off the O(L·B) cliff.

**Measured** (`nlp.applyDRY`, sequence-breaker set):
- `runtime.mapaccess1_fast64` was **1.14s of the function's 1.99s cumulative** — 57% of
  its own time, for a prepass whose comment already recorded killing the O(L²) probes.
- `BenchmarkApplyDRY` **19.52µs → 15.87µs, −18.72%** (p=0.002, n=6, interleaved, both arm
  orders); B/op and allocs/op unchanged; `mapaccess1_fast64` left the profile entirely.
- The benchmark's own 256KB logits reset is ~25% of each op, so the function improved by
  more than the reported delta.

**Crossover, measured rather than assumed.** Forced onto each arm on an M2 Pro: at 8
breakers the scan wins (62.7µs vs 64.7µs), at 16 it has already lost (68.0µs vs 65.4µs),
at 64 it loses badly (97.7µs vs 66.3µs). Hence a threshold of 8 — and hence this is a
SMALL-SET transform, not a general one.

**Silent on** a set written after its build loop (a mutable working set genuinely needs a
map — autograd's einsum `avail`), and on a build already guarded by a size THRESHOLD on
the source, which is code that has taken this advice already. **Not** silenced by an
emptiness guard: `len(src) > 0` is a nil check whose branch is the only path, not a
fallback, and conflating the two made the check silent on the one true site in this repo.

**Blind spot.** Only the value form `for _, v := range src` is recognized; the index form
`for i := range src { set[src[i]] = true }` is missed. No instance exists in this repo.

---

## Discipline

- **Verify or revert.** No change ships without a pre/post benchmark on
  representative data. If no benchmark harness exists for the target, either
  build one or don't ship (a correct-but-unbenchmarked change was reverted this
  way: `BPETokenizer.Encode`).
- **Bit-identical or same-optimum.** Prefer fixes that preserve results exactly
  (raw-slice loops, radix on monotonic keys, disjoint-write parallelism). Where
  the fix changes iteration order (e.g. a solver), require convergence to the
  same unique optimum within the golden tolerance.
- **Watch the goldens.** Exact-label goldens (kmeans, DBSCAN) forbid any
  summation reorder; keep reductions serial or cluster-partitioned.
- **Cross-platform.** Don't retune a constant that was deliberately calibrated
  for other hardware (e.g. the CPU pool `parThreshold` / dense-worker cap) on a
  single box's numbers — that needs multi-platform validation.

## PS2004 — poolable per-call scratch in a stateful method  *(scanner: static)*

A `make([]T, …)` inside a per-item loop of a **pointer-receiver method**, bound to a
**local that does not escape** the iteration (not returned, not stored into a field or
slot), is scratch reallocated on every call. On a reusable stateful object — an
optimizer's `Step` iterating over its parameters — that is pure GC churn: N allocs and
several MB per step, feeding the collector for the whole training run.

**Fix:** hoist the buffer to a reused receiver field grown on demand
(`u := growF64(o.uScr, n); o.uScr = u`), zeroing only when the code reads before it
writes. Params and steps run serially, so a single grow-able buffer per scratch is
safe, and — being fully overwritten before use — the reuse is bit-identical.

**Shipped:** Adafactor 1.32×, CautiousAdamW 1.44×, LAMB 1.19×, Grokfast 1.58×,
GrokfastMA 1.47× — each from 8–27 allocs/step to 0.

**Deliberately silent** on buffers that escape: a returned buffer, one stored into a
receiver field, or a ring slot (`ring[pos] = flat`). Those are still poolable but by a
different fix — pre-allocate the slot and overwrite it in place — so N does not
mis-advise "hoist to one reused field." Also silent on value-receiver methods and free
functions (no receiver to hang the reused buffer on), on `make()` outside any loop, and
on **pointer-element slices** (`make([]*T, …)`) — a handful of pointers used as
orchestration scaffolding (fully overwritten before a concat/reduce reads them), dwarfed
by the elements' own allocations, not the numeric value scratch the `growF64` win targets.

## PS5001 — divide by a loop-invariant scalar  *(scanner: static)*

A `/` (or `/=`) by a loop-invariant scalar on every iteration of an element-wise loop.
Hoisting `inv := 1/D` once and multiplying is **1.2–1.5×** when the divide is the
loop's standalone cost — a float divide is ≈20–40 cycles versus ≈4 for a multiply.

**Shipped:** SoftCap VJP 1.28×/1.29×, and the whole optimizer reciprocal-multiply family
(Adam/Cautious/LAMB/AdEMAMix/Adafactor bias-correction & moment divides, 1.1–1.3×).

**SAFE ONLY for a CONTINUOUS output** — a gradient, an optimizer moment, a probability —
whose ½-ulp reassociation rides a tolerance. **NEVER** when the result feeds a discrete
step: `math.Round`, quantization (the nf4 `am` scale is flagged precisely so you DON'T
blindly convert it), or an `argmax`. Keep every path (typed fast + generic) on the same
`inv` so their bit-identity holds. Verify the divisor is a float, and that the divide is
standalone rather than amortized behind other per-element work.

**Deliberately narrowed** to keep the signal actionable: silent on a divisor accumulated
via `+=`/`-=`/`*=` (a reduction — a softmax Σ or attention denominator, where the divide
is minor or parity-locked, not a config scalar), on loops already dominated by a
transcendental (K/L territory — the divide is in the noise), on integer INDEX divisions
(`a[i/stride]`), on divisors that vary across iterations, and on non-element-wise loops.

Also silent on an **index decomposition**: when the same `x / d` appears alongside a
matching `x % d` in the function (`iy, ix := i/m, i%m`). Go's `%` requires integer
operands, so `x` and `d` are provably integers there and the divide computes a discrete
index, never a float a reciprocal-multiply could replace — type-sound rather than
heuristic, so it cannot suppress a real float divide. The match is on the whole pair
(same numerator AND divisor, modulo parenthesization), not on the divisor name alone, so
a function that happens to use `d` as a modulus elsewhere still gets its float divide
reported.

## PS5002 — symmetric-matrix full accumulation  *(scanner: static)*

A nested `i`/`j` loop that accumulates a **symmetric** matrix in full — `m[i][j] += x[i]·x[j]`
(an outer product) or a gram reduction `acc += M[i][k]·M[j][k]` folded into `m[i][j]`. Every
off-diagonal entry is computed twice.

**Fix:** if the consumer reads only one triangle — a Cholesky factor (`gmmCholesky`), a symmetric
eigendecomposition (`SymEig`), an inverse-root preconditioner — accumulate the upper triangle +
diagonal and mirror it down once (`m[j][i] = m[i][j]`). Roughly **halves** the O(n·d²) accumulation.
Bit-identical where the product is commutative (`x[i]·x[j] == x[j]·x[i]`, exact in IEEE-754).

**Shipped:** GMM full-cov mStep (1.35×), PCA covariance (1.24×) — the same class as the SVD
column-major win.

**Detection is a same-base product AND an `m[i][j]` write** under the two loops: matmul
(`C[i][j] += A[i][k]·B[k][j]`) has *different* bases (A≠B) so it is not flagged. **Deliberately
silent** on already-triangular loops and on forms that pre-slice the row (`covi := m[i]; covi[j] +=
ci·c[j]`) — the hoisted 1-D write hides the `[i][j]` signal, so verify hot covariance/gram loops by
eye too. **Verify the consumer reads one triangle and benchmark** before shipping.

## PS4008 — a matmul whose inner loop is a serial scalar dot  *(scanner: static)*

A triple-nested loop whose innermost body is a single `acc += A[…] * B[…]`, with `acc`
declared in the middle loop and stored to an indexed destination after it. The accumulator
is a **serial dependency chain**: each FMADD waits on the previous one's ~4-cycle latency,
so the loop runs at the FMA's *latency* rather than its *throughput*, no matter how much
ILP the machine has.

**Fix:** transpose the k-dim operand once (`k·m` stores against `m·m·k` MACs — negligible)
and rewrite as ikj/axpy, `c[j] += av * bt[j]`, so the accumulators are independent across
the output index. If the two operands are the SAME slice the product is symmetric to the
last bit, so computing one triangle and mirroring halves the work again.

**Shipped:** `nn.matmulABt` — 0.92 → 0.32 ns/MAC, `BenchmarkMuonStepOnly` 418.3 → 200.0 ms
(**2.09×**), bit-identical, at the cost of one reused `[k,m]` scratch (+6% bytes/op).

**Bit-identity is claimable but must be PROVEN**, not assumed: the ikj form accumulates over
`p` in the same ascending order into an accumulator that also starts at +0, which is the
argument `backend/cpu/gemm.go` makes for its tolerance-0 gate — but only a cross-reference
test against the *pre-rewrite* form actually holds it. Two traps, both measured: reversing
the `p` order is caught by such a test, while carrying over matmulFlat's
`if av == 0 { continue }` skip is **NOT** caught by random fixtures (they contain no exact
zero) and silently drops `0·±Inf` NaNs — the exactness gate needs an explicit zero/Inf case.

**Deliberately silent** on the ikj/axpy form itself (applying the advice clears the
finding), on same-base reductions with no indexed store (a norm has no output index to make
independent), and when the inner loop does anything besides the accumulation (the dot is
then not the whole cost). Findings often overlap PS4006 when the operands are `[][]T`.

**Sometimes the rewrite just loses, and the dot form is already right.** The MLA score
recompute was implemented BOTH ways, gated bit-identical, and benchmarked: a pure reorder
is **0.85×**, and transposing the keys first — the very structure that made `matmulABt`
win — is **0.82×**, i.e. worse still. The reason is that its `j` loop already supplies
instruction-level parallelism (each `(i,j)` score is independent, so there is no serial
chain to break), while ikj adds `(dh+dR)×` the read-modify-write traffic on the score
array. Before assuming the accumulator is the bottleneck, ask whether an enclosing loop
already provides the independence.

**A·Bᵀ needs the TRANSPOSE to win, and the transpose costs an allocation — decide per
site.** Measured both ways. `nn.matmulABt` transposes the k-dim operand once, which makes
the inner loop contiguous, and won **2.09×**. The MLA score recompute was rewritten ikj
*without* a transpose — a pure reorder, no allocation — and **LOST 17–18%**
(`BenchmarkMLAVJPSeq128` 5.02 → 5.89 ms, `Seq256` 19.9 → 23.5 ms, control flat): the
strided reads across the output index cost more than the independent accumulators gain.
So the rewrite is not free-standing; it is worth it only when the inner loop ends up
contiguous, which for this shape means paying for a transposed copy. That is exactly why
`soap.go` and `galore.go` decline — pooled paths where a per-call allocation is the thing
being avoided — and the MLA measurement is the other half of the same lesson.

**"Both operands are already contiguous in the summation index" is NOT a reason to
decline** — a rationale used twice here before it was disproved. `nn.matmulABt` is exactly
that shape (`s += ai[p] * bj[p]`, both stride-1 in `p`) and the ikj rewrite still won
**2.09×**, because the serial FMADD chain, not the access pattern, was the cost; the
transpose paid for itself. A guard encoding that rationale was written and reverted when it
suppressed the rule's own canonical fixtures. The real reason those two sites were declined
is **allocation**: the transposed copy is a fresh buffer on every call, on paths that are
pooled precisely to avoid that. Site-specific and not statically detectable — decline in a
comment, not in the rule.

## PS4005 — an N-D odometer ticked once per ELEMENT  *(scanner: static)*
```go
for pos := range xs { // one element of work
	for d := nd - 1; d >= 0; d-- { // ...then a full odometer tick
		idx[d]++
		off += stride[d]
		if idx[d] < shape[d] {
			break
		}
		idx[d] = 0
		off -= stride[d] * shape[d]
	}
}
```
The innermost axis has a CONSTANT stride across a run of `shape[nd-1]` elements, so
that run is one straight walk — a `copy` at stride 1, a fill at stride 0 — and the
odometer need only tick once per run. Bit-identical: traversal order and the
per-element work are untouched, and the innermost axis contributes
`inner*stride − stride*inner = 0` to the offset over a full run.

Measured 4.49× (ref broadcast), 5.29× (cpu broadcast), 3.14× (tensor gather),
1.78× (ref argmax).

Matches the odometer's SHAPE, not the loop body — those four sites do a copy, a
cast and an accumulate respectively, so any rule keyed on the body catches them
only by accident. **A hoisted odometer keeps the same shape** and merely runs per
run, so the rule distinguishes them by where the walk starts: per-element ticks
every axis (`d := nd - 1`), hoisted skips the innermost (`d := nd - 2`). Without
that discriminator the rule reported the sites it had just helped fix.

## PS4006 — a `[][]T` matrix indexed inside a nested loop  *(scanner: static)*
One heap allocation per ROW, then `m[i][j]` two-deep in a nested loop. A column walk
(`m[k][p]`, k varying) dereferences one unrelated cache line per row. Flatten to a
single `[rows*cols]` buffer indexed `m[i*cols+j]`: index arithmetic only, so
bit-identical, and `rows+1` allocations collapse to 1.

Measured 2.15× (solvespd), 1.5× (cholesky), 1.35× (qr), 1.2× (SymEig, SVD).

**And 0.93× on `classic/linalg.go` cholSolve** — reverted. An OLS fit is dominated
by the O(N·d²) Gram-matrix build, so the O(d³) factorization this flags is a small
share of it. The flatten pays when the flagged loop IS the enclosing operation's
work; measure the OPERATION end to end, not the loop. Cheaper mitigation when it is
not: hoist `row := m[i]` above the inner loop — `naivebayes.go` and `gmm.go` already
do, which is why they were declined.

Requires BOTH the per-row allocation loop and a two-deep index inside ≥2 nested
loops. A `[][]T` merely passed around, indexed once, or genuinely ragged is not a
candidate.

**Biggest measured win of the rule, and a null beside it — same commit, same file.**
Interleaved (4 alternations): `ExpertChoiceCombine` **1.554×** (654,598 → 421,160 ns) by
hoisting BOTH `y[t]` and `expertOut[ex][i]` out of the innermost loop; `ExpertAffinity`
**1.167×** by hoisting the destination row. The third hoist in the same file, the gate
fill in `ExpertChoiceRoute`, measured **0.992×** — no effect, because that loop is
O(capacity) beside an O(n log n) sort. All three are bit-identical; only two pay. The
size of the enclosing work, not the shape, decides.


## Serial spines — found by scaling sweep, not by a static rule  *(scanner: `tools/scaling_sweep.sh`)*

Run a benchmark at `GOMAXPROCS=1` and at full width and divide. Substantial ns/op with a
ratio near **1.0** means nothing in it parallelizes — whatever dominates is serial.

**Why this is a script and not a PSxxxx rule.** Proving a loop's iterations are
independent is a dataflow question, and perfscan is AST-only. A static rule that guessed
at independence would advise data races. Scaling is an *observation*, so it is measured
rather than inferred — and the measurement is cheap enough to sweep a whole package.

**One sweep found four**, and the two acted on were the largest wins of their sessions:

| benchmark | before | speedup | outcome |
|---|---|---|---|
| `GBMHist_hist_80k` | 334 ms | 1.01× | fixed → **1.57×** |
| `GMMFitFull` | 77 ms | 1.00× | open |
| `MLAVJPSeq256` | 20 ms | 0.99× | open |
| `CholeskyVJP_128` | 4.3 ms | 1.09× | open |

The quantized prefill path was found the same way and is now **2.52×**.

**A ratio near 1.0 is a candidate, not a defect.** Plenty of work is legitimately serial,
and small benchmarks are dominated by dispatch cost rather than compute — hence the
2 ms floor before the flag is raised.

**A profile that is mostly `cond_wait` is not evidence of overhead.** After the exact GBM
fit was parallelized its profile read 72% runtime synchronization against 22% compute —
which looks like the pool eating three quarters of the machine. It is not: samples in
`pthread_cond_wait` are **parked** time, attributed but free, and a parked worker is not
consuming its core. Replacing the parking with a bounded yield-free spin measured **2–4%
SLOWER** across three different callers (GBM −4%, Muon −2%, KNN neutral) and was reverted,
which is exactly what `backend/cpu`'s own notes record for always-spin variants. **The
scaling curve is the evidence; the sync share is not.**

**A ratio of two noisy numbers is noisier than either, and this bit already.** The first
version of the sweep used `-benchtime 10x` and reported `CholeskyVJP_128` at **0.88×** —
*slower with more cores*, which reads as false sharing and is worth chasing. At 300× the
same benchmark measures **1.10×**: the 0.88 was one noisy sample, and it reached a commit
message and two records before anyone re-measured. The script now defaults to a generous
benchtime and takes the **minimum of 3 runs per arm** — benchmarks are contaminated upward
by interference, never downward. The corrected table above reflects the re-measurement.

**Watch for the loop order that blocks the split.** GBM's histogram was sample-major,
which reads the bin table contiguously and is the *faster serial form*, but makes every
feature's bins a shared write target. Feature-major is partitionable and 22% slower on one
core. Keep both and choose by whether the work will actually be split — otherwise a
constrained host pays for a speedup it will never collect.

## PS6006 — a receiver field used as a per-call temporary  *(scanner: static)*

```go
func (m *M) logGaussian(x []float64, c int) (float64, error) {
    y := m.yScratch            // receiver field, taken as a local alias
    for i := range d { … ; y[i] = s * id[i] }   // written…
    for i := range d { quad += y[i] * y[i] }    // …and read back, same call
}
```

**Two defects wearing one shape.** The method cannot be called concurrently, which
silently blocks parallelizing any loop over it; and when someone parallelizes anyway,
every worker writes the same cache line.

**Both were measured.** `classic.GaussianMixture.logGaussian` carried exactly this, with a
comment saying the method *runs serially* — the precondition was known, written down, and
violated the moment the E-step was parallelized. `-race` caught the correctness half. The
performance half is the striking one: the racy version measured **1.16×**, and moving the
buffer to a parameter took the same parallelization to **1.93×**. The contention cost more
than the allocation saved.

**The fix is always the same:** make it a parameter. The requirement then lives in the
signature instead of a comment, where the next caller cannot miss it.

**Two things this check needed that the obvious version lacks.**

*Alias tracking.* Real code writes `y := m.yScratch` then `y[i] = …`, not
`m.yScratch[i] = …`. A detector insisting on the literal selector found **nothing at all**,
including the method it was written from.

*A name discriminator.* Written-and-read-in-one-method is not sufficient: the first cut
flagged `m.Means`, persistent model state that the M-step legitimately fills and reads
back. Nothing in the AST separates *temporary* from *state I happen to finish building
here* — the difference is intent, and intent is recorded in the name. Hence
`scratch`/`scr`/`buf`/`tmp`/`work`; a project with a different convention configures its own.

**The name was the wrong discriminator, and two more instances proved it.** The first
version keyed on `scratch`/`buf`/`tmp`-style field names. It then MISSED `gbmBuilder.vals`
and `gbmBuilder.part` — two further blockers found the same way, in the split search and
the node partition — because neither is spelled like a buffer. Three independent
instances, one caught.

What actually separates a temporary from state is **structural**: a temporary is *indexed
in exactly one function*. The method that needs it uses it element-wise; everything else
at most allocates it. State (`m.Means`, `b.cols`) is indexed by several. Three more
filters were needed to make that usable, each mutation-probed load-bearing:

- **exported fields are never temporaries** — they are API, readable from outside the file
- **slice-typed only** — the false positives were maps used as registries
- **primitive elements only** — `[]soapState` and `[]*tree` are collections someone keeps;
  `[]float64` is working space
- **a slice expression is a read** — `copy(dst, b.part[:r])` is how the partition consumed
  its scratch, and requiring an *indexed* read missed it entirely

Precision went **24 → 15 → 8** findings across those filters while all three real
instances stayed caught. Six of the surviving eight are genuine, including four CART
builder buffers in `tree.go` that a deferred task had predicted would be there.

Silent on plain functions (no receiver, no sharing) and on fields only written and never
read back, which are outputs rather than temporaries.

## PS6007 — a per-item search that chooses where to accumulate  *(scanner: static)*

```go
for _, x := range data {
	b := nearest(x, cent) // expensive, independent per item
	cnt[b]++              // …but this is order-dependent
	for t := range dim {
		sums[b][t] += x[t]
	}
}
```

The loop **looks** partitionable — every item is independent — and is not. The
accumulation is a reduction over items in order, and per-chunk partial sums reassociate
it. What makes this shape worth its own check is that the index *disguises* the
dependency: with `sums[b]` rather than `total`, the write appears to belong to the item.

**Split it.** Run the search in parallel into an assignment array, then fold sequentially
in the original order. The expensive half parallelizes; the order-dependent half does not
move. Shipped twice — AQLM's k-means assignment (part of **990ms → 278ms**) and, in its
scalar form, the GMM E-step's log-likelihood total (part of **76.5ms → 18.7ms**). In both
the reduction was a small fraction of the work, so leaving it serial cost nothing
measurable and bought exactness outright.

**The wrong fix passes the tests.** Parallelizing the whole loop with per-chunk partials
is quicker to write, silently not bit-identical, and green under any test that checks
*reproducibility* rather than *preservation* — which is what the determinism tests in this
repo do (they run the same code twice).

**It missed its own motivating case first.** The k-means loop is `for _, x := range data`,
and the check used a helper that requires a **named** loop variable, so it found nothing.
Replaying against the pre-fix revision is what exposed it — fixtures written from the same
mental model as the detector all used named keys. Third rule in this campaign to fail that
way (see PS6010, PS6006); replay is now part of how a detector is validated.

Silent on scalar accumulations (`total += v` is every reduction loop ever written), on
plain indexed **stores** (idempotent — the last writer wins, no order is preserved), and
when the accumulation index is the loop variable rather than the searched value, since
each item then owns its slot.

## PS6008 — a buffer allocated inside a parallel body  *(scanner: static)*

```go
parallelFeatures(d, n, func(lo, hi int) {
    vals := make([]float64, n)   // once per DISPATCH, not once per program
    …
})
```

**Whether this is free or ruinous depends entirely on how often the dispatch runs**, which
is why the check reports rather than condemns. Both sides were measured here:

| dispatch frequency | site | memory |
|---|---|---|
| once per encode pass | AQLM ICM | 49 → 51 MB — fine |
| once per EM iteration | GMM M-step | 4 → 4 MB — fine |
| **once per tree node** | GBM exact | **64 → 2007 MB** |

The GBM figure is a **31× memory regression that shipped**, hidden behind a 2.80× speedup,
because the commit reported only ns/op. Identical code shape in all three; three orders of
magnitude difference in cost, decided by the enclosing call frequency alone.

**The fix is not to avoid the allocation but to move it** — one buffer per *chunk* on the
caller's struct, allocated once, selected by the chunk index (`parallel.RowsIdx`).

**No local loop is required to fire, deliberately.** The GBM case has none: `bestSplit`
contains no loop around its dispatch — `bestSplit` *itself* runs per node, one call frame
up. A check demanding a visible enclosing loop would have missed the only case that
mattered. That is also why this is not covered by PS2001/PS2004, which look for
allocation inside a loop and stay silent here.

Silent when the buffer is hoisted and chunk-indexed, on non-allocating defines in the
body, and on callbacks that are not parallel dispatches.

## PS6009 — `sort.Slice` allocates a reflect swapper on every call  *(scanner: static)*

`sort.Slice`/`SliceStable` reach the swap through `reflectlite.Swapper`, which **allocates
on every call regardless of slice length**. `slices.SortFunc`/`SortStableFunc` take the
same comparator, produce the same permutation for a total order, and monomorphize the swap.

**Triage by call frequency, not slice length** — the counter-intuitive part, and what the
measurements show:

| site | frequency | result |
|---|---|---|
| `classic/tree.go` `radixByFeature` | per node per feature | 1,095,700 → 352,027 allocs (**3.11×**), 182 → 161 MB |
| `classic/knn.go`, `spatialindex.go` | per node / per query | 36,004 → 24,003 allocs (**1.50×**) |
| five other sites | once per Fit/SVD/call | **declined** — one swapper each |

The KNN sorts handle *short* slices — k results, one node's indices — and still returned
1.50×, because the allocation is per **call**. A long sort called once is worth nothing; a
short sort called a million times is worth everything.

**Split out of PS3002 deliberately, and the reason is concrete.** That check reports the
same sites but bundles two unrelated remedies — an LSD radix on the key bits, and this swap
fix — and states it cannot verify the radix precondition. After `classic/tree.go` was
converted, PS3002 went on flagging the `slices.SortFunc` **replacement it had recommended**,
so the site could only be *silenced* with a suppression, never *cleared*. A check that
cannot recognize its own fix cannot tell you whether the work is done. PS6009 clears.

**Two conversion traps**, both real:

- Both forms are **unstable**, so ties may land differently. Check the comparator is a total
  order, or gate the output. Inverting a `(dist, idx)` tie-break left every KNN test green
  until a deliberately constructed tie was added.
- `sort.Slice` passes **indices**, `slices.SortFunc` passes **values** — so
  `key[order[a]] < key[order[c]]` becomes `key[a] < key[c]`, silently wrong if transcribed
  rather than re-derived.

Silent on `sort.Ints`/`Strings` (concrete, no reflection) and on same-named methods outside
the `sort` package.

## PS6003 — a fast path that covers only part of a variant family  *(scanner: static)*

A function short-circuits the general path for some members of a variant family, and a
switch in the same function shows the family is larger:

```go
if qt == Q8_0 && m == 1 { … return out, nil }   // fused: no row materialization
…
switch qt {                                      // …and six more types land here
case Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K:
```

The uncovered variants keep paying the general path, and nothing in the code says so —
a fast path reads as *this case is handled*, not *only this case is handled*.

**Found by its symptom, not its shape, which is the argument for automating it.**
`gguf.QMatMul` had the fused single-token path for Q8_0 and none for Q4_0, so Q4_0 decode
ran **slower than Q8_0 despite half the memory traffic** — backwards for the smaller
format, and the only reason anyone looked. Fusing it was **1.40×** on the enclosing
`QuantMamba2` decode step. Nobody reading QMatMul top to bottom would have suspected the
gap; it is visible only by comparing the guard against the switch fifty lines below.

**Advisory, and deliberately not a defect report.** A fast path may legitimately cover one
variant: the others may be rare, unfusable, or already fast. What the rule asserts is that
the asymmetry is intentional-or-not and the code does not distinguish those — so it is
worth one benchmark per uncovered variant.

**The guard must close before the switch opens.** Its only false positive on this tree was
`gguf`'s metadata reader, where an `if vt == vtI16 { … return }` sits *inside* one case
clause of `switch vt` — a sub-case of the dispatch that bypasses nothing. Positional
dominance is what separates a bypass from a branch.

Also silent on switches with fewer than three named members (a two-way switch is a branch,
and a fast path for one of two arms is an if/else written twice), on guards that do not
end in `return` (without it the general path still runs), and when every member is already
covered. Literal cases are excluded from the family — `switch n { case 1, 2, 3 }` is not a
set of formats. That last filter suppresses nothing on its own, since an all-literal switch
has no member matching the guard; what it does is keep a bare literal out of the reported
variant list when a switch mixes the two. Probing is what established the distinction.

## PS6010 — an output loop that re-reads an operand no output varies with  *(scanner: static)*

One accumulator per output, and an operand that is the same for every one of them:

```go
for ni := range n { // one output per iteration
	wr := w[ni*k:] // per-output
	var acc float64
	for i := range k {
		acc += row[i] * wr[i] // row[i] is re-loaded for every ni
	}
	outf[ni] = acc
}
```

Unrolling the OUTPUT loop by 4 amortizes each `row[i]` load and its float conversion
across 4 accumulators — register blocking, the m==1 dual of unroll-and-jam. Bit-exact:
every output keeps its own accumulator and still sees the same terms in the same order.

**Shipped:** `gguf.QMatMul`'s Q8_0 single-token path is blocked by 4 and measured
**526µs → 233µs** per decode step (**2.26×**) — larger than either fusing the dequant into
the dot (1.40–1.75×) or parallelizing the row loop (1.19–1.66×) delivered. It was blocked
on **one of seven** sibling paths. Q4_0, the closest in block shape, measured **1.55×**
when given the same treatment.

**Nothing already in perfscan flagged either.** PS4008 wants a plain `acc += A[i]*B[i]`,
and these loops unpack nibbles and convert float widths along the way, so it stays silent.
The remedy differs too: PS4008 breaks a serial FMADD dependency chain, while here the
chain is already broken by having one accumulator per output — what is wasted is the
repeated load of the shared operand. Different defect, different fix, separate check.

**The first version missed the loop it was written from**, which is the lesson worth
keeping. The per-output operand in the Q4_0 loop arrives as a **range value**
(`for i, q := range qs`), never as an index expression, so a derived-value closure that
walked only assignments left it unclassified and the check saw no per-output operand at
all. It reported 55 findings elsewhere while staying silent on its own motivating case —
a rule can be noisy and blind at once, and only replaying it against the original site
catches that. Replaying against the pre-blocking revision is now part of the validation.

Silent on already-blocked loops (a stride above 1 is someone having done this), when
**every** operand is output-invariant (that is PS5003, whose fix is to hoist the whole
computation out rather than unroll it), and when the accumulator never reaches an output
index — that last guard took the check from 145 findings tree-wide to 56.

## PS6004 — a dual-path kernel whose bit-identity claim is unverified  *(scanner: static)*
A function carrying a devirtualized fast path (guarded by a configured
`fastPathHelpers` entry in comma-ok form, or a `switch x.Dtype()` with a `default`
arm) AND a generic accessor fallback. That structure is a bit-identity CLAIM, and
the claim usually lives in a comment rather than a test.

**Not a performance finding** — nothing here is slow. It is an unverified invariant.
Four kernels with this shape were probed and all four were blind to a one-ulp change
in the fast path. The rule cannot tell whether a bit-exact test exists (test
sensitivity is not an AST property); it lists the population that needs one.

Probe with a deliberate one-ulp mutation to decide, confirm the mutated line
actually executes before believing a surviving mutation, and run the probe over
every package that could hold a cross-reference test — not just the kernel's own.

**A third form: the fast path that DECLINES to its caller.** `v, ok := x.data.([]float32)`
discriminates on concrete storage rather than through a configured comma-ok helper, and
the generic arm lives in the *caller*, reached by returning `false`. Neither the
helper test nor the same-function accessor test sees it. This was found the hard way —
by shipping one: `tensor.gatherHalfTyped` devirtualizes four half-cast arms for a
**3.19×** win, and PS6004 reported nothing. A verification rule that goes quiet on a
real dual path is worse than no rule, because the silence reads as "nothing to prove".
Detected by: a `bool` result, **two or more** comma-ok assertions to *slices of numeric
types*, and a `return false`.

The numeric-slice requirement is not decoration. Without it the widening fired on 13
functions inside perfscan itself — an AST visitor is wall-to-wall `x, ok := n.(*ast.Foo)`
followed by `return false`, structurally identical and semantically unrelated. Asserting
`[]float32` is the signal; asserting `*ast.Ident` is not. Tree-wide the tightened form
adds exactly one finding (56 → 57) instead of fourteen.


## PS0001 — a `//perfscan:ignore` that suppressed nothing  *(scanner: static)*

An inert suppression is **worse than no suppression**, because it reads as though it took
effect. A directive goes stale two ways:

```go
//perfscan:ignore PS3002 deliberate
y := len(x)                 // an edit inserted this…
sort.Slice(x, less)         // …and the directive no longer reaches its target
```

or the finding was genuinely fixed and the comment should be deleted. Both want the author's
attention, exactly as an unused lint suppression does.

This file already widened a directive's reach from its own line to its whole comment block
after **two directives here were found dead**. Widening reduces the failure mode; it cannot
detect it. This check detects it.

**It caught its own author.** Three directives written the same day — two in `beam.go`, one in
`diverse_beam.go` — had been separated from their targets by a later edit that inserted a
selection block between the comment and the sort. All three read as deliberate, considered
suppressions. None of them suppressed anything.

**Two preconditions, both mutation-verified:**

- **A directive, not a mention.** The token must open the comment, after the marker and any
  indentation. Matching it anywhere in the text made four of this file's own doc comments
  register as live directives — harmless only while no finding ever landed on a doc-comment
  line, and not harmless once unused directives became reportable.
- **Crediting must mirror suppression exactly.** A first attempt credited only a directive's
  own line and the next, on the reasoning that over-crediting merely hides a stale directive
  while under-crediting is safe. That was backwards: two directives stacked above one statement
  form a two-line block, so the upper one sits two lines from its target, and the tight span
  reported a *working* directive as unused. Under-crediting produces false reports — the very
  failure this check exists to prevent.

## PS6022 — a full sort feeding a truncation  *(scanner: static)*

A slice sorted in full and then resliced to a smaller bound:

```go
slices.SortFunc(cands, byScore)
if len(cands) > bPrime {
    cands = cands[:bPrime]     // the order of everything past bPrime is discarded
}
```

This is the third consumer shape in the sort-does-too-much family, and the one the other two
**structurally cannot see**. PS6013 requires a counted loop that indexes the slice; PS6001
requires a consumer that breaks on a threshold. Here the consumer is neither a loop nor a
break — it is a reslice, so there is no loop to match.

Both existing checks reported **zero hits across `nlp/`** while beam search and diverse beam
search each sort every candidate (beams × vocabulary) to keep the top few. At a 2048
vocabulary and width 8 that is a sort of **16384 elements to select 8**, once per generated
token. Do not read PS6001/PS6013 silence as evidence that a sort is well-sized.

**Soundness.** The bound must not be `len(target)` — that keeps everything. Any statement
between the sort and the reslice that indexes or ranges over the slice is disqualifying: it may
depend on the full order, and it also means PS6001 or PS6013 describes the site better. A
`len(target)` guard is not such a read, and needs no special case: `len` takes the slice as a
bare identifier, which is neither an index nor a range.

**Bit-safety is a precondition, not a consequence.** Replacing the sort with a selection is
bit-safe only when the comparator is a **total order** — with ties the retained *set* is not
unique, so a selection and a sort can legitimately keep different elements. In the motivating
case the comparator was deliberately written as a total order so it would reproduce a stable
sort's tie order, which is what makes the substitution safe there.

Every clause of this check is mutation-verified. An explicit `len()` exemption in the
intervening-read test was written first and **removed**: no floor depended on it, and by
stopping the AST descent it would also have skipped a genuine indexed read nested inside a len
argument, such as `len(target[i])`.

## PS6021 — a fan-out helper with no per-worker seam  *(scanner: static)*

A parallel helper whose callback receives **only a work index**:

```go
func knnParallelRows(n int, body func(i int))          // KNN: 3 allocations per QUERY
func nbPredictParallel(n, feat int, body func(i int))  // GaussianNB: 1 allocation per ROW
```

The callback runs once per **item**, so a buffer the caller allocates inside it is allocated
per item. Hoisting it above the helper makes it shared mutable state every worker races on,
and putting it on the receiver is the same bug with a longer fuse (PS6006). With no
per-worker seam in the signature, **per-item allocation is the only correct option
available** — which is exactly why these sites survive review. The code is not careless; the
interface is short a parameter.

Three wins in this repository were blocked by this one shape, all with the arithmetic already
parallel:

| site | after adding a seam |
|---|---|
| `GaussianNB.Predict` | **1.28x**, allocations −99.2%, bytes −73.6% |
| `KNNClassifier.Predict` | allocations −99.4%, bytes −94.1%, 2.1% faster |
| `DBSCAN.Fit` | allocations −78.8%, bytes −40.0% |

**The fix is the signature.** Either give the callback a scratch parameter the helper supplies
per worker (`gmmParallelRows`, `moeParallelTokens`, `wkvParallelChannels`), or take a
`func() T` constructor the helper calls once per worker and passes down.

Two floors keep this quiet enough to act on, and both are mutation-verified:

- **A `(lo, hi)` range callback is not reported.** The caller can allocate inside the chunk
  closure, which is per-chunk and therefore already per-worker. Most helpers here have that
  shape, so reporting them would drown the real hits.
- **A channel-creating helper is not reported.** That is a work-queue primitive
  (`parallelBuild`), where the callback *is* the job and other helpers build their seam on top
  of it by passing a worker count as the job count. Reporting the primitive reports the cure.

A clause excluding helpers that take a `func() T` constructor was written and then **removed**:
such a helper must pass the constructed value to its callback, so the callback already carries
a scratch parameter and fails the index-only test. Mutation testing showed no test could reach
the clause, and a predicate clause nothing exercises implies coverage it does not have.

## PS6019 — a jam loop whose remainder delegates  *(scanner: static)*

An unroll-and-jammed loop whose scalar remainder is handled by a **different code path**:

```go
for ; c+4 <= k; c += 4 {
	y0, y1, y2, y3 := y4[0], y4[1], y4[2], y4[3] // wide body: buffers passed in
	// …
}
for ; c < k; c++ {
	ld[c], _ = m.logGaussian(x, c) // tail: reads the buffer off the RECEIVER
}
```

Two code paths computing one thing, so every property established for the wide body has to be
re-established for the tail. **This shipped as a data race.** Parallelizing GMM's caller required
moving the wide body's four solve buffers off the receiver; the tail still read the receiver, and
the row scan raced for any component count not a multiple of four. Nothing caught it because
every benchmark and test used `k=8` — the race detector cannot flag a line that never runs, and a
parity test compares equal on a path that is empty. The defect was one modulo from the tested
case.

The properties that go stale in a tail are worth naming: **per-worker scratch** (a race),
**explicit FMA pinning** (a one-ulp divergence on the remainder only), and **hoisted bounds or
dtype checks** (a panic on the last elements).

**Delegation is the signal, not the tail.** A tail repeating the wide body inline shares its
edits by construction — same text. A tail calling a method inherits nothing. So the rule reports
only when the remainder invokes a receiver method and the wide body does not, which keeps it
quiet on the ordinary scalar remainder nearly every jammed kernel has.

**This rule does not go quiet when the bug is fixed, and that is deliberate** — unlike every
other rule here. Threading the scratch through a parameter fixed the race; it did not remove the
duplication, so the next property established for the wide body meets the same gap. It is a
maintenance hazard attached to a shape, not a defect with a closing state. Suppressing once state
is threaded was considered and rejected: telling "argument the wide body uses as a buffer" from
"argument it merely reads" needs alias analysis — the racy call passed `x`, which the wide body
reads too — and an unsound suppression here hides a race.

**The actionable response is a test, not an edit**: whenever you touch the wide body, run a trip
count NOT divisible by N. `PROC-UNROLL-TAIL-COVERAGE-001`.

## PS6018 — a function that mostly moves data  *(scanner: domain)*

Three or more dispatches of a pure movement op — slice, reshape, transpose, concat — in one
function with no fused raw-storage path:

```go
flat, _ := exec1(ctx, backend.OpReshape, wide, x)         // seven layout dispatches
rot, _ := exec1(ctx, backend.OpSlice, head, flat)         // around
pass, _ := exec1(ctx, backend.OpSlice, tail, flat)        // exactly
rotWide, _ := exec1(ctx, backend.OpReshape, flatten, rot) // one
rotWide, _ = exec1a(ctx, backend.OpRoPE, r, rotWide)      // arithmetic op
merged, _ := exec1(ctx, backend.OpConcat, axis1, rot, pass)
```

**Movement cannot change a value**, so gathering the operands out of storage and scattering the
result back is bit-identical BY CONSTRUCTION — no reassociation, no FMA question, no tolerance
argument. That is what makes this class worth flagging on sight where a bare "too many
dispatches" report would not be: the fix needs index arithmetic, not numerical judgment.

**Shipped three times, all measured**: `partialRoPE` **1.25–1.33×** with **38–43% fewer
allocations** across three architectures (31 call sites, one shared helper); Gemma2 capped
attention **1.21× / −27.6%**; DeepSeekV2 absorbed attention **1.12× / −9.3%**.

**PS4011 does not subsume this.** That rule requires the dispatches to sit in a sequential
loop, and `partialRoPE` is straight-line code — so PS4011 could not see the largest of the
three wins. The two rules cover the two shapes.

**Gate the fused arm on `ctx.Recorder == nil`.** Under a tape every one of those layout ops is
a gradient edge, and replacing them with raw copies detaches the graph. The rule suppresses
once a function has that guard, a configured fast-path helper, or a `Storage()` grab — a fixed
function must stop reporting, or the rule flags its own successes forever.

**Threshold three**: two movement ops around one arithmetic op is often the irreducible shape
of an operation (transpose then matmul). Three or more means the layout algebra has outgrown
the computation.

**Prove the layout algebra over a SWEEP of geometries, not one shape.** The claim is index
arithmetic, and a single `(seq, heads, rotaryDim)` triple can agree by coincidence when an
offset is wrong; `rotaryDim == hd` also takes a different early-return branch. Compare raw bits
between a plain and a taped context — that tests the arithmetic and the tape guard at once —
and panic-probe the fused branch, since a parity test whose arms take the same path passes
while proving nothing.

## PS6017 — a variadic helper called at an arity a sibling already covers  *(scanner: static)*

A variadic function called inside a loop with a fixed number of arguments, when the same
package declares a non-variadic sibling taking exactly those:

```go
for l, b := range m.Blocks {
	a, _ := exec1(ctx, backend.OpMHA, attn, q, kNew, vNew) // allocates a 3-element slice
}
```

`exec3` takes the same three tensors as named parameters and pools the slice it builds, so the
variadic call is a per-iteration allocation with a ready-made replacement. **422 candidates
tree-wide** across four families — `exec1` against `exec1a`/`exec2`/`exec3` in `nlp`, and
`rdropExec`/`hcExec` against `execPool1`/`execPool2` in `nn`, which nobody had connected.

**The sibling relation comes from signatures, no config.** A candidate must have identical
leading parameter types followed by exactly n parameters of the variadic element type, so the
call transfers argument for argument. The registry is built package-wide in a pre-pass,
because the variadic form and its siblings are usually declared in one file and called from
twenty — the same reason `intMapReg` exists.

**At least one FIXED leading parameter is required, and that single condition removed the only
wrong pairing in the tree.** With none, "same trailing types" is far too weak:
`concat1D(parts ...*tensor.Tensor)` matched every two-tensor function in its package. The
shared prefix is what makes a family — for `exec1` it is `(ctx, op, attrs)`, which names the
operation all the siblings perform.

**Rendering types with `go/printer` is load-bearing.** `exprText` has no `StarExpr` case and
returns empty for every pointer, which breaks the comparison in both directions: as a
placeholder all pointers collapse and `*backend.Context` equals `*tensor.Tensor`; as
unrenderable-and-skipped, every candidate with a pointer parameter drops out and the rule
reports **nothing at all**. The second is the worse failure, because a silent check reads as a
clean codebase — and note that a zero-expecting suppression test cannot detect it. The guard
is the positive test.

**What it cannot check** is whether the sibling is semantically equivalent rather than merely
type-compatible; that is a judgment about two bodies. Here the siblings pool only when
`ctx.Recorder == nil` and delegate to the variadic form otherwise, which is exactly the wanted
equivalence — elsewhere, read before swapping. Restricted to loop bodies: the allocation is
per call either way, but once per invocation is rarely worth a diff.

## PS6016 — a struct literal rebuilt every iteration from constants  *(scanner: static)*

A composite literal built inside a loop and passed straight to a call, whose every field
initializer is loop-invariant:

```go
for l, b := range m.Blocks {
	q, _ = exec1(ctx, backend.OpRoPE,
		backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: pos}, q)
}
```

Nothing here depends on `l` or `b`. Rebuilding the struct is cheap on its own; what is not
cheap is that the parameter is an INTERFACE, so each construction is also a heap box — once
per layer per decoded token. Hoisting these above the loop across six `nlp` decode paths
removed **8490 allocations** from a 500-token generate (−2.9%) and 391KB of garbage. Wall
time was a wash: these are small short-lived objects, so the payoff is garbage pressure, not
throughput, and it should be reported that way.

**Soundness.** The literal must be passed directly as a call argument and nowhere else — not
appended, not assigned, not address-taken. A literal that escapes into a slice needs its
per-iteration identity, and hoisting it would make every element alias one value: a
correctness change wearing an optimization's clothes. Field initializers must reference no
loop variable and nothing the loop assigns.

**Dedup is per SITE, not per type.** The q and k RoPE attrs in a decode loop are two distinct
literals of one type and both need hoisting; keying on the type name reported one and hid the
other.

**Half of this defect is invisible to any parser, and that is worth knowing.** The same waste
occurs when the struct is ALREADY hoisted but the interface conversion still happens at the
call site — `quant_llama_decode.go` hoisted its `AttnAttrs` as a concrete struct and escape
analysis still reported it escaping. That form is what an earlier pass on the float paths
produced and then missed, because hoisting the struct *looks* like the fix. Recognizing it
needs to know the parameter is an interface, which needs `go/types`; this scanner is
deliberately `go/ast`-only. The tool that sees both forms is the compiler:

```bash
go build -gcflags='github.com/jxsl13/goai/nlp=-m' ./nlp/
```

That is how both forms were actually found here. This check covers what a parser can prove
and points at escape analysis for the rest rather than approximating it unsoundly.

## PS6015 — a batch-of-one call the loop never reads  *(scanner: domain)*

A pure call made once per iteration on a single-element batch, whose result is used ONLY to
append to a slice that outlives the loop:

```go
for { // per environment step
	v, _ := forward(NewContext(), critic, [][]float64{obs}) // batch of one
	ro.values = append(ro.values, v.AtF64(0, 0))            // the only use
}
// ro.values consumed here, after the loop
```

The loop does not depend on the answer while it runs, so N batch-1 calls answer what one
batch-N call answers. Hoisting the critic out of `rl.rlRollout` was **1.59×** on collection
(29239 allocations down to 15471) and **1.19×** end to end — each batch-1 forward was five
backend dispatches on a one-row tensor.

**This is different advice from PS1003, not a refinement of it.** PS1003 matches the same
call shape and says *call a single-item API instead* — drop the wrapper allocation, keep N
calls. That is correct when the loop READS the result: the actor forward in this very loop
feeds a softmax that feeds the sampled action that feeds the environment, and it cannot move.
PS1003 also reports once per loop, so where a hoistable and a non-hoistable call share a
loop it flags only the first — here that was the actor, and the critic went unmentioned
entirely. This check reports per call site and only for the hoistable case. Both rules on the
same loop is the intended outcome.

**Purity licenses the hoist, and for a stronger reason than in PS6014**: a call that consumed
RNG would move draws out of the stream and change every later iteration, so the callee must
be named in `pureComputeFuncs`. Beyond that, every use of the result inside the loop must be
an append to a slice declared outside it. One use in a branch condition, one handed to
another call, one feeding the iteration state — the result is loop-carried and the check
stays silent rather than proposing a hoist that changes behavior. That suppression is the
load-bearing test: the actor case is one line different from the critic case and flagging it
would be worse than reporting nothing.

## PS6014 — the same pure call made twice  *(scanner: domain)*

Two syntactically identical calls to a function the project has declared pure, in one block,
with nothing between them that could change the answer:

```go
qPred, _ := forward(NewContext(), d.Net, states) // untaped preview
target := New(F64, Shape{len(batch), k})         // reads qPred only
q, _ := forward(tape.Context(), d.Net, states)   // SAME net, SAME input
```

The first call exists only to seed the entries of `target` whose gradient must be zero — and
the second already holds those numbers. Deleting the preview and reading the taped result
instead was **1.30×** at batch 32 and **1.35×** at batch 128 in `rl.DQN.learn`, because on a
per-step path at these widths the cost is one Context plus five backend dispatches, not
arithmetic. The shape recurs wherever an untaped preview precedes the real taped pass, which
is a natural way to write a TD target and a natural way to write it twice.

**The leading argument is ignored when comparing**, and that is the whole point: the two
calls differ in exactly that argument (a fresh Context versus the tape's) and are otherwise
identical. A comparison that included it would miss every instance. Soundness therefore
rests on the rest — every remaining argument must be a plain name or selector chain, and the
check scans everything between the two sites for an assignment to any name the call reads, a
non-pure call handed one of those names, or an `&x` that lets the address escape. The scan
descends into nested nodes, so a write buried in a loop suppresses correctly.

**Purity is not derivable from syntax, so it is not guessed.** Only callees listed in
`pureComputeFuncs` qualify; with that list empty the check reports nothing and says so via
the starved-vocabulary warning. Flagging repeated calls in general would fire on every
`rng.IntN(n)` and every `env.Step(a)` — calls whose entire purpose is to differ.

**A zero here is the healthy state, not a dead rule.** The tree reports no candidates because
the one instance was found and fixed; `PERF-SCANRULE-EMPTY-001` is about rules that never
found anything, which is not this.

**`Forward` was evaluated for the vocabulary and rejected.** Adding it would widen the rule
from one package to most of `nlp`, and it is not sound: a `Sequential.Forward` containing a
Dropout consumes RNG in training mode, and a quantized decode `Forward` mutates its KV cache.
Either makes a second identical call load-bearing, so declaring it pure would license
deleting a call that must run. The vocabulary is a purity ASSERTION, not a list of
expensive functions — that is the whole reason it is config and not a heuristic.

**Widening the vocabulary is how the receiver bug was found**, and the method generalizes:
run the rule against a deliberately over-broad list, then curate the hits. With only the
one rl-local name the rule looked clean, because that name is a package-level function with
the network as an explicit argument. Every hit the broad list produced in `nlp` was the same
false positive — `b.Wq.Forward(ctx, xn)`, `b.Wk.Forward(ctx, xn)`, `b.Wv.Forward(ctx, xn)`
keying identically, because `calleeName` collapses a qualified call to its last segment. The
comparison key now carries the full callee expression. A vocabulary of one hides
receiver-shaped bugs by construction.

**How the len/cap exemption was found is worth more than the exemption.** Names reachable only
through `len` or `cap` are excluded from the mutation scan, because reading a size is the
normal reason a name appears between two identical calls. Without that, `New(F64, Shape{len(states), k})`
counted as a possible write to `states` and suppressed the finding. Replaying the detector
against the real pre-fix source did **not** catch this — that source happened to size its
tensor from a different slice than it fed the forward, so it passed for an incidental reason.
The synthetic positive test is what exposed it. Replay proves a rule finds the case it was
built from; it does not prove the rule finds the *shape*.

## PS6011 — an inner loop that walks a flat buffer along the wrong axis  *(scanner: static)*

The inner loop variable appears MULTIPLIED by a stride, so consecutive iterations jump a
whole row:

```go
for c := range dk { // outer: additive in the index
	ac := at[c]
	for r := range dv { // inner: SCALED by dk
		S[r*dk+c] *= ac // each step lands in a different cache line
	}
}
```

Every iteration touches its own cache line to consume one of its eight doubles, and the
whole traversal repeats once per outer index. The correct spelling puts the inner variable
in the additive position (`S[r*dk+c]` iterated over `c`), which is what makes the two
distinguishable from the AST alone — the check asks only which loop variable is being
scaled, never what the buffer's type is.

Two fixes, and which one wins is a measurement rather than a rule. **Interchanging** the
loops makes the access sequential and is bit-neutral when the body is a pure elementwise
update. **Blocking four adjacent OUTER indices** keeps register accumulators and reuses the
line that was fetched anyway — the right choice while the buffer is cache-resident, per
`PERF-ACCUM-RESIDENCY-001`, and the one to reach for when the body accumulates.

**Head-to-head, measured** (`R-01KYQTN083E71`): the two `RetentionChunkwise` output
accumulations were rewritten *both* ways independently — interchange (accumulate i-outer
into a `d_v` buffer, scale, then add the V term in a third pass) and 4-way blocking over
the output channel (four register accumulators carrying both terms in one pass). Blocked
beat interchanged by **1.62×** chunkwise and **1.68×** recurrent, consistently and with no
overlap between arms. Interchange fixes the access pattern; blocking fixes it *and* keeps
the intermediate in registers *and* writes the output row once instead of
read-modify-writing it twice. When the body accumulates, reach for blocking first.

**When two rewrites both claim bit-identity, test them against each other.** Two
independent claims of bit-identity with the same original are a claim of bit-identity
between the rewrites — which is testable, where each claim on its own is only prose.
Digesting the output (bitwise sum plus xor of every element) under both arms is enough; the
Retention arms agreed exactly, and the pinned digest now guards the next rewrite. The
package's own tests were tolerance-based and would not have caught a disagreement.

**Shipped:** NSA's `attendMask` P·V walked `vs[j*dm+off+d]` over `j` for each output
channel — **2.40×** when blocked four channels at a time. KDA's decay loop scaled a column
of `S` once per key channel per timestep; interchanging it carried the larger share of that
module's **1.75×**. Sinkhorn's transposed half is the same pathology in `[][]float64` form
(**2.65×**), where PS4006 sees it instead.

**Triage first: is the strided index revisited?** A stride only costs what the cache cannot
absorb, and the deciding question is not the buffer's size but the FOOTPRINT OF THE STRIDED
WALK ITSELF. The wins above all share a property — the strided index is the *reduction*
axis, so each line is touched once per output and never returned to. When instead the same
addresses recur on every outer iteration, those lines stay resident and there is nothing to
recover.

The triangular back-substitutions in `linalg` are the recorded null case (`R-01KYQT329EEAD`).
`out[j*cols+c]` strides by 4KB at n=512 over a 2MB buffer, which reads exactly like the NSA
case — but for a fixed `c` the pass touches the *same* n addresses on each of its n outer
iterations. That is n lines, 32KB of footprint at n=512, resident after the first pass.
Predicted 2–5×; measured 1.005× (LU), 1.001× (Lstsq), 0.994× (Inverse), and 1.041× at one
site and one size that vanished at the next size up. So: before booking a PS6011 candidate,
ask whether the outer loop re-reads the strided range. If it does, expect nothing.

No suppression is added for this. Deciding it needs the outer trip count weighed against the
index expression, and an unsound rule here would hide the 2.40× cases — an advisory false
positive costs one A/B, a wrong suppression costs the win permanently.

**Three false-positive classes are excluded by construction.** A transpose
(`out[j*r+i] = x[i*c+j]`) strides on one side whichever way it is iterated, so interchange
only moves the problem — suppressed by detecting the mirrored shape. A nest whose two loop
variables never reach the same index expression has no interchangeable axes at all. And any
**permutation copy** — one indexed write fed by one read, with or without a conversion or
accessor — has no reduction to interchange; its real fix is tiling.

That last exclusion exists because the mirrored-shape test is syntactic and loses sight of
the stride the moment it is hoisted: `row := i * b` outside the inner loop makes `src[row+j]`
look unstrided, which is how `nlp`'s **already-tiled** gguf transposes were being flagged.
Suppressing permutation copies cut the tree-wide count from 120 to **71**. The suppression is
kept honest by a test asserting the check still fires on a strided *accumulation* written in
the same assignment shape.

**Validated by replay**, per the discipline this file records twice already: the first draft
searched only the outer body's direct statements and missed its own motivating case, because
NSA's P·V loop sits inside an `if sum > 0`. Guarded inner loops are the norm here, not the
exception. The regression test for it fails if that discovery is narrowed back.

## PS6012 — a fused path that pins some products against FMA but not all  *(scanner: static)*

Go contracts `a*b + c` into a single `FMADD` on arm64 and generally does not on amd64. A
fused fast path that must reproduce a chain of separately-rounded backend ops therefore has to
round **every** product explicitly:

```go
// inc is NOT pinned — one rounding here, and the compiler fuses it into the subtract below
inc := g[i] * th
s[i] = float64(s[i]*et) - inc
```

The failure this catches is not forgetting the technique — it is applying it **incompletely**,
which looks correct and passes on amd64 CI. The discriminator is therefore internal
consistency: only functions that already contain a `float64(a*b)` are considered, because
those have declared that contraction matters here. A function that pins nothing is not making
the claim.

**Naming a subexpression does not pin it.** `inc` above is a local with a single use, so the
compiler inlines it and emits `fma(-g[i], th, ...)` — one rounding where the path it must
match does two.

**Shipped:** this is the defect that cost **three attempts** on the Titans fused path. The
symptom was maddening rather than obvious: the `t == 0` branch computes `s = -inc`, a negation
with nothing to fuse into, so it always matched, while every later step was off by one ulp —
which reads as a bug in the momentum branch. Two other products in the same function were
already pinned correctly. Fixing it took the path from unverifiable to bit-exact and shipped
**2.7×** with allocations down from 24,525 to 3,305.

**Two exclusions keep it quiet**, both about separating float math from integer offset math
without type information. Index and slice subscripts are excluded outright, and the flagged
product must have an **indexed operand** — real value arithmetic reads memory (`g[i]*th`),
while offset arithmetic is plain identifiers (`oy*wo + ox`), including when it is computed
into a helper call where a subscript-only exclusion cannot see it. Separately, only
`float64(a*b)` counts as a pinning signal: `float32(a*b)` is overwhelmingly a store rounding
on an F32 path, and counting it fired in every typed F32 branch in the tree. Together these
took the tree-wide count from 132 to **31**.

## PS6013 — a full sort whose only reader is a counted prefix  *(scanner: static)*

The order past position k is computed and thrown away:

```go
slices.SortFunc(idx, byScoreThenIndex)
for r := 0; r < k; r++ {
	drop[idx[r]] = true // membership, not order
}
```

When the consumer asks only **which** elements are the k smallest, a selection
(quickselect / nth_element) answers it in O(n) against the sort's O(n log n).

**Shipped:** WandaPrune sorted every output column in full to decide which half to drop —
2048 sorts of 2048 elements, roughly 46M comparisons per call. Replacing the sort with an
in-place quickselect measured **282ms → 55ms (5.1×)**, and 348ms → 55ms (6.3×) together with
the panel transpose that preceded it.

**Bit-safe only under two conditions**, and the message states both because they are what a
reviewer must check rather than assume. The comparator must be a **total order** — Wanda's is
score ascending with ties broken by input index, and indices are unique, so no two elements
compare equal and the k-smallest set is uniquely determined. And the consumer must read
**membership rather than position**, since a selection leaves the prefix in arbitrary order.
With ties, or with a consumer that reads `idx[0]` as "the smallest", the two disagree.

That same total order is why Lomuto partitioning cannot degrade here on a column of identical
scores: index tie-breaking means there are no duplicate keys. Median-of-three pivoting is
still required — score columns are `|w|·‖x‖` products and frequently near-sorted, exactly the
shape that takes a first-element pivot quadratic.

**Soundness rests on the prefix loop being the only reader.** If anything else reads the
sorted slice afterwards the full order is load-bearing, and the check stays silent; writes do
not count, since re-initializing the index slice for the next column is a write. A loop
bounded by `len(idx)` is also silent — nothing is discarded.

**Validated by replay**, and it needed two rounds. The first draft matched only direct sort
calls and missed its own motivating case, because Wanda's sort sits behind a local closure
capturing the slice; the second failed because `calleeName` collapses a qualified call to its
selector, so matching `"slices.SortFunc"` never fired. Both are covered by tests now.

**Related:** PS6001 is the narrower relative — a descending vocabulary sort feeding a
consumer that breaks on a threshold. It matches nothing anywhere in this tree; this counted
form is the one that occurs.

## PS4011 — a loop dispatching backend ops per iteration, with no fused path  *(scanner: static)*

**Its precision was audited, and the result changed the check.** Every one of its 110 hits was
classified: **zero** were the sequential recurrence the message described. 57 were transformer
layer stacks, 35 more were per-head, per-window or per-expert fan-outs, 12 were movement-only
prep loops. The genuine class exists — six loops in this tree, every one carrying explicit state
across a `range seq` — and all six were already suppressed by the fused-path guard, so none
appeared in the hit list at all.

The fix is one negative condition: **skip the loop when its trip count comes from a field.**
`range m.Blocks`, `range cfg.Heads`, `range m.Experts`, `for r := 0; r < m.MaxRecursion; r++`
are architecture counts on the order of tens; a sequence length arrives as a local or a
parameter. Counted before it was written: prunes 84 of 110, keeps 6 of 6 on the real class,
leaves the three existing fixtures unchanged. Layer stacks drop from 57 sites to 1.

Four richer predicates were counted and **rejected**:

| signal | why it fails |
|---|---|
| loop-carried state | fires on every layer stack too — the residual *is* carried state, so this is what the two shapes have in common |
| carried value feeds ≥2 dispatches | loses the canonical single-chain recurrence; recall 4/6 |
| elementwise vs matmul ops | recall **0/6** — 22 layer stacks show no visible matmul (theirs sit behind method calls) while all six real recurrences do |
| literal row-slice attribute | recall 0/6 |

An AST walker cannot tell `range m.Blocks` (slice field) from `range cfg.Heads` (int field), and
does not need to: both are architecture counts, both belong on the suppressed side. Known cost:
a recurrence written `for t := range m.Config.Ctx` would be missed. There are none today.


A sequential loop that issues several `backend.Execute` calls per step, in a function with no
typed fast path. Each dispatch materializes a tensor, and on a per-timestep or per-window loop
that dominates everything the arithmetic does.

**Read the trip count before believing the finding.** Most of this rule's matches are per-layer
(`for l, b := range m.Blocks`) or per-head loops — order 32 — where the overhead is real but
bounded. The wins came from loops whose trip count scales with the *input*: a per-timestep
recurrence, or a per-window attention. Titans' `NeuralMemory.Scan` was doing roughly **191
allocations per timestep**; Swin's windowed attention roughly **30k per batched forward**.

### Four fix shapes, in increasing order of risk

| Shape | What it removes | Measured |
|---|---|---|
| Fuse an elementwise op chain in place | one tensor per op | Swin scale+bias+mask, 30.5k → 24.5k allocs |
| Reuse operand buffers, refilled per iteration | slice/transpose dispatches | Swin q/k/v blocks, 24.5k → 11.0k |
| Place outputs directly into one buffer | concat dispatches | Swin heads, 11.0k → 10.1k, **time neutral** |
| Fuse the whole step, keeping matmuls on the backend | most dispatches | Titans, **2.7×**, 24.5k → 3.3k allocs |

The last row is the one to scope carefully. A *fully* fused path has to reproduce the bits of
every op it replaces, and `PERF-FUSED-PATH-CHAIN-001` exists because a twenty-op chain resisted
three attempts. `ADR-01KYQ9PHNPEFC` decided the rule: keep the matmuls on the backend, where
their rounding is already correct, and fuse only the slicing and the elementwise work.

### These paths are inference-only, for two unrelated reasons

Both guard on `ctx.Recorder == nil`, and it is worth not conflating them:

1. **A learnable parameter needs its op on the tape.** Fusing an `Add` that applies a trainable
   bias removes the edge that carries its gradient. Swin's first attempt did exactly this and
   `TestSwinGradcheck` reported `param 8: nil grad`.
2. **A recorder captures tensors by pointer.** Refilling a reused buffer across iterations
   leaves the graph holding whatever the last iteration wrote. This applies even when no
   parameter is involved.

The dispatch path must stay intact for the taped case, not be restructured around the buffers.

### Verify with a fused-vs-dispatch parity test

A tape context takes the op chain and a plain context takes the fused path, so running both and
comparing on raw `float64` bits *is* the comparison. Cover **both dtypes**: the F32 chain
computes each op in float64 and rounds the *store*, so a fused arm must round after every term
rather than once, and a scalar operand arrives as an F32 *tensor* — multiplying by the float64
value instead of `float32(v)` is a one-ulp divergence. Swin's F64 arm passed on the first
attempt while F32 was wrong twice.

**A measurement of exactly zero change is a bug signal, not a null result.** Swin's first fusion
measured identical allocations because the benchmark model is F32 and only the F64 arm had been
written — the fused branch was never entered. Probe which branch runs before concluding a change
does not help.
