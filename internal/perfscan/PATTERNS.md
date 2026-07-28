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
alloc-in-loop, `PS4002` scalar-transcendental-vectorizable, `PS6001` unverified-dual-path
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

**Shipped, three times, all the same shape:** the GBM presort (**1.05×**, and **1.10×**
cumulative once the flat key made an LSD radix pass practical) and the ball-tree median
split (**1.088×** on KNN fit — the half that loses to sklearn). The CART builder had
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

## PS6001 — a dual-path kernel whose bit-identity claim is unverified  *(scanner: static)*
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
**3.19×** win, and PS6001 reported nothing. A verification rule that goes quiet on a
real dual path is worse than no rule, because the silence reads as "nothing to prove".
Detected by: a `bool` result, **two or more** comma-ok assertions to *slices of numeric
types*, and a `return false`.

The numeric-slice requirement is not decoration. Without it the widening fired on 13
functions inside perfscan itself — an AST visitor is wall-to-wall `x, ok := n.(*ast.Foo)`
followed by `return false`, structurally identical and semantically unrelated. Asserting
`[]float32` is the signal; asserting `*ast.Ident` is not. Tree-wide the tightened form
adds exactly one finding (56 → 57) instead of fourteen.

