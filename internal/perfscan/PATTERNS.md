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

perfscan detects the problems **independent of any one repo**. Ten checks are pure
language/stdlib shapes and run on any Go module with no configuration (PS2002–PS2005,
PS3001–PS3003, PS4001, PS4003, PS5001). The four **domain** checks — `PS1001`
per-element-dispatch, `PS1002` per-element-closure, `PS2001` alloc-in-loop, `PS4002`
scalar-transcendental-vectorizable — key on a project's own vocabulary (its element
accessors, allocators, fast-path helpers and vectorized kernels), which lives in a
**JSON config, not the engine**. With no config those four stay silent. Supply one
with `-config file.json` or a discovered `perfscan.json` / `.perfscan.json`:

```jsonc
{
  "elementAccessors":       ["AtF64", "SetF64"],     // PS1001/PS1002
  "fastPathHelpers":        ["flatF64", "flatF32"],   // PS1001 — presence silences a fallback loop
  "elementCountMethods":    ["Numel"],               // PS1001 — a loop bound over this reads as per-element
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
indirection/reflection, `PS4xxx` vectorization, `PS5xxx` arithmetic. `perfscan
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
functions (no receiver to hang the reused buffer on) and on `make()` outside any loop.

## PS5001 — divide by a loop-invariant scalar  *(scanner: static)*

A `/` (or `/=`) by a loop-invariant scalar on every iteration of an element-wise loop.
Hoisting `inv := 1/D` once and multiplying is **1.2–1.5×** when the divide is the
loop's standalone cost — a float divide is ~20–40 cycles versus ~4 for a multiply.

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
