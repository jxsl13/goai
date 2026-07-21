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
`make perfscan` / `go run ./internal/perfscan`); the code labels detectors A–L.
Sections P1/P2 below are the static detectors **I** and **J**; P3/P4 are
profile/benchmark heuristics (no static detector); K and L (below) are the newest
static ones — L generalizes K to transcendentals hidden one call deep in a helper.

## Suppressing findings (class-granular, staticcheck-style)

Silencing one class never hides another — accepting a site for class X still
surfaces a new, unrelated class Y there. Name a class by its **letter** (A–K) or
its **category** (copy-paste from a report line; `-list` shows both):

```go
//perfscan:ignore K reason        // silence ONLY class K on the next (or same) line
//perfscan:ignore K,I reason      // several classes at once
//perfscan:ignore                 // bare: silence ALL classes at that site
```

Repo-wide, pass `-exclude=K,per-element-closure`. Example: the f64
exp/log/tanh/sigmoid/gelu kernels are flagged by K but are exact-locked
(`TestCPUCrossReferenceExact`) — mark each `//perfscan:ignore K exact-locked ref`
so the one genuinely-open member of the family still reports.

---

## P1 (detector I) — per-element closure over a contiguous buffer  *(scanner: static)*

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

## P2 (detector J) — closure-comparator sort on a large keyed slice  *(scanner: static)*

**Smell.** `slices.SortFunc` / `sort.Slice` / `sort.Sort` sorting a large slice
by a float/int key looked up through the comparator. The per-comparison indirect
call dominates (profiled at ~50% of the op), and it's O(n log n) where the
consumer often needs only a threshold.

**Fix.** If the key is **monotonic** (for non-negative `float64`,
`math.Float64bits` is monotonic in the value; invert the bits for descending),
replace with an 8-pass **LSD radix sort** on the key bits — closure-free, O(n).
The value order is identical to the comparison sort for distinct keys. Gate on
`n >= ~2048` so small slices keep the lower-constant comparison sort.

**Wins (~50k-vocab flat distribution):**
- `nlp` top-p nucleus (`sortIdxDescByProb`) — **2.25×** (7.38→3.27 ms)
- `nlp` locally-typical (`sortIdxAscByScore`) — **1.89×** (7.14→3.77 ms)

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

## K — scalar transcendental where a vectorized sibling exists  *(scanner: static)*

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

## L — a loop calls a local helper that WRAPS a transcendental  *(scanner: static)*

**Smell.** A hot per-element loop calls a package-local elementwise helper
(`softplus`, `mish`, `swish`, `silu`, a `gaussianQuantile`, …) whose body hides a
scalar libm transcendental one call deep. Class K only sees a **direct** `math.X`
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
attention-score shape) had. Class K/L never see it: the transcendental lives in
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
