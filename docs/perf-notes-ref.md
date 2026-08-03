# Perf notes: `backend/ref` devirtualization (§T646 follow-up)

Last-percentile optimization pass over the reference backend. Constraint: ref is
the numerical truth (§V9) — every change must be **bit-identical**, so only data
access and loop mechanics changed, never the arithmetic sequence. All wins were
measured A/B with `go test -bench -benchmem -count=3` (Apple M2 Pro, arm64,
CGO_ENABLED=0); the benchmarks live in `backend/ref/perf_regress_test.go` as
regression guards.

## Techniques (researched + verified here)

### 1. Devirtualized flat traversal (`f64Data` / `outF64`, `backend/ref/devirt.go`)

The generic ref loop pays, per element: a `tensor.Unravel` **heap alloc**, a
variadic multi-index dot product, and an `AtF64`/`SetF64` dtype dispatch.
`f64Data` exposes a tensor once as a flat row-major `[]float64` (zero-copy for
contiguous F64; an **exact** `float32→float64` widened copy for F32 — widening
is lossless, so arithmetic on the widened slice is bit-identical to widening per
element). `outF64` gives a `[]float64` result buffer whose flush narrows once
per stored value — exactly what per-element `SetF64` did. Used in the
compute-bound kernels (GEMM, attention, conv, norms); O(n) elementwise kernels
keep per-dtype typed loops instead to avoid the extra copy.

Measured effect: 2–33× per kernel (table in the task report; e.g. MatMul 128³
16.1 ms → 1.66 ms, RMSNorm 2.6 ms → 0.26 ms, Concat 65 546 → 10 allocs/op).

### 2. Bounds-check elimination by row-slice hoisting + range loops

Hot inner loops read via hoisted subslices (`arow := as[i*k : i*k+k]`) and
`for j, v := range arow` — range-loop accesses are bounds-check free, and
re-slicing to a known length lets the compiler prove the paired index (`brow[j]`
after `brow := bt[j*k : j*k+k]`) in range. This is the standard Go BCE guidance
(SSA-based BCE since Go 1.7): hoist checks out of loops via subslicing, prefer
range iteration.
Sources: [Go 101 BCE](https://go101.org/optimizations/5-bce.html),
[Bounds checking in Go](https://unskilled.blog/posts/bounds-checking-in-go/).

### 3. Generic `refFloat` cores for read-modify-write gradients

Backward kernels (`mhaBackwardCore`, `conv2dBackwardCore`,
`retentionBackwardCore`) accumulate into their outputs. For F32 outputs the
generic path narrows on **every** update (`SetF64(AtF64+δ)` = widen/add/narrow),
which is not the same as accumulating in f64 and narrowing once — so the typed
core is generic over `T refFloat` (`float32 | float64`) and stores
`T(float64(out[i]) + δ)`. For `T=float64` the conversions are identities; for
`T=float32` they reproduce the exact narrowing chain. Go's gcshape stenciling
instantiates `float32` and `float64` **separately** (different sizes ⇒ different
gcshapes ⇒ no shared dictionary-based code for the arithmetic), so both
instantiations compile to direct scalar code.
Sources: [gcshape stenciling design](https://go.googlesource.com/proposal/+/master/design/generics-implementation-gcshape.md),
[Generics can make your Go code slower](https://planetscale.com/blog/generics-can-make-your-go-code-slower)
(pointer-shape dictionaries are the slow case; distinct scalar shapes are not).
Measured: MHABackward 5.86 ms → 0.45 ms (12.9×), Conv2DBackward 15.3 ms → 1.19 ms.

### 4. Incremental odometer offsets instead of per-element Unravel

Reductions/argmax/broadcast/cumsum need an output/input offset per element that
the generic path rebuilt from an Unravel'd multi-index (O(nd) + alloc per
element). An odometer walk (increment last axis, carry) maintains the same
offset **incrementally** with per-axis effective strides (0 for reduced or
broadcast axes). Element order stays ascending row-major, so every accumulator
sees the same combine sequence — bit-identical. Measured: Sum(all) 946 → 236 µs
with 65 544 → 9 allocs; Sum(axis) 5.5×; Broadcast 3.9×.

### 5. Invariant hoisting that reuses *identical* operations

Only hoists that recompute the *same operation on the same operands* are legal
(same value, bit-identical): RoPE's `cos/sin(pos·θᵢ)` hoisted out of the heads
loop (the generic loop recomputes them per head), retention's `γ^(n−m)` powers
precomputed once per distance, attention's `row[j]/sum` computed once per j
instead of per (d,j). Loop *interchange* is only used where each output element
still accumulates its own terms in the same order (attention P·V restructured
j-outer/d-inner; GEMM B pre-transposed so the p-dot is contiguous — each C[i,j]
still sums over ascending p). Measured: RoPE 1.63 ms → 0.21 ms (7.9×),
Retention 1.57 ms → 0.14 ms (11×).

### 6. Same-dtype block `copy()` for pure data movement

Concat/slice/reshape/transpose/embed move values without arithmetic; the
generic path still did two Unravels + an f64 round-trip per element. A
same-dtype round-trip through f64 is exact, so typed `copy()` of contiguous
blocks is **byte**-identical. Measured: Concat 1.89 ms → 57 µs (33×), Slice
853 → 25 µs (34×, 32 776 → 8 allocs), Reshape 14.6×, Embed 13×.

## What was deliberately NOT changed

- Accumulation order anywhere (§V10 f64 accumulation preserved; per-output-element
  term order preserved under loop interchange).
- The O(batch) scalar loss kernels (dpo/ipo/kto/cpo/ppo/simpo/grpo/gspo): rank-1
  `AtF64(i)` scans over tiny tensors, exp/log-dominated — no measurable win.
- `qr/svd/eigh/cholesky/solvespd/logdet`: their O(n³) cores already run on dense
  `[][]float64` working copies; input conversion is O(n²) `AtF64`. Flattening the
  working copies is a bigger rewrite with modest ROI — left for a future pass.
- `einsum` delegates to `backend.EinsumContract` (outside `backend/ref/`, out of
  scope for this pass).

## Generic fallbacks

Every devirtualized kernel keeps the original loop verbatim as the fallback for
exotic dtypes, so future dtypes (int/bf16) keep working through `AtF64/SetF64`.

## The masked-attention backward is the largest unoptimized op left (T1175, measured)

`OpMHAMaskedBackward` is registered in **ref only** — `PS3062` lists it among the
25 ops with no cpu kernel — so every caller on the default backend runs the
reference implementation. It is not a small op:

    BenchmarkMHAMaskedBackward          5.06 ms
    BenchmarkMHAMaskedBackward_256h8   23.2  ms
    BenchmarkMHAMaskedBackward_512h8   89.8  ms

97.35% of that profile is one closure, `mhaMaskedBackwardKernel.func6`, and its
line-level breakdown against 5.55 s of flat samples is:

    1.11s  dqrow[d] += ds*krow[d]  /  dkrow[d] += ds*qi[d]   (dQ/dK projection)
    0.98s  maskBuf[mbBase+j] = dscore
    0.83s  d += gi[c]*vrow[c]  /  dvrow[c] += rj*gi[c]        (dV accumulation)
    0.34s  sc += qi[d]*krow[d]                                (score dot)

All three arithmetic loops are the shapes with the strongest measured record in
this tree: a score dot on ONE accumulator over a query row re-streamed once per
key, a dV accumulation whose `gi` does not vary with the key, and a dQ/dK loop
whose `dqrow` and `qi` are both shared across keys. The identical transform on
the **cpu** masked backward measured −45.8% (T1153), and on the NSA masked
attention −49.7% (T1173).

`PS6010` reports two of these sites as of T1174 — they were excluded before that
round by the non-foldable-call filter, because the loop's mask guard calls
`math.IsInf`.

### What the conversion has to get right

- **The loop appears TWICE**, once per mask variant (shared `[sq,sk]` and
  per-head `[heads,sq,sk]`). A patch anchored on the loop body matches both, and
  a one-site edit silently measures as no change — which is how this note came
  to be written instead of the conversion.
- **There is no live oracle inside ref.** The generic `AtF64` arm is dead for
  the registered dtypes: `f64Data` succeeds for both F32 and F64, so nothing
  reaches it. The gate must therefore be a digest frozen on the pre-change code
  (PERF-TOLERANCE-ORACLE-001), not a comparison between the two arms — and the
  existing tests do not cover it either, since the parallel-equivalence test
  compares the kernel with ITSELF and the F32 test is a 1e-5 tolerance.
- **Adding a cpu kernel instead** keeps ref as the oracle for a future
  optimized path, which is what PS3062 recommends and what OpCholesky did in
  T1127 (ref 21.1 ms → cpu 7.0 ms, 3.0×, bit-identical). It costs a 361-line
  port; jamming ref in place costs about 60 lines across the two sites. The
  choice is between a preserved oracle and the smaller diff, and it should be
  made deliberately rather than by whichever is quicker to type.

## `math.Min`/`math.Max` are calls; the builtins are instructions (T1180)

`math.Min` and the `min` builtin are not the same function, and the gap between
them is expensive. On arm64 the library function compiles to

```
TEXT MathMin(SB), ABIInternal, $48-16
CALL math.archMin(SB)
CALL runtime.morestack_noctxt(SB)
```

while the builtin compiles to a single `FMIND` in a leaf with no frame at all.
Inside an element loop that is the difference between one instruction and a
non-inlinable call, and there are 86 such call sites in the tree
(`perfscan -checks PS3082`).

### Why the substitution is not a rename

`math.Max` documents `+Inf` as beating `NaN` and `math.Min` documents `-Inf` as
beating `NaN`; the builtins propagate `NaN` unconditionally, as the language
spec requires. Over every ordered pair drawn from
`{NaN, ±Inf, ±0, ±1, ±MaxFloat64, ±SmallestNonzero}` the two formulations
disagree on exactly four: `Max(NaN, +Inf)`, `Max(+Inf, NaN)`, `Min(NaN, -Inf)`
and `Min(-Inf, NaN)`.

That divergence is reachable in this code, not theoretical. A log-probability of
`-Inf` — an ordinary `log 0` — makes the PPO ratio exactly `+0`, and `+0` times
an infinite advantage is the `NaN` that pairs with the `-Inf` the other
surrogate branch produces. A raw substitution turns a finite loss into `NaN`
there.

### The recovery

The two can only ever disagree **on a NaN result**: whenever they differ, the
builtin is the one returning `NaN`. So take the instruction and consult `math`
only when the instruction says `NaN`. That is `internal/fmath`, and the branch
predicts perfectly on ordinary data.

A `Clamp(v, lo, hi)` wrapper was written and removed: composing the two bodies
puts it over the inliner's cost budget, which turns it into exactly the
per-element call the package exists to avoid.

### Measured

| Benchmark | before | after | delta |
|---|---|---|---|
| `BenchmarkPPOClip_4096` (ref) | 100.5 µs | 41.6 µs | **−58.7%** |
| `BenchmarkGRPO_4096` (ref) | 126.7 µs | 79.8 µs | **−37.0%** |
| `BenchmarkGSPO` (ref) | 21.6 µs | 19.8 µs | −8.4% |

GSPO is the small one because its clamp runs once per sequence against a
256-token inner sum — the site is real and ranked out by execution count.

### Rejected, with numbers

- **Converting an existing comparison chain.** The PPO VJP already used the
  chain PS3077 recommends. Rewriting it onto `fmath` measured 51.8 → 58.6 µs,
  **+13%**, and was reverted. `fmath` replaces *calls*; against branchless code
  it only adds a guard. Rank a site by whether the call is still there.
- **The GSPO VJP clamp.** Converted, measured flat (254.4 → 250.5 µs, inside the
  spread), reverted. One clamp per sequence against a 256-token inner loop.
- **Reslicing the operands so the bounds checks fold.** 63.6 → 64.7 µs on the
  PPO loop: no effect, slightly negative.
- **The naive comparison chain for the outer `math.Min(surr1, surr2)`.** Worth
  another 24% and semantically wrong: `math.Min(+0, -0)` is `-0`, and a `<`
  chain keeps `+0`.

### What the gate has to get right

A kernel that reduces to a scalar **cannot see this divergence**. The first
version of `rl_minmax_oracle_test.go` swept the whole hostile grid in one batch
and was green under the raw-builtin rewrite it exists to reject: one `NaN`
anywhere poisons the sum, and both formulations then agree on `NaN`. The gate
plants **one hostile triple per kernel call**.

Two further traps in the same file, both found by the control run rather than by
reasoning:

- The oracle must mirror the kernel's accumulator, including its initial `+0`.
  `0 + -0` is `+0`, so an oracle that negates the term directly reports a sign
  flip the kernel never produces.
- `GRPOAttrs.WithDefaults` rewrites a zero `Beta` to `0.04`, so a case passing
  `0` tests `0.04` against an oracle using `0` and fails on the KL term.

NaN *payload* is deliberately not compared: the kernel's `NaN` comes from a
runtime helper and the oracle's from inlined arithmetic, and they differ. Payload
is not part of the contract; `NaN`-versus-not-`NaN` and the sign of a zero are.
