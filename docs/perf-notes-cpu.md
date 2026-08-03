# backend/cpu perf notes (last-percentile sweep, 2026-07)

Techniques researched and applied during the §V22-measured optimization pass of
`backend/cpu/`, with the A/B numbers observed on the darwin/arm64 (M-series,
GOMAXPROCS=12) host, `CGO_ENABLED=0`. Every change kept the documented parity
class (bit-identical where claimed, ulps otherwise); no tolerance was relaxed.

## 1. Bounds-check elimination via hoisted row subslices + `range`

The single biggest lever of the pass. The typed attention/norm/softmax cores
indexed 2-D data as `x[base+d]` inside `for d := 0; d < dk; d++` loops: every
access pays a bounds check plus full address arithmetic, and the generic
(`[T float32|float64]`) instantiations made it worse. Rewriting each inner loop
to hoist a three-index subslice per row and drive the loop with `range`

```go
qr := q[qBase : qBase+dk : qBase+dk]
kr := k[kBase : kBase+dk : kBase+dk]
for d, qv := range qr {
	s += float64(qv) * float64(kr[d])
}
```

lets the compiler prove the range index in-bounds (range-loop accesses on the
ranged slice are check-free) and shrinks the addressing to a single induction
variable. The three-index form also pins `cap`, which helps the prove pass and
prevents accidental aliasing growth.

Measured (median of interleaved A/B, `-count≥4`):

| kernel | shape | before | after |
|---|---|---|---|
| `mhaBwd` | 512×512, 8h causal f32 | 53.0 ms | 21.6 ms (−59%) |
| `mhaFwd` (incl. §2) | same | 17.4 ms | 14.2 ms (−18%) |
| `flashAttnTyped` (incl. §2) | 512×512 8h blk64 | 27.4 ms | 13.4 ms (−51%) |
| `retentionBwd` | 512×64 | 6.24 ms | 5.39 ms (−14%) |
| `softmaxTyped` | 2048² f32 | 6.3 ms | 5.6 ms (−8%) |

Counter-example: the same rewrite on the norm *forwards* (rmsNormFwd /
layerNormFwd) measured flat — those loops are memory-bandwidth bound, so the
saved checks hide behind loads. Reverted per §C3. BCE only pays where the loop
is issue-limited, not bandwidth-limited.

Sources: [Go 101 — BCE](https://go101.org/optimizations/5-bce.html),
[Bounds checking in Go](https://unskilled.blog/posts/bounds-checking-in-go/),
[Ardan Labs — BCE in Go](https://www.ardanlabs.com/blog/2018/04/bounds-check-elimination-in-go.html).

## 2. Loop-interchange to contiguous streams, division hoisting

`mhaFwd`/`flashAttnTyped`/`retentionFwd` accumulated the output with `d` outer /
`j` (keys) inner, reading `v[j*kvDM+kvOff+d]` — a stride-`kvDM` walk per output
element, and `mhaFwd` recomputed `row[j]/sum` for every `d` (sk·dk divisions).
Swapping to `j` outer / `d` inner with a small f64 accumulator buffer keeps each
output element's ascending-`j` summation order (bit-identical values — division
is deterministic, computing `row[j]/sum` once yields the same float) while
making `v` reads contiguous and collapsing the divisions to one per key.
Measured alone on `mhaFwd`: 21.2 → 16.5 ms (−22%); `retentionFwd` 3.2 → 1.88 ms
(−40%); `flashAttn` 27.4 → 22.8 ms (−17%) before the §1 rewrite stacked on top.

The same reassociation-free trick tiles the norm backwards' dγ/dβ column
reductions: keep each column's ascending-row order but walk the matrix row-major
in per-worker column bands (`layerNormBwd` 818 → 623 µs, `rmsNormBwd` 809 →
540 µs on 512×1024 f32).

## 3. FP contraction (FMA) is context-dependent — pin it at parity edges

The Go spec allows the compiler to fuse `x*y + z` into one rounded FMA "possibly
across statements", and gc's choice is context-dependent: during this pass, the
norm-backward dγ accumulation flipped between fused and unfused when *unrelated
code elsewhere in the package changed*, moving the result a few ulps and
tripping the 1e-15-relative parity test against `backend/ref` (whose own
contraction also shifts as it is edited). Two spec-guaranteed pins exist:

- `acc += float64(u * x)` — an explicit conversion **forces rounding**, i.e.
  plain mul-then-add, never fused;
- `acc = math.FMA(u, x, acc)` — always fused (single rounding; compiles to a
  native FMADD on arm64, no call).

Both are deterministic regardless of what the optimizer does around them.
`norm.go` now fences the tiled dγ/dβ accumulations with the conversion form,
matching ref's devirtualized core (which rounds `g·x̂` through a scratch slice
before accumulating). Cost: not measurable vs the unfenced loop on the norm
benches. Lesson for this package: any reduction that must satisfy an
ulps-tolerance against another compilation unit should pin its contraction
explicitly rather than trust the default.

Sources: [go#17895 — spec: allow FMA](https://github.com/golang/go/issues/17895)
(explicit casts force rounding; parentheses do not),
[go#25819 — math: add guaranteed FMA](https://github.com/golang/go/issues/25819),
[FMA & FP consistency](https://siboehm.com/articles/23/Inlining-FMA-FP-consistency).

## 4. Fork/join barriers dominate medium ops — fuse row-parallel passes

Profiling `conv2d` (581 µs/op) showed ~75% of samples in
`pthread_cond_wait`/`_signal`/`findRunnable`: the kernel ran im2col, GEMM and
the output scatter as three sequential `parallelWork` barriers, and the pool
workers parked/rewoke between each. All three stages operate on the same
disjoint row bands, so they fuse into ONE pass per band (fill rows → GEMM band →
scatter band): identical per-row operations, bit-identical output, two fewer
fork/joins, and the band's im2col rows are still cache-hot for the GEMM.
Measured: conv2d fwd 588 → 480 µs (−18%); conv2d bwd (fill+dXcols fused, and
col2im+dX-store fused per image) 1376 → 1299 µs (−6%).

## 5. Kill transposes: transposed-A GEMM band

conv2d-backward materialized `colsᵀ` (k×rows, all strided writes — 12% of the
kernel's flat profile) just to feed `dW = colsᵀ·dO` to the row-major band GEMM.
`gemmATF64Band` reads A as stored `[K,M]` (`A[p*m+i]`) with the same 4-row
register blocking and the same ascending-p per-element accumulation —
bit-identical to transpose-then-multiply, zero transpose traffic, and the
4-element `A[p*m+i:i+4]` reads are contiguous. Part of the 1557 → 1299 µs
(−17% total) conv2d-backward win.

## 6. Devirtualize hot unary kernels — but only where the call was the cost

`unOp(f)` calls its `func(float64) float64` per element (indirect call).
Direct-call kernels measured: Exp −10%, Tanh −5%, Log −3% (64K f64). A sigmoid
twin was built, measured **flat**, and reverted — sigmoid's cost is the
`math.Exp` + divide, not the dispatch. Devirtualization pays only when the
callee is cheap relative to an indirect call.

## 7. Misc measured wins

- `pool2dMax`: compare natively in `T` instead of converting every tap to f64
  (f32→f64 is exact and monotonic → same winner, same stored bits; NaN
  never wins under strict `>` in both forms). With the `isMax` branch hoisted
  out of the pixel loop and window rows subsliced: MaxPool −3%, AvgPool −8%
  (8×16×64×64, k2 s2 f32).
- `addBiasKernel`: each row is a same-length vector add → route through
  `internal/simd.AddF64/F32` (BCE'd tight loop here, archsimd AVX on
  amd64+GOEXPERIMENT=simd for free). −4–5% on 512×1024 f32, bit-identical.
- conv2d-backward dBias: one row-major pass with `f` accumulators instead of
  `f` stride-`f` column sweeps (ascending-r order per filter preserved →
  bit-identical). ~1% of the backward kernel — marginal but strictly less
  memory traffic on every shape.

## Not touched, deliberately

- `gemm.go`/`gemm_nosimd.go` band kernels: BLIS-style blocking was already
  built, measured and discarded on this host (§T74/§B28 note in gemm.go);
  the remaining headroom is wider FMA SIMD, host-blocked on arm64 (§T11b).
- `binOp`/`broadcastContig`: elementwise binaries are memory-bound through the
  simd primitives already; the Add/Mul benches are dominated by tensor
  allocation, not the loop.
- `parThreshold`/pool dispatch internals: retuning the crossover or chunk
  count reproduced current behavior at the measured break-even; the fused-pass
  work (§4) attacks the same overhead with better returns.

## Where the allocation actually goes (T1168, measured)

A byte-per-op sweep of the largest cells puts almost all of it in ONE place,
and it is not a leak, a missing pool or a churn pattern — it is the op API.

    BenchmarkMTAForward_ch16   283 ms   500 MB/op    4942 allocs
    BenchmarkCoPE_512x256_h4    25 ms   179 MB/op     808 allocs

Allocation profiles of both:

- MTA: **85.6% is `tensor.heapAllocator.allocF32` reached through
  `tensor.NewOn`** — the output tensor every kernel constructs for its result.
- CoPE: **99.6% is `allocF64`, 78.5% of it under `backend.Execute`** — the
  same thing, with almost nothing else in the profile.

At ≈101 KB per allocation on MTA and ≈221 KB on CoPE, these are whole
intermediate tensors, one per op in the graph.

### Three things this is NOT, each ruled out by measurement

- **Not GC clearing the scratch pool.** `sync.Pool` is emptied every GC cycle,
  which would make a pooled buffer allocate on nearly every get under this
  much pressure. Raising GOGC from 100 to 1600 moved MTA only 498 → 461 MB/op
  (−7.5%) and did not improve the clock, so at most a fourteenth of the
  allocation is GC-induced pool loss.
- **Not a leaking or thrashing scratch pool.** Instrumented over an
  MHA/Conv/MatMul benchmark set, `getF64` recorded 105 gets against 105 puts —
  balanced, so nothing leaks — with 12 misses of which only 2 were a buffer
  being GROWN; the other 10 were first-time creation per P. An 88% hit rate
  with balanced puts is a pool working as designed, and its bytes in a profile
  are the one-time fill of the per-worker working set.
- **Not per-job gather churn.** That pattern is real and was fixed where it
  occurs (T1167, forest gather, −38.4% bytes), but it does not appear here.

### Why it stays deferred

Kernels RETURN their output tensor to the caller, so recycling one needs
ownership and lifetime semantics the op API does not have: nothing tells a
kernel whether its result is a graph temporary that dies at the next op or a
value the caller keeps. An arena with explicit release, or a tensor whose
buffer is checked back in by the executor once its consumers have run, is the
shape of the fix; a pool cannot be bolted on underneath the current contract.

Recorded so the next allocation sweep does not re-derive it: on this tree the
allocation axis for `nn` and `backend/cpu` ends at the op API, and the pooling
opportunities BELOW it are already taken.

## The WKV recurrence: `math.Max` in the innermost position (T1181)

WKV's numerically-stable scan takes a running maximum twice per token per
channel — the innermost position there is. Moving those onto `internal/fmath`
(see `perf-notes-ref.md` for why the builtins are not a drop-in) measured:

| Benchmark | before | after | delta |
|---|---|---|---|
| `nn.BenchmarkWKV` | 17.71 ms | 15.28 ms | **−13.7%** |
| `cpu.BenchmarkWKVF32_512x1024` | 3.273 ms | 2.688 ms | **−17.9%** |
| `cpu.BenchmarkWKV_512x1024` (F64) | 3.275 ms | 3.211 ms | −2.0% |

The −13.7% lands almost exactly on `math.archMax`'s 13.1% share of the `nn.WKV`
profile.

### The F64 arm was flat because the call is in another package (corrected T1182)

T1181 recorded this as "the F64 path dispatches into SIMD assembly, so there is
nothing left to convert." That explanation was wrong. `simd.WKVScanRangeF64` is
**portable Go** outside the AVX build — `internal/simd/wkv_scalar.go` — and it
carries two `math.Max` calls per token per channel. The F64 arm was flat because
the sweep never opened that file, not because the work had left Go.

Converting it (T1182) moved the arm the earlier round could not:

| Benchmark | before | after | delta |
|---|---|---|---|
| `cpu.BenchmarkWKV_512x1024` (F64) | 3.234 ms | 2.863 ms | **−11.5%** |
| `cpu.BenchmarkWKVF32Ref_512x1024` | 3.190 ms | 2.632 ms | **−17.5%** |
| `autograd.BenchmarkWKVBackward_256x1024` | 3.039 ms | 2.750 ms | **−9.5%** |

The generalizable form: a kernel file's own call sites are not the kernel's call
sites. Follow the dispatch into helper packages before concluding an arm has
nothing left — and confirm the routing rather than assuming it. Mutating the max
in `internal/simd/wkv_scalar.go` reddens exactly the two CPU-F64 digests below
and neither Ref case, which is what proves where that path lands.

### Rejected: the SVM solver

`classic/svm.go` has eight of these calls in its working-set loops. Converted,
`BenchmarkSVCFit/n4000_rbf` measured 6.11 → 6.27 ms — slightly **worse** — and
`SVCPredict` was flat at 600 µs. Its profile is dominated by
`pthread_cond_wait`/`signal`: the fit is parallel and the scalar loop is not
what it is waiting on. Reverted.

### Unmeasured, and kept anyway

`nlp/rwkv_decode.go` and `nn/rwkv_block.go` carry the same recurrence in
single-token decode form. There is no decode benchmark in the tree, so these two
are converted on the strength of the measured twin rather than on their own
number. The arithmetic is identical and `fmath` is bit-identical to `math`, so
the only open question was speed, which the twin answers.

### What the gate proves, and what it does not

`nn/wkv_minmax_bitidentity_test.go` freezes five digests, two of them with
`+Inf`, `-Inf` and `NaN` planted in the same channel. Those hostile cases do
**not** distinguish `math.Max` from the raw builtin here: WKV carries the
running maximum forward, so the one pairing the two disagree on turns the
channel's state to `NaN` either way within the same step. A raw-builtin mutation
of all six sites leaves every digest unchanged.

The equivalence therefore rests on `internal/fmath`'s own exhaustive pair test.
These digests catch a **mis-edit** — substituting `Min` for `Max` at one site
reddens three of the five cases — and the infinities keep the branch-skipping
equality tests around the max exercised rather than only its ordinary path.

## MoBA's block-selection set: a map in the innermost loop (T1185)

Mixture-of-Block-Attention picks the top-K past blocks per query and attends only
to those. The membership test lived in a `map[int]bool` **built fresh for every
query and probed once per key**, which put `runtime.mapaccess1_fast64` at 5.4% of
the profile on top of an allocation per query. The gate slice was rebuilt per
query too.

The keys are block indices, dense in `[0, nBlocks)`. A slice is the natural
container, and a **generation counter** makes the per-query reset free: bump the
counter, write it at the selected indices, and test equality instead of
membership — no clear proportional to `nBlocks`.

| Measure | before | after | delta |
|---|---|---|---|
| `BenchmarkMoBAAttention` | 10.96 ms | 9.94 ms | **−9.3%** |
| bytes/op | 2.967 MB | 2.464 MB | **−16.9%** |
| allocs/op | 20533 | 10301 | **−49.8%** |

### The `len` trap

The selection loop was bounded on `len(selected)`, which a slice does not have.
An explicit counter is only equivalent when the inserted keys are **distinct** —
here they are, because the gate loop visits each past block exactly once, so
every insert is new and counting inserts reproduces the map's length. That has to
be read off the key source, not assumed.

### The gate is a digest, not a tolerance

A membership set that differs by one element changes *which* keys get scored, and
every resulting number stays entirely plausible. Four shapes are frozen: a
sequence that is not a multiple of the block size (so the last block is short),
a `topK` above the block count (every past block selected), a block-aligned
shape, and `topK=1` (only the current block ever in the set).

### Not converted

The generic `AtF64` fallback arm carries the same map. It is the slow path taken
only by dtypes the flat-`float64` fast path cannot expose, and `PS3083` still
reports it — correctly.

### The P·V accumulation, jammed (T1186)

The same kernel's output row does not vary with the attended key, so the
accumulation loaded and stored the whole row once per key for a single
multiply-add each. Eight keys per pass hold `orow[d]` in a register across eight
of them.

| Benchmark | before | after | delta |
|---|---|---|---|
| `BenchmarkMoBAAttention` | 9.92 ms | 7.49 ms | **-24.5%** |

Widths 6 and 8 tie (7.45 vs 7.44 ms); width 2 leaves half the win (8.35). Every
width from 2 to 8 leaves the digests unchanged, which is what says the
association is right — the accumulator is an explicit local, not a compound
assignment over the sum of eight products.

**The obvious alternative is both slower and wrong.** Most attended keys sit
outside the top-K blocks, so their score is `-Inf`, their softmax weight is
exactly `+0`, and the whole inner loop adds nothing. Skipping them with a
per-key branch measured **8.46 ms against the jam's 7.44** — worse. It would not
be equivalent either: `0` times an infinite or NaN value is `NaN`, not zero, so a
`v` holding a non-finite entry would get a different answer. No digest over
ordinary data can see that, which is the reason to state it rather than to test
for it.

Still reported by `PS3075`: the per-block key-sum at `moba.go:84`. It is
`O(seq·dk)` per head against this loop's `O(seq^2·dk)` and was not measured.

## RetNet's recurrent decode, and a limit on "bit-identical" (T1189)

`RetentionRecurrent` carries a `[dk,dv]` state across steps. Two loops over the
key dimension dominate it: the state update `S[i,j] = gamma*S[i,j] + k_i*v_j`
(35% of the profile) and the output dot `out[j] += q_i*S[i,j]` (40%).

Only the second one shipped.

| Benchmark | before | after | delta |
|---|---|---|---|
| `BenchmarkRetentionRecurrent_256x256` | 21.52 ms | 16.51 ms | **-23.3%** |
| `BenchmarkRetentionRecurrent_512x128` | 14.23 ms | 11.22 ms | **-21.1%** |
| `BenchmarkRetentionRecurrent_256x64` | 1.540 ms | 1.124 ms | **-27.0%** |

### Count the multiplies before promising bit identity

The state update looks like the easier of the two — every `S[i,j]` is written
once from its own old value, so the keys are genuinely independent and grouping
them changes no arithmetic at the source level. Jamming it moved the output.

It moved **even at `dk=1`**, where the jammed body never executes and the
surviving tail is textually identical to the code that was already there. The
whole-function FMA count went from 192 to 196.

The cause is the expression shape. `gamma*S + k*v` **adds two products**, so two
fused-multiply-add contractions are legal — fuse either multiply into the add —
and they round differently. Which one the compiler picks is not stable across
restructuring the enclosing function. The output dot's `out[j] + q_i*S[i,j]` has
**one** multiply and therefore one legal contraction, which is why it jams
cleanly.

So the rule is not about the loop at all: an accumulation is safely jammable when
its expression holds a single multiply. With two, "the arithmetic is unchanged"
is a statement about the source that the compiler is free to contradict.

This is also the clearest case yet for freezing bits rather than comparing to a
tolerance. A one-ulp drift in a recurrence compounds across steps, and no
gradient check or accuracy assertion would have said a word.

## Mamba-2's SSD scan, and the two-product rule confirmed (T1190)

`SSDRecurrent` has the same two-loop shape as RetNet: a state update
`h[i,j] = a_t*h[i,j] + b_i*x_j` and an output dot `y[j] += c_i*h[i,j]`, at 43%
and 38% of the profile.

The state update was jammed first, deliberately, to test T1189's finding on a
second kernel. **All five digests moved**, exactly as RetNet's did. The output
dot, one product, jammed cleanly.

| Benchmark | before | after | delta |
|---|---|---|---|
| `BenchmarkSSDRecurrent_256d256n128` | 11.33 ms | 8.96 ms | **-20.9%** |
| `BenchmarkSSDRecurrent_512d128n64` | 5.988 ms | 4.471 ms | **-25.3%** |
| `BenchmarkSSDRecurrent_256d64n16` | 0.459 ms | 0.459 ms | flat (control) |

The flat cell is a control by construction: at `n*d = 1024` the kernel takes its
strided output path instead of the interleaved one, and that path is untouched.

### PS3084

The hazard is now a check: an in-place update inside a nested loop whose
right-hand side adds two products and reads the destination back, where the
destination varies with the outer loop and some factor does not. **101 sites**,
overwhelmingly optimizer update rules of the form
`m[i] = beta*m[i] + (1-beta)*g[i]` — a population worth knowing about, since
optimizer element loops are a standing perf target here.

It is a warning, not a candidate list: the advice is that jamming these is a
one-ulp change and must be described as one, and that the neighbor loop adding a
single product is usually the better target anyway.

Four conditions, each pinned by a fixture that reddens when it alone is removed.
Two of those fixtures had to be rewritten before they isolated anything: a `+=`
form is rejected by the operator test and a bare `h = a*h` by the add test, so
neither ever reached the product count that a mutation was weakening.

## Gated linear attention: the third kernel with this pair (T1191)

GLA carries a `[dk,dv]` state and has the now-familiar two loops — a state update
`S = S*g_i + k_i*v` and an output dot `o[j] += q_i*S[i,j]`. Only the second was
touched, on the rule established in T1189 and confirmed in T1190.

| Benchmark | before | after | delta |
|---|---|---|---|
| `BenchmarkGLA_512x128` | 13.27 ms | 9.99 ms | **-24.7%** |
| `BenchmarkGLA_256x64` | 1.242 ms | 0.908 ms | **-26.9%** |

No new rule this round: `PS3075` found the jammable loop and `PS3084` marks the
one beside it as off limits, which is exactly the division those two checks
exist to draw. Four recurrent kernels now follow it — RetNet, Mamba-2 SSD, GLA
and MLA.

### Not taken

`Mamba2Forward_512` is the largest benchmark in this area at 177 ms and was
profiled first. It is **72.8% `gemmF64Band`** — the CPU GEMM — so the leverage
there is in a kernel with a known BLAS-shaped ceiling, not in a loop this
technique reaches. Left alone deliberately rather than for lack of size.
