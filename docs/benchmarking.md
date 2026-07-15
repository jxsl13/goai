# Benchmarking & the cgo gate (§V5, §C3)

> **In plain terms:** this document explains how we prove that a "faster"
> version of the code is actually faster. Every speed claim in GoAI comes
> with a measured before/after number, produced by a repeatable benchmark —
> and a change that cannot show such a number does not get merged. The rest
> of the page records those numbers and the rules for producing them.


**Abbreviations in this document:** **GELU** = the Gaussian-error activation function; **BLIS** = a modern BLAS-like kernel library; **CE** = cross-entropy.

Every optimization task (§T11, §T12, …) must show a benchmark delta against the
Pure-Go reference before it lands, and cgo is only considered after the Pure-Go
path is optimized to its ceiling and still loses by the §C3 threshold.

## Running

```sh
make bench                       # all benchmarks, CGO_ENABLED=0
go test ./backend/ref -bench .   # a single package
```

Inputs come from `internal/bench` with explicit seeds, so numbers are
reproducible across runs and machines.

## Full comparison matrix — every variant vs the industry (§T606)

> **In plain terms:** the tables below put every GoAI backend variant next to
> the mainstream Python stacks measured **on the same machine**, so you can see
> exactly where this library stands — including where it is still behind.

Method: identical shapes and seeds, one warm-up before timing, 30 iterations,
Apple M-series (darwin/arm64), macOS; Python side is numpy 2.5.1 and
torch 2.12.1 (`torch-mps` = torch on Apple's Metal Performance Shaders GPU
backend; `GFLOP/s` = billions of floating-point operations per second, higher
is better; `tok/s` = generated tokens per second). Regenerate with:

```sh
make bench-compare bench-python > /tmp/bench.txt
go run ./internal/benchcompare/rendertables /tmp/bench.txt
```

Variants: `goai-ref` is the plain readable reference (numerical truth, not
speed), `goai-cpu` the optimized pure-Go CPU backend, `goai-metal`/`goai-vulkan`
the GPU backends (Metal via Apple's GPU API; Vulkan via MoltenVK, a layer that
runs Vulkan on Metal).

| Workload | goai-ref | goai-cpu | goai-metal | goai-vulkan | industry (same box) |
|---|---|---|---|---|---|
| Conv2D/n8c16hw32 | 353.50 ms (0.2136 GFLOP/s) | 2.41 ms (31.37 GFLOP/s) | 853.9 µs (88.41 GFLOP/s) | 551.0 µs (137.0 GFLOP/s) | torch-cpu: 621.12 GFLOP/s; torch-mps: 2418.93 GFLOP/s |
| Conv2D/n8c64hw56 | 8.74 s (0.2116 GFLOP/s) | 35.87 ms (51.57 GFLOP/s) | 2.60 ms (711.7 GFLOP/s) | 4.55 ms (406.4 GFLOP/s) | torch-cpu: 621.12 GFLOP/s; torch-mps: 2418.93 GFLOP/s |
| FlashAttn | 767.25 ms | 25.89 ms | 10.77 ms | 12.19 ms |  |
| GPTForward | — | 224.69 ms (1139 tok/s) | 30.49 ms (8396 tok/s) | 42.52 ms (6020 tok/s) |  |
| GPTTrainingStep | — | 1.05 s (244.9 tok/s) | 88.66 ms (2887 tok/s) | 130.52 ms (1961 tok/s) |  |
| LayerNorm | 72.92 ms | 6.79 ms | 1.77 ms | 1.79 ms |  |
| LayerNormBackward | 175.48 ms | 23.74 ms | 2.52 ms | 2.43 ms |  |
| MHABackward | 2.73 s | 45.27 ms | 3.05 ms | 4.83 ms |  |
| MHAForward | 795.20 ms | 20.56 ms | 1.78 ms | 1.78 ms | torch-cpu: 0.71 ms/op; torch-mps: 0.42 ms/op |
| MatMul/1024 | 6.96 s (0.3083 GFLOP/s) | 31.71 ms (67.72 GFLOP/s) | 1.56 ms (1376 GFLOP/s) | 3.69 ms (581.4 GFLOP/s) | numpy-cpu: 2653.45 GFLOP/s; torch-cpu: 2684.50 GFLOP/s; torch-mps: 4170.66 GFLOP/s |
| MatMul/256 | 108.04 ms (0.3106 GFLOP/s) | 760.1 µs (44.15 GFLOP/s) | 409.6 µs (81.92 GFLOP/s) | 493.5 µs (68.00 GFLOP/s) | numpy-cpu: 2653.45 GFLOP/s; torch-cpu: 2684.50 GFLOP/s; torch-mps: 4170.66 GFLOP/s |
| MatMul/512 | 862.64 ms (0.3112 GFLOP/s) | 4.50 ms (59.61 GFLOP/s) | 659.0 µs (407.3 GFLOP/s) | 1.03 ms (261.0 GFLOP/s) | numpy-cpu: 2653.45 GFLOP/s; torch-cpu: 2684.50 GFLOP/s; torch-mps: 4170.66 GFLOP/s |
| RMSNorm | 70.74 ms | 4.27 ms | 1.68 ms | 1.73 ms |  |
| RMSNormBackward | 118.76 ms | 18.19 ms | 2.30 ms | 2.28 ms |  |
| Retention | 88.90 ms | 10.57 ms | 10.98 ms | 10.92 ms |  |
| RetentionBackward | 320.93 ms | 25.29 ms | 20.22 ms | 20.31 ms |  |
| Softmax | 77.51 ms | 7.76 ms | 1.73 ms | 1.76 ms |  |

| Decode workload | variant | rate |
|---|---|---|
| LlamaDecode | batched-metal | 263.1 tok/s |
| LlamaDecode | batched-vulkan | 245.0 tok/s |
| LlamaDecode | per-op-metal | 160.3 tok/s |
| LlamaDecodeLongContext | batched-metal | 71.56 tok/s |
| LlamaDecodeLongContext | batched-vulkan | 66.84 tok/s |
| LlamaPrefill | sequential-metal | 276.9 tok/s |
| LlamaPrefill | stepn-metal | 11068 tok/s |

| Python stack workload | numpy-cpu | torch-cpu | torch-mps |
|---|---|---|---|
| Conv2D/8x16x32sq_k3 | — | 621.12 GFLOP/s | 2418.93 GFLOP/s |
| MHAForward/512x8x64 | — | 0.71 ms/op | 0.42 ms/op |
| MHAFwdBwd/512x8x64 | — | 2.26 ms/op | 1.15 ms/op |
| MatMul/1024 | 2653.45 GFLOP/s | 2684.50 GFLOP/s | 4170.66 GFLOP/s |
| MatMul/256 | 981.68 GFLOP/s | 989.00 GFLOP/s | 486.97 GFLOP/s |
| MatMul/512 | 2141.68 GFLOP/s | 2172.86 GFLOP/s | 1328.09 GFLOP/s |


### Conv2D fused implicit-GEMM on Vulkan (§T620, rung 1)

The Vulkan Conv2D was lowered as im2col (materialize the column matrix to global
memory) → tiled GEMM → scatter. The `col` write+read is the bottleneck (§T620).
The fused `conv_igemm.comp` kernel instead gathers `X` directly in the tiled-GEMM
tile load — no `col` buffer, one dispatch. Same-machine A/B
(`BenchmarkVulkanConv2D`, n8c16hw32 f32 k3, 100×3):

| path | GFLOP/s | Δ |
|------|---------|---|
| im2col + tiled GEMM (old) | ≈68 | — |
| fused implicit-GEMM (§T620) | ≈83.5 | +≈20–23% |

Correctness is bit-checked vs the CPU reference (`TestVulkanConv2DCrossReference`,
5 shape cases). NOTE: absolute numbers here are ≈2× below the healthy-machine
baseline (137 GFLOP/s, §T606 matrix above) because of transient local GPU
degradation (§B55) — the **same-machine relative delta** is the signal, not the
absolute. A naive single 30× run first read 67.55 and looked like a regression;
the paired same-machine A/B corrected that (verify-don't-assume).

**Rung 2 (vec4-coalesced gather):** giving each thread NB=4 output columns and
gathering the B panel as a contiguous vec4 (one 16-byte load) on the stride-1
fast path adds another ≈7% (≈80→≈87.5 GFLOP/s, tightly-interleaved same-machine
A/B). fp16 was measured-and-skipped: this kernel is latency/occupancy-bound
(≈6 GB/s « the M2 Pro's ≈200), not bandwidth-bound, so half-precision has no
bottleneck to relieve.

**Metal took a different path.** A hand-tiled fused kernel on metal was measured
≈1.7× *slower* than im2col+MPS (MPS's tuned GEMM can't be matched by a hand-written
kernel), so custom lowering is the wrong lever there. The win instead is Apple's
**native MPSGraph convolution2D** — the same tuned primitive torch-mps calls, which
picks implicit-GEMM/Winograd internally. Swapping the im2col→MPSMatrixMultiplication
path for MPSGraph conv (same-machine A/B, 100×4):

| shape | im2col+MPS | native MPSGraph | Δ |
|---|---|---|---|
| n8c16hw32 k3 | ≈189 GFLOP/s | ≈216 GFLOP/s | +14% |
| n8c64hw56 k3 | ≈787 GFLOP/s | ≈1850 GFLOP/s | 2.35× |

On the compute-bound shape that closes the torch-mps gap (≈2418 GFLOP/s) to ≈1.3×,
from ≈3.6×. Correctness stays within a documented 2× cross-tolerance (MPSGraph uses
a different f32 reduction order). The vulkan fused+vec4 kernel, the metal MPSGraph
switch, and these findings are §T620.


### MPSGraph attention prefill on Metal (§T621, experiment — opt-in)

The Metal attention prefill (sq==sk, window==0) default runs `mtl_mha_mps`: a
per-head loop of `MPSMatrixMultiplication` (Q·Kᵀ) → custom softmax kernel →
`MPSMatrixMultiplication` (P·V), sharing one `[seq,seq]` scratch, so the heads are
serialized. `mtl_mha_mpsgraph` instead builds ONE cached `MPSGraph` (reshape →
transpose → batched matmul QKᵀ → +causal −∞ mask → fused softmax → batched matmul
P·V → transpose back) that runs **all heads in a single graph** — Apple's native
graph compiler picks the schedule. Manual matmul/softmax graph (macOS 12.3+), not
the macOS-15 `scaledDotProductAttention` primitive, so it runs on-OS. Tightly
interleaved same-machine A/B (`BenchmarkMHA[MPSGraph]_512x8x64_metal`, seq=512
heads=8 dk=64 causal, 1000×, 5 rounds; MPSGraph won **every** round):

| path | µs/op | GFLOP/s | Δ |
|------|-------|---------|---|
| custom per-head MPS (`mtl_mha_mps`, default) | ≈1417 | ≈379 | — |
| MPSGraph batched (`mtl_mha_mpsgraph`, §T621) | ≈1124 | ≈478 | **+≈26% (1.26×)** |

Correctness is cross-referenced vs the f64 Pure-Go reference
(`TestMetalMHAMPSGraphCrossReference`: mha/causal/GQA/MQA/attn-scale + the seq=512
shape) within the **same** `crossTol(dk+seq)` as the custom path — no tolerance
relaxation was needed. Absolutes are ≈2× below a healthy machine (§B55); the
same-machine relative delta is the signal.

**Disposition — opt-in, NOT the default.** The win is real for *fixed* shapes,
but the graph carries a **one-time ≈15–30 ms compile per unique
`(seq,kvHeads,rep,dk,causal)`** (the custom path compiles nothing). A fixed-shape
training loop amortizes this to zero and should flip it on
(`metal.SetMHAUseMPSGraph(true)`); variable-length inference prefill would
recompile per prompt length and regress. So the custom path stays the compiled-in
default and MPSGraph is a proven, tested opt-in. Flipping the global default is
blocked pending a small shape-keyed graph cache + a variable-length A/B.


### CPU kernel devirtualization (§T596/§T602 pattern, class-audit 2026-07-14)

Several optimized CPU kernels read and wrote every element through `f64at`/`f64set`
per-element closures (one indirect call per element, plus a float32↔float64 convert
for f32 tensors). Routing each hot kernel through a generic `[]T` core (`T =
float32|float64`) over the concrete storage slice lets the compiler inline every
access; f64 accumulation and the per-row operation order stay unchanged, so
reference parity holds within ulps. Same-machine A/B (CGO_ENABLED=0, ref+cpu only):

| kernel | before (closures) | after (generic `[]T`) | speedup |
|---|---|---|---|
| RMSNorm (2048²) | 4.42 ms | 1.26 ms | ≈3.5× |
| LayerNorm (2048²) | 6.85 ms | 1.83 ms | ≈3.7× |
| Softmax (2048² f32) | 7.71 ms | 5.82 ms | ≈1.33× |
| Retention fwd (512×64 f32) | 10.66 ms | 2.77 ms | ≈3.85× |
| Retention bwd (512×64 f32) | 24.12 ms | 5.69 ms | ≈4.25× |
| ReLU (1M f32) | 1.14 ms | 0.87 ms | ≈1.32× |
| SiLU (1M f32) | 2.00 ms | 1.43 ms | ≈1.40× |
| GELU (1M f32) | 2.12 ms | 2.00 ms | ≈1.06× |
| Neg (1M f32) | 0.37 ms | 0.22 ms | ≈1.68× |
| StopGradient (1M f32) | 0.37 ms | 0.17 ms | ≈2.14× |
| AvgPool2D (n8c64hw56 k3s2 f32) | 1.83 ms | 0.65 ms | ≈2.83× |
| MaxPool2D (n8c64hw56 k3s2 f32) | 2.83 ms | 1.74 ms | ≈1.63× |
| Conv2D fwd (im2col fill, f32) | 0.71 ms | 0.59 ms | ≈1.21× |

The unary **activations** (relu/silu/gelu) sat on the same closure path — `unOp(f
func(float64)float64)` called an indirect `f` per element plus a float32↔float64
round-trip; SiLU even nested a second closure (`x·sigmoid(x)`). Native per-op `[]T`
kernels remove that. These are **base ops** — every FFN in every layer runs an
activation — so the win compounds across the whole stack. GELU gains little (its
cost is `math.Erf`, an f64 transcendental, so the round-trip is inherent); relu/silu
gain most (little math, so the closure overhead dominated).

### L0 tensor-layer hot paths (the same anti-pattern, one level down)

The devirtualization sweep of the compute kernels missed a deeper layer: the L0
`tensor` operations everything is built on. `Contiguous()` (materialize a
transposed/permuted view — hit by every attention Q/K/V reshape), `Cast()` (the
F64↔F32 conversion every tensor pays to reach the f32-only GPU backends), and the
elementwise `broadcastContig` helper all walked the tensor **element by element**,
and each element called `Unravel(pos, shape)` — which **heap-allocates a fresh
`[]int` multi-index per element** — then read/wrote through the dtype-dispatching
`AtF64`/`SetF64`. Replacing that with a running multi-index whose source offset is
maintained **incrementally** (no per-element alloc) plus typed `[]T` access:

| L0 op | before | after | speedup |
|---|---|---|---|
| Cast (512² f64→f32, contiguous) | 2.65 ms | 0.16 ms | **≈17×** |
| broadcastContig (512→512², via mul) | 2.79 ms | 0.73 ms | ≈3.8× |
| Contiguous (512² transposed) | 1.72 ms | 0.58 ms | ≈3.0× |
| conv-backward grad scatter | (17% of op) | typed copy | ≈1.23× on the op |

`Cast` is the standout: it's on the hot path to every GPU op (F64 model → F32
kernels), and it was allocating one `[]int` per element (≈262 k allocs for a 512²
tensor). These are the **base of the base** — L0 ops that compound into every layer
above — so the win is stack-wide even though no model code changed. Guarded by
`BenchmarkContiguousStrided`/`BenchmarkCastF64toF32` (§V5).

The same closure pattern sat on the **CV path** — pooling read every window tap
through a `get`/`set` closure pair, and the conv im2col fill (forward and backward)
read `X`/`W`/`dO` through per-element closures. These are the base ops CNN layers
build on, so devirtualizing them lifts every CV model. AvgPool wins most (2.83×):
its inner loop is a pure sum over the window, so the closure indirection was almost
all of it. Conv gains ≈1.21× on the whole op (the im2col fill is memory-bound and
a minority of conv time — the blocked GEMM dominates and was already typed).

The wins scale with how closure-heavy the kernel's inner loops were: softmax's
per-element `exp` dilutes the closure overhead (small win), while retention's nested
score/aggregate loops call the accessor many times per output (largest win). After
this class-audit the only remaining `f64at`/`f64set` uses in `backend/cpu/` are
comments — every hot CPU kernel is devirtualized.

### BPE tokenizer merge — allocation-free byte-pair merge (§T625, 2026-07-14)

Tokenization is a base path in the same sense: it runs on every prompt, upstream of
all inference. `bpeMerge` (`nlp/bpe.go`) was the textbook-naive byte-pair merge — a
list of copied byte-slices, and on every candidate pair on every iteration it built
the merge-table key as `string(parts[i]) + string(parts[i+1])`: three string
allocations per pair, an O(L²) allocation load per piece of length `L`.

The fix is tiktoken's `byte_pair_merge` (§R33): keep only byte **offsets** into the
immutable piece and rank each pair with `ranks[string(piece[a:c])]`. The Go compiler
special-cases `m[string(byteSlice)]` — when the converted string is used only as a
map key it elides the allocation entirely — so the whole merge does **zero** map-key
allocations. The per-pair minimum is maintained incrementally (only the two
neighbours of a merge recompute their pair rank). The merge order (leftmost lowest
rank) and the final token boundaries are bit-identical, so encode parity is exact.

Same-session A/B (`BenchmarkBPEMergeNew` vs `…Naive`, a mixed natural-language / code
/ digit-run / CJK / URL corpus, interleaved ×3):

| | ns/op | B/op | allocs/op |
|---|---|---|---|
| naive scan (old) | 502 µs | 68656 | 2084 |
| offset merge (new) | 78.7 µs | 42456 | 601 |
| factor | **6.4×** | 1.62× | 3.47× |

Parity is locked by the real-tiktoken golden (`TestBPEEncodeParityTiktoken`), a new
piece-for-piece old-vs-new check (`TestBPEMergeNaiveParity`), and the existing
2.5M-exec round-trip fuzz — all green, CGO0 vet + full `nlp` suite green.

### Sampler top-k via quickselect (§T626, 2026-07-14)

`Sampler.Dist` is the other end of every generated token: it turns a logit vector
into a probability distribution, applying whatever truncation the sampler is
configured for. Each truncation path (top-k, top-p, locally-typical) sorted the
**entire** vocabulary with `sort.Slice` just to keep a handful of tokens. Measured
floor at V=50257 (the plain softmax is the baseline):

| path | ns/op | over softmax |
|---|---|---|
| plain softmax | 441 µs | — |
| top-k=40 | 5637 µs | +5.2 ms (full sort) |
| top-p=0.9 | 5856 µs | +5.4 ms |
| typical | 6970 µs | +6.5 ms |

Top-k only needs the k largest, not a total order, so it now uses **quickselect**
(`kthLargest`, deterministic median-of-three pivot, O(V) average) to find the k-th
largest logit and mask everything below it. Same-session A/B (`BenchmarkDistTopK40`
vs `…Naive`, interleaved):

| top-k=40 | ns/op | allocs/op |
|---|---|---|
| full sort (old) | 5637 µs | 5 |
| quickselect (new) | 562 µs | 3 |
| factor | **10.0×** | — |

Output is bit-identical (`TestDistQuickselectParity` compares against the old
full-sort across 9 sampler configs × {64, 1024, 50257} vocab × 3 seeds ≤ 1e-12;
`TestKthLargest` checks the selector against a full sort on adversarial inputs). The
top-p and typical paths were switched to a typed `slices.SortFunc` (no reflection),
but that alone measured only ≈1.06× — modern `sort.Slice`'s reflection overhead is
small and the O(V log V) sort work dominates.

### Sampler top-p via bounded nucleus selection (§T627, 2026-07-14)

The real top-p win is algorithmic. The nucleus is almost always a tiny fraction of
the vocabulary, so `nucleusTopP` quickselects the top-512 candidates (far more than
any realistic nucleus), sorts only those, and accumulates in exact descending order
until the mass crosses `p` — falling back to a full sort only when the nucleus
genuinely exceeds the candidate set. Because the accumulation order is unchanged the
result is bit-identical, and the fallback means it is never worse than the old sort.

| top-p | logits | ns/op (old → new) | factor |
|---|---|---|---|
| p=0.9 | peaky (realistic, small nucleus) | 3651 µs → 596 µs | **6.1×** |
| p=0.99 | flat (nucleus > candidates → fallback) | 5980 µs → 5512 µs | 1.08× (no regression) |

Real LLM logits concentrate their mass on a handful of tokens, so the peaky row is
the case that matters in practice. The flat row confirms §C3 — even when the bounded
probe misses and the code re-sorts, the typed fallback beats the old reflection sort.
Parity is locked by `TestDistQuickselectParity` (top-p configs, ≤ 1e-12 vs the old
full sort over {64, 1024, 50257} × 3 seeds).

Locally-typical sampling (§T628) gets the same bounded selection, with two wrinkles:
it orders by the typical score `|−log p − H|` (ascending) while accumulating
*probability*, and it keeps the exact index prefix rather than a value threshold —
`|score|` can tie across two distinct probabilities symmetric about `e^−H`, so a
threshold would be ambiguous. The win is smaller because the per-token entropy and
score computation (two logs over the whole vocab) is unavoidable O(V) and dominates
once the sort is removed:

| sampler | logits | ns/op (old → new) | factor |
|---|---|---|---|
| typical τ=0.9 | peaky | 4603 µs → 1618 µs | **2.85×** |
| typical τ=0.95 | flat (fallback) | 6883 µs → 6585 µs | 1.05× (no regression) |

Together the three truncation paths — top-k (§T626, 10×), top-p (§T627, 6.1×) and
typical (§T628, 2.85×) — take `Sampler.Dist` off its per-token full-sort of the
vocabulary, bit-identically.

### The combined-config trap: quadratic nucleus selection (§T629, §B58)

Measuring the *full* host cost per token (`SampleWithHistoryRealistic`: temp +
top-k=200 + top-p=0.9 + repeat penalty, V=50257) surfaced a surprise — **7.5 ms**,
worse than the plain softmax. The cause was in the nucleus quickselects: their
two-way partition degrades to O(n²) when most keys are equal, and a top-k that
already masked all but 200 tokens to zero left ~50k identical zeros for the top-p
quickselect to partition one at a time. The per-truncation benchmarks never showed
it (each path alone sees distinct keys), and parity was always correct — only the
combined config exposed it.

A three-way (Dutch-flag) partition resolves the equal-key band in a single pass:

| combined sampler (top-k + top-p + penalty) | ns/op |
|---|---|
| two-way partition | 7527 µs |
| three-way partition | 611 µs (**12.3×**) |

Lesson: a selection helper fed post-truncation data has *many* duplicate keys — make
it duplicate-robust, and benchmark the **combined** sampler config, not each
truncation in isolation. The 0.61 ms host cost this leaves is what makes the
SIMD+GPU overlap analysis in ADR-0021 valid (≈13% of a ~4 ms GPU decode step); the
7.5 ms figure would have inverted that conclusion.

### The CGO0-default fallback devirtualization (§T645/§T646, 2026-07-15)

The `cpu` backend registers only ~12 optimized kernels (matmul, conv2d, pools, MHA,
softmax, retention, add-bias forward); every other op falls back to the `ref`
backend (the §V9 numerical-truth reference), which is written per-element
(`AtF64`/`SetF64` dispatch). A `GOAI_LOG_FALLBACK=1` audit (§T401 method) on a real
CPU Mamba/MoE/MLA char-LM training exposed how much of the CGO0-default training path
this covers — fallbacks per training run, by call count:

| op | calls/run | was |
|---|---|---|
| `addbias_backward` | 6250 | per-element ref |
| `silu_backward` | 2500 | per-element ref |
| `crossentropy` + `_backward` | 1500 | per-element ref |
| `conv1d`, `softplus` (fwd) | 805 each | per-element ref |
| `ssm`, `mla`, `moe*`, `wkv` | 250–805 | per-element ref |

Every one of these was devirtualized in `backend/ref` (typed dtype-switched fast
paths + generic fallback, bit-identical, each with a new F32 forward/backward parity
test since the suites were F64-only). The per-op speedups match the sibling
devirtualizations (≈7–15× on the typed kernel). Note the audit's fallback *counts*
are unchanged afterward — and that is correct: devirtualizing `ref` does not remove
the `cpu→ref` fallback (cpu still has no kernel for these ops), it makes the `ref`
path fast. The fallback dispatch itself (a map miss + one indirection) is negligible
next to the now-typed kernel, so a fast `ref` fallback is the complete fix, not a
reason to duplicate every op into `cpu`.

Two method lessons: (1) a crude "grep the op name in the cpu package" audit is
unreliable — registration patterns vary; `GOAI_LOG_FALLBACK` on the real workload is
the definitive gap-finder. (2) Re-running that audit measures *coverage* (which ops
route to ref), not speed — the speed win is per-op-typed, so verify it with the op's
own parity/gradcheck, not a fallback-count diff.

### Head-to-head: llama.cpp on the SAME weights (§T607)

The decode benchmark's llama (17.7 M parameters, dim 512, 6 layers, GQA 8/2,
F32) is exported as a llama.cpp-loadable GGUF by
`go run ./internal/benchcompare/exportgguf`, so both engines time **identical
weights** (llama-bench b9960, 3 repetitions, same machine):

| Engine | prefill (pp64) | decode (tg64) |
|---|---|---|
| llama.cpp Metal | 15,900 ± 751 tok/s | 1,098 ± 72 tok/s |
| llama.cpp CPU (Accelerate BLAS) | 8,778 ± 355 tok/s | 1,143 ± 35 tok/s |
| GoAI metal (batched decoder) | 11,068 tok/s | 263 tok/s |
| GoAI vulkan (batched decoder) | — | 245 tok/s |

Read: our batched prefill is within 1.4× of llama.cpp's Metal prefill — the
one-command-buffer recording strategy holds up. Single-token decode is ≈4.2×
behind: llama.cpp's hand-tuned kernels and years of decode-path engineering
(and at this toy size, Apple's Accelerate BLAS alone already saturates it).
Honest caveat: 17.7 M parameters is far below production size; the gap
composition will differ at 7B-class sizes where memory bandwidth dominates.

Honest read of the gaps (as of 2026-07-14):

- **CPU vs vendor BLAS** (BLAS = the decades-old optimized linear-algebra
  libraries behind numpy/torch): our pure-Go f64-accumulating GEMM (general
  matrix multiply) reaches ≈68 GFLOP/s where torch-cpu reaches ≈2700 — that
  class of gap needs f32 SIMD kernels (SIMD = single-instruction-multiple-data,
  the CPU's vector math unit) and is tracked as the §T11b track; the same
  applies to attention (torch's fused SDPA at 0.7 ms vs our 20.6 ms).
- **Metal matmul at 1.4 TFLOP/s vs torch-mps 4.2** reflects Apple's MPS-tuned
  kernels; measured history in §T434.
- **The 2026-07-14 CPU arc** (norms, softmax, attention, conv output path)
  multiplied the end-to-end CPU transformer to 6.4× forward / 6.1× training of
  its previous-day numbers — the per-op rows above are AFTER that arc.
- FlashAttn and Retention received their typed parallel CPU kernels on
  2026-07-14 (29.6×/8.4×/12.7×, §T610) — no `ref ≡ cpu` row remains.

## Baselines (Pure-Go reference, §T5–§T9)

Reference kernels favor clarity over speed (index math via `Unravel` allocates
per element) — the high alloc counts are exactly what §T11 removes. Recorded so
the optimized kernels have a concrete target:

| Benchmark | ref ns/op | cpu ns/op | speedup | note |
|-----------|-----------|-----------|---------|------|
| AddF64 4K | ≈1.27e5 (4104 allocs) | ≈1.2e4 (9 allocs) | ≈10× | §T11, bit-identical |
| MatMulF64 128³ | ≈9.1e6 (0.46 GFLOP/s) | ≈2.97e5 (14.1 GFLOP/s) | ≈31× | §T12/§T12b, bit-identical |
| MatMulF64 256³ | — | ≈9.22e5 (36.4 GFLOP/s) | +31% vs ikj | §T12b 4-row blocking |
| MatMulF64 512³ | — | ≈5.30e6 (50.6 GFLOP/s) | — | §T12b |

(Indicative darwin/arm64 numbers; treat the committed CI run as the source of
truth. Machine/arch is recorded alongside any comparison. §T12b's 4-row register
blocking preserves per-element k-order → still bit-identical to ref (V11 tol 0).
Next GFLOP/s gains: archsimd FMA microkernel on amd64, §T11b.)

## Cross-language baseline: GoAI vs PyTorch (§R67, 2026-07-06)

The per-kernel tables above compare GoAI against its own Pure-Go reference. This
one compares the optimized `cpu` GEMM against **PyTorch** (the parity target),
head-to-head on identical sizes/dtypes. Harness: `backend/cpu/gflops_bench_test.go`
(Go, `GFLOP/s` metric) + `testdata/bench_torch.py` (torch). Dense N×N = 2·N³ flops.

```sh
go test ./backend/cpu -run '^$' -bench 'GEMM.*gflops' -benchtime=1s
.venv/bin/python testdata/bench_torch.py
```

| dtype · N | GoAI cpu | PyTorch (Accelerate) | GoAI / torch |
|-----------|---------:|---------------------:|-------------:|
| f64 512   | 60.6  | 660.3  | 9.2% |
| f64 1024  | 69.3  | 683.6  | 10.1% |
| f32 512   | 58.5  | 2312.8 | 2.5% |
| f32 1024  | 70.4  | 2734.7 | 2.6% |

(Indicative darwin/arm64, Apple M2 Pro, torch 2.12.1, Go 1.26. Committed CI is the
source of truth.) Two levers, in priority order (see
`docs/research/02-frontier-and-perf-2026-07-06.md` for the full roadmap):

1. **f32 SIMD width** — GoAI f32 ≈ f64 (≈70 GFLOP/s), but torch f32 is ≈4× its f64.
   Our kernel does not exploit f32's 2× lanes; a SIMD microkernel roughly doubles
   f32 throughput (portable, no AMX needed).
2. **BLIS/Goto blocking + asm microkernel** — packing + register blocking lifts a
   scalar kernel to ≈50-60% of scalar peak; asm/`avo`/Go-1.26-`simd` closes most of
   the rest. The residual (Apple AMX / AVX-512) is a documented cgo-gate candidate,
   not a pure-Go target.

## The first cgo gate: Metal/MPS (§T20, PASSED 2026-07-05)

All three §C2 conditions held before merge: (1) Pure-Go GEMM at its documented
ceiling (§T12/§T12b); (2) benchmark over the §C3 threshold (≥1.5×); (3)
`CGO_ENABLED=0` default build untouched (`-tags metal` isolation).

| f32 GEMM | cpu (Pure-Go ceiling) | metal (MPS) | speedup |
|----------|----------------------|-------------|---------|
| 512³  | 4.50ms · 59.7 GFLOP/s | 0.99ms · **272 GFLOP/s** | **4.6×** |
| 1024³ | 30.2ms · 71.1 GFLOP/s | 2.37ms · **906 GFLOP/s** | **12.7×** |

Cross-tolerance (§V11): rtol(K) = 1e-6·√K (MPS accumulates in f32 and reorders).
Run: `make metal-test` / `make metal-bench` (darwin + cgo only).

## Cross-backend coverage: `make bench-compare` (§T92, extended §T338)

`internal/benchcompare` times each accelerated op on ref/cpu/metal/vulkan side by
side. Rows (2026-07-12): MatMul (256/512/1024), MHAForward/MHABackward, FlashAttn,
Retention/RetentionBackward, Softmax, RMSNorm, Conv2D (n8c16hw32 latency probe +
n8c64hw56 ResNet-block, §T341). Snapshot after the
§T335–§T337 GPU-overhead work (M2 Pro, medians, 30x):

| Op (shape) | ref | cpu | metal | vulkan |
|------------|----:|----:|------:|-------:|
| FlashAttn (512·8·64, causal) | 761ms | 757ms | 12.1ms | 12.2ms |
| Retention (512×64, γ .968)   | 87.7ms | 87.7ms | 10.6ms | 10.8ms |
| RetentionBackward (512×64)   | 319ms | 319ms | 20.2ms | 20.4ms |
| Softmax (2048×2048)          | 83.0ms | 82.8ms | 1.65ms | 1.71ms |
| RMSNorm (2048×2048)          | 78.8ms | 80.7ms | 1.80ms | 1.72ms |
| LayerNorm (2048×2048)        | ≈80ms  | ≈80ms  | 1.66ms | 1.70ms |

Regression check 2026-07-13 (§T524, after the T504–T523 era incl. the cpu worker
pool): every row re-measured within noise or slightly better (FlashAttn metal
12.1→10.8ms, RMSNorm metal 1.80→1.63ms; MatMul 1024 metal 1645 GFLOP/s, vulkan
582 vs 594 GFLOP/s recorded at §T335). No drift to investigate.

§T528 (2026-07-13): vulkan MHABackward got the metal-style matmul decomposition
(strided matmuls + packed softmax + jacobian, 7 staged submits): 71.5ms → 4.74ms
(15.1×), now 1.5× behind metal's MPS path. The per-op vulkan MHA FORWARD stays on
flash DELIBERATELY: §T398 measured the decomposed forward at 6× in isolation but a
real-GPT LOSS (3191→2957 tok/s — vulkan's forward is FFN-bound, not attention-bound);
the backward is different because it was 24× off, far above dispatch overhead.
§T530 real-workload confirmation: full training step (D512 S256 L6, the metal bench
shape, new `BenchmarkGPTTrainingStepVK`): 934.7 → 1591 tok/s = **1.70× vulkan GPT
training** from the backward chain alone (A/B/A file-toggle, medians of 3, a one-off
thermal outlier ruled out with 6 further chain runs).
§T531 profile-driven follow-up: GOAI_TIME_OPS put the FORWARD mha at 19% of the
training step — the §T398 rejection had measured a costlier per-head-submit chain.
Rebuilt in the §T528 structure and A/B/A'd: flash 1590 → chain 1878 tok/s (+18%),
now the DEFAULT for the sq==sk no-window shape (flash keeps window + error fallback).
Cumulative §T528+§T531: **934.7 → 1882 tok/s = 2.01× vulkan GPT training** —
metal-class (its §T399 rework gave 2.04×). Remaining profile: matmul 51% (GEMM
ceiling), attention now ≈14%.
§T534/§B49: profiling METAL's step found the residual adds at 14.6% — the per-op
GPU binary kernels violated ADR-0008 on host-resident tensors. Binary ops now route
to the optimized cpu backend on BOTH GPU backends (with the recorder STRIPPED on the
re-dispatch — §B49: keeping it doubled every routed op's gradients). Final honest
numbers: metal 2985→3219 tok/s (+7.8%), vulkan 1882→1992 (+5.8%, arc cumulative
935→1992 = 2.13×).

History of the row-parallel norm/softmax kernels (all measured 2048×2048, medians):
- Original one-thread-per-row, ≈1024-wide threadgroups → metal 10ms / vulkan 2.8ms.
- §T339 capped metal at 64 threads/threadgroup (dispatch-granularity fix) → ≈3ms.
- §T345/§T346 made them COOPERATIVE — one 256-thread threadgroup per row, coalesced
  strided access + a threadgroup tree reduction (the old kernel's neighbouring threads
  read addresses `dim` floats apart, fully uncoalesced, hitting ≈10% of bandwidth) →
  ≈1.7ms, metal and vulkan at parity — the coalesced kernel is the real, measured win.
- The remaining ≈1.7ms floor was *assumed* to be the host↔device memcpy, but §T348/§B42
  **measured** it: a same-session A/B with `bytesNoCopy` zero-copy vs the copy path showed
  **no difference** (≈1.73 vs ≈1.75ms). The copy is NOT the bottleneck. The kernels move
  ≈48MB at only ≈25 GB/s (≪ the ≈200 GB/s the hardware allows), so the floor is per-op GPU
  dispatch / `waitUntilCompleted` latency + reduction-barrier serialization. The next lever
  for this family is **fewer per-op GPU round-trips** (batch ops into one command buffer with
  barriers, as §T343 did for conv; or a persistent encoder / graph submission) — not
  zero-copy, which was built, measured, and reverted (ADR-0018).

## Real-workload throughput: end-to-end GPT (§T350–§T355)

The per-op tables above are micro-benchmarks; §C3/§B10 require judging optimizations
against a real model. `BenchmarkGPTForward` / `BenchmarkGPTTrainingStep` time a full
transformer forward (and forward+backward) on a realistically-sized synthetic GPT
(vocab 4096, 512-dim, 8 heads, 6 layers, 256 tokens) across every backend. In
`internal/benchcompare` (cpu/metal/vulkan via `make bench-compare`) and, for cpu-vs-metal,
in `backend/metal/gpt_test.go`.

Snapshot (M2 Pro, 2026-07-12, tokens/s, higher is better):

| Workload | cpu (Pure-Go) | metal | vulkan |
|----------|--------------:|------:|-------:|
| GPT forward       | 181 | 4168 | 3647 |
| GPT training step | 41  | 535  | 497  |

Both GPU backends win ≈20× (forward) / ≈13× (training) over the Pure-Go cpu backend.

**BUT autoregressive DECODE is the opposite (§T360).** `BenchmarkGPTDecode` times one-token-per-step
generation with a KV cache — the real inference workload. Each step's ops are tiny (seq=1), so the
per-op GPU dispatch / `waitUntilCompleted` round-trip (≈200 µs, and a decode step is ≈95 ops)
dominates, and the CPU — which runs the tiny compute with no round-trip — **wins**:

| Workload | cpu | metal |
|----------|----:|------:|
| GPT decode (tok/s, higher better) | ≈101 | ≈44 |

So Metal is ≈2.3× **slower** than cpu for decode *on the per-op path*. The systemic fix — batch a
whole decode step into one command buffer (one submit + one wait instead of ≈95) — is **done**:
the recorder / `llamagpu` program below (ADR-0019, §T404–§T432).

But this is **size-dependent** (§T361): as the model grows, the per-op GPU *compute* eventually
outweighs the fixed dispatch overhead and the GPU wins decode too. Measured decode, metal/cpu ratio
(lower = GPU faster):

| model | metal / cpu decode |
|-------|-------------------:|
| dim 512, 6 layers    | 2.7× (cpu wins) |
| dim 1024, 12 layers  | 0.99× (a wash)  |
| dim 2048, 24 layers  | 0.62× (GPU wins) |

So a blanket "decode on cpu" would be *wrong* for large models. `GPT.Generate` / `Llama.Generate`
run decode on `backend.Default()` but accept `nlp.WithBackend(be)` so the caller — who knows the
model size and hardware — can put small-model decode on the CPU. For serious GPU decode
throughput, use the batched decoders below instead of the per-op path.
Getting here was measurement-driven: the forward jumped 3.3× (1264→4168) once §T352 found
that **GELU and bias-add were silently falling back to the CPU reference** — they, not the
norm/attention kernels earlier fires had tuned, were ≈half the forward. The training step
rose as §T353/§T354 moved the GELU and bias-add **backwards** onto the GPU too. The lesson
(§V22): profile the real workload to find the bottleneck before optimizing a kernel. The
next measured training bottleneck is the MHA backward (≈21 ms/layer, a naive
one-thread-per-query atomic kernel).

## Batched GPU decode: the recorder & `llamagpu` (ADR-0019, §T404–§T432)

The per-op decode problem above (≈95 dispatch round-trips per token) is solved by the
**recorder**: record every op of a decode step into ONE command buffer over device-resident
weights and KV cache, then submit + wait once. `llamagpu` is the public API
(`New`/`NewVulkan`/`NewQuant`/`NewGPT` → `Decoder.Step`/`StepN`/`Generate`, plus lossless
`SpeculativeGenerate` and `PromptLookupGenerate`). Standing benchmarks in
`internal/benchcompare/decode_bench_test.go` (needs `-tags vulkan`; §T425/§T430):

| Benchmark (M2 Pro, 2026-07-12, tok/s) | metal | vulkan | per-op metal |
|---------------------------------------|------:|-------:|-------------:|
| `BenchmarkLlamaDecode` (1 token)                | 205  | 200  | 7.5 (**27× slower**) |
| `BenchmarkLlamaPrefill` (64-token prompt, StepN) | 5054 | —    | 140 sequential (**36×**) |
| `BenchmarkLlamaDecodeLongContext` (@pos 1920)    | 72.3 | 71.0 | — |

Long context needed its own lever (§T428–§T432): the original two-pass MHA kernel is serial in
sequence length, a cliff at large KV (242 ms/step @2k context). A **cooperative kernel** — one
32-lane simdgroup (Metal) / subgroup (Vulkan) per (query row, head), online-softmax partials
merged via lane shuffles — covers every attention surface: recorder decode (§T428/§T429,
242→13.8 ms, 17.6×), recorder prefill windows (§T431, 291→104 ms), and the per-op `OpMHA`
path (§T432, sq=1 @sk=1920: ≈40→2.18 ms). Quantized decode (`NewQuant`, §T413–§T416) trades
≈16% speed for 4× less weight memory (value = memory, not speed — measured, §T416). Its QUALITY
cost, measured on a trained model (§T477): Q8_0 and Q4_0 are both near-lossless there —
teacher-forced CE deltas within noise (+0.001 / −0.013 bits) and 99% / 97% argmax agreement
with f32. Measure agreement TEACHER-FORCED; a free-running comparison diverges at the first
mismatch and then scores different contexts (a metric artifact).

### Speculative decoding on dispatch-bound decoders: drafting must be free (§T434/§T446)

Both measured on the same in-repo-trained char-level base (dim 96, 3 layers, M2 Pro,
2026-07-13; medians of 3, real trained drafts/heads, greedy):

| Scheme | Acceptance | Lossless | tok/s vs plain | Speedup |
|--------|-----------:|:--------:|---------------:|--------:|
| Draft-model speculative (1-layer draft, §T434) | 81% | yes | 1150 → 1293 | 1.12× |
| **Medusa chain (3 linear heads, §T446/§T455)** | 97% | no (typical acceptance) | 1152 → 3546 | **3.08×** |
| **Prompt-lookup, n-gram (§T452)**              | 15% | yes | 1121 → 2020 | **1.80×** |

Round cost is everything on a dispatch-bound decoder. Medusa's first version paid 2 steps per
round (a StepHidden to draft, a StepN to verify) and reached 1.81×; §T455 folded the drafting
into the verify pass — StepNHidden returns the window's hidden rows, so the heads draft the
NEXT window from the CURRENT verification (the §T419 lastTok-lead-window convention) — making
a round ONE step for up to K+1 tokens: 3.08×. Prompt-lookup's round was always one step but
draws its candidates from history matching, so it needs repetitive output (here: the grammar
corpus) — at 15% acceptance it still yields 1.80× because ≈2.2 tokens/step. Medusa works on
any text its heads learned. Draft-model speculative pays a real decoder step per drafted token
and needs compute-bound (large) targets.

Draft QUALITY is a separate lever from round cost (§T472): acceptance measures how well the
draft matches the TARGET's distribution — which is what distillation optimizes directly. On the
same task, same draft architecture, same step budget and same init, a draft trained
independently on the corpus reached 73% acceptance while a draft DISTILLED from the target's
logits (`nn.GKDLoss`, forward KL) reached 88% (`TestDistilledDraftImprovesSpeculative`). If you
need a draft model, distill it from the target.

Same setup, opposite outcomes — the acceptance rate was never the problem. A batched decode
step is dispatch-bound (cost ∝ recorded ops, not compute), so a 1-layer draft model still pays
≈1/3 of a target step per drafted token and k·t_draft+t_verify eats the win. Medusa's heads
draft **host-side for free** (one [dim,vocab] projection each), so a round costs ≈2 steps for
up to K+1 tokens. Lesson pair: on dispatch-bound decoders use free-drafting schemes (Medusa,
prompt-lookup); draft-model speculative needs compute-bound (large) targets where
t_draft/t_target is small. Standing measurements: `TestSpeculativeWithTrainedModels` and
`TestMedusaGenerateGPTTrainedThroughput` in `llamagpu` (skipped under `-short`) train their
models in-repo and re-measure on every full suite run.

## Regression policy (§V5)

- An optimized kernel PR includes `benchstat old.txt new.txt`.
- CI fails a benchmark that regresses beyond noise on the reference baseline.
- The cgo gate (§C3): merge a cgo backend only when it beats the **optimized**
  Pure-Go kernel by ≥1.5× or reaches ≥80% of the C++ baseline the Pure-Go path
  cannot — measured on a real workload, not a micro-bench (§B10).

## Real-model-size decode (§T543)

A 124M-parameter GPT-2-small-shaped model (12 layers, d=768, vocab 50257 — synthetic
random weights, converter-mechanics fire; real weights are download-gated) through
`GPT2FromHF` → the batched decoders: **metal 51 tok/s, vulkan 59 tok/s** f32 greedy
decode (batched tokens bit-matching the analysis-path full forward on metal; note
the small-scale ordering INVERTS at d=768 — vulkan's decode kernels lead here). First at-scale figures for the
GPT pipeline.

Quantized decode at the same class (§T545/§T546, 124M-class Llama d=768/12L/GQA
12-4):

| 124M decode (tok/s) | f32 | Q8_0 | Q4_K |
|---------------------|----:|-----:|-----:|
| metal               | 76.0 | 57.3 | 66.4 |
| vulkan              | 61.8 | 70.6 | **72.4** |

Two findings: Q4_K outruns Q8_0 at this width on BOTH backends (less weight
traffic per token), and the backends INVERT on quant-vs-f32 — metal's MPS f32
matmuls are fast enough that dequant costs, vulkan's tiled kernels are
bandwidth-bound so quant WINS (Q4_K is the fastest 124M decode measured on
either backend). Weight memory: f32 ≈500MB → Q8 ≈130MB → Q4_K ≈70MB.
Language-quality numbers need real weights (download-gated).

## KV-cache memory tiers (§R108, §T619)

The key/value cache dominates long-context inference memory. GoAI offers two compressed
tiers beyond raw f32; both are pure-Go and drop into the decode path. Footprint measured at
dim=512 over 100 rows (keys + values), `TestTurboQuantMemoryHierarchy`:

| Cache                      | Bytes   | vs f32 | vs Q8_0 |
|----------------------------|---------|--------|---------|
| f32 (uncompressed)         | 409,600 | 1.0×   | —       |
| Q8_0 8-bit (`QuantKVCache`, §R108) | 108,800 | 3.8×   | 1.0×    |
| TurboQuant 2-bit (`TurboQuantKVCache`, §T619) | 41,600 | 9.8×   | 2.6×    |

Q8_0 is the near-lossless safe tier; TurboQuant is the extreme sub-4-bit tier
(arXiv:2504.19874) — a fixed random rotation plus a per-coordinate Lloyd-Max quantizer and a
1-bit QJL residual that keeps attention scores unbiased. Data-oblivious: no calibration, no
training. The scalar-per-row overhead (norm plus residual norm) amortizes as dim grows.

Throughput (`BenchmarkTurboQuant*`, dim=512): the fast Hadamard-Rademacher rotation (O(d log m)
via the in-place Walsh-Hadamard butterfly) replaces the naive dense O(d²) rotation, cutting
compress from 165µs to 60µs per row (2.8×) and reconstruct from 43µs to 23µs (1.9×). The isolated
rotation is 119× faster (`BenchmarkRotationHadamard` 2.5µs vs `BenchmarkRotationDense` 294µs); the
QJL sketch was then also moved to its fast-JL (Hadamard) form: the full chain is now Append 165→9.9µs (16.6×) and reconstruct 43→3.0µs (14.3×) vs the original dense implementation, both stages O(d log m), with unbiasedness re-verified.

## CUDA GPU inference vs llama.cpp (worker: linux/amd64 + RTX 3060, §PERF)

The `-tags cuda` backend runs a full TinyLlama-1.1B decoder resident on an NVIDIA
RTX 3060 (12 GB), built from scratch on nvrtc kernels + cuBLAS (no nvcc — the pip
CUDA wheels ship only ptxas). End-to-end it loads the GGUF, tokenizes, decodes on
device, and detokenizes real text (`TestCUDATinyLlamaGenerate{,Sampled}`).
Industry baseline: llama.cpp `llama-bench` on the **same** GGUF + GPU via its
prebuilt **Vulkan** build (a CUDA build is impossible here — no nvcc, no Linux
CUDA prebuilt; Vulkan on NVIDIA is a first-class backend). Reproduce the industry
side with `scripts/bench-llamacpp.sh`; the goai side with
`GOEXPERIMENT=simd go test -tags cuda -bench 'TinyLlamaDecode|TinyLlamaPrefill' ./backend/cuda/`.

| metric (TinyLlama-1.1B, RTX 3060) | llama.cpp Vulkan Q8_0 | goai CUDA | goai/llcpp |
|---|---|---|---|
| decode tg (Q8 + fixed-buffer + graph + on-device argmax) | 243 tok/s | **164.7 tok/s** | 0.68× (1.48× behind) |
| prefill pp32 (f32) | 3298 tok/s | 1330 tok/s | 0.40× |
| prefill pp128 (f32) | 8389 tok/s | 2353 tok/s | 0.28× |

**Decode optimization journey** (26 → 164.7 tok/s = 6.3×, every step correctness-
gated token-for-token vs the CPU reference; each bottleneck diagnosed by
measurement, not assumption):

| step | tok/s | lever / diagnosis |
|---|---|---|
| KV-cache decode (f32, per-op alloc) | 26 | O(1)/step cache; ≈250 `cudaMallocAsync`/token |
| launch-reduction fusions | ≈26 (paired-A/B +30%) | residual-into-matmul, out-of-place RMSNorm, fused attn-softmax |
| **fixed persistent buffers** | 69 | **decode was ALLOC-bound**, not launch-bound — kill per-token allocs (2.56×) |
| CUDA graph replay | 71 | +2% — decode already mostly GPU-bound after fixed buffers |
| **Q8 quantized weights** | 154→161 | **now MEMORY-bound** — int8 = 4× less weight bandwidth (2.3×); was −20% when alloc/launch-bound (masked) |
| on-device argmax | 164.7 | +2% — 4-byte token id, not the 128 KB logit vector; confirms GPU-bound |

Rejected with data (§C3): Q4_0 (correct kernel, 6.5% L1 = 16× Q8, but no speed win
post-Q8 — decode no longer weight-bandwidth-bound — and too lossy for 1.1B greedy);
gate-up projection fusion (−4%, strided-SwiGLU coalescing); GQA pointer-array
scratch (≈1%). Key CUDA-graph gotcha found by bisection: cuBLAS-batched attention
built its pointer arrays on the **host** and memcpy'd them — the shared host buffer
is overwritten across layers during capture, so every replayed layer read the last
layer's pointers; fixed by building the pointer arrays **on device**.

Takeaway: a from-scratch Go CUDA decode reaches **within 1.48× of a mature,
hand-tuned implementation** on the same GPU + model — competitive. The remaining
gap needs a better quantized kernel (Q4_K) or flash-style attention; prefill (less
optimized) trails more because llama.cpp fuses + batches attention.

### Unified llamagpu API vs the bespoke graph engine (worker linux-amd64)

The bespoke engine above is a hand-tuned test harness. The public, backend-agnostic
`llamagpu` decoder (one `Decoder` core shared by metal, vulkan and now CUDA — `Generate`,
`Step`, and every sampler for free) drives the CUDA `Recorder` instead. What does that
API uniformity cost in throughput? `TestCUDAUnifiedVsGraphDecodeThroughput` measures both
on the **same** TinyLlama-1.1B Q8 weights in one quiet window:

| CUDA decode path | tok/s | notes |
|---|---:|---|
| unified `llamagpu.NewQuantCUDA` (recorder, eager submit, no graph) | 44.4 | per-op launches + full-logit D2H + host argmax each token |
| bespoke (fixed buffers + CUDA graph + on-device argmax) | 160.6 | one graph replay + device argmax per token |
| **graph / unified** | **3.61×** | the launch-bound cost of the uniform API |

The unified path is launch-bound — hundreds of un-collapsed kernel launches per token
plus a per-token host round-trip (download the whole `[1, 32000]` logit vector, argmax on
the CPU). The bespoke path collapses the launches into one replayed CUDA graph and picks
the token on-device. This is consistent with the decode journey above: CUDA graph capture
is the dominant decode lever (≈2×) and the per-token host round-trip is most of the rest.

Takeaway: the uniform API buys correctness and the entire `Generate`/sampler surface for
free at ≈0.28× of peak — a legitimate trade. Recovering peak means wiring graph capture
into the shared `llamagpu` `Decoder` (a capture hook in `decoder.go`); the CUDA recorder
already submits eagerly onto one stream, which is the capturable shape.

### T631 CPU-offload viability (ADR-0021 measurement gate, worker linux-amd64)

To run a model that EXCEEDS device VRAM, T631 would offload the overflow layers to
the amd64 CPU-SIMD backend (llama.cpp-style partial offload). `TestT631OffloadViability
Probe` measures the per-token cost first (TinyLlama-1.1B, RTX 3060 vs the amd64 CPU):

| layers offloaded to CPU | decode tok/s | vs all-GPU |
|---|---|---|
| 0 (all GPU) | 164.7 | 100% |
| 1 | 30.1 | 18% |
| 2 | 16.5 | 10% |
| 4 | 8.7 | 5% |
| 8 | 4.5 | 3% |

Per layer: CPU-SIMD ≈27.5 ms vs GPU-Q8 ≈0.28 ms → **CPU ≈99× the GPU per layer**.
So even one offloaded layer becomes the bottleneck. VERDICT: CPU-SIMD offload is
FUNCTIONAL (runs >VRAM models with no hard OOM) but throughput collapses fast — T631
must spill the MINIMUM overflow layers, keeping the hottest ones on the GPU. Caveat:
this is the f64 `nlp.Llama` CPU path vs the Q8 GPU path; a Q8/f32 CPU offload path
would narrow the ratio (though offloaded layers would still dominate).

REFINED with the OPTIMIZED f32 SIMD GEMM (the offload path a real T631 would use,
not the f64 `nlp.Llama` path): CPU ≈8.2 ms/layer → **≈30× the GPU per layer** (vs
99× for f64 — the SIMD GEMM is ≈3× faster). Offload N=1 → 72 tok/s (44% of all-GPU),
N=2 → 46 (28%), N=4 → 27 (16%). So with the optimized f32/Q8 offload path, spilling
1-2 layers stays usable — **T631 is genuinely practical for models slightly over
VRAM**, not just a hard-failure fallback. (Matmul-only estimate; real T631 adds the
CPU attention/norm + the device↔host transfer at the split boundary.)

## amd64 SIMD decode-GEMV parallelism (worker linux-amd64)

The `GOEXPERIMENT=simd` F32 GEMM path (`gemm_simd.go`, archsimd/AVX) is amd64-only, so
these numbers can only be measured on the amd64 worker (the M2 dev machine is ARM and
cannot compile the path). The small-m branch (m ≤ 3 — the **decode GEMV** shape, one
token × the weight matrix) parallelizes over COLUMN blocks because there are too few rows
to split. A GEMV is memory-bandwidth-bound (each B element is read once, no reuse), so its
throughput scales with the number of cores streaming B — the column-block size must yield
≈one block **per worker**.

It previously used a fixed 512-column block, giving only `ceil(n/512)` blocks — e.g. **4**
for the common `n=2048`, leaving 12 of 16 cores idle. Sizing the block to ≈`n/workers`
(floored to a whole 32-wide tile, capped at 512 to keep each worker's `k×jblk×4B` B slice
L2/L3-friendly) puts every core to work:

| decode GEMV `[1,2048]·[2048,2048]`, 16 cores | GFLOP/s |
|---|---:|
| fixed 512-col blocks (4 workers) | 14.4 |
| adaptive ≈`n/workers` blocks (16 workers) | **27.5** (+90%) |

The fits-in-tile GEMM path (m ≥ 4) is untouched — `MatMul/512` stays ≈230 GFLOP/s,
`512×2048×8192` ≈200 GFLOP/s. The block ranges stay disjoint, so the result is
bit-identical (parity vs the reference holds at every n, `TestGemmSmallMCrossReference`).

## Further reading

- Hoefler & Belli, *Scientific Benchmarking of Parallel Computing Systems* (SC '15) — the canonical treatment of run variance, warm-up and honest reporting that this document's rules follow.
- Georges, Buytaert & Eeckhout, *Statistically Rigorous Java Performance Evaluation* (OOPSLA '07) — why repeated runs with variance beat single numbers, in any language.
- The Go blog, *Profiling Go Programs* and the `testing` package's benchmark docs — the mechanics behind every number here.
