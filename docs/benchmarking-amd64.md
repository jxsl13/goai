# AMD64 CPU baseline — the GEMM floor for §T11b / §T74

> **In plain terms:** the GoAI perf history was measured almost entirely on
> an Apple M-series (arm64) laptop. Two GEMM tasks — §T11b and §T74 — were
> *parked* for one reason only: their payoff is an **x86 f32-SIMD
> microkernel** (AVX2/FMA via Go 1.26 `simd/archsimd` behind the
> `goexperiment.simd` tag), and there was no amd64 host to build or
> cross-verify it on. This page records the first measurement of the
> **pure-Go GEMM floor on an actual amd64 host** — the number the SIMD
> kernel must beat before it can land (§V22: measure the floor *before* the
> rewrite).

## Host

| | |
|---|---|
| CPU | AMD Ryzen 7 5700G (Zen 3, 8C/16T) |
| ISA (f32-relevant) | AVX2 + FMA — **no AVX-512** |
| Toolchain | Go 1.26.5, `linux/amd64`, gcc 16.1.1 |
| GPU | NVIDIA GeForce RTX 3060 (12 GB, driver 610.43.02); CUDA toolkit (`nvcc`) not yet installed |

This is the "ubuntu-amd64 runner class" that §T562 / §T542 recorded as the
precondition for graduating §T11b / §T74. It is a distinct machine from the
arm64 development host and works **only on dedicated branches via pull
requests**.

## Method (§V22)

Pure-Go, no C toolchain:

```sh
CGO_ENABLED=0 go test -run '^$' \
  -bench 'BenchmarkGEMM_F(32|64)_(512|1024)_gflops' \
  -benchtime 1s -count 3 ./backend/cpu/
```

Warm-up excluded by the `testing` framework; 3 samples per case; median
reported. The `_gflops` benchmarks report `2·M·N·K / ns` directly.

## Results — pure-Go GEMM floor (this host)

| kernel | 512³ (median) | 1024³ (median) |
|--------|---------------|----------------|
| F64    | 40.8 GFLOP/s  | 42.3 GFLOP/s   |
| F32    | 42.9 GFLOP/s  | 42.8 GFLOP/s   |

(Run-to-run spread ≤ 3% across the 3 samples in each cell.)

For reference, the arm64 M-series pure-Go ceiling recorded in the history
was ≈50 GFLOP/s (§T597 ceiling analysis) — so this Zen 3 host sits a little
below the M-series on the same scalar kernel, as expected for a desktop
Zen 3 core versus an M-series core on a bandwidth-friendly streaming loop.

## The finding that matters: F32 ≈ F64

The f32 kernel runs at **essentially the same GFLOP/s as f64** (≈43 vs ≈42).
The current pure-Go GEMM accumulates f32 products in f64 (§V10 precision
policy), so on a scalar core an f32 element costs the same as an f64 element
and **none of f32's 2×-wider SIMD-lane advantage is captured**.

This is exactly the headroom §T11b / §T74 exist to unlock:

- AVX2 packs **8 × f32** per vector vs **4 × f64**; with 2 FMA units a Zen 3
  core's f32 throughput ceiling is far above the scalar ~43 GFLOP/s measured
  here.
- The reference vendor-BLAS class (torch-cpu, f32 SIMD) was measured at
  ≈600 GFLOP/s on the arm64 host (§R237) — an order-of-magnitude gap that is
  a *pure-Go scalar-vs-SIMD* gap, not an algorithmic one (the blocking rung
  was already tried and found bandwidth-bound, §B41/ADR-0017).

## Floor to beat

**≈43 GFLOP/s (f32, 1024³, this host)** is the number the AVX2+FMA f32
microkernel (§T11b) must beat, with:

- bit-/tolerance-parity vs the ref f64 GEMM held (§V3/§V11), scalar fallback
  staying green under `CGO_ENABLED=0` / non-simd builds (§V7/§V23);
- an A/B delta measured on *this* host (§V22), no `git stash` toggling
  (§B52 process note — scratchpad file-copy toggle only).

## Vendor-BLAS gap on THIS silicon (resolved)

The ≈600 GFLOP/s figure was carried over from the arm64 host (§R237). Measured
directly here (`testdata/bench_torch.py` + numpy, same 2·N³/t methodology, warm-up
excluded, 30 reps), 1024³ f32/f64, GFLOP/s:

| GEMM | goai scalar | goai SIMD | torch-cpu | numpy (OpenBLAS) |
|------|-------------|-----------|-----------|------------------|
| F64 1024³ | 40 | **84** (nr=8) | 177 | 227 |
| F32 1024³ | 43 | 43 (scalar) | **580** | 485 |

Gap of best-goai vs best-incumbent on this Zen 3:

- **F64: ~2.7×** (84 vs numpy 227). goai's SIMD kernel is deliberately
  bit-exact — `Mul`+`Add`, *not* fused `MulAdd` — which alone costs ~2× of peak
  FMA throughput; the rest is cache blocking (vendor BLAS packs panels; goai
  streams, §B41). A bit-exact kernel cannot fully close this.
- **F32: ~13×** (44 vs torch 580). This is the headline gap and it is almost
  entirely *scalar-vs-SIMD*: goai's F32 GEMM is still the scalar kernel (the
  f64-accumulating SIMD twin regressed, see `simd-gemm-amd64.md`). An
  f32-*native* 8-wide kernel is the single biggest CPU-GEMM lever — but it
  trades §V10's f64 accumulation for a tolerance, so it needs an ADR / policy
  decision before it can land.

**Thread-count finding:** torch is *faster at 8 threads than 16* on this
8-core/16-thread part (f64 1024³: 177 → 134 GFLOP/s when forced to 16) —
GEMM is compute-bound, so SMT oversubscription just adds contention. goai's
`parallelWork` uses `GOMAXPROCS` (16 here), so it likely leaves the same
headroom on the table; capping GEMM parallelism at the physical-core count is
a cheap, bit-exact experiment worth running.

## Roadmap status (from this doc's original "next")

1. ✅ Same-host vendor-BLAS A/B (above).
2. ◐ F64 AVX SIMD landed (`simd-gemm-amd64.md`, ~2× scalar, bit-exact);
   f32-*native* SIMD still pending the §V10 policy decision.
3. ✅ CUDA toolkit + backend stood up on the RTX 3060 (`cuda-amd64.md`).
