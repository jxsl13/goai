# AMD64 archsimd GEMM — the F64 microkernel (§T11b / §T74)

> **In plain terms:** the matrix-multiply inner kernel is where GoAI spends
> most of its CPU time. This page records the first SIMD (AVX) version of that
> kernel on the amd64 worker, what it sped up, and — just as important — the
> one variant that made it *slower* and was thrown away.

Builds on `simd-amd64.md` (the elementwise landing) and `benchmarking-amd64.md`
(the ~43 GFLOP/s scalar floor).

## Design — vectorize the free dimension, stay bit-exact

`backend/cpu/gemm_simd.go` (`//go:build amd64 && goexperiment.simd`) replaces
the scalar band kernels of `gemm_nosimd.go`. The key idea that keeps it
**bit-identical** to the reference (§V3/§V11, tol 0) rather than merely
within-tolerance:

- **Vectorize the free dimension `j`** (4-wide `Float64x4`), never the
  reduction `k`. Each `C[i][j]` still sums its k-products in ascending-p order,
  the same order as the scalar/ref kernel → same rounding, same bits.
- **`Mul` then `Add`, never `MulAdd`.** A fused multiply-add rounds once;
  the scalar `c[j] += a*b` rounds twice. Using separate `Mul`+`Add` reproduces
  the two-rounding result exactly. (FMA is reserved for the future f32-native
  kernel, which accepts a tolerance.)
- **Load the accumulator from C before the p-loop, store after.** This
  preserves the band kernel's `C += A·B` contract verbatim — conv's im2col
  scatter (`gemmF64Band` is shared, §T597) passes zeroed buffers and relies on
  it. 4-row register blocking is kept; a scalar column-tail (`n%4`) and
  row-remainder finish the edges. `archsimd.X86.AVX()` runtime-gates the
  intrinsics (§I4).

Bit-exactness is verified by the existing cpu-vs-ref suites running under the
experiment build (`TestGemmCrossReferenceExact`, `TestConvCrossReferenceExact`,
`TestConv2DBackwardCrossReference`) plus `TestGemmSimdTailResiduesExact`, which
nails every column residue 0..3 and non-4-multiple `m`.

## A/B (§V22) — Ryzen 7 5700G (Zen 3, AVX2+FMA), count=3 medians

| GEMM | scalar | archsimd nr=4 | archsimd nr=8 | scalar→nr8 |
|------|--------|---------------|---------------|------------|
| **F64** 512³  | 40.8 GFLOP/s | 63.5 | **80.5** | **1.97×** |
| **F64** 1024³ | 42.3 GFLOP/s | 62.4 | **82.3** | **1.95×** |
| F32 512³  | 43.1 GFLOP/s | 43.1 | 43.1 | 1.00× (scalar, unchanged) |
| F32 1024³ | 43.1 GFLOP/s | 43.1 | 43.1 | 1.00× (scalar, unchanged) |

**Register-blocking tune (nr=4 → nr=8).** The first cut kept one `Float64x4`
accumulator per row (4 live chains). At ~4 GFLOP/s/core that was well under the
Zen 3 f64 ceiling, and the tell was that it scaled with *more independent
accumulators*, not more bandwidth: the `acc = acc.Add(a·b)` chain is
latency-bound (each Add depends on the previous), and 4 chains barely cover the
~4-cycle add latency. Widening to **two column groups per row = 8 chains**
(`nv8 = n − n%8`, a 4-wide cleanup for `n%8∈[4,7]`, scalar tail for `n%4`) gives
enough ILP to hide it: **+28–32%** on top of nr=4, a **~1.96× cumulative**
bit-exact win over scalar (same ascending-p order → tol 0, parity suites green).
8 accumulators + 2 B-loads + 4 A-broadcasts sit within the 16 YMM registers.

Because `gemmF64Band` is shared, **F64 conv (im2col→GEMM) inherits the same
~2×** at bit-exact parity (its cross-ref suites stay green under the experiment
build); a dedicated conv A/B is a follow-up.

## The discarded F32 SIMD variant (§C3/V-CGO)

The natural f64-accumulating F32 SIMD twin — per inner iteration
`LoadFloat32x4Slice(B).ConvertToFloat64()` then `Float64x4` `Mul`/`Add` — was
built and measured, and **regressed ≈25×** (512³/1024³: ≈43 → ≈1.7 GFLOP/s).
The 128-bit f32 load + `ConvertToFloat64` widen in the hot loop is pathological
on this path. Per §C3 (never ship a non-winning opt) it was **discarded**; F32
GEMM keeps the blocked scalar kernel under the experiment too — unchanged and
still bit-exact, no regression.

## Why F32 is a separate, bigger task

f64 accumulation (§V10) caps *any* f32 kernel at the f64 throughput (~63
GFLOP/s here) — an order of magnitude short of vendor-BLAS SGEMM (~600). The
real f32 win needs **f32-native 8-wide accumulation** (`Float32x8` + `MulAdd`),
which:

1. changes the §V10 f64-accumulation policy → needs an ADR + a **tolerance**
   parity gate (f32 accumulation error ≈ √K·ε_f32), not tol-0;
2. can use FMA (`MulAdd`) since bit-exactness is no longer the contract.

That is the next task and where the headline speedup lives. This F64 landing
proves the bit-exact SIMD-GEMM structure (free-dim vectorization, `+=`-
preserving accumulator, tail handling) it will build on.
