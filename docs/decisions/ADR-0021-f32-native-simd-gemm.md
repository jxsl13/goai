# ADR-0021 — f32-native accumulation for the amd64 SIMD GEMM fast path

- Status: Accepted (measurement-driven, experiment-gated)
- Date: 2026-07-14
- Task: §T11b / §T74 (f32 SIMD GEMM), amends §V10 (accumulation precision)
- Related: ADR-0017 (cache blocking parked), `SPEC-worker-linux-amd64-cuda.md` §CPU-3,
  `SPEC-worker-linux-amd64-cuda.md` §GAP (the ≈13× F32 vendor gap that motivated this)

## Context

§V10 requires the `cpu` backend to accumulate f32 products in **f64**, so f32
matmul is bit-identical to the f64 reference (§V3/§V11, tol 0). That policy caps
any f32 GEMM at the f64 throughput: on this Zen 3 host the bit-exact f64-SIMD
kernel reaches ≈84 GFLOP/s while `cpu` **F32 GEMM stayed scalar at ≈43** (the
f64-accumulating F32 SIMD twin regressed 25× on the per-iteration f32→f64
convert — `SPEC-worker-linux-amd64-cuda.md` §CPU-3). Measured directly against vendor BLAS on this
silicon (`SPEC-worker-linux-amd64-cuda.md` §GAP), that left F32 GEMM **≈13× behind torch-cpu
(580 GFLOP/s)** — the single largest CPU-GEMM gap, and almost entirely
scalar-vs-SIMD.

Closing it requires **f32-native accumulation** (8-wide `Float32x8` + fused
`MulAdd`), which is exactly what vendor SGEMM does — and which is **not**
bit-identical to an f64-accumulating reference.

## Decision

Under the `amd64 && goexperiment.simd` build only, `cpu` **F32** matmul
(`gemmF32Band`, `backend/cpu/gemm_simd.go`) accumulates in **f32** using
`Float32x8.MulAdd`. This is a scoped, opt-in amendment to §V10:

- **Default build is unchanged.** Without the experiment, F32 matmul keeps the
  scalar f64-accumulating kernel and stays bit-exact vs ref. The experiment is
  opt-in (a perf build), never the `CGO_ENABLED=0` product default (§V7).
- **F64 is unchanged in both builds** — still f64-accumulated, bit-exact.
- Acceptance moves from tol-0 to a **tolerance** for F32-under-experiment
  (§V11/§V16): a relative `2e-3` + absolute `1e-4` bound vs the f64 reference.
  Observed max relative error over the parity shapes (K ≤ 128) is ≈1e-4 for
  genuine outputs (near-zero outputs are covered by the absolute floor) — three
  orders of magnitude inside the bound. The bound is far tighter than an f16
  path would need and comparable to what torch's own f32 GEMM differs from f64.

## Consequences

- **Win:** F32 GEMM 42.6 → 128.3 GFLOP/s (1024³) = **3.0×**, closing the vendor
  gap from ≈13× to ≈4.5×. Bit-exactness is a *default-build* guarantee; the
  experiment build trades it for the SIMD density the whole §T11b/§T74 line
  exists to capture.
- **Test policy:** F32 matmul parity is now build-tagged
  (`gemm_f32policy_{default,simd}_test.go` → `gemmF32Tolerant`); the two
  cross-reference tests assert exact on the default build and within-tolerance
  under the experiment via `assertMatMul`. F64 parity stays exact everywhere.
- **Blast radius (measured):** running the *entire* experiment-build test suite,
  the only F32-bit-exactness failures were those two `backend/cpu` tests
  (now tolerance-gated). `nn`, `nlp`, `autograd`, `linalg` do not assert
  bit-exact F32 matmul and are unaffected. CI builds — but does not test — the
  experiment, so this changes no CI-observed behavior.
- **Not for conv:** `gemmF64Band` (shared by conv's im2col path) is untouched
  and stays bit-exact; only the dedicated F32 matmul kernel changes.

## Alternatives rejected

- **f64-accumulating F32 SIMD** — regressed 25× (convert cliff), and even at
  best caps at the f64 rate (≈84), leaving most of the vendor gap open.
- **A separate opt-in op / runtime flag** instead of the build tag — more API
  surface for no benefit; the experiment tag is already the project's SIMD
  opt-in and keeps the default untouched.
- **Keeping §V10 tol-0 for F32** — leaves a measured ≈13× gap permanently on the
  table for the most common ML dtype; contrary to the §T11b/§T74 mandate.
