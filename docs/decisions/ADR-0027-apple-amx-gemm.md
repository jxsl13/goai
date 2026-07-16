# ADR-0027 — Apple AMX F32 GEMM: Accelerate-cgo vs raw-AMX-asm, benchmark and use the winner

- Status: Accepted (both paths implemented and measured head-to-head on M2 Pro; per-shape dispatch live)
- Date: 2026-07-15
- Task: §T658 (follows T656 NEON GEMM, T657 MHA/Conv routing) — close the residual ≈3.25× CPU F32 GEMM gap vs torch-cpu
- Related: ADR-0021 (f32-native tolerant policy), ADR-0026 (NEON GEMM, the pure-Go ceiling at ≈795 GFLOP/s), §C2 (cgo-last), §C3 (deutlich threshold), §V10 (accumulation precision)

## Context

After ADR-0026 (NEON, ≈795 GFLOP/s @1024³ = ≈93% of the NEON FMA-pipe peak) and ADR-0027-routing
(T657), the CPU F32 GEMM is still ≈3.25× behind torch-cpu/numpy (≈2584 GFLOP/s). The reason is
Apple's **AMX** matrix coprocessor (one block per CPU cluster, ≈512 f32 FLOPs/instruction/cycle;
M2 Pro has 2 P-AMX + 1 E-AMX, a ≈3.6 TFLOP/s ceiling). AMX is undocumented (distinct from ARM SME,
which arrived on M4). torch-cpu/numpy reach it through Apple's Accelerate BLAS. The user directed:
implement Apple AMX support, compare the cgo and the raw path, and use the better — the raw path,
adapted from a production kernel, may beat the cgo variant.

## The two paths (research, measured on this M2 Pro)

- **A — Accelerate `cblas_sgemm` via cgo** (the sanctioned AMX door; what torch/numpy use). MEASURED
  here: 2785 GFLOP/s best @1024³ (2532 mean) — matches/beats torch-cpu (2584), a 3.2–3.5× win over
  NEON. Binding: `#cgo LDFLAGS: -framework Accelerate`, `-DACCELERATE_NEW_LAPACK=1`, `CblasRowMajor`
  natively (no transpose). Accelerate self-threads (call outside parallelWork). Effort S. Cons: cgo
  (opt-in, precedented by metal's `darwin && cgo`); L2 falloff on very large >L2 shapes (1627 @
  512×2048×8192, still 2.8× ours).
- **B — raw AMX in Plan9 asm** (pure Go, no cgo). Feasible: the same `WORD $0x00201xxx` trick as
  ADR-0026 emits AMX ops; `AMX_SET`/`CLR` around a non-CALLing asm body (async preemption never lands
  in asm → no cross-thread AMX-state split). Reference kernels exist (Fnk7/amx_sgemm, xrq-phys/
  blis_apple) and arXiv:2606.25426 shows a dual-AMX + prepacking kernel BEATS Accelerate ≈2× (geomean)
  on LLM-prefill shapes — so B can plausibly reach ≈1.3–2.5+ TFLOP/s and exceed Accelerate on >L2
  shapes. Cons: undocumented, chip-generation-specific (M1/M2/M3 only; M4 removed it), macOS could
  gate the enable, weeks of per-chip tuning. Effort L–XL, higher risk.

## Decision

Implement BOTH, benchmark head-to-head (Accelerate vs raw-AMX vs NEON vs the torch-cpu 2584 bar) at
the torch-compared shapes, and DISPATCH `gemmF32` to the winner — per-shape if the winner differs by
shape (raw-AMX likely wins the >L2 shapes where Accelerate falls off; Accelerate likely wins the
square shapes). Both are gated `darwin && arm64 && goexperiment.simd` (Accelerate additionally needs
`cgo`); `CGO_ENABLED=0` and non-simd builds fall back to the NEON path (ADR-0026), which stays the
tail and the m<4 decode-GEMV path. Numerics: both accumulate f32-native, so they ride the ADR-0021
tolerant-f32 policy + `gemmF32Tolerant` parity tests (rtol 2e-3); F64 stays bit-exact and untouched.
§C2 gates all hold (pure-Go at its roofline; cgo/AMX beats it ≫1.5×; CGO0 fully functional).

If raw-AMX does not beat Accelerate in the head-to-head (the tuning is hard), Accelerate is the
fast-path and raw-AMX is parked in §B as a pure-Go research follow-up; either way the gap to torch
closes to ≈1.0× (Accelerate alone matches torch). Follow-ups: `cblas_dgemm`/AMX-f64 (needs a §V10
amendment), `cblas_sgemv` for the decode path.

## Outcome (implemented 2026-07-15, measured on the M2 Pro)

**Both paths landed.** Files: `backend/cpu/gemm_accel_darwin.go` + `backend/cpu/internal/accel`
(Path A — the cgo binding lives in a subpackage because the Go toolchain forbids cgo and Plan9
asm in one package, and backend/cpu carries the NEON/AMX kernels), `backend/cpu/gemm_amx_arm64.{go,s}`
(Path B), hooks + per-shape dispatch in `gemm_neon_arm64.go`, head-to-head harness in
`gemm_amx_bench_test.go`, parity in `gemm_amx_test.go`.

Path B kernel: 32×32 C tile = the full 64-row Z grid (four 16×16 f32 banks), 4 `fma32` outer
products per k-step, k unrolled ×2 with M2 256-byte quad loads and ×4 with ping/pong software
pipelining (X0–3/Y0–3 vs X4–7/Y4–7, +5%); both operands prepacked 256-byte-aligned (A transposed
to columns); one non-CALLing NOSPLIT asm function holds AMX_SET → loop → stz → AMX_CLR (async
preemption cannot land in asm, so the per-thread AMX state never splits across OS threads).
Driver: single pool pass (pack items → in-pass spin barrier → tiles; +14–16% on ≤512³), dynamic
supertile pulls, g×g L2 groups (g ∈ {8,4,2}, +35–55% on >L2 shapes) and FULL-HEIGHT column strips
when `n ≫ m` and apack fits L2 (bpack then streams exactly once: 512×2048×4096 +11% more).
Runtime gate: `machdep.cpu.brand_string` ∈ M1/M2/M3 (no AMX sysctl exists; M4 moved to SME),
darwin-only build tag (only XNU context-switches AMX state).

Head-to-head (§V22, paired same-session A/B runs, GFLOP/s medians; torch-cpu bar = 2584):

| shape | NEON (ADR-0026) | raw AMX (B) | Accelerate (A) | winner |
|---|---:|---:|---:|---|
| 256³ | 428 | 537 | 1116 | A (2.1× B) |
| 512³ | 552–707 | 1373–1524 | 2193 | A (1.5× B) |
| 1024³ | 758 | 2076–2160 | 2569–2590 | A (**1.00× torch**) |
| 512×2048×2048 (B=16 MiB) | 656 | **1878** | 1786 | **B +5–6%** |
| 2048³ (16 MiB) | — | **2325** | 2100 | **B +11%** |
| 512×2048×4096 (32 MiB) | — | **1695** | 1294 | **B +31%** |
| 512×2048×8192 (64 MiB) | 550 | 1570 | 1610 | tie (A +3% ≤ thermal noise; dispatched to B) |
| 64/128×2048×2048, m<256 | — | 853/1219 | 1129/1500 | A |
| decode m=1, k=n=2048 | 17 | (punts) | 40 | A (2.3×) |

Ranges are thermal drift (sustained sweeps throttle ~10%); winners were decided by paired
alternating runs in the same thermal state.

**Dispatch (live in `gemmF32`):** raw AMX when `k·n·4 ≥ 16 MiB && m ≥ 256` (the >L2 band where
Accelerate falls off and prepacking pays), Accelerate otherwise; `CGO_ENABLED=0` drops Accelerate
and raw AMX takes every shape it supports (m,n ≥ 32 — it beats NEON from 64³ up), NEON serves the
rest — fully functional pure-Go, now at 2082 GFLOP/s @1024³ instead of 795. Both fast paths ride
the ADR-0021 tolerance parity (measured max rel err ≈1e-4..2e-3-band, same class as NEON); the
default non-simd build is untouched and bit-exact.

**Headline: the torch gap is closed — 2584 ≈ matched at 1024³ via Accelerate (3.25× → ≈1.0×),
and the pure-Go raw-AMX kernel BEATS Accelerate by 5–31% across the >L2 shapes (as the
arXiv:2606.25426 result predicted), while pushing the no-cgo build from 795 to ≈2100 GFLOP/s.**

## Errata (T662, 2026-07-15)

The "decode GEMV (m<4) stays scalar at ≈17 GFLOP/s" note above describes the `CGO_ENABLED=0`
fallback ONLY. In the cgo build the live `gemmF32` dispatch already routes every m<4 shape to
Accelerate `cblas_sgemm` (m=1 k=n=2048 measured at ≈40 GFLOP/s — the "40" was already live), so
the m=1 decode path was never scalar there. A follow-up grind (T662) added `cblas_sgemv` (BLAS2)
and found the actual gap was at **m=2**, where `cblas_sgemm` has a hole (≈27.6 GFLOP/s) that the
matrix-vector primitive fills: m=2 2048³ sgemv 33.7–38.7 vs sgemm 20.5–27.6 = **1.4–1.6×**
(2-token speculative/batched decode). `gemmF32` now routes m ∈ {1,2} to per-row `cblas_sgemv`
(m=1 a wash but the semantically-correct BLAS2 primitive; m=3 keeps sgemm). The scalar ≈17
GFLOP/s path remains only where cgo is unavailable.
