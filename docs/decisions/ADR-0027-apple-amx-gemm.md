# ADR-0027 — Apple AMX F32 GEMM: Accelerate-cgo vs raw-AMX-asm, benchmark and use the winner

- Status: Proposed (research done, measured on M2 Pro; implementing both paths per user directive)
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
