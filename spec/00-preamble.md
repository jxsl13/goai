# SPEC — GoAI

Go-native AI library, full spectrum, Pure-Go-first / cgo-last.
Encoding: caveman (see `FORMAT.md`). Target Go 1.26.
Source research: `docs/research/00-landscape.md`. Framing: `PLANNING_PROMPT.md`.

WORKER SUB-SPECS (hardware-capability-scoped, one per secondary machine; this root spec = cross-platform truth, sub-specs record host-specific SIMD/GPU work + measured A/B by capability):
- `SPEC-worker-linux-amd64-cuda.md` — Linux/amd64 (Zen 3 AVX2+FMA CPU-SIMD) + NVIDIA RTX 3060 (CUDA/cuBLAS). Owns §T11b/§T74 amd64-SIMD + the CUDA backend, PR-only. = THE single derived runner spec for that machine (consolidation 2026-07-15): hardware capabilities (§H/§Iw/§CPU/§GPU/§GAP/§PERF), task log (§Tw), runner protocol (§RUN — PR/merge/gates/GPU discipline, worker deltas over LOOP.md), local model assets (§MODELS — formerly models/README.md).
