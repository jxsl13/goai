---
name: linux-amd64-worker-role
description: "This machine's role in the goai project — Linux/amd64+NVIDIA secondary worker, PR-only, runs LOOP.md every 1m"
metadata: 
  node_type: memory
  type: project
  originSessionId: 491e34f2-de40-467f-89a0-f918efb8a6b3
---

This machine is a **secondary** goai build worker (user directive, 2026-07-14). The primary dev machine is a MacBook Pro M2 developing on `main`. This one is Linux/amd64 (AMD Ryzen 7 5700G, Zen 3, AVX2+FMA no AVX-512, 16T; Go 1.26.5; gcc 16) with a **dedicated NVIDIA RTX 3060** (driver present, CUDA toolkit/`nvcc` NOT yet installed). `brew` available for deps.

**Working rules:**
- Work **SOLELY on dedicated branches via pull requests** — never commit to `main`. This overrides LOOP.md's "no push without permission" default; PR delivery IS the granted permission.
- Push via `gh` — see [[gh-for-pushing]].
- A `*/1 * * * *` cron runs the LOOP.md per-iteration procedure (one task per fire). Git identity set repo-local: John Behm / john.behm@googlemail.com.
- Honor §C16 push throttle (≤1/hour) and §T567 (no machine-local info in commits: no abs paths/hostnames/usernames).

**This worker's high-value niche** (things the arm64 Mac could NOT do): amd64 f32-SIMD GEMM (§T11b/§T74, parked purely for lack of an amd64 host) and the CUDA backend (`register_cuda.go`, first NVIDIA GPU available).

**Progress:**
- Fire 1, **PR #1** `linux-amd64/baseline-audit` (docs): pure-Go sweep green (22 ok/0 FAIL, CGO0); amd64 GEMM floor F32≈F64≈43 GFLOP/s @1024³ (scalar f32 captures none of its 2× SIMD density) → `docs/benchmarking-amd64.md`.
- Fire 2, **PR #2** `linux-amd64/simd-f32-elementwise` (code): first real archsimd kernels — `internal/simd` Add/Sub/Mul/Div F32/F64 AVX overrides behind `amd64 && goexperiment.simd` (ADR-0005 slice of §T11b). Bit-exact parity; A/B **F32 2.2×** (26→60 GB/s, now bandwidth-bound), F64 1.15×. Proven pattern: `simd/archsimd` (Go 1.26.5) works here — `Float32x8`/`Float64x4`, `.MulAdd` (FMA), `archsimd.X86.AVX()/.FMA()` gates; CI gate = `GOEXPERIMENT=simd go build -tags=simd ./...`.

**WORKER SUB-SPEC (PR #10, user /spec directive 2026-07-14):** this worker now has a DEDICATED caveman spec `SPEC-worker-linux-amd64-cuda.md` at repo root, referenced from main SPEC.md's "WORKER SUB-SPECS" block near the top. It consolidates the former `docs/{benchmarking,simd,simd-gemm,cuda}-amd64.md` (now DELETED) — sections §H hardware, §Iw invariants, §CPU/§GPU results, §GAP vendor gaps, §Tw task log, §NEXT. **Future fires: record worker results/tasks in THIS sub-spec (§CPU/§GPU/§Tw), not scattered docs.** SPEC.md edits still main-machine-owned EXCEPT the sanctioned worker-sub-spec reference.

**Coordination:** do NOT edit main SPEC.md §G/§C/§I/§V/§T / CHANGELOG.md on PR branches (main machine owns them). Own doc = the worker sub-spec + ADRs. Branch per task off `origin/main` (independent PRs). Push: `git push` (gh credential helper) then `gh pr create`.

- Fire 3, **PR #3** `linux-amd64/simd-gemm-f64` (code): archsimd **F64 GEMM microkernel 1.5×** bit-exact (`backend/cpu/gemm_simd.go`). Key bit-exact tricks: vectorize the FREE dim j (`Float64x4`) not the reduction k; `Mul`+`Add` NOT `MulAdd` (FMA fuses to 1 rounding, scalar does 2); load/store accumulator to preserve the `C += A·B` contract (conv shares `gemmF64Band`). F64 conv inherits ≈1.5×. **F32 SIMD DISCARDED** (§C3): f64-accumulating twin with per-iter `LoadFloat32x4Slice`+`ConvertToFloat64` regressed ≈25× (pathological 128-bit load+widen in hot loop).

**KEY LESSON (archsimd perf):** never put a 128-bit `LoadFloat32x4Slice`+`ConvertToFloat64` in a GEMM hot loop — ~25× regression. Pure-256-bit-YMM paths (`Float64x4` only) are fast.

- Fire 6 PR #5 (merged): CUDA device-buffer pool 1.24×/1.29×. Fire 7 **PR #6 (merged)**: F64 GEMM **nr=4→nr=8** register blocking (2 accumulators/row = 8 ILP chains; was latency-bound) → 512 63→80.5, 1024 62→82 GFLOP/s = **~2.0× cumulative over scalar**, bit-exact, conv inherits it. Auto-merge now active (watch CI green → `gh pr merge --merge`, see [[auto-merge-prs]]).

- Fire 8 **PR #7 (merged)**: vendor-BLAS gap on this Zen3 (torch-cpu + numpy via `.venv`, `testdata/bench_torch.py`). 1024³: F64 goai 84 vs numpy 227 (≈2.7×); **F32 goai 43 vs torch 580 (≈13×!)**. `docs/benchmarking-amd64.md`.

**KEY FINDINGS (fire 8):** (1) F32 GEMM gap is 13× (scalar-vs-SIMD) → f32-native SIMD is the biggest CPU lever. (2) F64 gap ≈2.7× partly from bit-exact Mul+Add (not FMA, ≈2× of peak) + vendor cache blocking. (3) **torch is FASTER at 8 threads than 16 on this 8c/16t part** (SMT contention on compute-bound GEMM) → goai `parallelWork` uses GOMAXPROCS=16 → **capping GEMM parallelism at physical cores is a cheap BIT-EXACT experiment worth trying** (candidate next fire, low-risk auto-merge-safe).

- Fire 9 **PR #8 (merged)**: **f32-NATIVE SIMD GEMM 3.0×** (42.6→128.3 GFLOP/s, closes vendor gap 13×→4.5×). `Float32x8`+`MulAdd`, widen-to-f64-carrier ONCE per tile on store (no per-p convert). **ADR-0021 amends §V10** (experiment-only; default build stays bit-exact f64-accum). Blast radius MEASURED tiny: only 2 backend/cpu tests (now build-tagged `assertMatMul`+`gemmF32Tolerant`, exact default / rel-2e-3 tolerance experiment); nn/nlp/autograd DON'T assert F32-exact matmul. FYI: user should mirror the §V10 amendment into SPEC (I can't edit SPEC).

**MEASURED FACTS (reuse):** (1) my earlier F32-blast-radius fear was WRONG — nn/nlp use tolerance for F32, so f32-native only broke 2 cpu gemm tests. (2) GOMAXPROCS=8 SLOWER than 16 for goai GEMM (SMT hides stalls; unlike torch) → don't cap. (3) LOCAL ENV false test failures under any build: `internal/mdlint` TestRepoMarkdownIsClean scans `.venv*/**/*.md` (my pip venvs) → fails locally, passes in CI; `ops`/`format/npy` `*VsNumpy` tests fail with numpy 2.4.4 in `.venv`. Run mdlint on specific files, not `./...`.

**Authoritative task log = the worker sub-spec `SPEC-worker-linux-amd64-cuda.md` §Tw** (not this memory). As of PR #11: F32 GEMM at nr=16 = 153 GFLOP/s (3.6× scalar, vendor gap 3.8×).

**Next options (see sub-spec §NEXT):** Nx1 CUDA activation residency (§V14/ADR-0019 full — big, user priority); Nx2 remaining = cache blocking (ADR-0017 re-open this large-cache x86) + FMA-saturation microkernel; Nx3 batched cuBLAS Sgemm for attention. `.venv`=numpy+torch-cpu, `.venv-cuda`=CUDA 12.9.

**§V41 PICKUP (2026-07-20):** the spec corpus moved to a `spec/` source tree; `SPEC-worker-linux-amd64-cuda.md` is now a GENERATED view. After your next `git pull` of main: book §Tw rows via `go run ./internal/specgraph task add -worker linux-amd64-cuda -cites <ids> "<text>"`, edit prose sections in `spec/worker/linux-amd64-cuda/` + `make spec-render`, add `go run ./internal/specgraph verify` to your RUN4 gates (see RUN9 in §RUN). Hand-edits to the rendered file are warn-only for now and will turn CI-red once workerRenderSyncHard flips.
