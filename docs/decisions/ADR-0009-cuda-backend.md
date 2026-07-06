# ADR-0009 — CUDA/cuBLAS backend (§T42)

Status: accepted (2026-07-06). Extends ADR-0008 (GPU strategy) to NVIDIA.

## Context

The user's top priority is GPU/accelerator support for **both training and
inference**. ADR-0008 established the strategy: offload compute-bound GEMM to the
GPU behind the backend interface, everything else falls back to Pure-Go (§I4).
The Metal backend (§T20/§T30) proved the pattern on Apple Silicon. NVIDIA CUDA is
the dominant training/inference accelerator and must be served the same way.

CUDA has **no pure-Go path** (proprietary driver + cuBLAS C ABI, §R38) → it is
necessarily a cgo, build-tag-gated backend. This host is arm64/macOS with no
NVIDIA GPU or CUDA toolkit, so the backend is **not host-verifiable** (§B35),
exactly like the amd64 archsimd work (§B13).

## Decision

1. **Mirror the Metal backend structure.** `backend/cuda` has a tag-free
   `doc.go` (so `go build ./...` and every `CGO_ENABLED=0` platform stay green —
   the package is just documentation without the tag) plus cgo files gated by
   `//go:build cuda && cgo && (linux || windows)`. Same `Backend`/`device`/
   `Kernel`/`Available`/`init→Register` shape as `metal.go`, so the dispatch
   layer, autograd tape (`NewTapeOn`), and fallback all work unchanged.

2. **cuBLAS SGEMM for f32 MatMul only** (fwd + both backward GEMMs of the matmul
   VJP → serves training AND inference). Row-major C=A·B is computed via the
   column-major idiom C^T=B^T·A^T: `cublasSgemm(h,OP_N,OP_N,N,M,K,&α,B,N,A,K,&β,
   C,N)` — operands B then A, dims N,M,K, leading dims N,K,N. Confirmed vs NVIDIA
   cuBLAS docs (§R43). All other ops fall back to Pure-Go (§I4).

3. **Synchronous, host-resident tensors** (copy H2D → sgemm → D2H → sync per
   call). Honest about transfer cost; async batching and device-resident tensors
   are a later optimization the stable interface (§V14) admits without a break.

4. **CI-gated verification (§B35).** The §V3 cross-reference (cuBLAS == ref within
   rtol(K)=1e-6·√K, §V11), the rectangular-shape guard on the row-major idiom, the
   fwd+bwd GPU-training test, and the §C3 gate benchmarks (cuda vs ceiling
   Pure-Go cpu) all live behind the tag and run on Linux/Windows CUDA runners.
   The cgo gate (§C2/§C3) — merge only if the benchmark beats optimized Pure-Go —
   is evaluated there, not on this host.

## Consequences

- **+** NVIDIA training+inference acceleration behind the same interface; zero
  impact on the pure-Go default build (verified: `CGO_ENABLED=0` green ×14,
  cross-compiles).
- **+** The row-major idiom is guarded by a cross-reference test that fails
  loudly on any operand/dim/ld mistake — the one subtle correctness risk.
- **−** Not verifiable on the dev host; correctness of the cgo/cuBLAS layer rests
  on the NVIDIA-doc-confirmed idiom (§R43) + CI until a CUDA runner exists.
- **−** Per-call H2D/D2H copies dominate small GEMMs; only large compute-bound
  GEMMs will clear the §C3 gate. Device-resident tensors are the follow-up.

## Alternatives rejected

- **Pure-Go CUDA (PTX generation / driver ioctl):** no maintained path, enormous
  surface, no cuBLAS-grade kernels. Rejected — cgo is the only realistic route.
- **Compile the cgo files unconditionally:** would break `CGO_ENABLED=0` and
  every non-CUDA platform. Rejected — the tag-gate + doc-only package is
  mandatory (§V7).
