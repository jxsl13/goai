# ADR-0010 — Portable Vulkan compute backend (§T43)

Status: accepted (2026-07-06). Extends ADR-0008 (GPU strategy); sibling of
ADR-0009 (CUDA).

## Context

Metal (Apple) and CUDA (NVIDIA) are vendor-locked. The user wants broad GPU/
accelerator support. Vulkan is the one vendor-neutral compute API: the same
SPIR-V shader runs on AMD, Intel, NVIDIA, and — via MoltenVK — Apple. It is the
portability play for the accelerator strategy.

Unlike cuBLAS, **Vulkan ships no BLAS** (§R39) — it is raw GPGPU. The GEMM must be
a hand-written compute shader. And, like CUDA, this host has no Vulkan SDK, so the
backend is not host-verifiable (§B36).

## Decision

1. **Mirror the Metal/CUDA structure.** `backend/vulkan` has a tag-free `doc.go`
   (keeps `CGO_ENABLED=0 go build ./...` green — package is doc-only without the
   tag) plus cgo files gated by `//go:build vulkan && cgo`. Same `Backend`/
   `device`/`Kernel`/`Available`/`init→Register` shape, so dispatch, autograd
   tape, and Pure-Go fallback (§I4) work unchanged. Serves inference AND training
   (the matmul VJP's two backward GEMMs dispatch here via `NewTapeOn`).

2. **Hand-written GLSL compute-shader GEMM.** `shaders/matmul.comp`: row-major
   f32 C=A·B, one invocation per output element, 16×16 local workgroup, dims via
   push constants, three std430 storage buffers. The host orchestration
   (instance → compute queue → host-visible|coherent buffers → 3-binding
   descriptor set → compute pipeline → dispatch ceil(N/16)×ceil(M/16) →
   queueWaitIdle) follows the Vulkan spec, confirmed vs the spec + compute
   tutorials (§R44).

3. **SPIR-V is a build artifact.** `matmul.spv` is produced from `matmul.comp` by
   `make vulkan-spv` (glslc, Vulkan SDK) and embedded via `//go:embed`. Building
   `-tags vulkan` therefore requires running that target first — exactly as
   `-tags cuda` requires the CUDA toolkit. Committing a generated binary was
   rejected in favor of committing the readable GLSL source + the codegen step.

4. **CI-gated verification (§B36).** The §V3 cross-reference (shader == ref within
   rtol(K)=1e-6·√K, §V11), an unaligned-rectangular guard on the dispatch-overhang
   mask and row-major indexing, the fwd+bwd GPU-training test, and the §C3 gate
   benchmarks live behind the tag and run on Vulkan-capable CI runners.

## Consequences

- **+** One portable backend covers every major GPU vendor; zero impact on the
  pure-Go default (verified: `CGO_ENABLED=0` green ×14, cross-compiles, gofmt-clean).
- **+** The naive shader is simple and its correctness is guarded by the
  unaligned cross-reference test — the dispatch-overhang mask is the one subtle
  risk and is exercised directly.
- **−** Not verifiable on the dev host; correctness of the cgo/Vulkan layer rests
  on the spec-confirmed sequence (§R44) + CI until a Vulkan runner exists.
- **−** The naive one-element-per-invocation kernel is far from peak (no tiling/
  shared memory) and per-call buffer setup dominates small GEMMs; it will not
  clear the §C3 gate yet. Shared-memory tiling + device-resident tensors are the
  follow-up before any cgo-gate merge decision.

## Alternatives rejected

- **Runtime GLSL→SPIR-V (shaderc/glslang):** adds a heavy cgo dependency just to
  avoid a build step. Rejected — glslc-at-build is the standard, lighter path.
- **Compute via WebGPU/wgpu instead:** higher-level but immature from Go and still
  cgo; Vulkan is the lower, more portable substrate. Deferred, not chosen now.
