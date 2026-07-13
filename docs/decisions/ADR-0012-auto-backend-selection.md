# ADR-0012 — Automatic backend selection by performance preference (§T46)

Status: accepted (2026-07-06). Builds on ADR-0008..0011 (the accelerator arc).

## Context

The accelerator backends exist (cpu-SIMD, Metal, CUDA, Vulkan), but using one
meant the caller explicitly did `backend.Get("vulkan")` + `WithBackend`. The user
asked for the opposite default: do as little as possible to run on a GPU/accel —
ideally nothing. Either auto-detect, or let the user give an ordered preference,
with a sensible default ordered by descending performance (GPU → … → CPU),
validated by research + benchmarks; cgo should be the only real gating.

Prior `Default()` was "last `RegisterDefault` wins" — import-order dependent and
never selected a GPU (accel backends only call `Register`, not `RegisterDefault`).

## Evidence (§R46)

- **ONNX Runtime**: an ordered execution-provider list, tried per-op, CPU EP the
  guaranteed final fallback.
- **PyTorch idiom**: `cuda if available else mps else cpu` — the library/user
  picks; there is no automatic global default beyond CPU.
- **ggml/llama.cpp**: a backend registry with a per-device score; GPU preferred,
  CPU always registered as fallback.
- **On-host benchmark** (Apple M2 Pro / MoltenVK): the Vulkan matmul beat CPU-SIMD
  **1.47× at 512³ and 3.38× at 1024³** — GPU-first is empirically justified, not
  aspirational.

## Decision

1. **Preference-ordered auto-selection.** The registry holds a descending-
   performance order, default `["cuda", "metal", "vulkan", "cpu"]`. `Default()`
   returns the first REGISTERED backend in that order, else the reference — the
   guaranteed final fallback, always present.
2. **Registration IS detection.** Build-tagged accel backends already call
   `Register` from `init()` only when their device is present (`Available()`), so
   the registry reflects exactly what can run here (§V4 — no silent claim of an
   absent device). cgo/build tags gate which backends compile in; everything else
   is detected at runtime (§C11).
3. **Zero-config for the user.** `NewContext` and `autograd.NewTape` call
   `Default()`, so merely building with an accel tag routes real work (matmuls,
   and both backward GEMMs of the matmul VJP → training too) to the GPU with no
   code change. Proven on-host: `-tags vulkan` makes `Default()` return `vulkan`
   and training run on the Apple GPU.
4. **Overridable.** `SetPreference(order...)` replaces the order (unknown names
   are skipped safely, so listing every backend you might build with is fine);
   `Context.WithBackend` still forces one backend per call. `RegisterDefault` is
   kept (§V8) and now appends its backend to the preference so a custom optimized
   backend still beats the bare reference, without displacing GPU preference.

## Consequences

- **+** The library "just uses the GPU" when built for it — the requested UX. No
  regression to the pure-Go build (`CGO_ENABLED=0` green ×15; the default order's
  GPU names simply never register).
- **+** Selection is deterministic and benchmark-justified, not import-order luck.
- **−** The default assumes GPU > CPU for the compute-bound matmuls we offload;
  for tiny matmuls transfer overhead can make a GPU slower. Mitigation: only
  compute-bound ops are offloaded (everything else falls back per §I4), and the
  order is overridable. A future per-op/size-aware cost model (like ggml's score)
  can refine this without an API change.
- **−** NPUs are absent from the order (no op-level backend, §R45/ADR-0011); they
  would slot below GPU and inference-only if a model-level runner ever lands.

## Alternatives rejected

- **Whole-machine probe + micro-benchmark at startup to rank backends**: slower
  start, nondeterministic, overkill — the static order matches every major
  framework and is overridable. Rejected for now (revisit with a score model).
- **Keep manual selection only**: fails the user's core "do as little as possible"
  requirement. Rejected.
