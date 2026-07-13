# ADR-0013 — Tag-free Metal & zero-config accel registration (§T47)

Status: accepted (2026-07-06). Refines ADR-0009/0010/0012.

## Context

After auto-selection (ADR-0012) a user still had to pass `-tags metal` (and
blank-import the backend) to get GPU acceleration. The question: can the user pass
NO build tags at all? Build tags gate whether cgo backends compile — dropping them
naively would force cgo + every vendor SDK on every build, breaking `CGO_ENABLED=0`
(§V7) and generic cross-compiles. But the constraint differs per backend (§R47):

- **Metal/MPS/Foundation** are macOS *system frameworks*, present on every macOS.
  A cgo file guarded by `darwin && cgo` linking `-framework Metal` always compiles
  and links — no SDK, no tag. cgo is the default for native macOS builds.
- **CUDA cuBLAS** ships only in the CUDA Toolkit; link-time `-lcublas` breaks a
  generic Linux build. Needs an opt-in tag (or runtime dlopen).
- **Vulkan** links `-lvulkan`, also a build-time dep; but the loader
  `libvulkan.so.1` is explicitly designed for runtime dlopen (`vkGetInstanceProcAddr`)
  — a future tag-free path.

Confirmed vs Apple/NVIDIA/Khronos docs (§R47).

## Decision

1. **Metal is tag-free.** Its build constraint changes from `metal && darwin &&
   cgo` to `darwin && cgo`. A plain native `go build`/`go test` on macOS (cgo on
   by default) compiles it; `CGO_ENABLED=0` still excludes it (§V7 intact).
2. **Zero-config registration via companion files** in the top-level `goai`
   package: `register_darwin.go` (`//go:build cgo`) blank-imports Metal;
   `register_cuda.go`/`register_vulkan.go` (tag-gated) import those. So importing
   the library auto-registers every backend usable on this build, and — combined
   with preference auto-selection (ADR-0012) — `backend.Default()` returns the GPU
   with no tags and no selection code on macOS.
3. **CUDA/Vulkan stay tag-gated** because their link dependency is not guaranteed;
   runtime dlopen (libcuda.so / libvulkan.so.1) is recorded as the future path to
   make them tag-free too, without a link dependency.

## Consequences

- **+** On macOS the answer to "must I pass build tags?" is **no** — proven on an
  Apple M2 Pro: a plain `go test` auto-selects Metal and runs GPT inference AND
  training on the GPU. Metal is now host-verified (previously only tag-gated).
- **+** `CGO_ENABLED=0` pure-Go build unchanged (verified green ×15).
- **−** A native macOS `go build` now compiles the Metal cgo (needs the Xcode
  Command Line Tools' clang — always present when cgo works on macOS) and slightly
  longer build. Opt out with `CGO_ENABLED=0` for a pure-Go binary.
- **−** CUDA/Vulkan still need a tag until the dlopen rework — an honest asymmetry
  driven by their SDK/link model, not a design gap.

## Alternatives rejected

- **Drop tags for CUDA/Vulkan too (unconditional link):** breaks every build on a
  machine without the SDK. Rejected — only dlopen makes them safely tag-free.
- **A runtime micro-benchmark to decide inclusion:** inclusion is a compile-time
  fact, not runtime; irrelevant here. Rejected.
