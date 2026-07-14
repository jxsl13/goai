# GoAI — Landscape & Feasibility Report (Bootstrap Phase 0)

> Created: 2026-07-05 · Target Go version: 1.26 · Policy: cgo-last (Pure Go first)
> Status: Iteration 1 of the autonomous loop. Sources below. `?` = not yet confirmed,
> to be hardened in Phase 1 (`/research`).
>
> Note on methodology: The `deep-research` workflow failed due to a
> harness schema error (StructuredOutput retry cap). Per the autonomy rule, work
> was rerouted to direct, targeted WebSearch verification of the version-sensitive
> facts + domain knowledge. Claims not adversarially cross-checked are marked
> with `?`.

---

## 1. State of the Art in Go ML — and the Gap

| Project | Maturity / maintenance (2025–26) | Acceleration | Assessment |
|---|---|---|---|
| **GoMLX** | Most active Go ML stack, v0.26.0 (Dec 2025) | OpenXLA JIT (CPU/GPU/TPU) **via `gopjrt` = cgo**; optional pure-Go backend (unoptimized), WASM | Strong, but peak performance depends on C++ XLA; the pure-Go path is only a fallback |
| **Gorgonia** | Largely **dormant/legacy**, functional | Graph-based, partly cgo (CUDA) | No longer a viable foundation |
| **gonum** | Active, solid numerics | Pure-Go **BLAS incomplete**, float32/float64 only; optional cgo→OpenBLAS | Good basis for numerics building blocks, but no DL/autograd/GPU |
| **tract (Rust)** | Reference for ONNX inference | — | Comparison benchmark, not Go |

**Consequence (the gap that justifies building anew):** There is no
Go-native, **pure-Go-first / cgo-last** full-spectrum stack that (a) fully
exploits the new Go 1.26 `simd` package as the primary accelerator, (b) guarantees
a reference-valid pure-Go ground truth with numerical parity, and (c) treats
cgo/GPU only as an optional, benchmark-triggered add-on. GoMLX inverts the priority
(XLA/cgo first); Gorgonia is dead; gonum covers only numerics. → GoAI occupies
exactly this niche.

## 2. CPU SIMD in Pure Go (the core of the pure-Go ceiling)

- **`simd/archsimd`** (Go 1.26, `GOEXPERIMENT=simd`, released Feb 2026): for the
  first time, explicit SIMD intrinsics **without cgo and without hand-written asm
  stubs**. Vector types as structs (`Int8x16`, `Float64x8`, …), 128/256/512-bit,
  **AVX2 + AVX-512**. **Currently AMD64-only.** The API always generates the AVX
  form for now; `X86` variable for feature detection (AVX2/AVX512). Source:
  golang/go #73787, go1.26 release notes, pkg.go.dev/simd/archsimd. `?` API
  stability (experimental, subject to change).
- **Performance ranking (Oct 2025, Callista benchmark):** `simd` package **inlined
  ≈4× faster** than the next-best solution and ≈16× vs. a plain Go loop; `avo`
  ≈3× vs. plain loop (cannot inline due to `.s` stub); `simd` (non-inlined)
  ≈30% above avo. → **Order of preference:** `simd` package > `avo` > Plan9 asm >
  auto-vectorization.
- **ARM64 (Apple Silicon, the primary developer target here):** the `simd` package
  does **not yet** cover ARM64/NEON → there, pure Go via **Plan9 asm (NEON)** or
  an `avo` equivalent, or fallback. `?` Timeline for ARM64 in the `simd` package.
- **Realistic pure-Go ceiling:** For BLAS-1/2 (elementwise, dot, axpy) and
  well-tileable GEMM, a **substantial fraction of OpenBLAS/oneDNN** is achievable
  with `simd`+blocking+goroutines; the exact percentage is op- and
  hardware-dependent and is measured per kernel (sets the §C cgo threshold).
- **Further building blocks without cgo:** `math/bits`, `segmentio/asm`, `go-highway`
  (portable SIMD abstraction with pure-Go fallback). `?` Maturity of go-highway.

## 3. GPU Paths from Go — all effectively require cgo

| Path | cgo required | Platform | Note |
|---|---|---|---|
| CUDA / cuBLAS / cuDNN | **Yes** (C libs) | Linux, Windows | Highest peak perf for NVIDIA |
| Metal | **Yes** (Obj-C/cgo) | macOS | Mandatory for Apple GPU/ANE proximity |
| Vulkan compute | **Yes** (loader is C) | portable | One backend for many GPUs |
| WebGPU / wgpu | **Yes** (wgpu = Rust, via C ABI) | portable/WASM | Promising, young |
| ROCm / HIP | **Yes** | Linux | AMD |

**Finding:** There is **no practicable pure-Go path to discrete GPU compute.**
This is not a contradiction of the policy but its core: GPU is exactly the class
where cgo earns its place **after** the pure-Go CPU ceiling has been exhausted —
as an optional build-tag backend with a pure-Go fallback. `?` Maturity of
individual Go bindings (Metal cgo, vulkan-go) to be checked in Phase 1.

## 4. NPU / Accelerator — mostly an honest non-goal (for now)

- **Apple Neural Engine (ANE):** only addressable indirectly via **CoreML**
  (cgo/Obj-C); no direct access. Realistic as a late optional backend.
- **Windows DirectML:** via cgo/COM. Feasible, but effort.
- **Intel oneDNN (incl. NPU paths):** cgo.
- **Recommendation:** mark NPU as an **explicit non-goal of the first expansion
  stage** (no silent promise); re-evaluate after GPU maturity.

## 5. Reference Baselines for Numerical Parity

- **BLAS/GEMM:** OpenBLAS / Eigen as perf baseline; NumPy (`@`) as correctness
  golden. Tolerance f64: rtol≈1e-12, f32: rtol≈1e-5 (`?` fix per op).
- **DL ops (Conv/Norm/Attention/Optimizer):** PyTorch/ATen as golden source;
  values reproducible via a small Python script (torch, fixed seed) exported to
  `testdata/golden/*.npy` → loaded in Go via npy reader.
- **LLM inference:** llama.cpp/ggml as perf and bit reference (quantization).
- **Golden generation:** deterministic (seed, dtype, shape documented),
  checked in; Python 3.14 + NumPy/torch available locally.

## 6. Model Interop Formats

| Format | Pure-Go effort | Priority |
|---|---|---|
| **safetensors** | Low (JSON header + raw tensors, zero-copy) | **First** |
| **GGUF** | Medium (header + quant blocks; pure-Go readers exist) | For LLM inference |
| **ONNX** | High (protobuf schema + large opset) | Later, incrementally by opset |
| HuggingFace | = safetensors + tokenizer/config | With the NLP layer |

## 7. Verification Methodology

- **V-PARITY:** golden tests against NumPy/torch within fixed tolerances.
- **V-GRAD:** numerical gradient check (central finite differences,
  threshold ≈1e-4 rel.) for every differentiable op.
- **V-PROP:** property-based tests (shape algebra, linearity, associativity where
  mathematically guaranteed) via `testing/quick` or rapid.
- **V-CROSS:** backend result == pure-Go reference (differential testing).
- **Fuzzing:** Go-native `go test -fuzz` for shape/numerics edge cases.
- **CI matrix:** {macOS, Windows, Linux} × {pure-Go fallback (always) + available
  accel}; missing accel ⇒ skip with log, never a silent pass; `CGO_ENABLED=0`
  build must stay green everywhere (V-CGO).

---

## (a) Load-Bearing Architecture Bets

1. **Pure-Go `simd` package as the primary accelerator.** It measurably beats
   `avo` and needs no C toolchain → it carries the pure-Go ceiling on AMD64. ARM64
   via Plan9 NEON until the `simd` package supports ARM64.
2. **A backend-agnostic `Backend`/`Kernel` interface** with the pure-Go reference
   as ground truth; cgo/GPU/NPU only as interchangeable, benchmark-triggered
   add-ons behind build tags. `CGO_ENABLED=0` remains fully functional.
3. **Golden parity as acceptance.** Every op is accepted against NumPy/torch
   goldens within fixed tolerances before it is optimized.
4. **GPU = cgo, deliberate and late.** Since no pure-Go GPU path exists, GPU is
   the canonical place where the cgo gate applies — after the CPU ceiling has
   been exhausted.

## (b) Biggest Risks + Mitigations

| Risk | Mitigation |
|---|---|
| `simd` package is experimental, API may break | Encapsulate behind a thin internal SIMD wrapper; pure-scalar fallback always present; pin to the Go 1.26 version |
| ARM64 (Apple Silicon = dev host) missing from the `simd` package | Plan9 NEON kernels + scalar fallback; measure the perf target there separately |
| GPU/NPU force cgo → portability break | Strict build-tag separation, `CGO_ENABLED=0` CI job as a mandatory gate |
| Goldens from torch not bit-reproducible | Fix seeds/dtype/shape, tolerances with §R justification, no loosening |
| Scope explosion (entire AI spectrum) | Strict §T ordering, one shippable increment per task |

## (c) Recommended Build Order

1. **L0 Core:** tensor, dtype (f32/f64 first), device, allocator, strides/views.
2. **L1 reference compute (pure Go, scalar):** elementwise, reduce, dot, GEMM +
   golden tests + bench harness. **Ground truth, unoptimized.**
3. **L1-Opt (separate):** GEMM/elementwise via `simd` package (AMD64) / NEON (ARM64),
   blocking, goroutines — against the reference from (2), with benchmark delta.
4. **L2 autograd:** tape + VJP rules of the L1 ops, V-GRAD.
5. **L3 NN:** linear, activations, loss, SGD/Adam — end-to-end on CPU.
6. **L5 IO:** safetensors first.
7. **L1b GPU (cgo gate):** first GPU backend (Metal on macOS / CUDA) as an
   optional build-tag backend, only when the §C threshold is breached.
8. **L4 domains:** transformer/LLM inference (GGUF), then CV, classical ML, RL.

---

## Sources

- [Go 1.26 Release Notes](https://go.dev/doc/go1.26)
- [golang/go #73787 — simd/archsimd intrinsics under GOEXPERIMENT](https://github.com/golang/go/issues/73787)
- [golang/go #76175 — simd CPU feature vet check](https://github.com/golang/go/issues/76175)
- [pkg.go.dev/simd/archsimd](https://pkg.go.dev/simd/archsimd)
- [Go 1.26 features overview — saraikin.com](https://saraikin.com/posts/go-1-26-features/)
- [Go 1.26 interactive tour — antonz.org](https://antonz.org/go-1-26/)
- [Go SIMD part 1 (Benchmark simd vs avo), Callista, Oct 2025](https://callistaenterprise.se/blogg/teknik/2025/10/20/trying-out-go-simd-support/)
- [go-highway — portable SIMD with pure-Go fallback](https://github.com/ajroetker/go-highway)
- [segmentio/asm](https://github.com/segmentio/asm)
- [gonum/blas](https://pkg.go.dev/gonum.org/v1/gonum/blas)
- [GoMLX (GitHub)](https://github.com/gomlx/gomlx) · [gopjrt](https://pkg.go.dev/github.com/gomlx/gopjrt)
