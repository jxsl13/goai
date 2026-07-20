# GoAI — Master Planning and Implementation Prompt

> Purpose of this file: A reusable prompt set for planning, building, and
> verifying a full-fledged Go AI library from architecture to optimized
> implementation, using the skills `/deep-research`, `/research`, `/spec`,
> `/review`, `/build`, and `/loop`.
>
> Usage: Each section ("PHASE …") is a self-contained prompt block.
> Copy the respective block after the matching slash command.
> Order: PHASE 0 → 1 → 2 → 3 → then continuous operation with `/loop`.

---

## 0. North Star (applies to ALL phases — always include)

**Mission:** An idiomatic, modular Go library for the entire AI spectrum
(linear algebra, autograd, classic ML, deep learning, NLP/LLM inference, CV,
RL, probabilistic models), whose core operations come as close as possible to
equivalent C/C++ implementations (PyTorch/ATen, llama.cpp, ONNX Runtime, Eigen,
oneDNN) via SIMD assembly and GPU/NPU accelerators.

**Top principles (non-negotiable):**
1. **Correctness before speed.** Every function first exists as a
   reference-valid, well-tested pure-Go implementation. Optimization is a
   *second, separate* step against exactly this reference.
2. **Mathematical/scientific grounding.** Every unit cites the underlying
   definition (paper, textbook, canonical reference implementation).
   Numerics decisions (stability, accuracy, overflow) are documented
   explicitly, not made implicitly.
3. **Numerical parity as the acceptance criterion.** "Done" means: results
   match a reference implementation (NumPy/PyTorch/reference C) within
   defined tolerances (ULP/rtol/atol) — demonstrated,
   not asserted.
4. **Performance is measurable or does not exist.** No "faster" without a
   benchmark, roofline classification, and comparison against a C/C++ baseline.
5. **Platform and hardware portability from the start.** Native support
   for **macOS, Windows, Linux** on **CPU, GPU, and NPU**. Every accelerated
   operation has a pure-Go fallback that runs everywhere.
6. **One responsibility per module, deep modules with a narrow interface.**
   Backend details (CUDA/Metal/Vulkan/SIMD) live behind stable interfaces.

**cgo policy (hard gate — applies in every phase and every task):**
- **The default is pure Go.** Every op is first implemented in pure Go and
  then optimized as far as possible in pure Go: algorithm, cache blocking/
  layout, `math/bits`, the experimental **`simd` package** (GOEXPERIMENT, if
  available for the target Go version — verify status in §R), `avo`-generated
  x86 asm, Plan9 asm (NEON/ARM64), goroutine parallelization.
- **cgo is the exception, not the default.** cgo (or external C/C++ libs such as
  OpenBLAS/cuBLAS/cuDNN/oneDNN) may only be introduced once ALL of the following
  conditions are met:
  1. The pure-Go version exists, is §V-green, and has demonstrably been
     optimized up to its practical ceiling (documented stages + roofline classification).
  2. A benchmark shows a **significant** performance superiority of the
     cgo variant against exactly this fully optimized pure-Go version (set the
     threshold in §C, e.g. "≥ X× or ≥ Y % closer to the C++ baseline"; "significant"
     is defined as a number, not by gut feeling).
  3. The cgo variant lives behind build tags as an OPTIONAL backend; the
     pure-Go path remains fully functional and is the cross-compile
     default (fallback everywhere).
  If any of the three conditions is not met ⇒ cgo is rejected, the pure-Go
  version remains the shipping state, and the cgo idea is parked as a candidate
  in §B/§T instead of being merged.
- Pure Go vs. generated assembly (`avo`) vs. Plan9 asm — weigh per backend
  (portability, build complexity, cross-compilation, performance ceiling). All
  of these paths count as "pure Go" in the sense of the policy (no C-toolchain requirement).
- Licensing model and dependency policy (which cgo/vendor libs are allowed?).
- Memory/tensor layout (row-major, strides, views, alignment, arena alloc,
  zero-copy to GPU).
- Numeric types: f64/f32/f16/bf16/int8 quantization — which first, how tested.
- Threading model (goroutines vs. fixed worker pool, NUMA, GOMAXPROCS interaction).

---

## PHASE 0 — Landscape & Feasibility  →  `/deep-research`

**Slash:** `/deep-research`

**Prompt:**
> Produce a deeply researched, source-backed report as the foundation for
> building a full-fledged AI library in Go that is meant to approach C/C++
> performance via SIMD assembly and GPU/NPU accelerators. Answer
> fact-based (with sources, dates, version numbers):
>
> 1. **State of the art in Go ML:** What already exists (Gorgonia, gonum,
>    GoMLX, go-torch/cgo bindings, tract, candle comparison)? Where exactly do
>    they fail on performance/portability/maintenance? Which gap justifies a rebuild?
> 2. **CPU SIMD in pure Go:** Realistically achievable performance WITHOUT cgo with
>    (a) the experimental `simd` package of the Go stdlib (GOEXPERIMENT — clarify
>    current availability/maturity status and target Go version), (b) `avo`-generated
>    x86 asm (AVX2/AVX-512), (c) ARM NEON/SVE (Plan9 asm, Go support status),
>    (d) auto-vectorization by the Go compiler, (e) `math/bits`. Benchmarks/evidence
>    for the achievable fraction of Eigen/oneDNN/OpenBLAS — and where a pure-Go
>    ceiling realistically lies, beyond which only cgo helps.
> 3. **GPU paths from Go:** cgo→CUDA/cuBLAS/cuDNN; Metal (macOS) via cgo/Objective-C;
>    Vulkan compute; WebGPU/wgpu; ROCm/HIP; SYCL. Trade-offs: build complexity vs.
>    portability vs. peak performance. Zero-copy options.
> 4. **NPU/accelerator paths:** Apple Neural Engine via CoreML, Windows DirectML,
>    Intel oneDNN/NPU, Qualcomm/ARM. What is even addressable from Go?
> 5. **Reference baselines for parity:** Which C/C++ libraries serve as the
>    correctness and performance reference per domain (Eigen, OpenBLAS, oneDNN,
>    llama.cpp/ggml, ONNX Runtime, PyTorch/ATen)? How does one reproducibly
>    extract golden reference values?
> 6. **Model interop formats:** ONNX, GGUF, safetensors, HuggingFace — maturity,
>    effort.
> 7. **Verification methodology in the field:** Numerical gradient checking,
>    property-based testing, ULP tolerances, differential testing against a reference,
>    fuzzing for numerics.
>
> Deliver at the end: (a) a recommendation for the 3–4 load-bearing architecture bets,
> (b) the biggest technical risks with countermeasures, (c) a justified
> order in which domains/backends should be built.

**Output expectation:** A report (becomes the raw source for §R in the spec).

---

## PHASE 1 — Targeted Detail Research  →  `/research`

**Slash:** `/research`  (multiple times, one round per open decision)

One focused round for each still-open core decision from PHASE 0. Examples:

> `/research` Compare `avo` (generated x86 asm) against handwritten
> Plan9 assembly against cgo→OpenBLAS for GEMM in Go: maintainability,
> cross-compile capability, measured GFLOP/s ceiling, alignment requirements.
> Recommend a default strategy and name when to deviate. Record the
> findings as §R in the spec.

> `/research` Clarify the current maturity of the experimental `simd` package of the
> Go stdlib (GOEXPERIMENT): availability per Go version, supported architectures
> (x86 AVX2/AVX-512, ARM64 NEON), API stability, measured performance vs.
> `avo`/Plan9 asm, cross-compile behavior. From that, define the concrete
> "significant" threshold for the cgo gate (speedup factor / % of the C++ baseline).
> Record finding + threshold as §R/§C.

> `/research` Determine the canonical, numerically stable algorithm + the
> usual reference tolerance for: Softmax, LayerNorm, GELU, Adam, Conv2d (im2col
> vs. Winograd vs. direct), Attention (naive vs. FlashAttention). Each with a
> source. Record as §R.

**Rule:** Every claim with a source; anything unsourced is marked as `?`, never
written as fact. Results land in the **§R (research log)** of the spec.

**Research mechanics (mandatory):** Do NOT use the built-in `/deep-research`
workflow (forces a StructuredOutput schema → crashes under rate limits). Always use
`research-lite` (`.claude/workflows/research-lite.js`): one focused question
per run, schema-free, compressing sub-agents, graceful with dead agents.
For details see `LOOP.md` → "Research rule".

---

## PHASE 2 — Spec & Architecture  →  `/spec`

**Slash:** `/spec`

**Prompt:**
> Generate `SPEC.md` for the Go AI library "GoAI" based on the North Star
> (above) and the §R findings. Structure:
>
> - **§G (Goals):** The North Star, condensed. Measurable definition of "full-fledged".
> - **§C (Constraints):** Resolved decisions from PHASE 0/1 (backend strategy
>   per platform, numeric-type roadmap, tensor layout, threading, license/deps).
>   MUST fix the **cgo policy** as a measurable gate: pure Go is the default;
>   cgo only after a fully optimized pure-Go version + benchmark-proven
>   "significant" superiority defined as a concrete number; cgo always optional behind
>   build tags with a pure-Go fallback. Fix the "significant" threshold here
>   numerically (e.g. speedup factor and/or % of the C++ baseline).
> - **§I (Invariants of the architecture):** The layer model and its hard
>   boundaries — e.g.:
>   - `L0 core`: Tensor, Dtype, Device, memory/allocator, strides/views.
>   - `L1 compute`: backend interface (`Backend`, `Kernel`) + pure-Go reference backend
>     that runs EVERYWHERE and is the definition of truth.
>   - `L1b accel`: swappable backends (cpu-simd, cuda, metal, vulkan, npu),
>     all against the same interface, with feature detection + fallback.
>   - `L2 autograd`: tape/graph, VJP rules per op.
>   - `L3 nn`: layers, init, optimizer, loss, data pipeline.
>   - `L4 domains`: classic ML, vision, nlp/llm-inference, rl, probabilistic.
>   - `L5 io`: ONNX/GGUF/safetensors, serialization, model zoo.
>   INVARIANT: Higher layers know no backend internals. Every op has a
>   pure-Go fallback. No `cgo` in `L0`. Public API is backend-agnostic.
> - **§V (Verification invariants — the acceptance rules):** e.g.
>   - V-PARITY: Every op passes a golden test against the named reference
>     within the tolerance fixed in §R (rtol/atol/ULP).
>   - V-GRAD: Every differentiable op passes numerical gradient checking
>     (finite differences) under a defined threshold.
>   - V-CROSS: Backend-X result == pure-Go reference within backend tolerance.
>   - V-PLATFORM: CI green on {macOS, Windows, Linux} × {CPU fallback + available
>     accel}. Missing accel ⇒ skip with log, never a silent pass.
>   - V-BENCH: Every optimized op has a benchmark + baseline comparison number;
>     regressions break CI.
>   - V-PROP: Property-based tests for invariants (shape, linearity, associativity
>     where mathematically guaranteed).
>   - V-CGO: No cgo in the shipping path without (a) a green, fully optimized pure-Go
>     reference and (b) a checked-in benchmark that exceeds the §C threshold.
>     The pure-Go build (without a C toolchain) must stay green on all platforms.
>   - V-STABLE: Public API changes only via a documented deprecation path.
> - **§T (Task backlog):** The actual work list. Every task is ONE
>   shippable, testable increment, ordered by dependency. Every task
>   carries: goal, affected layer, reference for parity, definition of done
>   (which §V rules it must satisfy). Ordering guideline:
>   1. `L0` Tensor/Dtype/Device + allocator + pure-Go reference backend.
>   2. `L1` GEMM/elementwise/reduce as reference + golden tests + bench harness.
>   3. **Only then** the first optimization (SIMD GEMM) — as a separate task against the
>      reference from step 2.
>   4. Autograd core + VJP rules of the L1 ops.
>   5. NN basics (Linear, Activation, Loss, SGD/Adam) end-to-end on CPU.
>   6. Only after that the GPU backend, then transformer/LLM inference, then further domains.
> - **§B (Backprop log):** initially empty; bugs/failures are condensed here into new
>   §V invariants.
>
> Keep §T deliberately in small steps: "correctness first, optimization as its
> own follow-up task" must be physically reflected in the task structure.

---

## PHASE 3 — Adversarial Review of the Spec  →  `/review`

**Slash:** `/review`

**Prompt:**
> Red-team the `SPEC.md` before any code is written. Check in particular:
> - Are the §V rules REALLY sufficient to demonstrate C++ parity, or do
>   they allow silent loss of accuracy?
> - Is the backend abstraction (§I) viable for CUDA AND Metal AND Vulkan AND
>   NPU, or does one backend inevitably leak into the API?
> - Is the task order free of hidden cycles (autograd vs. kernels vs.
>   device placement)?
> - Where does the plan tempt one to optimize too early?
> - Which numerical traps (f16 overflow, reduction order,
>   non-associative FP summation, determinism across backends) are missing in §V?
> Every objection with evidence (file:line in the spec or a §R source). Surviving
> findings harden §V. End with an explicit go/no-go.

After the go: the spec is frozen as the truth (source: `spec/` tree; `SPEC.md` = generated view, mutations via `internal/specgraph`, §V41). Only now does the build begin.

---

## CONTINUOUS OPERATION — continuous implementation  →  `/loop` + `/build`

The build runs via the `/build` skill (plan-then-execute against SPEC.md, with
automatic `backprop` on test/build failures). `/loop` drives this
self-paced, task by task.

### /loop definition (self-paced, no fixed interval)

**Slash:** `/loop`  (no interval ⇒ the model paces itself per task)

**Prompt (pass exactly as is):**
> Work FULLY AUTONOMOUSLY, without questions back to the user. Implement GoAI
> continuously, strictly according to `SPEC.md`. Per iteration, carry EXACTLY ONE task
> to completion, and only stop the loop when all §T tasks carry the status
> "done". Flow per iteration:
>
> 0. **Bootstrap (if `SPEC.md` is missing or incomplete):** First autonomously
>    produce the planning foundation, ONE phase per iteration, in this order:
>    (a) `/deep-research` → report to `docs/research/00-landscape.md`; (b)
>    `/research` for the open core decisions → §R/§C; (c) `/spec` →
>    `SPEC.md` with §G §C §I §V §T §B; (d) `/review` → findings harden §V, result
>    as a go/no-go note. Only once `SPEC.md` exists and carries a review "go",
>    proceed to step 1. Use `PLANNING_PROMPT.md` as the template for the phase prompts.
>
> 1. **Selection:** Pick the next unfinished §T task whose dependencies
>    are satisfied. Name its ID and definition of done before you begin.
> 2. **Build:** `/build` this task. If it is an optimization task, the
>    reference-valid pure-Go version MUST already exist and be green — otherwise
>    build its correctness task first.
> 3. **Verify (acceptance per §V):**
>    - V-PARITY: golden test against the reference named in the task within
>      the §R tolerance. If a golden file is missing, generate/update it
>      reproducibly from the reference and commit it.
>    - V-GRAD (if differentiable): numerical gradient checking.
>    - V-CROSS (if a backend task): result == pure-Go reference.
>    - V-PROP: pertinent property tests.
> 4. **Measure + cgo gate (optimization tasks only):** First optimize in
>    pure Go up to the ceiling (algorithm → layout/blocking → `simd`/`avo`/
>    NEON → goroutines), each stage with a green §V and a documented benchmark
>    delta (GFLOP/s, speedup, % of the C++ baseline, roofline). Only once the pure-Go
>    ceiling is reached, evaluate a cgo candidate ONLY if the §C threshold
>    seems plausibly reachable: build it as an optional build-tag backend,
>    benchmark it against the fully optimized pure-Go version. If it exceeds the
>    §C threshold ⇒ V-CGO satisfied, merge (pure Go remains the default fallback).
>    Otherwise ⇒ discard, pure Go remains the shipping state, park the cgo idea as a note in
>    §B. No optimization counts as done without the benchmark number.
> 5. **Error handling:** On a test/build/parity/regression failure, invoke `backprop`:
>    trace the cause, check whether a NEW §V invariant would catch the regression
>    in the future, extend §B, then fix. Never skip a red test
>    or soften tolerances to get green — tolerance changes require
>    a §R justification.
> 6. **Platform check:** Ensure that the pure-Go paths remain platform-neutral
>    and accelerated paths sit cleanly behind build tags/feature detection
>    (macOS/Windows/Linux × CPU/GPU/NPU), with fallback. Unavailable
>    accel backends ⇒ documented skip, no silent pass.
> 7. **Wrap-up:** Set the task to "done" in §T, a short changelog (what, parity,
>    benchmark), and — only if the user has allowed commits — commit. Then
>    on to the next iteration.
>
> Invariants of the loop: Only one task open at a time. Correctness before
> performance. No progress without a passed §V acceptance. No "optimized" without
> a benchmark number.
>
> **Autonomy rule (instead of stopping for the user):** On structural ambiguity or
> an open design decision, do NOT stop and do NOT ask the user.
> Instead: (a) choose the scientifically/mathematically best-justified default
> option (with a source, if possible); (b) record the decision as an entry in
> `docs/decisions/ADR-<n>.md` (context, options, choice, rationale,
> revisability) AND note it as a `?` marker or amendment in `SPEC.md` (§C/§B);
> (c) keep building with the assumption made this way. Only GENUINE hard
> blockers (missing toolchain, broken environment that the loop cannot
> repair itself) justify a halt — then send a brief
> `PushNotification` and await the next iteration. Never
> soften tolerances or skip tests to fake progress —
> a red test is a `backprop` case, not a reason to loosen.
> Commits/pushes remain forbidden as long as the user has not explicitly
> allowed them; until then, the loop works only in the working tree.

### Optional: interval-paced variant (for long runners/overnight)

> `/loop 30m` + the same prompt — useful when individual tasks entail long builds/
> benchmarks and you want regular progress checkpoints.
> For pure compute load, the self-paced variant (no interval) is usually
> better, because it only advances after a genuine task completion.

---

## Cross-cutting: Verification & Performance Standard (applies in every task)

**Correctness proof (before any optimization at all):**
- Golden tests against a named reference (NumPy/PyTorch/reference C), tolerances
  from §R, reproducibly generated and checked in.
- Numerical gradient checking for all differentiable ops.
- Property-based tests for mathematically guaranteed properties.
- Cross-backend differential testing against the pure-Go reference.

**Performance methodology (the second, separate step):**
1. Analyze roofline/complexity (compute- vs. memory-bound).
2. Optimize **in pure Go** in stages: algorithm → cache blocking/layout →
   SIMD (experimental `simd` package / `avo` / NEON) → multithreading. After
   EVERY stage: correctness green again + benchmark delta documented.
3. Comparison against the C/C++ baseline in % of achieved peak performance.
4. **cgo gate** only after the pure-Go ceiling is reached: merge cgo/external lib only
   if the benchmark breaks the §C threshold against the fully optimized pure-Go version;
   otherwise discard. GPU/NPU offload follows the same logic (optional backend,
   pure-Go fallback remains).
5. Benchmark regression protection in CI (V-BENCH); the pure-Go build without a C toolchain
   stays green on all platforms (V-CGO).

**Platform/hardware matrix (target coverage):**
- OS: macOS, Windows, Linux.
- CPU: x86-64 (AVX2/AVX-512) via `avo`; ARM64 (NEON, possibly SVE) via Plan9 asm;
  pure-Go fallback everywhere.
- GPU: CUDA/cuBLAS/cuDNN (Linux/Windows), Metal (macOS), Vulkan compute
  (portable), ROCm (optional).
- NPU: CoreML/ANE (macOS), DirectML (Windows), oneDNN (Intel) — insofar as
  addressable from Go; otherwise marked as a non-goal (no silent promise).

---

## Order at a glance

```
/deep-research   → landscape, bets, risks, domain order                (→ raw source)
/research (×N)   → factually resolve open decisions                   (→ §R)
/spec            → SPEC.md: §G §C §I §V §T §B                          (→ truth)
/review          → red-team, harden §V, go/no-go                      (→ freeze)
/loop + /build   → build task by task, verify, measure                (→ continuous operation)
   └─ backprop on every failure → new §V invariants
```
