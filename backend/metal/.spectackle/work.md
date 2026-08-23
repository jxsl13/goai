---
schema: v1
---

## T-01KYJREJ34FN9BEZDREAF6WYT8 Fix the broken BenchmarkGPTDecode — it blocks honest sizing of dispatch work
kind: task
state: draft
created: 2026-07-27

BLOCKER, not an optimization. Small, but it gates a whole line of work.

SITE: backend/metal/gpt_test.go:303 BenchmarkGPTDecode.

SYMPTOM: it fails on both sub-benchmarks with: nlp: DecodeStep position 8 does not continue the cache: the next token is at position 9 (cached rows: 9).

WHY IT MATTERS: this is the ONLY benchmark in the tree that measures per-op dispatch cost at decode shapes (seq=1). Prefill benchmarks cannot see it — the researching agent measured BenchmarkGPTForward/metal at 29 ms for roughly 93 ops, so about 200 ns/op of dispatch overhead is 0.06%, i.e. invisible. At decode shapes the same fixed cost is roughly 10% (BenchmarkGemmaDecode averages 406 ns per Execute at dim=16). Without this benchmark the Execute-memoization task cannot be honestly sized, and any claim about it would be an estimate dressed as a measurement.

FIX: diagnose the position/cache-row mismatch. The error says the loop advanced position to 8 while the cache already held 9 rows, so either the benchmark seeds the cache with a prompt and then restarts positions from a stale index, or it reuses one cache across b.N iterations without resetting. Establish which before changing anything — the fix is either resetting per iteration or deriving the position from the cache length rather than from a counter.

VALIDATION: the benchmark runs to completion on both sub-benchmarks and produces a stable tok/s figure across -count=3. Then confirm it actually exercises the intended regime by checking with GOAI_TIME_OPS=1 that per-token Execute counts match the model's layer count, so it is measuring decode and not accidentally re-running prefill.

SCOPE: fix the harness only. Do not optimize anything in the same change — the entire point is to obtain a trustworthy measuring instrument before optimizing, and a benchmark repaired in the same commit as the thing it measures cannot serve as the A/B baseline.

## ADR-01M0FVWNPKEX6917B1N0VBM0FJ How should synchronous host-resident F32 Metal bias gradients route after the CPU reduction optimization?
kind: adr
state: done
created: 2026-08-20
context: Three independent count-7 M2 campaigns show the production CPU selector is 3.263x to 199.71x faster than direct Metal through 2,097,152 elements, with worst candidate spread 1.788x, exact reference parity, and 0.994x end-to-end GPT throughput.
decision: Route measured shapes through CPU and preserve direct Metal above the bound
consequences: F32 rank-2 gradients with positive dimensions and at most 2,097,152 elements use the exact optimized CPU kernel with recorder suppression. Larger, unsupported, or CPU-unavailable cases retain the isolated direct Metal implementation. Future device-resident graph execution requires a new benchmark boundary and does not inherit this host-resident decision.
status: accepted

kind: radio
option: Route measured shapes through CPU and preserve direct Metal above the bound
option: Retain direct Metal for all shapes
option: Remove the direct Metal implementation
blocks: T-01M0FVGM88EWMRQCHFN4B748AV
choice: Route measured shapes through CPU and preserve direct Metal above the bound

## ADR-01M0M9MPAGFM9T4AMA7D97HWJD Which native Metal Q4_1 kernel shape should be the production baseline on M2?
kind: adr
state: done
created: 2026-08-22
context: Q4_1 uses a 20-byte affine block with d and m. Direct decoding preserves compressed residency and avoids transformation traffic. Q4_0 already proves the scalar and cooperative occupancy shapes.
decision: Separate scalar and two-SIMD-group cooperative pipelines derived from Q4_0
consequences: The kernel decodes Q4_1 directly from 20-byte resident blocks. Synchronous, resident, and recorder paths share the same cached pipelines. Scalar is the capability fallback; cooperative is the M2 path. No dense expansion or transient Q4_0 transformation is allowed.
status: accepted

kind: radio
option: Separate scalar and two-SIMD-group cooperative pipelines derived from Q4_0
option: Reuse Q4_0 after transforming Q4_1 weights or activations
option: Materialize dense F32 weights before Metal GEMM
blocks: P-01M0M9B6FRFCZA18408PMM2WGH
choice: Separate scalar and two-SIMD-group cooperative pipelines derived from Q4_0

## R-01M0QQFB4CE6C9Q5EXBWC3SFNB Measure precompiled Q4_K metallib cold-start leverage
kind: research
state: active
created: 2026-08-23
targets: objc:metal_bridge.ensure_qmatmul_q4k

Apple documents that MSL source strings compile first to GPU-independent Metal IR and then to a device pipeline, while an offline metallib removes the runtime source-to-IR stage. Sources pinned on 2026-08-23: https://developer.apple.com/documentation/metal/metal-libraries and https://developer.apple.com/documentation/metal/building-a-shader-library-by-precompiling-source-files. Seven fresh incumbent Q4_K processes measured first-call durations 18.643, 3.466, 3.714, 3.680, 14.473, 4.125, and 3.705 ms; median 3.714 ms. Their second-call median was 0.468 ms. Research whether loading an exact offline-compiled Q4_K metallib through newLibraryWithURL removes at least half of first-call latency without changing warm output or throughput. The diagnostic uses an environment-selected artifact path and retains source compilation as fallback. Any production proposal must define reproducible artifact generation, source/hash coupling, deployment target, architecture compatibility, fallback behavior, and repository size impact.

## P-01M0QQG05VFSN83HQ7JTG43PVY Load an exact precompiled Q4_K metallib for cold Metal startup
kind: proposal
state: draft
created: 2026-08-23
targets: objc:metal_bridge.ensure_qmatmul_q4k

Diagnostic phase: compile the existing scalar and cooperative Q4_K MSL into one macOS Metal-IR library with a pinned deployment target, then let ensure_qmatmul_q4k load that file only when GOAI_METAL_Q4K_METALLIB names it. The incumbent source-string path remains the default and fallback. Compare fresh processes from one binary, alternating control and artifact arms. Frozen gates: existing Q4_K cross-reference and cooperative parity tests remain unchanged; metallib functions expose the same names and pipeline width; three independent campaigns each contain seven fresh processes per arm; median first-call control divided by candidate is at least 2.00x in every campaign; candidate second-call median retains at least 0.97x incumbent warm throughput; missing, malformed, and incompatible artifact paths fall back to source compilation and return correct output. No artifact, environment seam, or source duplication may enter production unless the diagnostic passes. Production requires one canonical MSL source, deterministic generation, recorded compiler and SDK versions, a source hash, and portable fallback.
