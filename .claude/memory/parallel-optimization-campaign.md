---
name: parallel-optimization-campaign
description: "User directive (2026-07-15) — run 4 parallel fable subagents for file-by-file low-level perf grinding, bug-hunt + regression tests, research Go opt techniques when stuck"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 491e34f2-de40-467f-89a0-f918efb8a6b3
---

User directive 2026-07-15: "start 4 parallel subagents in the background for the file-by-file optimization tasks with the fable model. grind for the last percentile of performance improvements. check for bugs while optimizing and add regression tests for each bug. start from the low level layers. if you hit a limit, research online for further Go optimization techniques and document them."

**How to apply:**
- Spawn background Agent subagents, `model: "fable"`, `isolation: "worktree"`, disjoint file scopes (so PRs never conflict). Start LOW (tensor core → cpu SIMD kernels → up).
- Each subagent: benchmark-first, keep ONLY measured wins (§C3, revert noise/regressions + report them as rejected negatives), bug-hunt + add a regression test per bug (fail-before/pass-after), research Go perf techniques online (WebSearch) if stuck and document, open a PR but DO NOT merge (I review+merge + watch CI green).
- First wave RESULTS (all verified by me — full test suites incl. downstream nlp/-race — then merged):
  - PR#40 opt-gemm-simd MERGED: packing-free L2 column blocking +88% on large LLM shapes, BCE hoist 197→230 GFLOP/s, small-m GEMV path ~3-5× (decode shape), F64 bit-exact. Rejected mr=6×16 (Go Y0–Y14 regalloc ceiling, golang/go#76969).
  - PR#43 opt-elementwise MERGED: broadcast fast paths −73% (bias-add), wide-softmax core-split −54%, softmax allocs 26→10; signed-zero/arg-order guard tests.
  - PR#42 opt-tensor-core MERGED: Cast 16-dtype typed paths (−78/88%), pool 2→1 allocs, view alloc cuts, defaultCPU pre-boxed (device.go). Verified vs full nlp Cast gate.
  - opt-norm-attn (a7fe): ran much longer, nudged to converge; PR pending.
- KEY LESSON: box is CPU-contended by 4 concurrent bench agents → launch-bound/memory-bound benches need PAIRED/INTERLEAVED A/B in the SAME window, not sequential medians (agents + I both hit this; my CUDA fused-softmax +4.4% only showed under paired A/B).
- My job: watch each PR's CI, sanity-review the diff (foundational code + smaller model = verify numerics/tests myself), merge serially. Then launch the next wave up the stack.

**Why:** user wants last-percentile perf across the whole low-level stack, parallelized. Fits my niche [[linux-amd64-worker-role]] (amd64 SIMD) + the hyper-optimize arc [[inference-benchmark-hyperoptimize]].

See [[user-directives-cuda-bottomup]].
