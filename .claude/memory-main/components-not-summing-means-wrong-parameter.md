---
name: components-not-summing-means-wrong-parameter
description: "When measured components don't sum to the measured whole, suspect an assumed parameter (shape/count), not the per-component timings — instrument the actual arguments."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c9844d26-52eb-4368-8358-6d744bfde57b
  modified: 2026-08-15T05:33:55.872Z
---

If per-component measurements don't add up to the measured total, the error is far more likely in an **assumed parameter** than in the timings. Log the *actual* arguments the code is called with before re-measuring the pieces.

**Why:** measured 2026-08-15. A 22-layer decode was 20.3 ms/token with 17.5 ms GPU-busy, while measured matmuls (4.1 ms) plus isolated small-kernel costs came to ~5 ms. I spent three rounds refining per-op microbenchmarks that were each individually accurate and collectively meaningless — because both the estimate *and* the microbenchmark used the shape I assumed the op ran at. Logging real element counts found it instantly: `Binary calls/token=70.1 n=2097152 x46.8` — elementwise ops running over the whole ctx-sized buffer (1024 rows) for a single decoded row, ~2.8 GB/token against ~520 MB of actual weights. Fixing it gave **2.83× end-to-end (48.7 → 138.0 tok/s)**, bit-identical output.

**How to apply:**
- Establish the total first and split it (GPU-busy vs wall via `GPUStartTime`/`GPUEndTime`), then subtract measured components. An unexplained remainder is a *finding*, not noise to absorb.
- Eliminate hypotheses by measurement, not plausibility: host overhead (86% GPU-busy → no), matmul cost (4.1 ms over 520 MB of distinct weights → as estimated), dependency-chain serialization (4.12 vs 4.18 ms → no).
- Instrument argument histograms, not just call counts. A call count times an assumed size is the same mistake twice.
- Structural smell worth grepping for: an op whose length comes from a *buffer* when every sibling op takes an explicit `rows` count. That asymmetry was the whole bug.
- Verify by output equivalence, not just tests — a bit-identical token stream over a long generation proves the skipped tail was never read.
- Related: [[benchmark-comparisons-must-be-in-session]], [[gpu-kernel-runtime-loop-bounds]], [[base-perf-sweep]].
