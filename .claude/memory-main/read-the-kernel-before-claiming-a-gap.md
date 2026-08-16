---
name: read-the-kernel-before-claiming-a-gap
description: "Before proposing to build a design, read the implementation to check it isn't already there — and measure which half of a cost dominates before optimizing the half that merely looks fixed."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c9844d26-52eb-4368-8358-6d744bfde57b
  modified: 2026-08-15T09:23:15.981Z
---

Two failures of the same kind, both on 2026-08-15: reasoning about an implementation from its *timings* instead of reading it.

**1. Claiming a design was missing.** From a crossover table I concluded "neither arm dequantizes a K-tile once into threadgroup memory, then runs simdgroup matrix ops" and proposed building it. `qmatmul_q4k_mm` already was exactly that — 32×32 tile, weight staged once in `sW`, consumed by all 32 rows via matrix units. I committed the wrong claim into a test comment before catching it. Reading the kernel took one command and also handed me the real defect: the tile is rebuilt once per **M-tile**, so its dequant work is O(M/BM) where the rival's is O(1) in M — which is precisely why the measured gap widened 0.77×→0.36× as M grew.

**2. Optimizing the half that only looks dominant.** A fixed cost independent of M *looks* like the reason small-M is weak. Measured, the expansion was 37.7% at M=64 and 5.9% at M=1024 — the minority everywhere. The GEMM was the weak half, throttled at ~99 GB/s reading the 46 MB expanded weight (47% of FLOP peak at M=64, vs 78% at M=1024 where it turns compute-bound). Same conclusion (try f16) but a different target and a different expected magnitude.

**How to apply:**
- Before "let's build X", grep for X. A capability claim is checkable in one command; the timings will not tell you whether the code exists.
- Establish the total, then split it, before optimizing any part. Compute achieved GB/s and % of FLOP peak per shape — those two numbers say *which ceiling* you are against, and a term can be fixed in M yet still be the minority.
- A confirmed hypothesis can still be insufficient: doubling the M-tile gave the predicted 1.22× at large M and *cost* 0.71× at small M (padding waste), against a 2.75× deficit. Revert, and keep the number in the test comment so the next attempt doesn't re-run it.
- Correct a wrong claim already committed, in its own commit, saying what was wrong and what reading the code showed.
- Related: [[components-not-summing-means-wrong-parameter]], [[benchmark-comparisons-must-be-in-session]], [[integration-audit-method]].
