---
name: benchmark-comparisons-must-be-in-session
description: Never compare a benchmark against numbers from an earlier session; interleave variants in one session and keep untouched arms as a control.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c9844d26-52eb-4368-8358-6d744bfde57b
  modified: 2026-08-15T04:09:27.710Z
---

A perf claim is only valid if both variants were measured **in the same session, interleaved**. Numbers from an earlier session are not a baseline — machine state (thermal, background load, cache) drifts enough to manufacture a 1.13–1.25× "improvement" out of nothing.

**Why:** measured 2026-08-15. After vectorizing the Metal Q2_K/Q5_K cooperative kernels, a cross-session table showed the two vectorized formats topping the list (+23%, +27%) — which read as clean confirmation. It was drift: **all seven** formats improved, including the five whose kernels were untouched, and Q8_0 (untouched) gained +25%, as much as vectorized Q2_K. A proper in-session A/B (3 alternations, 9 samples each, non-overlapping distributions) gave the real answer, 1.22×.

The drift-contaminated table happened to land on 1.233×, within noise of the true 1.22×. A broken method can return the right answer — that is why the method, not the plausibility of the number, is what to check.

**Eight more trap shapes, all hit on goai in one session (2026-08-15), all distinct from drift:**
- **Probe order inside one run.** Measuring dtype A fully then dtype B gave "f32 beats f16 everywhere"; probing f16 first inverted it. Alternate the arms — two runs then agreed to 0.3 points.
- **State pollution across iterations.** A sweep that ran short prompts before long ones left a cache populated, so the long-prompt arm never measured the uncached path it existed to test. A `cache=0.00 GB` reading exposed it; removing the gate it had wrongly justified gained 4.4%.
- **GPU-only vs wall time compared as equivalent**, and **prefill-inclusive vs prefill-exclusive throughput** — each manufactured a gap (12%, 4x) that did not exist.
- **A metric covering one part read as the whole.** `LastGPUSeconds` reports only the last command buffer; recorder dispatch counts count what is ENQUEUED, not executed (speculative pre-encoding doubled them).
- **Cache-resident microbenchmarks.** A 6.5 MB read reported 369 GB/s — twice hardware peak. Sweep the size until the rate stops rising.
- **A knob that never reached the code.** Twice: a cap already above the computed value, and a setter inert while another mode was on. The tell is *identical* numbers rather than a noisy trend.
- **Isolated per-op timing is BIASED, not merely noisy.** N identical dispatches serialise on write-after-write hazards: it overstated rmsnorm 8x and understated attention. Direction depends on whether the op overlaps its neighbours, so averaging cannot fix it — only leave-one-out on the real sequence can.

**How to apply:**
- Rebuild both variants and alternate them in one session; report min/max overlap, not just means.
- **Keep untouched arms in the sweep as a control group.** Here five unmodified quant formats silently reported the drift magnitude. If everything moved, nothing was measured.
- Expect end-to-end dilution vs a leaf win (1.72× leaf → 1.22× decode). An end-to-end gain *equal* to the leaf gain signals a measurement error, not a better result.
- Same failure one level down: [[base-perf-sweep]] and the FIRST-BENCHMARK-SAMPLE-IS-NOT-COMPARABLE-001 record (warm-vs-cold within a run). Same control catches both.
- Related: [[perf-gap-vs-python]] on recording incumbent versions so comparisons don't rot.
