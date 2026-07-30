---
name: perf-floor-is-scoped
description: A declared performance floor is scoped to the kernel shape it was measured against and to the thresholds calibrated against that shape — change the shape and both go stale. Re-open "measured and discarded" verdicts when the code beneath them changes.
metadata:
  node_type: memory
  type: feedback
---

The portable GEMM band kernels in `backend/cpu` were documented as finished in two
places. `gemm.go` recorded that BLIS-style panel packing had been built, measured on
this arm64 host and DISCARDED — "the M-series memory subsystem already feeds this
streaming kernel near-optimally, so panel packing is net overhead here" — and
`docs/perf-notes-cpu.md` listed the kernels under "Not touched, deliberately" on that
basis. Both were honest and both were wrong by the time they were read.

They were wrong because the verdict was scoped to a kernel that no longer existed in
that form. The band then ran p-outer with j inner and STREAMED its accumulator: four
f64 loads and stores per j to issue four FMAs. It was load/store bound on C, so B
locality could not be the limit and packing genuinely was overhead. Giving the band a
4x4 register tile removed the C traffic; B locality then BECAME the limit and the same
packing measured -11.1% (f32) and -28.1% (f64).

Cumulative on the vision benchmark suite, verified by swapping the whole `backend/cpu`
directory between the pre-arc commit and HEAD: **-17.76% geomean** (ViT batched -28.1%,
MLPMixer batched -29.9%, Swin batched -19.7%).

## The second-order effect: every threshold calibrated against the old kernel went stale

Four of them, and two were actively costing:

- `gemmPackMinWork` (f32) — crossover moved two orders of magnitude once the operand
  widenings were hoisted; the same n=256 went from +2.78% to -17.17%.
- `gemmPackMinRows` — never measured on its own axis at all. The work gates were
  calibrated on SQUARE matrices, where m and n move together and m was always large, so
  the row term went unmeasured and shipped at a value where BOTH dtypes paid 13-18%.
- `matmulInlineWork` — the serial/parallel crossover. At the gate value itself the
  fan-out measured +37.26% slower.
- The pack gates needed to be PER-DTYPE: f32 turns over between 1MB and 4MB of B, f64
  between 128KB and 512KB. One shared gate either leaves a win unclaimed or pays for a
  loss.

## What to do differently

- Treat a "measured and discarded" note as scoped to the code it was measured against.
  Re-open it when that code changes shape.
- When a gated kernel gets faster, re-sweep its gate IN THE SAME CHANGE.
- Sweep every axis a gate's condition reads. A square sweep cannot exercise a row term.
- Make thresholds `var`, not `const`, and give each a benchmark that forces both arms
  inside one binary. Four went stale precisely because nobody could re-sweep them.
- CHECK THE SCANNER FIRST. `perfscan` PS6023 flags exactly this ("a tuning constant
  gating two paths that no test names") and had `gemm.go:95` — `matmulInlineWork` — in
  its output the whole time. It was re-derived by hand instead of read.

## Measurement discipline that paid off

- A CONTROL CASE is the cheapest way to tell a measurement from a mirage. A benchmark
  arm where the change provably cannot execute reported +22.19% at 20-40% spread; the
  same comparison at a size with millisecond ops showed the control inert and the real
  effect (-3.4% to -5.8%) appear.
- A wide-variance "no change" is not a null result. Two figures reported as ~ at +-40%
  came back at -16% and -50% when re-run alone.
- Do not fit a gate to a non-monotonic sweep. Pinning blocks-per-band and pack size and
  varying only the leftover rows past the tile swung the result from +71% to -28% — a
  hundred points on a variable neither axis read.
