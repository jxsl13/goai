# M2 Metal Bias-Gradient Route Revalidation (2026-08-20)

## Outcome

Retained with a measured ceiling. At GoAI's synchronous host-resident tensor boundary, the later
bit-exact parallel CPU column reduction is 3.263x to 199.712x faster by campaign median than the
incumbent Metal upload/reduce/download route across the frozen matrix. Production routes F32
shapes containing at most 2,097,152 elements through CPU and preserves direct Metal above that
measured bound.

Spectackle proposal: `P-01M0FVFC9BFDC`. Task: `T-01M0FVGM88EWM`. Decision:
`ADR-01M0FVWNPKEX6`. Contract: `MEASURED-METAL-BIAS-GRAD-ROUTE-001`.
General perfscan findings: [jxsl13/perfscan#773](https://github.com/jxsl13/perfscan/issues/773)
and [jxsl13/perfscan#774](https://github.com/jxsl13/perfscan/issues/774).

## Why the route changed

The Metal route was introduced when its alternative was a scalar reference column sum. The CPU
backend later gained a typed row-major, column-owned parallel reduction with F64 accumulation, but
the Metal wrapper kept the historical execution choice. At this API boundary the wrapper copies
the full gradient into a pooled shared Metal buffer, submits and synchronizes a one-thread-per-
output reduction, and copies the small bias gradient back. The optimized host kernel avoids that
boundary cost and is bit-identical to the reference accumulation contract.

This is not a claim about a future graph whose gradients remain Metal-resident. The direct Metal
function remains the production route above the measured ceiling and the same-binary benchmark
control for future crossover work.

## Frozen setup

- Base: `1f5b07d48d770bc201dfa220b5f8d48ce5e0e30d`.
- Machine: Apple M2 Pro, macOS 26.5.1, Go 1.26.6, darwin/arm64.
- Control: preserved direct Metal upload/reduce/download implementation.
- Candidate: the production selector, which reaches the optimized CPU backend inside the bound.
- Gate: all three campaign medians at least 1.10x faster; candidate max/min spread at most 3.0x;
  exact reference parity; direct Metal above the bound; full GPT step at least 0.99x; full Metal
  and repository tests.

The first `-benchtime=100x` pilot was non-gating: with one untimed warmup, the sub-microsecond
`[1,512]` candidate ranged from 823.8 ns to 3,737 ns, a 4.54x spread exceeding the frozen 3.0x
ceiling. Production still used direct Metal. Ten untimed warmups reduced that pilot's worst
candidate spread to 1.32x. The final evidence below uses the production selector and was collected
only afterward as three independent processes:

```text
go test ./backend/metal -run '^$' \
  -bench '^BenchmarkMetalAddBiasBackwardRoute$' -benchtime=100x -count=7
```

## Results

Each cell is direct-Metal median / production-selector median in nanoseconds, followed by
control/candidate.

| Shape `[rows,cols]` | Campaign 1 | Campaign 2 | Campaign 3 | Worst speedup |
|---|---:|---:|---:|---:|
| `[1,512]` | 163,857 / 907.9 = 180.479x | 176,905 / 885.8 = 199.712x | 165,117 / 947.9 = 174.192x | 174.192x |
| `[7,512]` | 167,165 / 2,080 = 80.368x | 168,830 / 1,992 = 84.754x | 164,458 / 2,301 = 71.472x | 71.472x |
| `[65,128]` | 161,748 / 3,745 = 43.190x | 163,487 / 3,756 = 43.527x | 153,662 / 3,869 = 39.716x | 39.716x |
| `[256,512]` | 321,943 / 15,663 = 20.554x | 321,813 / 17,099 = 18.821x | 325,877 / 16,764 = 19.439x | 18.821x |
| `[256,2048]` | 365,175 / 76,972 = 4.744x | 372,285 / 75,321 = 4.943x | 370,128 / 74,939 = 4.939x | 4.744x |
| `[512,4096]` | 676,262 / 207,260 = 3.263x | 711,754 / 192,377 = 3.700x | 715,848 / 191,214 = 3.744x | 3.263x |

All 18 medians clear the 1.10x gate. Candidate spreads remained below 3.0x; the worst observed was
1.788x in the sub-microsecond `[1,512]` case.

## End-to-end and correctness

`BenchmarkGPTTrainingStep` was run with `-benchtime=1x -count=7` before and after production
routing. The six-layer D=512, S=256 Metal training step changed from 118,260,625 ns median to
118,976,584 ns, or 0.994x baseline throughput; median throughput changed from 2,165 to 2,152
tokens/s. This clears the 0.99x no-regression gate while keeping the claim appropriately modest.

The winner zone is bit-identical to the reference backend's row-order F64 accumulation for
contiguous and transposed F32 inputs. `TestMetalAddBiasBackwardMeasuredThresholdRoutesBothArms`
pins both sides of `maxHostBiasGradElements`: `[256,512]` matches the CPU route bytewise, while
`[4097,512]` (2,097,664 elements) matches direct Metal bytewise. The complete Metal suite passed
in 46.74 seconds, repository preflight passed, focused in-tree perfscan reported no candidates in
the new route files, and Spectackle reported no drift or errors (only pre-existing repository-wide
advisories and its Go 1.25 versus repository Go 1.26 typed-call limitation).
