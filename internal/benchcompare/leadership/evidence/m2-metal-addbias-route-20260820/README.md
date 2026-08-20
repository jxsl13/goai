# M2 Metal Bias-Add Route Revalidation (2026-08-20)

## Outcome

Retained with a measured ceiling. At GoAI's synchronous host-resident tensor boundary, the
optimized CPU row-broadcast kernel is 2.506x to 288.131x faster by campaign median than the
incumbent Metal upload/add/download route across the frozen matrix. Production routes valid F32
shapes containing at most 8,388,608 elements through CPU and preserves direct Metal above that
measured bound.

Spectackle proposal: `P-01M0FX1QZ9F08`. Task: `T-01M0FX2EBSFC2`. Decision:
`ADR-01M0FX3E7JFNB`. Contract: `MEASURED-METAL-BIAS-ROUTE-001`.
General perfscan finding: [jxsl13/perfscan#773](https://github.com/jxsl13/perfscan/issues/773#issuecomment-5358275330).

## Why the route changed

ADR-0008 already routes same-shape binary elementwise work through the optimized CPU because a
single host pass beats synchronous GPU transfer and dispatch. AddBias retained an older direct
Metal route even after the CPU backend gained a typed SIMD row-broadcast kernel. At this API
boundary the Metal wrapper copies the activation and bias into shared buffers, submits and waits
for one memory-bound kernel, then copies the result back. The CPU kernel allocates the same logical
output but completes the broadcast in one host pass.

This is not a claim about a future graph whose inputs remain Metal-resident. The direct Metal
function remains the production route above the measured ceiling and the same-binary benchmark
control for future crossover work.

## Frozen setup

- Base: `27f4a64d1868dab15a8543a004169b2bea0e4d96`.
- Machine: Apple M2 Pro, macOS 26.5.1, Go 1.26.6, darwin/arm64.
- Control: preserved direct Metal upload/add/download implementation.
- Candidate: the production selector, which reaches the optimized CPU backend inside the bound.
- Gate: all three campaign medians at least 1.10x faster; candidate max/min spread at most 3.0x;
  exact reference parity; direct Metal above the bound; full GPT step at least 0.99x; full Metal
  and repository tests.

A non-gating `-benchtime=20x -count=3` pilot kept production on direct Metal and established that
CPU won from 512 through 8,388,608 elements with worst candidate spread below 3.0x. The final
evidence below uses the production selector and was collected afterward as three independent
processes with ten untimed warmups per arm:

```text
go test ./backend/metal -run '^$' \
  -bench '^BenchmarkMetalAddBiasRoute$' -benchtime=100x -count=7
```

## Results

Each cell is direct-Metal median / production-selector median in nanoseconds, followed by
control/candidate.

| Shape `[rows,cols]` | Campaign 1 | Campaign 2 | Campaign 3 | Worst speedup |
|---|---:|---:|---:|---:|
| `[1,512]` | 145,506 / 505.0 = 288.131x | 138,383 / 490.0 = 282.414x | 132,611 / 561.7 = 236.089x | 236.089x |
| `[7,512]` | 136,136 / 1,800 = 75.631x | 133,985 / 1,787 = 74.978x | 136,336 / 1,717 = 79.404x | 74.978x |
| `[65,128]` | 137,891 / 5,755 = 23.960x | 137,064 / 5,892 = 23.263x | 137,828 / 6,257 = 22.028x | 22.028x |
| `[256,512]` | 189,453 / 74,760 = 2.534x | 190,787 / 76,141 = 2.506x | 192,530 / 74,302 = 2.591x | 2.506x |
| `[256,2048]` | 376,459 / 142,514 = 2.642x | 370,668 / 139,940 = 2.649x | 355,310 / 133,361 = 2.664x | 2.642x |
| `[512,4096]` | 871,330 / 307,347 = 2.835x | 856,941 / 307,827 = 2.784x | 891,520 / 330,613 = 2.697x | 2.697x |
| `[1024,4096]` | 1,643,479 / 495,013 = 3.320x | 1,648,335 / 496,846 = 3.318x | 1,659,560 / 503,858 = 3.294x | 3.294x |
| `[2048,4096]` | 2,687,171 / 1,037,012 = 2.591x | 2,689,747 / 1,034,032 = 2.601x | 2,714,836 / 1,033,066 = 2.628x | 2.591x |

All 24 medians clear the 1.10x gate. Candidate spreads remained below 3.0x; the worst observed was
1.631x in the sub-microsecond `[1,512]` case.

## End-to-end and correctness

`BenchmarkGPTTrainingStep/metal` was run with `-benchtime=1x -count=7` on an isolated
`origin/main` worktree and on the candidate. The six-layer D=512, S=256 Metal training step changed
from 122,156,167 ns median to 120,187,833 ns; median throughput changed from 2,096 to 2,130
tokens/s, a 1.016x improvement. This clears the 0.99x no-regression gate.

The winner zone is bit-identical to the reference backend for contiguous and transposed F32
inputs. `TestMetalAddBiasMeasuredThresholdRoutesBothArms` pins both sides of
`maxHostBiasElements`: `[256,4096]` matches the CPU route bytewise, while `[2049,4096]`
(8,392,704 elements) matches direct Metal bytewise.

The complete Metal suite passed in 45.253 seconds, repository preflight passed, and focused
in-tree perfscan reported no candidates in the new route files. Spectackle reported no drift or
errors; only pre-existing repository-wide W001/W002 advisories, the existing `metal_test.go:712`
VAC warning, and its Go 1.25 versus repository Go 1.26 typed-call limitation remain.
