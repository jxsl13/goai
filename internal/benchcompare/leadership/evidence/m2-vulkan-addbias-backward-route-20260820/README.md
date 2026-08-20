# M2 Vulkan Bias-Gradient Route Revalidation (2026-08-20)

## Outcome

Retained with a measured ceiling. At GoAI's synchronous host-resident tensor boundary, the later
bit-exact parallel CPU column reduction is 3.27x to 451.29x faster by campaign median than the
incumbent MoltenVK upload/reduce/download route across the frozen matrix. Production routes F32
shapes containing at most 2,097,152 elements through CPU and preserves direct Vulkan above that
measured bound.

Spectackle proposal: `P-01M0FS2S8AFRK`. Task: `T-01M0FS3HRCE44`. Decision:
`ADR-01M0FS4JSXFD4`. Contract: `MEASURED-VULKAN-BIAS-GRAD-ROUTE-001`.
General perfscan finding: [jxsl13/perfscan#773](https://github.com/jxsl13/perfscan/issues/773).

## Why the route changed

The Vulkan route was introduced when its alternative was a scalar reference column sum. The CPU
backend later gained a typed row-major, column-owned parallel reduction with F64 accumulation, but
the Vulkan wrapper kept the historical execution choice. At this API boundary the wrapper uploads
the full gradient, submits and synchronizes a one-thread-per-output reduction, and downloads the
small bias gradient. The optimized host kernel avoids that boundary cost and is bit-identical to
the reference accumulation contract.

This is not a claim about a future graph whose gradients remain Vulkan-resident. The direct Vulkan
function remains the production route above the measured ceiling and the same-binary benchmark
control for future crossover work.

## Frozen setup

- Base: `ba625e6e8ab5d4d8e430335d17ed1385330c9e5a`.
- Machine: Apple M2 Pro, macOS 26.5.1, Go 1.26.6, darwin/arm64.
- Vulkan path: MoltenVK via `/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json`.
- Control: preserved direct Vulkan upload/reduce/download implementation.
- Candidate: the production selector, which reaches the optimized CPU backend inside the bound.
- Gate: all three campaign medians at least 1.10x faster; candidate max/min spread at most 3.0x;
  exact reference parity; direct Vulkan above the bound; full GPT step at least 0.99x; full Vulkan
  and repository tests.

The initial `-benchtime=1x` pilot was non-gating: a 3–19 microsecond candidate case had a 5.57x
max/min spread, exceeding the frozen 3.0x ceiling. No production route had been enabled. The final
command uses 100 operations per sample to move timer and scheduler granularity below the signal,
and was run as three independent processes:

```text
go test -tags=vulkan ./backend/vulkan -run '^$' \
  -bench '^BenchmarkVulkanAddBiasBackwardRoute$' -benchtime=100x -count=7
```

## Results

Each cell is direct-Vulkan median / production-selector median in nanoseconds, followed by
control/candidate.

| Shape `[rows,cols]` | Campaign 1 | Campaign 2 | Campaign 3 | Worst speedup |
|---|---:|---:|---:|---:|
| `[1,512]` | 467,539 / 1,036 = 451.292x | 278,798 / 899.6 = 309.913x | 362,068 / 1,108 = 326.776x | 309.913x |
| `[7,512]` | 343,469 / 2,200 = 156.122x | 267,075 / 2,032 = 131.435x | 230,289 / 2,134 = 107.914x | 107.914x |
| `[65,128]` | 360,360 / 4,175 = 86.314x | 395,980 / 4,038 = 98.063x | 270,846 / 4,073 = 66.498x | 66.498x |
| `[256,512]` | 399,215 / 17,685 = 22.574x | 436,827 / 23,061 = 18.942x | 443,262 / 16,954 = 26.145x | 18.942x |
| `[256,2048]` | 536,202 / 78,073 = 6.868x | 461,633 / 83,647 = 5.519x | 529,015 / 75,416 = 7.015x | 5.519x |
| `[512,4096]` | 888,450 / 210,235 = 4.226x | 941,936 / 192,720 = 4.888x | 727,267 / 222,582 = 3.267x | 3.267x |

All 18 medians clear the 1.10x gate. Candidate spreads remained below 3.0x; the worst observed was
1.758x in the sub-microsecond `[1,512]` case.

## End-to-end and correctness

`BenchmarkGPTTrainingStepVK` was run with `-benchtime=1x -count=7` before and after production
routing. The six-layer D=512, S=256 training step improved from 224,503,875 ns median to
222,645,416 ns, or 1.008x; median throughput increased from 1,140 to 1,150 tokens/s. This clears
the 0.99x no-regression gate while keeping the claim appropriately modest.

The winner zone is bit-identical to the reference backend's row-order F64 accumulation for
contiguous and transposed F32 inputs. `TestVulkanAddBiasBackwardMeasuredThresholdRoutesBothArms`
pins both sides of `maxHostBiasGradElements`: `[256,512]` matches the CPU route bytewise, while
`[4097,512]` (2,097,664 elements) matches direct Vulkan bytewise. The full tagged Vulkan/MoltenVK
suite and repository preflight passed; focused perfscan reported no candidate anti-patterns after
the two-arm threshold test was added.
