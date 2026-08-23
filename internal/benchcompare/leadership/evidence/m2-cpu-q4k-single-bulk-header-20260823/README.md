# M2 CPU independent Q4_K bulk header decode

## Scope

This tranche removes repeated indexed helper calls from the independent ARM64
Q4_K row kernel. Each super-block now loads one packed coefficient header once
per low/high pair and directly decodes both 6-bit scale/min fields. Coefficient
slots, floating-point operands, the NEON block kernel, super-block subtotals,
and the F64 row reduction remain unchanged.

The route remains narrow: only the existing ARM64 independent Q4_K M1 row path
changes. Paired rows, other quantization formats, non-ARM64 builds, and all
portable/prefill paths keep their established implementations.

## M2 Pro result

Seven alternating 500 ms process pairs isolate a production-shape K=2,048
independent row. Every candidate process wins, with zero allocations in both
variants and an exact Mann-Whitney result of U=49, p=0.0005828.

| independent Q4_K row | `origin/main` | bulk header | delta |
|---|---:|---:|---:|
| median time | 312.0 ns | 288.6 ns | **-7.50% / 1.081x** |
| allocations | 0 | 0 | unchanged |

The same seven-pair campaign at the TinyLlama mixed Q4_K/Q4_K/Q6_K QKV
boundary favors five candidate processes. Two late processes in each arm were
affected by unrelated host work, so this boundary is supporting evidence, not
a statistical claim (U=36, p=0.1649).

| K=2,048 mixed QKV triple | `origin/main` | bulk header | delta |
|---|---:|---:|---:|
| median time | 190.126 us | 182.325 us | **-4.10% / 1.043x** |
| allocations | 35 | 35 | unchanged |

Five clean-window production pairs alternate baseline/candidate order at eight
threads. Each process excludes one model/cache warm-up and retains three
64-step decode samples. Four of five pairs favor the candidate and the median
paired ratio is 1.060x. Each arm contains one large outlier, so aggregate
medians are recorded for regression safety but are not promoted as a
publishable end-to-end throughput claim.

| 64-step CPU decode | `origin/main` | bulk header |
|---|---:|---:|
| median time | 1.8048 s | 1.5965 s |
| median throughput | 35.461 tok/s | 40.086 tok/s |
| median allocated bytes | 172,411,672 | 172,408,624 |
| median allocations | 200,767 | 200,762 |

Every production sample produced final-logit digest `ea3df5516f17df83`.

An earlier 12-thread campaign is retained as invalidated evidence. Two
unrelated `perfscan/checks` race suites consumed roughly four to six CPU cores
during that run; both binaries drifted from about 1.8 seconds to 3.1 seconds.
It is not used for the integration decision.

## Validation and claim boundary

A frozen helper-based test oracle exercises arbitrary packed coefficient bytes
and proves the optimized row is bit-identical to the old header path. The
complete GGUF test binary passes under pinned Go 1.26.6 and Go 1.27.0. The
package cross-compiles for Darwin/AMD64, the local preflight passes, and
Spectackle reports no spec/code drift beyond pre-existing lint warnings.

This is an internal current-main M2 improvement, not a cross-library
leadership claim. A matched competitor result still requires identical model,
quantization, token stream, thread policy, and measurement boundaries.

The reusable packed-header pattern and this second data point are reported in
[perfscan issue #838](https://github.com/jxsl13/perfscan/issues/838#issuecomment-5384272037).
