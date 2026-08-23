# M2 CPU Q4_K bulk header decode

## Scope

This tranche removes repeated indexed helper calls from the paired ARM64 Q4_K
row kernel. Each super-block now loads the two packed coefficient headers once
per low/high pair and directly decodes both 6-bit scale/min fields. The
coefficient slots, floating-point multiplication order, NEON block kernel, and
row reduction order remain unchanged.

The change is deliberately narrow: it affects only the existing ARM64 paired
Q4_K row path. Independent rows, other quantization formats, non-ARM64 builds,
and all fallback paths retain their established implementations.

## M2 Pro result

Seven interleaved 500 ms process pairs isolate a production-shape K=2,048
paired row. Every candidate process wins, with zero allocations in both
variants and an exact Mann-Whitney result of U=49, p=0.0005828.

| paired Q4_K row | `origin/main` | bulk header | delta |
|---|---:|---:|---:|
| median time | 571.4 ns | 536.4 ns | **-6.13% / 1.065x** |
| allocations | 0 | 0 | unchanged |

The same seven-pair campaign at the TinyLlama FFN paired-apply boundary also
favors every candidate process:

| N=5,632/K=2,048 FFN pair apply | `origin/main` | bulk header | delta |
|---|---:|---:|---:|
| median time | 555.135 us | 528.617 us | **-4.78% / 1.050x** |
| allocations | 30 | 30 | unchanged |

Five fresh-process production pairs alternate baseline/candidate order. Each
process excludes one model/cache warm-up and retains three 64-step decode
samples at the measured 12-thread host optimum. Three of five pairs favor the
candidate; the paired median ratio is 1.017x and aggregate process medians are
neutral-to-positive. The small end-to-end delta is not presented as a
statistically significant throughput claim.

| 64-step CPU decode | `origin/main` | bulk header | delta |
|---|---:|---:|---:|
| median time | 1.8885 s | 1.8838 s | -0.25% / 1.003x |
| median throughput | 33.889 tok/s | 33.974 tok/s | +0.25% |
| median allocated bytes | 173,698,088 | 173,706,984 | +0.005% |
| median allocations | 246,359 | 246,355 | -4 |

Every production sample produced final-logit digest `ea3df5516f17df83`.

## Validation and claim boundary

The arbitrary-header test exercises every packed coefficient byte with a
deterministic noncanonical bit pattern and proves both paired outputs are
bit-identical to independent ARM64 row calls. The complete GGUF test binary
passes under pinned Go 1.26.6 and Go 1.27.0. The package cross-compiles for
Darwin/AMD64, and the leadership harness plus markdown validation pass.

This is an internal current-main M2 improvement, not a cross-library
leadership claim. A matched competitor result still requires identical model,
quantization, token stream, thread policy, and measurement boundaries.

The generalized packed-header opportunity is reported in
[perfscan issue #838](https://github.com/jxsl13/perfscan/issues/838).
