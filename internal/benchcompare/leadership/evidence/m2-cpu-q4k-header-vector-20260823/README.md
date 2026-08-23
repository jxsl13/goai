# M2 CPU paired Q4_K vector header decode

## Scope

This tranche replaces the scalar Q4_K coefficient builder inside the merged
paired ARM64 row kernel with two NEON packed-header decodes. Each row loads its
12-byte scale/min header once, extracts all eight scale/min pairs with vector
masks, shifts, zips, and table lookups, converts the bytes in four lanes, and
multiplies by alternating exact f16-table `d`/`dmin` values.

The dual-dot schedule, per-super-block F32 reduction, ordered F64 row
accumulation, non-ARM64 builds, and portable fallback are unchanged. Raw FMUL
encodings retain the Go 1.26 compiler floor; Go 1.27 disassembly verifies the
intended V20/V22 vector multiplies.

## Apple M2 Pro results

Seven retained fresh-process pairs follow one discarded warm-up pair. The
K=2,048 paired row wins every retained pair with zero allocations.

| paired Q4_K row | merged PR 1178 | vector header | delta |
|---|---:|---:|---:|
| median time | 511.7 ns | 480.9 ns | **-6.02% / 1.064x** |
| allocations | 0 | 0 | unchanged |

At the TinyLlama N=5,632/K=2,048 FFN paired-apply boundary, six of seven
retained processes favor the candidate. One isolated candidate process is a
large thermal/scheduler outlier; the retained median remains decisive.

| TinyLlama FFN pair apply | merged PR 1178 | vector header | delta |
|---|---:|---:|---:|
| median time | 532.716 us | 475.400 us | **-10.76% / 1.121x** |
| allocations | 30 | 30 | unchanged |

Seven production process pairs alternate execution order. Each process omits
one model/cache warm-up and retains five 64-step decode samples. Every pair
favors the candidate; the median paired ratio is 1.060x and aggregate process
medians improve by 1.066x.

| 64-step CPU decode | merged PR 1178 | vector header | delta |
|---|---:|---:|---:|
| median time | 1.9114 s | 1.7929 s | **-6.20% / 1.066x** |
| median throughput | 33.483 tok/s | 35.697 tok/s | +6.61% |
| final-logit digest | `ea3df5516f17df83` | `ea3df5516f17df83` | exact |

An earlier five-pair production pilot is retained in
`production-excluded.tsv` but excluded from acceptance because an unrelated
level-3 perfscan process in another workspace occupied a core throughout it.
The uncontaminated confirmation began only after that process exited.

## Generalized finding

Repeated scalar expansion of a fixed packed header inside an otherwise SIMD
loop can dominate the remaining kernel cost. Loading the header once and
performing bounded field extraction, conversion, and coefficient formation in
vectors removed that staging bottleneck without changing arithmetic order.

The reusable detector opportunity is reported as
[perfscan issue #841](https://github.com/jxsl13/perfscan/issues/841).

## Validation and claim boundary

Arbitrary packed headers remain bit-identical to two independent ARM64 Q4_K
rows. The complete GGUF binary passes with Go 1.26.6 and Go 1.27.0; Darwin/AMD64
and Linux/ARM64 compile. Spectackle lint has zero errors, and external perfscan
completed with its documented pre-existing findings.

The repository-wide run passed every package except the unrelated deterministic
`nlp/TestDiffusionLMGrammarE2E` generation assertion, which reproduced in an
isolated package-wide retry; this branch has no NLP changes. Required CI must
still be fully green before merge.

This evidence establishes an internal current-main M2 improvement. A
cross-library leadership claim still requires matched model bytes,
quantization, token stream, thread policy, warm/cold state, transfers, and
measurement boundaries.
