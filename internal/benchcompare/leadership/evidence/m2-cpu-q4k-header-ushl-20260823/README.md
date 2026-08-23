# M2 CPU paired Q4_K table-shift header decode

## Scope

This tranche shortens the packed scale/minimum expansion inside the merged
paired ARM64 Q4_K row kernel. Three constant vectors describe the alternating
lane shifts, paired low-field indexes, and duplicated high-field indexes. Two
`VTBL` operations, one lane-local `USHL`, masks, and one final `VEXT` now build
each row's 16 coefficient bytes.

The prior fixed shuffle network required six more instructions per row. The
new path adds one three-register constant load per super-block, for a net
reduction of ten instructions across its two headers. Coefficient conversion,
the dual-dot schedule, per-super-block F32 reduction, ordered F64 row
accumulation, non-ARM64 builds, and the portable fallback are unchanged. A raw
`USHL` opcode preserves the Go 1.26 compiler floor; Go 1.27 disassembly verifies
the intended instruction.

## Apple M2 Pro results

Seven retained fresh-process pairs follow one discarded warm-up pair. The
K=2,048 paired row wins every retained pair with zero allocations.

| paired Q4_K row | merged PR 1179 | table-shift header | delta |
|---|---:|---:|---:|
| median time | 499.1 ns | 482.4 ns | **-3.35% / 1.035x** |
| paired wins | 0/7 | 7/7 | candidate |
| allocations | 0 | 0 | unchanged |

At the TinyLlama N=5,632/K=2,048 FFN paired-apply boundary, six of seven
retained processes favor the candidate.

| TinyLlama FFN pair apply | merged PR 1179 | table-shift header | delta |
|---|---:|---:|---:|
| median time | 795.902 us | 738.569 us | **-7.20% / 1.078x** |
| paired wins | 1/7 | 6/7 | candidate |
| allocations | 30 | 30 | unchanged |

Seven retained production pairs also follow one discarded warm-up pair. Each
process omits one model/cache warm-up and retains five 64-step decode samples.
Six pairs favor the candidate, and aggregate process medians improve by 1.036x.

| 64-step CPU decode | merged PR 1179 | table-shift header | delta |
|---|---:|---:|---:|
| median time | 1.791947 s | 1.729485 s | **-3.49% / 1.036x** |
| median throughput | 35.715 tok/s | 37.005 tok/s | +3.61% |
| final-logit digest | `ea3df5516f17df83` | `ea3df5516f17df83` | exact |

An earlier partial production pilot is retained in `production-excluded.tsv`
but excluded because an unrelated perfscan suite restarted during the run and
occupied CPU. The clean confirmation began only after its parent orchestrator
and level-3 perfscan process exited.

## Generalized finding

An already-vectorized fixed packed-field decoder can still carry an excessive
shuffle network. Constant table indexes plus signed per-lane shifts can replace
repeated rotates and zips when the source mapping is static and bounded.

The reusable detector opportunity is reported as
[perfscan issue #842](https://github.com/jxsl13/perfscan/issues/842).

## Validation and claim boundary

Adversarial packed headers remain bit-identical to two independent ARM64 Q4_K
rows. The complete GGUF binary passes with Go 1.26.6 and Go 1.27.0;
Darwin/AMD64 and Linux/ARM64 compile. Spectackle lint has zero errors, and
external perfscan completed with its documented pre-existing findings.

The repository-wide run passed every package except the unrelated deterministic
`nlp/TestDiffusionLMGrammarE2E` generation assertion, which reproduced in an
isolated package-wide retry with the same output. This branch has no NLP
changes; required CI remains authoritative before merge.

This evidence establishes an internal current-main M2 improvement. A
cross-library leadership claim still requires matched model bytes,
quantization, token stream, thread policy, warm/cold state, transfers, and
measurement boundaries.
