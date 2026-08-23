# M2 CPU independent Q4_K whole-row assembly

## Scope

This tranche moves independent ARM64 Q4_K row orchestration across the native
boundary. The former Go loop decoded each 256-weight header, staged 16 F32
coefficients, and called the NEON block dot once per super-block. The new
`dotQ4KRowNeon` entry point keeps the merged table-indexed header decode, block
loop, F32 subtotal reduction, and ordered F64 row accumulation inside one
assembly call.

The zero-length path remains in Go. Non-ARM64 routing is unchanged. The new
path preserves the former independent-row output bits rather than changing its
arithmetic schedule.

## Profile basis

On merged main `9f1801c8`, a 15-repetition 64-step CPU profile attributed 4.08%
flat time to `dotQ4KBlockNeon` and 1.16% flat time to the surrounding
`dotQ4_KRowASM`, for 5.26% cumulative independent-row time. That exposed a
larger controllable boundary than further paired-row inner-loop tuning.

## Apple M2 Pro results

Seven retained fresh-process pairs follow one discarded warm-up pair. The
K=2,048 independent row wins every retained pair with zero allocations.

| independent Q4_K row | merged PR 1180 | whole-row assembly | delta |
|---|---:|---:|---:|
| median time | 283.5 ns | 245.4 ns | **-13.44% / 1.155x** |
| paired wins | 0/7 | 7/7 | candidate |
| allocations | 0 | 0 | unchanged |

The M1/N=4,096 Q4_K matrix boundary retains five of seven process wins. Its
29 allocations are scheduler/output ownership and remain unchanged.

| Q4_K M1/N=4096 | merged PR 1180 | whole-row assembly | delta |
|---|---:|---:|---:|
| median time | 157.548 us | 147.343 us | **-6.48% / 1.069x** |
| paired wins | 2/7 | 5/7 | candidate |
| allocations | 29 | 29 | unchanged |

Seven retained production pairs also follow one discarded warm-up pair. Each
process omits one model/cache warm-up and retains five 64-step decode samples.
Every retained pair favors the candidate.

| 64-step CPU decode | merged PR 1180 | whole-row assembly | delta |
|---|---:|---:|---:|
| median time | 1.788325 s | 1.651625 s | **-7.64% / 1.083x** |
| median throughput | 35.788 tok/s | 38.750 tok/s | +8.28% |
| final-logit digest | `ea3df5516f17df83` | `ea3df5516f17df83` | exact |

## Generalized finding

An allocation-free assembly leaf can still be constrained by a Go loop that
performs fixed-size metadata staging and one ABI crossing per block. Moving the
loop across the native boundary can amortize orchestration while retaining the
same block reductions and numerical output.

The reusable detector opportunity is reported as
[perfscan issue #843](https://github.com/jxsl13/perfscan/issues/843).

## Validation and claim boundary

Adversarial packed headers, paired-versus-independent outputs, and the complete
GGUF package remain bit-identical. The package passes with Go 1.26.6 and Go
1.27.0; Darwin/AMD64 and Linux/ARM64 compile. Spectackle lint has zero errors,
and external perfscan completed with its documented pre-existing findings.

The shared-host repository run passed the changed path and all reported
non-GPU packages, but recorded three unrelated environmental or pre-existing
exceptions: Metal timestamp CV exceeded its noise gate, `llamagpu` reached its
10-minute timeout inside a Metal matmul, and the deterministic NLP grammar
assertion reproduced. This branch changes only ARM64 GGUF CPU code; required CI
remains authoritative before merge.

This evidence establishes an internal current-main M2 improvement. A
cross-library leadership claim still requires matched model bytes,
quantization, token stream, thread policy, warm/cold state, transfers, and
measurement boundaries.
