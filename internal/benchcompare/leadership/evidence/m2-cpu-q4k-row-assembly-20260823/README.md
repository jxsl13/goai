# M2 CPU Q4_K assembly-resident paired row

## Scope

This tranche moves the complete paired ARM64 Q4_K row loop into one assembly
call. The kernel decodes each packed Q4_K header through the existing f16
lookup table, writes fully overwritten coefficient scratch into its own stack
frame, retains the established dual-output NEON arithmetic, reduces each
super-block in F32, and accumulates those subtotals in ordered F64.

The change removes repeated Go coefficient-array staging, bounds/index work,
and one assembly crossing per 256-value super-block. Independent Q4_K rows,
other quantization formats, non-ARM64 builds, and portable fallbacks are
unchanged.

## Apple M2 Pro results

Seven retained fresh-process pairs follow one discarded warm-up pair. The
K=2,048 paired-row candidate wins every retained pair with zero allocations.

| paired Q4_K row | `origin/main` | assembly row | delta |
|---|---:|---:|---:|
| median time | 545.5 ns | 510.3 ns | **-6.45% / 1.069x** |
| allocations | 0 | 0 | unchanged |

At the TinyLlama N=5,632/K=2,048 FFN paired-apply boundary, every retained
process also favors the candidate.

| TinyLlama FFN pair apply | `origin/main` | assembly row | delta |
|---|---:|---:|---:|
| median time | 561.721 us | 509.870 us | **-9.23% / 1.102x** |
| allocations | 30 | 30 | unchanged |

Five production process pairs alternate execution order. Each process omits
one model/cache warm-up and retains three 64-step decode samples. The
candidate wins three pairs; the paired median ratio is 1.025x and aggregate
process medians improve by 1.042x. This is a production non-regression and
directional gain, not a statistical cross-library throughput claim.

| 64-step CPU decode | `origin/main` | assembly row | delta |
|---|---:|---:|---:|
| median time | 1.8487 s | 1.7734 s | -4.07% / 1.042x |
| median throughput | 34.619 tok/s | 36.089 tok/s | +4.25% |
| final-logit digest | `ea3df5516f17df83` | `ea3df5516f17df83` | exact |

## Rejected alternatives and generalized finding

Folding pointer updates into post-index loads was neutral. Batching eight
blocks with larger Go staging arrays grew the wrapper frame from 224 to 1,216
bytes, zeroed 1,088 bytes, and regressed the leaf by 3.4%. Batch two still
regressed by 1.3%. The winning design eliminates Go staging rather than
enlarging it.

That reusable compiler/kernel-boundary finding is reported as
[perfscan issue #839](https://github.com/jxsl13/perfscan/issues/839).

## Validation and claim boundary

Arbitrary packed headers are bit-identical to two independent ARM64 Q4_K row
dots. The GGUF suite passes with Go 1.26.6 and Go 1.27.0; Darwin/AMD64 and
Linux/ARM64 compile. The full repository run reached all affected packages;
its two unrelated failures are documented in `tests.txt`, including an exact
clean-`origin/main` reproduction of the NLP failure and isolated baseline plus
candidate passes for the host-contended Metal timeout.

This evidence establishes an internal current-main M2 improvement. A
cross-library leadership claim still requires matched model bytes,
quantization, token stream, thread policy, warm/cold state, transfers, and
measurement boundaries.
