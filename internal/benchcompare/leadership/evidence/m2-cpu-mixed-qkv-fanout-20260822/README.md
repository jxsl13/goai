# M2 CPU mixed-shape QKV fan-out

## Scope

This tranche executes eager, single-token CPU Q, K, and V projections under
one work-sized fan-out. `QMatMulTriple` accepts unequal row counts and a mixed
Q4_K/Q6_K quant set while retaining each established row-dot kernel. Every
output is therefore bit-identical to three independent `QMatMul` calls.

The route is deliberately narrow: all three weights must share the input
dimension and use Q4_K or Q6_K, and the activation must be contiguous,
offset-zero F32 with shape `[1,K]`. Recorded execution, accelerators, batches,
unsupported quants, mismatched inputs, and noncontiguous tensors retain the
three independent `QuantLinear` calls.

## M2 Pro result

Twelve fresh-process pairs alternate baseline/candidate order. Every process
excludes one model/pool warm-up and measures one 64-step TinyLlama-1.1B Q4_K_M
CPU decode pass. All twelve pairs favor the candidate.

| 64-step CPU decode | merged main | grouped QKV | delta |
|---|---:|---:|---:|
| time | 1.889 s | 1.787 s | **-5.36%** (`p=0.001`, n=12) |
| allocated bytes | 199.2 MiB | 197.8 MiB | **-0.71%** (`p<0.001`) |
| allocations | 255.7k | 205.0k | **-19.82%** (`p<0.001`) |

Every process produced the exact final-logit digest `ea3df5516f17df83`. The
model, source baseline, and both benchmark binaries are pinned in
`manifest.json`.

The permanent production-shape leaf benchmark isolates scheduler coalescing:

| Q4_K/Q4_K/Q6_K M1, N=2048/256/256, K=2048 | independent | grouped | delta |
|---|---:|---:|---:|
| time | 365.7 us | 287.9 us | **-21.27%** (`p<0.001`, n=10) |
| allocated bytes | 12.34 KiB | 11.31 KiB | **-8.35%** |
| allocations | 63 | 27 | **-57.14%** |

The final-source three-repetition production binary retained the exact digest,
207,360,864 median allocated bytes, and 204,988 median allocations. Its timing
was 2.975 s under variable shared-host load; the order-alternating A/B above is
the accepted timing claim.

## Rejected adjacent design

The first exact prototype concatenated Q, K, and V row ranges before splitting
them. That removed the same allocations but concentrated the slower Q6_K V
tail on the last workers and lost six of eight alternating production pairs.
It was replaced completely. The retained implementation proportionally
partitions every matrix across every scheduler chunk, giving each TinyLlama
worker 256 Q rows, 32 K rows, and 32 V rows.

## Claim boundary

This is an internal merged-main improvement, not a cross-library leadership
claim. The existing llama.cpp comparison remains unmatched because KV dtype,
token stream, and starting position differ. A shared-semantics harness remains
required before publishing an incumbent ratio.

The general sibling-fan-out opportunity remains in
[perfscan issue #830](https://github.com/jxsl13/perfscan/issues/830). The new
heterogeneous load-balance lesson is reported in
[perfscan issue #831](https://github.com/jxsl13/perfscan/issues/831).
