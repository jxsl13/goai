# M2 CPU paired Q4_K decode projections

## Scope

This tranche executes the Q4_K gate and up projections of eager, single-token
CPU QuantSwiGLU under one output-row fan-out. `QMatMulPair` allocates the two
outputs, chooses one worker layout, and calls the existing Q4_K row dot once
per matrix and output row. The row arithmetic is unchanged, so both outputs
are bit-identical to two independent `QMatMul` calls.

The route is deliberately narrow: both weights must be Q4_K with equal input
and output dimensions, and the activation must be contiguous, offset-zero F32
with shape `[1,K]`. Recorded/autograd execution, non-CPU backends, other quant
types, batches, and unsupported layouts retain the established independent
projection path.

## M2 Pro result

Ten retained fresh-process samples follow one discarded process pair. Order is
alternated baseline/candidate. Every process excludes one model/pool warm-up
and reports the median of three measured 64-step TinyLlama-1.1B Q4_K_M decode
passes.

| 64-step CPU decode | merged main | paired projections | delta |
|---|---:|---:|---:|
| time | 2.029 s | 1.948 s | **-3.99%** (`p=0.043`, n=10) |
| allocated bytes | 199.9 MiB | 199.2 MiB | **-0.36%** (`p=0.000`) |
| allocations | 281.0k | 255.7k | **-9.02%** (`p=0.000`) |

All runs produced the exact final-logit digest `ea3df5516f17df83`. The model
and both benchmark binaries are pinned in `manifest.json`.

The permanent paired-projection leaf benchmark isolates the shared fan-out:

| Q4_K M1/N4096/K1024 | independent | paired | delta |
|---|---:|---:|---:|
| time | 301.3 us | 245.5 us | **-18.52%** (`p=0.029`, n=10) |
| allocated bytes | 34,368 B | 33,824 B | **-1.58%** |
| allocations | 42 | 24 | **-42.86%** |

The final-source seven-repetition production binary retained the exact digest,
208,847,664 median allocated bytes, and 255,674 median allocations. Its timing
was 2.146 s under variable shared-host load; the order-alternating A/B above is
the accepted timing claim.

## Rejected adjacent design

A preceding work-first scheduler experiment made the caller compute one chunk
and launched one fewer goroutine per `qmatmulParallelChunks` call. It preserved
exactness and reduced leaf allocations by 9.52%, but the production median
regressed 5.66% (`p=0.019`, n=10). That rewrite was fully removed. The accepted
design reduces the number of complete fan-outs while retaining the established
scheduler inside each fan-out.

## Claim boundary

This is an internal merged-main improvement, not a cross-library leadership
claim. The existing llama.cpp CPU comparison remains unmatched because its KV
dtype, token stream, and starting position differ. A shared-dtype/token-stream
harness remains required before publishing an incumbent ratio.

The reusable sibling-kernel opportunity is reported in
[perfscan issue #830](https://github.com/jxsl13/perfscan/issues/830). The
rejected scheduler-only lesson is tracked in
[perfscan issue #829](https://github.com/jxsl13/perfscan/issues/829).
