# M2 CPU quantized SwiGLU in-place fusion

## Scope

This tranche removes two full hidden-width tensors from eager quantized
SwiGLU execution. The CPU backend overwrites the private gate projection with
`SiLU(gate) * up` and passes that storage directly to the down projection.
Recorded/autograd execution and unsupported backends retain the established
explicit `OpSiLU` then `OpMul` graph.

The capability is deliberately narrow: equal-shape, contiguous, offset-zero,
distinct-storage F32 projections only. It never mutates a public input or the
`up` projection. Exact-bit tests cover scalar and vector-width boundary sizes,
and an end-to-end QuantSwiGLU test proves the eager result matches the recorded
fallback while the recorder still observes both public operations.

## M2 Pro result

Ten retained fresh-process samples follow one discarded process pair. The
production campaign alternates baseline/candidate order; every process runs
one excluded model/pool warm-up and three measured 64-step TinyLlama-1.1B
Q4_K_M decode passes. The process median is the statistical sample.

| 64-step CPU decode | merged main | in-place fusion | delta |
|---|---:|---:|---:|
| time | 2.211 s | 1.982 s | **-10.35%** (`p=0.011`, n=10) |
| allocated bytes | 266.7 MiB | 199.9 MiB | **-25.04%** (`p=0.000`) |
| allocations | 296.5k | 281.0k | **-5.22%** (`p=0.000`) |

All baseline and candidate runs produced the exact final-logit digest
`ea3df5516f17df83`. The model SHA-256 is pinned in `manifest.json`.

The permanent hidden-width leaf benchmark isolates the removed middle:

| 5,632-wide SwiGLU middle | composed ops | in-place fusion | delta |
|---|---:|---:|---:|
| time | 44.01 us | 39.53 us | **-10.18%** (`p=0.000`, n=10) |
| allocated bytes | 49,800 B | 64 B | **-99.87%** |
| allocations | 12 | 1 | **-91.67%** |

The permanent leaf benchmark uses a 500 ms benchtime; its first sample per
sub-benchmark is discarded and the two sub-benchmark names are normalized to
one benchstat key. The candidate seven-repetition profile reports 2.003 s
median, 31.952 tok/s,
209,612,176 median allocated bytes, 281,015 median allocations, and the same
digest. Relative to the baseline profile, cumulative `heapAllocator.allocF32`
space falls from 2,195.30 MB to 1,694.68 MB. CPU samples remain dominated by
worker-pool condition waits, which is not interpreted as an independent
bottleneck claim.

After the unsupported-layout fallback was cleaned up to reuse an already
computed `up` projection, the exact final-source binary was rebuilt and run
for seven measured repetitions. It confirms a 1.999 s median, 209,612,192
allocated bytes, 281,015 allocations, and digest `ea3df5516f17df83`. The
accepted CPU fusion hot path measured by the alternating campaign is
unchanged; both A/B and final-verification binary hashes are retained.

## Claim boundary

This is an internal merged-main improvement, not a cross-library leadership
claim. The existing llama.cpp CPU comparison remains intentionally unmatched
because its KV dtype and token stream differ. A shared-dtype/token-stream
harness is still required before publishing an incumbent ratio.

The generalizable last-use/private-intermediate opportunity is reported in
[perfscan issue #828](https://github.com/jxsl13/perfscan/issues/828).
