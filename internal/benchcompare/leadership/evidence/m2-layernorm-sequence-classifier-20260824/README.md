# M2 LayerNorm sequence-classifier evidence (2026-08-24)

## Claim and scope

This evidence compares merged `main` at
`8a52aa93ffe860cc733ff8374e0940109a1903aa` with implementation commit
`def1abfe1461e6d47830ab8e0363237d36409f8d`. Both arms use Go 1.27.0 on
macOS 26.5.1 and an Apple M2 Pro with 32 GiB unified memory. The pinned F32
workload is ViT B=8, image 32x32, patch 4, sequence 65, dimension 128, hidden
dimension 512, depth 4, 4 heads, and 10 classes.

The control normalizes all 520 packed rows, executes eight Slice operations and
one Concat to retain the class-token rows, then applies a biased classifier.
The candidate applies the same row-local LayerNorm only to the eight retained
rows before the projection. Forward and the exact five-input VJP are one
autograd operation. The measured production geometry uses the reference host
kernel on Darwin ARM64 and makes zero Metal submissions; unmeasured supported
geometries retain a cached one-submission MPSGraph route. Unsupported backends,
dtypes, layouts, or missing backend directions execute the original composite.

## Frozen gates

- Output, all five operation gradients, complete ViT logits, every model
  parameter gradient, and input immutability must match the control.
- Three fresh-process, order-alternated count-seven campaigns per benchmark,
  `GOMAXPROCS=1`, one-second adaptive samples, and built-in warmups.
- Boundary median speedup at least 1.20x.
- Complete ViT training-step median speedup at least 1.05x in every campaign.
- Every aligned complete-step sample pair at least 1.03x.

## Result

Campaign order is control/candidate, candidate/control, then control/candidate.

| Boundary campaign | Control median | Candidate median | Speedup | Worst aligned pair |
|---|---:|---:|---:|---:|
| 1 | 1.539 ms | 0.083 ms | 18.4516x | 17.9355x |
| 2 | 3.715 ms | 0.088 ms | 42.2081x | 17.4523x |
| 3 | 3.050 ms | 0.136 ms | 22.4701x | 14.5473x |

| ViT campaign | Control median | Candidate median | Speedup | Worst aligned pair |
|---|---:|---:|---:|---:|
| 1 | 17.288 ms | 14.228 ms | 1.2150x | 1.1477x |
| 2 | 23.956 ms | 20.832 ms | 1.1500x | 1.0487x |
| 3 | 19.387 ms | 13.120 ms | 1.4777x | 1.3877x |

All frozen gates pass. Absolute latency shifts with sustained system load, but
the candidate wins in every accepted pair. At the boundary it reduces
4,635,736 to 462,384 B/op and 245 to 111 allocs/op. At complete-step scope it
reduces 15,752,384 to 11,575,984 B/op and 1,271 to 1,008 allocs/op: 4,176,400
bytes and 263 allocations removed per step.

One earlier same-binary control-first screening run passed the median gate but
failed one aligned pair when a candidate sample spiked to 18.285 ms. It was
rejected rather than hidden or used to weaken the gate. Separating boundary and
complete-step campaigns into fresh processes produced the accepted evidence
above; the rejected samples remain in `samples.txt`.

## Design finding

A cached MPSGraph version improved the composite boundary during screening, but
the small synchronous production boundary remained much faster on the CPU over
Apple unified memory. The larger win is semantic dead-work elimination:
LayerNorm is row-local, so selecting the class rows before normalization avoids
processing 512 rows that cannot affect the classifier. This generalized static
analysis opportunity is tracked in
[perfscan issue 874](https://github.com/jxsl13/perfscan/issues/874).

The exact benchmark binary is
`goai-metal-sequence-classifier-def1abfe.test`, SHA-256
`b5107beb1216543169c510c717ee97b54ad9f230aa5a241d47f98e7907b44595`.
Raw measurements, commands, correctness gates, and scanner results are stored
beside this file.
