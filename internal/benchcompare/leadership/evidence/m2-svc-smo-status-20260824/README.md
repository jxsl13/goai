# M2 SVC SMO active-set status cache

## Scope

The SVC SMO solver scans every variable twice per iteration. Both scans used to
recompute `I_up` and `I_low` membership from the immutable label and the current
alpha value. An accepted SMO step changes only two alpha values, so the retained
implementation stores both predicates in one status byte per variable and
refreshes only those two entries.

The cache preserves the original ascending traversal, comparison operators,
tie behavior, alpha arithmetic, kernel routing, and public API. Alpha and the
error vector now share one backing allocation; the status byte slice replaces
the allocation this removes. Spectackle proposal `P-01M0TMV3V1EZR`, task
`T-01M0TMY9SGFS7`, and rules `SVC-SMO-STATUS-CACHE-EXACT-001` /
`SVC-SMO-STATUS-CACHE-GATE-001` bind the implementation and retention gate.

## Correctness gate

The fixed 4,000x20 RBF workload still converges in exactly 79 SMO steps and
retains 42 support vectors. Scalar and SIMD decision signs match, their maximum
absolute decision delta remains `3.3306690738754696e-15`, and repeated SIMD fits
remain bit-identical. A separate portable table test proves that the two cached
bits match the original predicates at alpha zero, inside the box, and at C for
both labels.

`go test ./classic` passes on the measured host. A SIMD short whole-tree run
also passed `classic`; unrelated existing architecture-sensitive failures were
observed in `autograd`, `nlp`, and `nn`, so that run is not used as a candidate
correctness claim.

## Apple M2 Pro result

The primary gate and reversed-start confirmation use frozen binaries from
baseline `40af00d467b7825c3af2280f607677637c2b0d1b` and implementation
`5c7281458b206055a82f395eb68d09e032be1292`. Each fresh process runs 300
complete 4,000x20 RBF fits with `GOMAXPROCS=12`. The two seven-pair campaigns
reverse their starting arm, leaving the combined 14-run set exactly balanced:
each binary runs first seven times.

| frozen pre/post | baseline | status cache | result |
|---|---:|---:|---:|
| median time, 14 runs | 6.701902 ms | 6.230004 ms | **1.0757x faster** |
| paired wins | 4/14 | 10/14 | candidate |
| median bytes/op | 1,618,940.5 | 1,623,086 | +4,145.5 B/op |
| median allocations/op | 1,038 | 1,038 | unchanged |

The original seven-pair gate independently reports 5.902029 to 5.466776 ms,
or **1.0796x**, with five paired wins. The reversed-start confirmation is much
noisier in absolute time; the combined balanced-order result above is therefore
the primary claim. One candidate sample rounded to 1,039 allocations/op while
the other thirteen candidate samples and all fourteen baselines reported
1,038; the median allocation count and source-level allocation substitution are
unchanged.

Absolute samples span roughly 2x because unrelated perfscan jobs remained
active on the shared host. This evidence is an internal M2 improvement signal,
not an absolute latency or cross-library leadership claim. The complete stream
is in `frozen-prepost.tsv`.

## Reproduction

```sh
GOCACHE=/private/tmp/goai-svcstatus-gocache GOEXPERIMENT=simd \
  go test -c -o /private/tmp/goai-svcstatus.test ./classic
GOMAXPROCS=12 /private/tmp/goai-svcstatus.test \
  -test.run '^$' \
  -test.bench '^BenchmarkSVCFit/n4000_rbf$' \
  -test.benchmem -test.benchtime=300x -test.count=1
```

The baseline executable SHA-256 is
`59e11cba7df72ff2a0691b4efc997fa6c251f0c0f28edfed2d108484abac2196`;
the candidate SHA-256 is
`2314ea3d3d4d1ff2632cb50beade61cc04a3564ad725ed04be3e01bb2c2c396a`.
The host used Go 1.27.0, Darwin 26.5.1, and ARM64.

## Generalized finding

Repeated full scans can cache a compact finite-state predicate when each
iteration mutates only a bounded number of elements. A safe detector must also
identify every mutation path and require exact refresh, traversal, and tie
semantics. The reusable opportunity is tracked as
[perfscan issue #901](https://github.com/jxsl13/perfscan/issues/901).
