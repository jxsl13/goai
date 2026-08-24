# M2 PS4008 KDA and reference MLA assessment

## Scope

Spectackle task `T-01KYMDP9EMFTB` required profile-first assessment of the
remaining PS4008 scalar-accumulator sites in Kimi Delta Attention and reference
Multi-head Latent Attention. The task was drafted against an older tree, so the
first gate was to re-run external perfscan v1.81.0 and profile current main
before editing either implementation.

## KDA result: already resolved

Current `nn/kda.go` emits no PS4008 finding. Its state update already walks one
contiguous row at a time, uses independent AXpy-style element updates, and sends
both reductions through the existing four-accumulator `dot4`. The committed
`BenchmarkKDA_F64_256x128` reports 7.245968 ms/op, 397,992 B/op, and eight
allocations on Apple M2 Pro. A 50-iteration CPU profile attributes 85.29% of
samples cumulatively to `KimiDeltaAttention`, including 20.59% flat in `dot4`;
there is no surviving scalar PS4008 target. Existing bit-identity coverage
already spans state sizes inside and outside L1. No KDA code was changed.

## MLA transform

The reference F64 MLA path still emitted one PS4008 finding in its decoupled
RoPE score dot. The same j-outer/e-inner shape existed in the F32 and generic
paths, although their nearby suppressions happened to apply. A baseline profile
of the existing F64 reference benchmark attributes 130 ms of 840 sampled ms to
the RoPE score loop.

For every attention row, the content score for each key j is still accumulated
over d in ascending order. The shared RoPE key is then transposed once from
`[seq,dR]` to `[dR,seq]`; the loop visits e in the same ascending order while its
inner loop updates independent scores j from contiguous keys. Thus every score
observes exactly the original sequence of floating-point operations, without
multi-accumulator reassociation. Query RoPE, row-major key RoPE, and the
transposed key share one capacity-clamped backing allocation.

## Exactness gate

`TestMLAAddRoPEScoresMatchesFrozenLoop` compares the interchanged helper with a
frozen j-outer/e-inner loop at tolerance zero for full and causal prefixes and
for deliberately dirty initial accumulators. Reversing e makes the test fail:
one manufactured score changes from `-4.999999999999995e+15` to
`-4.999999999999994e+15`, proving the gate detects accumulation-order drift.

`TestMLACPUByteIdenticalToRef` remains bit-identical for F32 and F64 across
three geometries and causal on/off. Full `nn`, `backend/ref`, and `backend/cpu`
package suites pass. External perfscan v1.81.0 reports no PS4008 finding in
either `nn/kda.go` or `backend/ref/mla.go`.

## Apple M2 Pro result

The final campaign uses frozen binaries from merged baseline
`1be2a7ef7e40109b9a00af9aea8d8d6a16e6e6d1` and implementation
`943b2df0cb2e81bc256ff74230b1658093d106a5`. Each fresh process uses
`GOMAXPROCS=1` and runs ten complete F64 reference MLA calls at seq=512,
heads=8, content width=64, RoPE width=32, and causal masking. Seven pairs
alternate process order.

| frozen pre/post | baseline | interchanged | result |
|---|---:|---:|---:|
| median time | 245.062725 ms | 222.883521 ms | **1.0995x faster** |
| paired wins | 2/7 | 5/7 | candidate |
| median bytes/op | 3,282,433 | 3,413,504 | +131,071 B/op |
| median allocations/op | 15 | 14 | one removed |

The median paired speedup is 1.1222x. Absolute times still vary under unrelated
shared-host load, so the conservative separate-sample median is the release
number. The added bytes hold the transposed key view; combining all RoPE scratch
in one backing buffer removes one allocation. The claim is limited to explicit
reference `OpMLA`; the optimized CPU backend and model-level MLA are unchanged.

The raw stream is in `frozen-prepost.tsv`. Baseline executable SHA-256 is
`341869e27ab39d6f19833ce4e86d0fcf4c8667aa2b8c7b1d7eefbb975c543777`;
candidate SHA-256 is
`8b0f174131704ddf37f4f3465cd994e7adac6ad95637f5f023e2e3fa30b0cb26`.
The host used Go 1.27.0, Darwin 26.5.1, and Apple M2 Pro ARM64.

## Reproduction

```sh
GOCACHE=/private/tmp/goai-ps4008-mla-gocache \
  go test -c -o /private/tmp/goai-ps4008-mla.test ./backend/cpu
GOMAXPROCS=1 /private/tmp/goai-ps4008-mla.test \
  -test.run '^$' -test.bench '^BenchmarkMLA_ref_512$' \
  -test.benchmem -test.benchtime=10x -test.count=1
```

The exact-output-axis interchange opportunity is reported as
[perfscan issue #902](https://github.com/jxsl13/perfscan/issues/902).
