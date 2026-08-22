# M2 Metal IQ4_XS evidence — 2026-08-22

This directory records exact GGUF type-23 IQ4_XS admission and the native Metal
leadership gate on base commit
`0d18e097a5f293cb5750d181a922c53a63b8c989`.

## Environment

- MacBook Pro `Mac14,10`
- Apple M2 Pro, 12 CPU cores (8 performance + 4 efficiency), 32 GB unified memory
- macOS 26.5.1 (25F80), arm64
- Go 1.26.6 darwin/arm64
- Benchmark binary reports `Apple M2 Pro`
- Runtime MSL compilation is the production path; no offline Metal compiler is used

The exact test-binary hashes and tool versions are pinned in `manifest.json`.

## Correctness, GGUF admission, and reachability

`correctness.txt` compares exact 136-byte IQ4_XS super-blocks against
`gguf.QMatMul` across scalar, two-SIMD-group, direct, resident, and recorder
paths. The maximum relative error was `1.086e-5` against the reference and
`1.976e-5` cooperative versus scalar, below the `1e-4` contract. It also
retains Inf/-Inf/NaN classes, rejects malformed layouts, verifies generic
host-bound fallback, and checks input immutability.

`loader.txt` proves that the raw GGUF loader accepts IQ4_XS without expanding or
changing compressed bytes and produces finite logits. `whole-model.txt` proves
that the full resident decoder reaches type 23 and preserves greedy token
sequences. Its median throughput changed from `50.17 tok/s` scalar to
`250.42 tok/s` cooperative: **4.991x**.

## Three independent campaigns

Each campaign is a fresh process, uses 16 distinct compressed resident weights
per shape, keeps all seven samples, performs eight warmups, and reports Metal
GPU timestamps, host API-boundary time, ARM64 CPU time, and an unchanged scalar
control. Each sample runs three measured iterations. No outlier is removed.

The table reports the conservative minimum of the three per-campaign medians.

| Shape (M=1) | Cooperative/scalar GPU | Cooperative/scalar host boundary | Resident Metal/ARM64 wall | Max unchanged-control median |
|---|---:|---:|---:|---:|
| N=512, K=2048 (KV) | 17.67x | 4.683x | 1.811x | 1.002x |
| N=2048, K=2048 (square) | 12.07x | 6.806x | 2.957x | 1.003x |
| N=5632, K=2048 (gate) | 5.179x | 3.920x | 3.933x | 1.002x |
| N=2048, K=5632 (down) | 12.87x | 10.40x | 4.101x | 1.005x |

The conservative minimum across all shape/campaign cells is **5.179x** for the
device kernel, **3.920x** at the scalar-versus-cooperative host boundary, and
**1.811x** for resident Metal over the ARM64 control. The unchanged-control
median stays within 0.5%, below the recorded 1% host-noise band.

Command:

```sh
/private/tmp/goai-iq4xs-metal-final.test -test.run '^$' -test.bench '^BenchmarkMetalIQ4XSCooperativeCampaign$' -test.count=7 -test.benchtime=3x -test.timeout=15m
```

## Routing boundary

`host-route.txt` is a stricter generic-call gate: both candidates start with
host X and compressed W and finish with a host F32 result. Eight distinct
weights, eight warmups, paired order, five measured iterations, and seven
retained samples prevent a cache-hot or ordering decision.

| Shape (M=1) | CPU median | Direct Metal median | Paired CPU advantage |
|---|---:|---:|---:|
| N=512, K=2048 (KV) | 82,740 ns | 210,112 ns | 2.539x |
| N=2048, K=2048 (square) | 188,978 ns | 266,173 ns | 1.400x |
| N=5632, K=2048 (gate) | 432,369 ns | 441,150 ns | 1.043x |
| N=2048, K=5632 (down) | 444,561 ns | 460,167 ns | 1.035x |

This is why generic `Backend.QMatMul` and `Backend.UploadQuant` decline IQ4_XS,
while `llamagpu` explicitly chooses compressed Metal residency and amortizes a
whole decode graph.

Command:

```sh
/private/tmp/goai-iq4xs-metal-final.test -test.run '^$' -test.bench '^BenchmarkMetalIQ4XSHostRouteCampaign$' -test.count=7 -test.benchtime=5x -test.timeout=15m
```

Raw outputs are committed unchanged in the sibling text files.
