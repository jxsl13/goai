# M2 Metal IQ4_NL evidence — 2026-08-22

This directory records the native GGUF type-20 IQ4_NL admission and Metal
leadership gate on the exact base commit
`cbd1ab2448150bfb8d6119deb7fa92f8d08a0df5`.

## Environment

- MacBook Pro `Mac14,10`
- Apple M2 Pro, 12 CPU cores (8 performance + 4 efficiency), 32 GB unified memory
- macOS 26.5.1 (25F80), arm64
- Go 1.26.6 darwin/arm64
- Benchmark binary reports `Apple M2 Pro`
- The standalone `xcrun metal` toolchain is not installed; runtime MSL compilation
  through Metal is the production path exercised here.

## Correctness and reachability

`correctness.txt` compares exact GGUF type-20 block bytes against
`gguf.QMatMul` across scalar, cooperative, direct, device-resident, and recorder
paths. The maximum relative error was `9.961e-7` against the reference and
`3.851e-6` cooperative versus scalar, below the `2e-5` contract. It also
retains Inf/-Inf/NaN classes, rejects malformed layouts, verifies generic
host-bound fallback, and checks input immutability.

`whole-model.txt` proves that the full resident decoder reaches IQ4_NL and
preserves greedy token sequences. Its median throughput changed from
`49.93 tok/s` scalar to `212.82 tok/s` cooperative: **4.262x**.

## Three independent campaigns

Each campaign is a fresh process, uses 16 distinct compressed resident weights
per shape, keeps all seven samples and outliers, alternates AB/BA ordering, and
reports Metal GPU timestamps plus wall time. Each sample runs three measured
iterations after warm-up.

| Shape (M=1) | Minimum campaign median cooperative/scalar GPU | Minimum campaign median resident Metal/CPU wall |
|---|---:|---:|
| N=512, K=2048 (KV) | 10.04x | 1.114x |
| N=2048, K=2048 (square) | 9.215x | 2.089x |
| N=5632, K=2048 (gate) | 4.250x | 2.646x |
| N=2048, K=5632 (down) | 10.71x | 2.932x |

The conservative minimum across all shape/campaign cells is **4.250x** for the
kernel transform and **1.114x** for the amortized resident Metal path over the
ARM64 CPU control.

Command:

```sh
/private/tmp/goai-iq4-metal-final.test -test.run '^$' -test.bench '^BenchmarkMetalIQ4NLCooperativeCampaign$' -test.count 7 -test.benchtime 3x -test.timeout 15m
```

## Routing boundary

A separate equal-boundary benchmark uploads/downloads activations and returns a
Go tensor for both candidates while keeping only the compressed Metal weight
resident. The seven-sample medians show that ARM64 must remain the generic
single-operation route:

| Shape (M=1) | CPU median | Metal host-I/O median | CPU advantage |
|---|---:|---:|---:|
| N=512, K=1024 | 47,378 ns | 162,620 ns | 3.432x |
| N=2048, K=2048 | 171,442 ns | 229,528 ns | 1.339x |
| N=4096, K=1024 | 206,267 ns | 390,327 ns | 1.892x |

This is why `Backend.UploadQuant` and generic `Backend.QMatMul` decline
IQ4_NL, while `llamagpu` explicitly chooses compressed Metal residency and
amortizes command execution across the full decode graph.

Command:

```sh
/private/tmp/goai-iq4-metal-final.test -test.run '^$' -test.bench '^BenchmarkMetalIQ4NLLeadership$' -test.count 7 -test.benchtime 200ms -test.timeout 10m
```

Raw outputs are committed unchanged in the sibling text files.
