# M2 native Metal Q4_1 evidence

Date: 2026-08-22
Base: origin/main 8e0a0d8216124320bf02d9d06ceaae1b9796c36e
Hardware: Apple M2 Pro, 12 CPU cores, 32 GiB unified memory
Software: macOS 26.5.1 (25F80), Go 1.26.6 darwin/arm64, Spectackle 0.9.3

## Claim boundary

This tranche does not claim that every Metal call beats the M2 CPU. It publishes two separate cells:

- Host input and output: generic Backend.QMatMul and Backend.UploadQuant keep Q4_1 on the fused ARM64 CPU path because a standalone Metal submission loses.
- Device-resident decoder: llamagpu explicitly uploads exact GGUF type-3 blocks and records Q4_1 inside a whole decode command. The two-SIMD-group kernel is the retained production route.

Both paths use the same Q4_1 semantics: 32 values per 20-byte block, little-endian f16 d and m, split-half nibbles, value d*q+m.

## Three fresh-process count-seven campaigns

Every campaign used 16 distinct resident weights per shape to avoid a cache-hot single-weight artifact. GPU metrics use Metal command-buffer timestamps; wall metrics include recorder construction, encoding, submit, and wait. AB/BA order alternates. All 84 rows, including outliers, are in campaign-{1,2,3}.txt.

| Shape (N,K) | Campaign | Cooperative/scalar GPU | Metal/CPU steady-state wall |
|---|---:|---:|---:|
| KV (512,2048) | 1 | 8.487x | 1.462x |
| | 2 | 8.360x | 1.810x |
| | 3 | 8.359x | 1.529x |
| Square (2048,2048) | 1 | 6.497x | 2.148x |
| | 2 | 6.208x | 2.126x |
| | 3 | 6.500x | 2.174x |
| Gate/up (5632,2048) | 1 | 2.750x | 2.730x |
| | 2 | 2.745x | 2.704x |
| | 3 | 2.754x | 2.718x |
| Down (2048,5632) | 1 | 6.585x | 2.890x |
| | 2 | 6.538x | 3.298x |
| | 3 | 6.604x | 2.964x |

The minimum campaign median is 2.745x cooperative/scalar and 1.462x Metal/CPU wall, above the 1.10x retention gate.

The generalizable cache-hot benchmark false-negative was reported as perfscan issue #820:
https://github.com/jxsl13/perfscan/issues/820

The pinned external `github.com/jxsl13/perfscan/perfscan@v1.71.0` exact-base ratchet found zero new
findings in changed production or test files; see `perfscan.txt`.

## Whole-model leverage

The existing TinyLlama-shaped six-layer reachability test was extended to Q4_1. With identical model, prompt, generation length, and toggle-only control:

- scalar Metal: 72.00 token/s
- cooperative Metal: 182.52 token/s
- gain: 2.535x
- TestMetalCooperativeEndToEnd: PASS in 80.36 s for the full quant-format matrix

## Negative host-route evidence

The count-seven equal host input/output benchmark is in host-route.txt. Median CPU versus resident Metal:

| Shape | CPU ns/op | Metal ns/op | Decision |
|---|---:|---:|---|
| N512 K1024 | 48,474 | 158,012 | CPU 3.26x faster |
| N2048 K2048 | 179,801 | 221,421 | CPU 1.23x faster |
| N4096 K1024 | 180,217 | 193,611 | CPU 1.07x faster |

Generic Metal dispatch therefore returns backend.ErrQuantUnsupported for type 3, while llamagpu uses UploadQWeightQ4_1 for the resident recorder path.

## Commands

```sh
GOCACHE=/private/tmp/goai-q41-gocache go test -c -tags metal ./backend/metal -o /private/tmp/goai-q41-metal.test
/private/tmp/goai-q41-metal.test -test.run '^$' -test.bench '^BenchmarkMetalQ4_1CooperativeCampaign$' -test.count 7 -test.benchtime 3x -test.timeout 15m
/private/tmp/goai-q41-metal.test -test.run '^$' -test.bench '^BenchmarkMetalQ4_1Leadership$' -test.count 7 -test.benchtime 200ms -test.timeout 10m
GOCACHE=/private/tmp/goai-q41-gocache go test -c -tags metal ./llamagpu -o /private/tmp/goai-q41-llamagpu.test
/private/tmp/goai-q41-llamagpu.test -test.v -test.run '^TestMetalCooperativeEndToEnd$' -test.timeout 20m
```
