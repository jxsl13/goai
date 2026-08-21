# M2 ARM64 IQ3_S fused row dot

## Scope

This tranche adds portable F32/F64 `QMatMul` semantics for `IQ3_S` and compares
its exact scalar row dot with the Apple ARM64 fused 9-bit-grid/direct-sign row
dot in the same candidate binary. It does not claim llama.cpp or cross-library
leadership: merged main rejected `IQ3_S` in `QMatMul`, and current llama.cpp
uses a quantized Q8_K activation boundary rather than GoAI's direct F32 input.

The unchanged IQ4_XS M1 decode benchmark is the baseline/candidate negative
control. A separate baseline/candidate comparison validates that delegating
the existing tensor decoder to caller-owned storage does not regress it.

## Environment

- Apple M2 Pro, macOS 26.5.1 (25F80), Darwin arm64
- Go 1.26.6, `darwin/arm64`
- Baseline binary: source state `ec1d39fa`, code-equivalent to merged main
  `d1daf49033de16b26711aca26fca849200f15345`; intervening commits are
  Spectackle-only
- Candidate: this evidence directory's containing commit
- Spectackle v0.9.3; external perfscan module v1.71.0
- llama.cpp ARM64 audit pinned at
  `3af988fabcf79fd81f8720505e684d2aa5bfc786`
- Benchmark binaries were compiled once per source state before sampling

## Method

- A three-sample 500ms pilot established the duration and was excluded.
- Ten retained fresh-process samples use `-test.benchtime 500ms`; invocation
  order alternates scalar-first and NEON-first. Every sample is present in the
  committed streams and none was removed.
- The IQ4_XS control and tensor-dequant checks use ten fresh-process
  baseline/candidate pairs at the same duration, alternating binary order.
- `benchstat` compares the committed normalized streams.

## Results

| Cell | Portable scalar | ARM64 fused | Speedup | Significance |
| --- | ---: | ---: | ---: | ---: |
| IQ3_S leaf, K=4096 | 5.254 us | 1.080 us | 4.86x | p=0.000, n=10 |
| IQ3_S QMatMul, M1/N64/K1024 | 84.83 us | 17.93 us | 4.73x | p=0.000, n=10 |
| IQ3_S QMatMul, M1/N4096/K1024 | 895.0 us | 245.9 us | 3.64x | p=0.000, n=10 |
| IQ4_XS M1/N4096/K1024 negative control | 209.9 us | 213.7 us | flat | p=0.971, n=10 |
| Existing IQ3_S tensor dequant | 239.0 us | 236.9 us | flat | p=0.436, n=10 |

Every accelerated cell clears the proposal's hard 2x threshold. Bytes and
allocations are statistically flat: the leaf remains zero-allocation, N64
remains at four allocations, and N4096 remains at 29 allocations. The
negative control is flat in time, bytes, and allocations.

## Correctness gates

- Exact all-positive block golden: `256`.
- Maximum scalar-relative error over 100 arbitrary packed rows at K=256, 512,
  and 4096: `2.223032812824975e-15`.
- Cancellation-heavy paired-half input remains within the `1e-4` relative gate.
- Inputs are immutable and the ARM64 row wrapper allocates zero bytes.
- Caller-owned IQ3_S decode is bit-exact to the tensor decoder.
- The portable fused row dot is bit-exact to materialized decode plus dot.
- Portable F32/F64 and M1/M3 `QMatMul` match the fully dequantized reference.
- M>1 scratch allocations are invariant from N=1 to N=31.
- Selector tests restrict the ARM64 leaf to contiguous F32 M1; F32 M>1 and F64
  use the portable caller-owned path.
- Linux ARM64 and AMD64 package test binaries compile successfully.

## Static analysis and validation

External perfscan module v1.71.0 ran with `GOPROXY=direct` and a fresh
task-local Go cache. A focused rescan reports no finding in a new production
or test file. The two strided f16 scale loads carry narrow PS4001 suppressions:
one half value per 110-byte heterogeneous block cannot become a same-layout
bulk copy. The external binary self-labels as `dev` and warns that eleven keys
from GoAI's newer internal config schema are unknown, so this is an advisory
external scan, not a strict config-complete pass.

`make preflight` passed build, vet, tidy drift, API/perfscan meta-tests, and the
full pure-Go short suite. The full GGUF package passes normally and under the
race detector. `make preflight-metal` passed both the M2 Metal backend and
llamagpu short suites.

## Reproduction

```sh
go test -c -o /tmp/goai-iq3s-gguf.test ./format/gguf
/tmp/goai-iq3s-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ3SPaths|BenchmarkQMatMulIQ3SPaths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
GOAI_GGUF_IQ3S_NEON_FIRST=1 /tmp/goai-iq3s-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ3SPaths|BenchmarkQMatMulIQ3SPaths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
```

Repeat those invocations five times each, alternating them. Normalize the
`/scalar` and `/neon` suffixes as in the committed streams, then run
`benchstat`. The direct test-binary `-test.run` filter is intentional; the
repository contract forbids `go test -run`.
