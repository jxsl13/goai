# M2 ARM64 IQ4_XS fused super-block dot

## Scope

This tranche adds portable `QMatMul` semantics for `IQ4_XS` and compares its
portable scalar row dot with the ARM64 fused 256-weight super-block dot in the
same candidate binary. It does not claim llama.cpp or cross-library leadership;
merged main rejected `IQ4_XS` in `QMatMul`, so there is no semantically matched
pre-change implementation to time.

The unchanged `Q5_K` M1 decode benchmark is the baseline/candidate negative
control. Dtype, shape, decoded arithmetic order, output allocation, and timed
operation boundary are identical between each IQ4_XS arm.

## Environment

- Apple M2 Pro, macOS 26.5.1 (25F80), Darwin arm64
- Go 1.26.6, `darwin/arm64`
- Baseline binary: source state `55f10bfa` (code-equivalent to merged main
  `61a34fed57c179222b895a74bdc21a4599b8d17d`; intervening commits are
  Spectackle-only)
- Candidate: this evidence directory's containing commit
- Spectackle v0.9.3; external perfscan module v1.71.0
- Benchmark binary was compiled once per source state before sampling

## Method

- A pilot/warmup campaign established the benchmark duration and was excluded.
- The first attempted retained campaign was rejected in full before statistics
  because later samples showed monotonic multi-millisecond host contention.
- The retained IQ4_XS campaign uses ten fresh-process paired samples at
  `-test.benchtime 500ms`, alternating scalar-first and NEON-first order every
  invocation. All ten samples per arm are retained.
- The `Q5_K` control uses ten fresh-process baseline/candidate pairs at the same
  duration, alternating which binary runs first. All samples are retained.
- `benchstat` compares the committed normalized benchmark streams. No sample was
  removed from either retained campaign.

## Results

| Cell | Portable scalar | ARM64 fused | Speedup | Significance |
| --- | ---: | ---: | ---: | ---: |
| IQ4_XS leaf, K=4096 | 5.146 us | 0.9365 us | 5.50x | p=0.000, n=10 |
| IQ4_XS QMatMul, M1/N64/K1024 | 81.35 us | 15.36 us | 5.30x | p=0.000, n=10 |
| IQ4_XS QMatMul, M1/N4096/K1024 | 1289.2 us | 412.2 us | 3.13x | p=0.000, n=10 |
| Q5_K M1/N4096/K1024 negative control | 188.5 us | 188.7 us | flat | p=0.631, n=10 |

Every accelerated cell clears the proposal's hard 2x retention threshold. The
IQ4_XS leaf remains at zero allocations; M1/N64 remains at 4 allocations and
M1/N4096 remains at a 29-allocation median. The negative control is statistically
flat in time, bytes, and allocations.

## Correctness gates

- Exact packed super-block golden: `-29568`.
- Maximum scalar-relative error over 100 arbitrary raw rows at K=256, 512, and
  4096: `4.5421753815656704e-15`.
- Cancellation-heavy paired-half input remains within the `1e-4` relative gate.
- Inputs are immutable and the ARM64 row wrapper allocates zero bytes.
- Caller-owned IQ4_XS decode is bit-exact to tensor decode.
- Portable F32/F64 and M1/M3 `QMatMul` results match the fully dequantized
  reference within `1e-4`.
- The M>1 scratch allocation count is invariant between N=1 and N=31.
- Selector tests prove the ARM64 row leaf is restricted to contiguous F32 M1;
  F32 M>1 and F64 use the portable caller-owned decode path.

## Static analysis

External perfscan v1.71.0 was run with `GOPROXY=direct` and a fresh task-local
Go build cache. It reports 1,681 pre-existing production findings and 3,527
when tests are included; every changed production and test file is clean. The
generalizable scalar-selector/fused-SIMD opportunity belongs to perfscan issue
[#799](https://github.com/jxsl13/perfscan/issues/799); this tranche's PR and
measurements are added there after publication rather than opening a duplicate.

## Reproduction

```sh
go test -c -o /tmp/goai-iq4xs-gguf.test ./format/gguf
/tmp/goai-iq4xs-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ4XSPaths|BenchmarkQMatMulIQ4XSPaths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
GOAI_GGUF_IQ4XS_NEON_FIRST=1 /tmp/goai-iq4xs-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ4XSPaths|BenchmarkQMatMulIQ4XSPaths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
```

Repeat those two invocations five times each, alternating them, then normalize
the `/scalar` and `/neon` suffixes as in `iq4xs-scalar.txt` and
`iq4xs-neon.txt` before running `benchstat`.
