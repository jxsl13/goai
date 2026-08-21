# M2 ARM64 MXFP4 fused row dot

## Scope

This tranche adds portable `QMatMul` semantics for `MXFP4` and compares its
portable scalar row dot with the ARM64 fused E8M0-scale/E2M1-lookup row dot in
the same candidate binary. It does not claim llama.cpp or cross-library
leadership: merged main rejected `MXFP4` in `QMatMul`, so no semantically
matched pre-change operation exists.

The unchanged `IQ4_XS` M1 decode benchmark is the baseline/candidate negative
control. Dtype, shape, decoded arithmetic, output allocation, and timed
operation boundary are identical between each MXFP4 arm.

## Environment

- Apple M2 Pro, macOS 26.5.1 (25F80), Darwin arm64
- Go 1.26.6, `darwin/arm64`
- Baseline binary: source state `d967e8b9` (code-equivalent to merged main
  `9b38bb89559ebdc749b62eae24903717801c5c51`; intervening commits are
  Spectackle-only)
- Candidate: this evidence directory's containing commit
- Spectackle v0.9.3; external perfscan module v1.71.0
- Benchmark binaries were compiled once per source state before sampling

## Method

- A five-sample pilot established the benchmark duration and was excluded.
- Ten retained fresh-process samples use `-test.benchtime 500ms`; invocation
  order alternates scalar-first and NEON-first. Every retained sample is
  present in `mxfp4-scalar.txt` and `mxfp4-neon.txt`; none was removed.
- The `IQ4_XS` control uses ten fresh-process baseline/candidate pairs at the
  same duration, alternating which binary runs first. Every sample is retained.
- The host showed substantial scheduling variance, particularly for N4096.
  The reported intervals expose that variance; all three deltas remain
  significant and clear the hard retention gate.
- `benchstat` compares the committed normalized streams.

## Results

| Cell | Portable scalar | ARM64 fused | Speedup | Significance |
| --- | ---: | ---: | ---: | ---: |
| MXFP4 leaf, K=4096 | 6.896 us | 1.044 us | 6.61x | p=0.000, n=10 |
| MXFP4 QMatMul, M1/N64/K1024 | 100.08 us | 17.04 us | 5.87x | p=0.000, n=10 |
| MXFP4 QMatMul, M1/N4096/K1024 | 2004.2 us | 671.0 us | 2.99x | p=0.000, n=10 |
| IQ4_XS M1/N4096/K1024 negative control | 662.3 us | 653.0 us | flat | p=0.631, n=10 |

Every accelerated cell clears the proposal's hard 2x threshold. The MXFP4 leaf
remains at zero allocations, M1/N64 remains at four allocations, and the
N4096 allocation distribution is statistically flat. The negative control is
flat in time, bytes, and allocations.

## Correctness gates

- Exact packed-block golden: `48`.
- Maximum scalar-relative error over 100 raw rows at K=32, 64, 256, and 4096:
  `8.925221112271966e-16`.
- Cancellation-heavy paired-half input remains within the `1e-4` relative gate.
- Inputs are immutable and the ARM64 row wrapper allocates zero bytes.
- All 256 E8M0 table entries and caller-owned MXFP4 decode are bit-exact.
- The portable fused row dot is bit-exact to materialized decode plus dot.
- Portable F32/F64 and M1/M3 `QMatMul` match the fully dequantized reference
  within `1e-4`.
- M>1 scratch allocations are invariant from N=1 to N=31.
- Selector tests restrict the ARM64 leaf to contiguous F32 M1; F32 M>1 and F64
  use the portable caller-owned path.

## Static analysis

External perfscan v1.71.0 ran with `GOPROXY=direct` and a fresh task-local Go
cache. No finding points at a new production hunk or new ARM64 file. Its only
new test finding was removed by replacing per-element tensor dispatch with
typed storage. The external binary warns that eleven fields from GoAI's newer
internal config schema are unknown; therefore this is recorded as an advisory
external scan, not misrepresented as a strict config-complete pass. Existing
findings outside the changed hunks remain out of scope.

The generalized scalar-selector/fused-SIMD opportunity belongs to perfscan
issue [#799](https://github.com/jxsl13/perfscan/issues/799); the published PR
and retained measurements are added there rather than opening a duplicate.

## Reproduction

```sh
go test -c -o /tmp/goai-mxfp4-gguf.test ./format/gguf
/tmp/goai-mxfp4-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotMXFP4Paths|BenchmarkQMatMulMXFP4Paths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
GOAI_GGUF_MXFP4_NEON_FIRST=1 /tmp/goai-mxfp4-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotMXFP4Paths|BenchmarkQMatMulMXFP4Paths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
```

Repeat those invocations five times each, alternating them. Normalize the
`/scalar` and `/neon` suffixes as in the committed streams, then run `benchstat`.
