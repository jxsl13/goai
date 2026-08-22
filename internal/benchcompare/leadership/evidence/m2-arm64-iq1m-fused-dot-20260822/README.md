# M2 ARM64 IQ1_M fused row dot

## Scope

This tranche adds portable F32/F64 `QMatMul` semantics for `IQ1_M` and
compares its exact scalar row dot with an ARM64 leaf that fuses split-f16
super-scale reconstruction, 2048-entry ternary-grid gathers, packed 11-bit
indices, paired odd multipliers, signed deltas, and activation dot in the same
candidate binary.

It does not claim llama.cpp or cross-library leadership: merged main rejected
`IQ1_M` in CPU `QMatMul`, while the pinned llama.cpp ARM64 kernel consumes
Q8_K-quantized activations rather than GoAI's direct F32 input.

The unchanged IQ1_S M1/N4096 benchmark is the baseline/candidate negative
control. A separate baseline/candidate comparison proves that routing the
existing IQ1_M tensor decoder through caller-owned storage is
performance-neutral.

## Architecture inputs

- Historical `ARCHITECTURE-RESEARCH.md` from commit `eb8b5a7f`, blob
  `a4b5ce34ce8db73f4b4c1ae01e7fcb0c1067755e`; it is a dated design input.
- Governing repository ADR: `docs/decisions/ADR-0016-quant-matmul-capability.md`.
- llama.cpp ARM quant audit pinned at
  `3af988fabcf79fd81f8720505e684d2aa5bfc786`; local
  `ggml/src/ggml-cpu/arch/arm/quants.c` blob
  `b988abf9963a192e16177661a7d99596effc0d36`, SHA-256
  `9fccd3897db24c9df89b8431b588175894e5f54697cf45768d0c6e6c5544093e`.
- Spectackle ADR `ADR-01M0K6C6PEF4J` preserves GoAI's direct-F32/F64 boundary.

## Environment

- Apple M2 Pro, macOS 26.5.1 (25F80), Darwin 25.5.0 arm64
- Go 1.26.6, `darwin/arm64`
- Baseline source: merged main `7bf789bbfd962a5ad2d0b3125d96f4aba68c9a63`
- Baseline binary SHA-256: `ae64761e41facf6417aa7ede510224237fa59881fb60b092c8a2afa8aead0777`
- Candidate source: this evidence directory's containing commit
- Candidate binary SHA-256: `11c67ef76e249d6e05bfc595216ffc61ed282e9358c264a32c14bb7ed075f803`
- Spectackle v0.9.3; external perfscan module v1.71.0

## Method

- Ten retained fresh-process samples use `-test.benchtime 500ms`; invocation
  order alternates scalar-first and NEON-first. Every final sample, including
  system-contention outliers, is committed and none was removed.
- IQ1_S and tensor-dequant controls use ten fresh-process baseline/candidate
  pairs at the same duration, alternating binary order.
- `benchstat` compares the committed normalized streams.
- A 2 KiB packed-QH offset table improved the initial correct native leaf by
  3.78%, N64 by 2.82%, and N4096 by 3.19% in five alternating pairs
  (`p=0.008` for each); this retained general technique is reported upstream.
- A later 2 MiB f16-by-odd-scale coefficient table was rejected after ten
  alternating pairs: no cell improved significantly and its geomean was 4.73%
  slower. It is absent from the candidate.

## Results

| Cell | Portable scalar | ARM64 fused | Speedup | Significance |
| --- | ---: | ---: | ---: | ---: |
| IQ1_M leaf, K=4096 | 6.1650 us | 0.7478 us | 8.24x | p=0.000, n=10 |
| IQ1_M QMatMul, M1/N64/K1024 | 98.25 us | 12.04 us | 8.16x | p=0.000, n=10 |
| IQ1_M QMatMul, M1/N4096/K1024 | 2,046.7 us | 498.9 us | 4.10x | p=0.000, n=10 |
| IQ1_S M1/N4096/K1024 negative control | 218.1 us | 229.5 us | flat | p=0.684, n=10 |
| Existing IQ1_M tensor dequant | 332.8 us | 340.9 us | flat | p=0.912, n=10 |

Every accelerated cell clears the proposal's hard 2x threshold. Allocation
counts are statistically flat: the leaf remains zero-allocation, N64 remains
at four allocations, and N4096 remains at 29 allocations. Both controls are
neutral in time and allocations.

## Correctness gates

- Exact known-value block golden: `-224`.
- Maximum scalar-relative error over 100 arbitrary packed rows at K=256, 512,
  and 4096: `6.617879878655745e-16`.
- Cancellation-heavy paired-half input remains within the `1e-4` relative gate.
- Inputs are immutable and the ARM64 row wrapper allocates zero bytes.
- The portable fused row dot is bit-exact to materialized decode plus dot.
- Existing gguf-py IQ1_M golden remains bit-exact through caller-owned decode.
- Portable F32/F64 and M1/M3 `QMatMul` match fully dequantized references.
- Invalid activation widths and truncated weight matrices return clean errors.
- M>1 scratch allocations are invariant from N=1 to N=31.
- Selector tests restrict the ARM64 leaf to contiguous F32 M1; F32 M>1 and F64
  remain portable.

## Static analysis and specification

External perfscan v1.71.0 ran with `GOPROXY=direct` and isolated task caches.
The exact-main production scan reports 1,925 raw findings and the candidate
1,923. A line-number-normalized comparison reports zero new diagnostics and
one removed IQ1_M diagnostic. Findings in touched shared `quant_matmul.go` are
identical to baseline.

The generalized packed-bitfield offset-table improvement is reported as
[perfscan issue #813](https://github.com/jxsl13/perfscan/issues/813).

Spectackle proposal `P-01M0KM3Z8YEME` and task `T-01M0KM5H42FBT` govern the
change. The four rules are `IQ1M-PORTABLE-QMATMUL-001`,
`IQ1M-PORTABLE-SCRATCH-001`, `ARM64-IQ1M-FUSED-DOT-001`, and
`ARM64-IQ1M-FUSED-DOT-SCOPE-001`. A fully paged check reports zero drift
records. Reindexing with the task's isolated Go cache records 6,750 typed call
edges and every new binding.

## Final validation

- The complete GGUF test binary and a separately rebuilt race-enabled binary
  pass. CGO-disabled Linux ARM64 and AMD64 GGUF test binaries compile.
- `make preflight` and the native `make preflight-metal` lane pass.
- Disassembly confirms one row-level native call, four FP64 FMA chains across
  every block, and no call inside the assembly leaf.
- The full external perfscan reports zero new production diagnostics.
- Spectackle lint reports 133 pre-existing warnings and zero errors.

## Reproduction

```sh
go test -c -o /tmp/goai-iq1m-gguf.test ./format/gguf
/tmp/goai-iq1m-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ1MPaths|BenchmarkQMatMulIQ1MPaths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
GOAI_GGUF_IQ1M_NEON_FIRST=1 /tmp/goai-iq1m-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ1MPaths|BenchmarkQMatMulIQ1MPaths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
```

Repeat those invocations five times each, alternating them. Normalize the
`/scalar` and `/neon` suffixes as in the committed streams, then run
`benchstat`. Direct test-binary filtering is intentional; repository policy
forbids `go test -run`.
