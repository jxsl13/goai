# M2 ARM64 IQ2_XXS fused row dot

## Scope

This tranche adds portable F32/F64 `QMatMul` semantics for `IQ2_XXS` and
compares its exact scalar row dot with an Apple ARM64 fused eight-wide-grid,
ksigns, and scale row dot in the same candidate binary. It does not claim
llama.cpp or cross-library leadership: merged main rejected `IQ2_XXS` in CPU
`QMatMul`, while the pinned llama.cpp ARM64 kernel consumes Q8_K-quantized
activations rather than GoAI's direct F32 input.

The unchanged IQ3_XXS M1/N4096 decode benchmark is the baseline/candidate
negative control. A separate baseline/candidate comparison proves that routing
the existing IQ2_XXS tensor decoder through caller-owned storage is
performance-neutral.

## Architecture inputs

- Historical `ARCHITECTURE-RESEARCH.md` from commit `eb8b5a7f`, blob
  `a4b5ce34ce8db73f4b4c1ae01e7fcb0c1067755e`; the file is absent from
  current main, so it is a dated design input rather than an evergreen claim.
- Governing repository ADR: `docs/decisions/ADR-0016-quant-matmul-capability.md`.
- llama.cpp ARM quant audit pinned at
  `3af988fabcf79fd81f8720505e684d2aa5bfc786`; local
  `ggml/src/ggml-cpu/arch/arm/quants.c` blob
  `b988abf9963a192e16177661a7d99596effc0d36`, verified against upstream.
- Spectackle ADR `ADR-01M0K6C6PEF4J` preserves GoAI's direct-F32/F64 boundary.

## Environment

- Apple M2 Pro, macOS 26.5.1 (25F80), Darwin 25.5.0 arm64
- Go 1.26.6, `darwin/arm64`
- Baseline source: merged main `6667d30f0aaa7e4214aa2a68de4f1402a527e1b0`
- Baseline binary SHA-256: `765a4376b77c621a3ca72f37bbab8d3b284a821cbabfb8e6f9eb9117f189522f`
- Candidate source: this evidence directory's containing commit
- Candidate binary SHA-256: `b4908db8296311d76e4c670d07dd13bc3141eec3d63ac3e0245612c6b83a96ed`
- Spectackle v0.9.3; external perfscan module v1.71.0

## Method

- A five-sample 300 ms screen established the likely effect size but is not
  part of the retained evidence.
- Ten retained fresh-process samples use `-test.benchtime 500ms`; invocation
  order alternates scalar-first and NEON-first. Every final sample is committed
  and none was removed.
- IQ3_XXS and tensor-dequant controls use ten fresh-process baseline/candidate
  pairs at the same duration, alternating binary order.
- `benchstat` compares the committed normalized streams.

## Results

| Cell | Portable scalar | ARM64 fused | Speedup | Significance |
| --- | ---: | ---: | ---: | ---: |
| IQ2_XXS leaf, K=4096 | 4.7340 us | 0.7548 us | 6.27x | p=0.000, n=10 |
| IQ2_XXS QMatMul, M1/N64/K1024 | 75.25 us | 12.58 us | 5.98x | p=0.000, n=10 |
| IQ2_XXS QMatMul, M1/N4096/K1024 | 761.2 us | 182.1 us | 4.18x | p=0.000, n=10 |
| IQ3_XXS M1/N4096/K1024 negative control | 188.5 us | 188.7 us | flat | p=0.529, n=10 |
| Existing IQ2_XXS tensor dequant | 232.1 us | 228.5 us | flat | p=0.143, n=10 |

Every accelerated cell clears the proposal's hard 2x threshold. Bytes and
allocations are statistically flat: the leaf remains zero-allocation, N64
remains at four allocations, and N4096 remains at 29 allocations. Both
controls are neutral in time, bytes, and allocations.

## Correctness gates

- Exact all-positive block golden: `2048`.
- Maximum scalar-relative error over 100 arbitrary packed rows at K=256, 512,
  and 4096: `8.731189028573922e-15`.
- Cancellation-heavy paired-half input remains within the `1e-4` relative gate.
- Inputs are immutable and the ARM64 row wrapper allocates zero bytes.
- The portable fused row dot is bit-exact to materialized decode plus dot.
- Existing gguf-py IQ2_XXS golden remains bit-exact through caller-owned decode.
- Portable F32/F64 and M1/M3 `QMatMul` match fully dequantized references.
- Invalid activation widths and truncated weight matrices return clean errors.
- M>1 scratch allocations are invariant from N=1 to N=31.
- Selector tests restrict the ARM64 leaf to contiguous F32 M1; F32 M>1 and F64
  remain portable.

## Static analysis and specification

External perfscan v1.71.0 ran with `GOPROXY=direct` and an isolated task cache.
The focused final scan reports no IQ2_XXS finding. Narrow PS4001 suppressions
cover the single strided f16 scale and the full 28-bit ksigns payload in each
heterogeneous 66-byte block; a same-layout bulk copy would be incorrect.
The repository-wide advisory scan completed and emitted 1,676 lines of
pre-existing findings; it is recorded separately from the clean focused scan
so unrelated repository debt cannot be mistaken for tranche regressions.
The missing cross-switch variant coverage detector is reported upstream as
[perfscan issue #804](https://github.com/jxsl13/perfscan/issues/804).

Spectackle proposal `P-01M0K8Z3S1ER2` and task `T-01M0K90SRGFGR` govern the
change. The four rules are `IQ2XXS-PORTABLE-QMATMUL-001`,
`IQ2XXS-PORTABLE-SCRATCH-001`, `ARM64-IQ2XXS-FUSED-DOT-001`, and
`ARM64-IQ2XXS-FUSED-DOT-SCOPE-001`.

## Validation

- The complete GGUF test binary and its race-instrumented equivalent pass.
- Linux ARM64 and Linux AMD64 GGUF test binaries cross-compile.
- `make preflight` and `make preflight-metal` pass; the latter exercises both
  the Metal backend and `llamagpu` suites on this M2 host.
- The final Spectackle reindex covers 2,657 files, 18,050 nodes, 33,311 edges,
  and 6,750 typed calls. Its fully paged check reports no IQ2_XXS drift or
  vacuous binding; only pre-existing warning debt and 16 pending anchors remain.
- `git diff --check` and the repository's formatting gates pass.

## Reproduction

```sh
go test -c -o /tmp/goai-iq2xxs-gguf.test ./format/gguf
/tmp/goai-iq2xxs-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ2XXSPaths|BenchmarkQMatMulIQ2XXSPaths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
GOAI_GGUF_IQ2XXS_NEON_FIRST=1 /tmp/goai-iq2xxs-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ2XXSPaths|BenchmarkQMatMulIQ2XXSPaths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
```

Repeat those invocations five times each, alternating them. Normalize the
`/scalar` and `/neon` suffixes as in the committed streams, then run
`benchstat`. Direct test-binary filtering is intentional; repository policy
forbids `go test -run`.
