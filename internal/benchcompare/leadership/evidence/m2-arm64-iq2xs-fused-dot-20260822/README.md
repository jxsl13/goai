# M2 ARM64 IQ2_XS fused row dot

## Scope

This tranche adds portable F32/F64 `QMatMul` semantics for `IQ2_XS` and
compares its exact scalar row dot with an ARM64 fused 512-entry eight-wide-grid,
ksigns, explicit-scale, and activation row dot in the same candidate binary.
ARM64 and AMD64 also view the contiguous 64-byte little-endian code plane
directly, while other architectures retain the byte-order/alignment-safe
decoder. The unaligned-input gate proves that the native view does not narrow
the public byte-slice contract.

It does not claim llama.cpp or cross-library leadership: merged main rejected
`IQ2_XS` in CPU `QMatMul`, while the pinned llama.cpp ARM64 kernel consumes
Q8_K-quantized activations rather than GoAI's direct F32 input.

The unchanged IQ2_XXS M1/N4096 decode benchmark is the baseline/candidate
negative control. A separate baseline/candidate comparison proves that routing
the existing IQ2_XS tensor decoder through caller-owned storage and the native
code-plane view is performance-neutral.

## Architecture inputs

- Historical `ARCHITECTURE-RESEARCH.md` from commit `eb8b5a7f`, blob
  `a4b5ce34ce8db73f4b4c1ae01e7fcb0c1067755e`; the file is absent from
  current main, so it is a dated design input rather than an evergreen claim.
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
- Baseline source: merged main `2f23e2a1ba303f65126fed9d243431c4ed263857`
- Baseline binary SHA-256: `9b96fb0be0efeb93de41406e984e498e587f93f87e095641b8ef145af3cea8b6`
- Candidate source: this evidence directory's containing commit
- Candidate binary SHA-256: `99eab0f32462078d0259b5faaed76716921fa2b4289a65f22ecba8f46516e43a`
- Spectackle v0.9.3; external perfscan module v1.71.0

## Method

- Short screens established the likely effect size but are not part of the
  retained evidence.
- Ten retained fresh-process samples use `-test.benchtime 500ms`; invocation
  order alternates scalar-first and NEON-first. Every final sample is committed
  and none was removed.
- IQ2_XXS and tensor-dequant controls use ten fresh-process baseline/candidate
  pairs at the same duration, alternating binary order.
- `benchstat` compares the committed normalized streams.
- The fixed two-trip assembly body was unrolled only after that form measured
  about 3.3% faster in a separate five-sample screen. Widening the two
  independent uint16 code loads into one word plus dependent extraction lost
  about 2% and was rejected.

## Results

| Cell | Portable scalar | ARM64 fused | Speedup | Significance |
| --- | ---: | ---: | ---: | ---: |
| IQ2_XS leaf, K=4096 | 4.8785 us | 0.8477 us | 5.75x | p=0.000, n=10 |
| IQ2_XS QMatMul, M1/N64/K1024 | 77.93 us | 14.22 us | 5.48x | p=0.000, n=10 |
| IQ2_XS QMatMul, M1/N4096/K1024 | 786.7 us | 199.3 us | 3.95x | p=0.000, n=10 |
| IQ2_XXS M1/N4096/K1024 negative control | 183.0 us | 181.8 us | flat | p=0.218, n=10 |
| Existing IQ2_XS tensor dequant | 233.7 us | 234.0 us | flat | p=0.796, n=10 |

Every accelerated cell clears the proposal's hard 2x threshold. Allocation
counts are statistically flat: the leaf remains zero-allocation, N64 remains
at four allocations, and N4096 remains at 29 allocations. N4096 bytes/op are
1.18% lower (`p=0.022`) rather than regressing; both controls are neutral in
time, bytes, and allocations.

## Correctness gates

- Exact all-positive block golden: `2048`.
- Maximum scalar-relative error over 100 arbitrary packed rows at K=256, 512,
  and 4096: `4.1954810760924744e-15`.
- Cancellation-heavy paired-half input remains within the `1e-4` relative gate.
- Inputs are immutable and the ARM64 row wrapper allocates zero bytes.
- The portable fused row dot is bit-exact to materialized decode plus dot.
- Existing gguf-py IQ2_XS golden remains bit-exact through caller-owned decode.
- Native unaligned and aligned code-plane views are bit-exact for decode and dot.
- Portable F32/F64 and M1/M3 `QMatMul` match fully dequantized references.
- Invalid activation widths and truncated weight matrices return clean errors.
- M>1 scratch allocations are invariant from N=1 to N=31.
- Selector tests restrict the ARM64 leaf to contiguous F32 M1; F32 M>1 and F64
  remain portable.

## Static analysis and specification

External perfscan v1.71.0 ran with `GOPROXY=direct` and an isolated task cache.
The focused final scan reports no finding in a touched IQ2_XS file. PS4001
identified the contiguous code plane and directly motivated the native view;
the portable fallback carries a narrow byte-order/alignment suppression.
The generalized fixed-trip assembly-loop opportunity is reported upstream as
[perfscan issue #805](https://github.com/jxsl13/perfscan/issues/805).

Spectackle proposal `P-01M0KBF17FE9Z` and task `T-01M0KBJMSPENE` govern the
change. The four rules are `IQ2XS-PORTABLE-QMATMUL-001`,
`IQ2XS-PORTABLE-SCRATCH-001`, `ARM64-IQ2XS-FUSED-DOT-001`, and
`ARM64-IQ2XS-FUSED-DOT-SCOPE-001`.

## Final validation

- The complete compiled `format/gguf` suite and its race-enabled equivalent
  pass with `-test.run . -test.count 1`.
- Linux ARM64 and AMD64 `CGO_ENABLED=0` test binaries cross-compile from the
  final source.
- `make preflight` and the native M2 `make preflight-metal` lane pass.
- The final Spectackle reindex covers 2,664 files, 18,079 nodes, and 33,361
  edges, including 6,750 typed calls. The fully paged check reports zero drift;
  its 133 warnings and 16 pending anchors are pre-existing repository debt and
  do not involve the four IQ2_XS rules.
- The repository-wide external perfscan advisory contains 1,674 lines and no
  finding in a touched IQ2_XS implementation or test file.

## Reproduction

```sh
go test -c -o /tmp/goai-iq2xs-gguf.test ./format/gguf
/tmp/goai-iq2xs-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ2XSPaths|BenchmarkQMatMulIQ2XSPaths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
GOAI_GGUF_IQ2XS_NEON_FIRST=1 /tmp/goai-iq2xs-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ2XSPaths|BenchmarkQMatMulIQ2XSPaths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
```

Repeat those invocations five times each, alternating them. Normalize the
`/scalar` and `/neon` suffixes as in the committed streams, then run
`benchstat`. Direct test-binary filtering is intentional; repository policy
forbids `go test -run`.
