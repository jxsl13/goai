# M2 ARM64 IQ2_S fused row dot

## Scope

This tranche adds portable F32/F64 `QMatMul` semantics for `IQ2_S` and
compares its exact scalar row dot with an ARM64 leaf that fuses the 1024-entry
eight-wide grid, packed two-bit index highs, direct sign bytes, explicit
sub-scales, and activation dot in the same candidate binary.

It does not claim llama.cpp or cross-library leadership: merged main rejected
`IQ2_S` in CPU `QMatMul`, while the pinned llama.cpp ARM64 kernel consumes
Q8_K-quantized activations rather than GoAI's direct F32 input.

The unchanged IQ2_XS M1/N4096 benchmark is the baseline/candidate negative
control. A separate baseline/candidate comparison proves that routing the
existing IQ2_S tensor decoder through caller-owned storage is
performance-neutral.

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
- Baseline source: merged main `f15eeb522e4d87af8f537e8b9568e9945eacf9c6`
- Baseline binary SHA-256: `69a2b0036e4e35724315d391692fa5929681ea6a4304884127b428ced9c35cf1`
- Candidate source: this evidence directory's containing commit
- Candidate binary SHA-256: `497de6c8514bafe7f71f84629ea76f6abe649dcdb8046b5441ed04dc0e204d03`
- Spectackle v0.9.3; external perfscan module v1.71.0

## Method

- Short screens established the likely effect size but are not part of the
  retained comparison.
- Ten retained fresh-process samples use `-test.benchtime 500ms`; invocation
  order alternates scalar-first and NEON-first. Every final sample, including
  visible system-contention outliers, is committed and none was removed.
- IQ2_XS and tensor-dequant controls use ten fresh-process baseline/candidate
  pairs at the same duration, alternating binary order.
- `benchstat` compares the committed normalized streams.
- A direct 256x8 sign-mask table beat two 16x4 nibble-table lookups by 6.63%
  in a separate five-pair screen. Four independent ARM64 `UBFX` operations
  then beat destructive shift-and-mask extraction by 2.20% in another
  five-pair screen. Both forms are retained and reported upstream.

## Results

| Cell | Portable scalar | ARM64 fused | Speedup | Significance |
| --- | ---: | ---: | ---: | ---: |
| IQ2_S leaf, K=4096 | 5.0925 us | 0.9362 us | 5.44x | p=0.000, n=10 |
| IQ2_S QMatMul, M1/N64/K1024 | 82.26 us | 15.82 us | 5.20x | p=0.000, n=10 |
| IQ2_S QMatMul, M1/N4096/K1024 | 962.4 us | 225.6 us | 4.27x | p=0.001, n=10 |
| IQ2_XS M1/N4096/K1024 negative control | 265.3 us | 274.3 us | flat | p=1.000, n=10 |
| Existing IQ2_S tensor dequant | 295.8 us | 288.7 us | flat | p=0.796, n=10 |

Every accelerated cell clears the proposal's hard 2x threshold. Allocation
counts are statistically flat: the leaf remains zero-allocation, N64 remains
at four allocations, and N4096 remains at 29 allocations. N4096 bytes/op are
1.27% lower (`p=0.028`) rather than regressing; both controls are neutral in
time and allocations.

## Correctness gates

- Exact all-positive block golden: `2048`.
- Maximum scalar-relative error over 100 arbitrary packed rows at K=256, 512,
  and 4096: `1.9996704013343956e-15`.
- Cancellation-heavy paired-half input remains within the `1e-4` relative gate.
- Inputs are immutable and the ARM64 row wrapper allocates zero bytes.
- The portable fused row dot is bit-exact to materialized decode plus dot.
- Existing gguf-py IQ2_S golden remains bit-exact through caller-owned decode.
- Portable F32/F64 and M1/M3 `QMatMul` match fully dequantized references.
- Invalid activation widths and truncated weight matrices return clean errors.
- M>1 scratch allocations are invariant from N=1 to N=31.
- Selector tests restrict the ARM64 leaf to contiguous F32 M1; F32 M>1 and F64
  remain portable.

## Static analysis and specification

External perfscan v1.71.0 ran with `GOPROXY=direct` and an isolated task cache.
The focused final scan reports no finding in a touched IQ2_S file. The
generalized full-byte table and independent bitfield-extraction opportunities
are reported as [perfscan issue #806](https://github.com/jxsl13/perfscan/issues/806)
and [perfscan issue #807](https://github.com/jxsl13/perfscan/issues/807).

Spectackle proposal `P-01M0KEEGADEWB` and task `T-01M0KEGBWFFG0` govern the
change. The four rules are `IQ2S-PORTABLE-QMATMUL-001`,
`IQ2S-PORTABLE-SCRATCH-001`, `ARM64-IQ2S-FUSED-DOT-001`, and
`ARM64-IQ2S-FUSED-DOT-SCOPE-001`.

## Final validation

- Rebuilding the final source after the assembly ABI-operand spelling fix
  reproduced the benchmarked candidate binary byte-for-byte at SHA-256
  `497de6c8514bafe7f71f84629ea76f6abe649dcdb8046b5441ed04dc0e204d03`.
- The complete GGUF test binary and a separately rebuilt race-enabled binary
  pass. CGO-disabled Linux ARM64 and AMD64 GGUF test binaries compile.
- `make preflight` and the native `make preflight-metal` lane pass.
- The full external perfscan reports 1,673 lines versus 1,674 on the exact
  merged-main baseline. A line-number-normalized set comparison finds no new
  diagnostic and one removed IQ2_S diagnostic. The three diagnostics in the
  touched shared `quant_matmul.go` file are identical to baseline findings.
- Final Spectackle reindex covers 2,669 files, 18,106 nodes, and 33,408 edges,
  including 6,750 typed calls with zero skipped files. The fully paged check
  reaches `ok`, emits no drift record, and retains 16 pre-existing pending
  anchors; repository-wide pre-existing lint warnings remain out of scope.

## Reproduction

```sh
go test -c -o /tmp/goai-iq2s-gguf.test ./format/gguf
/tmp/goai-iq2s-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ2SPaths|BenchmarkQMatMulIQ2SPaths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
GOAI_GGUF_IQ2S_NEON_FIRST=1 /tmp/goai-iq2s-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ2SPaths|BenchmarkQMatMulIQ2SPaths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
```

Repeat those invocations five times each, alternating them. Normalize the
`/scalar` and `/neon` suffixes as in the committed streams, then run
`benchstat`. Direct test-binary filtering is intentional; repository policy
forbids `go test -run`.
