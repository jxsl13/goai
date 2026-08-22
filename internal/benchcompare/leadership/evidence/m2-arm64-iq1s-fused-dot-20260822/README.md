# M2 ARM64 IQ1_S fused row dot

## Scope

This tranche adds portable F32/F64 `QMatMul` semantics for `IQ1_S` and
compares its exact scalar row dot with an ARM64 leaf that fuses native f16
scale lookup, the 2048-entry eight-wide ternary grid, packed three-bit index
highs, shared odd multipliers, signed deltas, and activation dot in the same
candidate binary.

It does not claim llama.cpp or cross-library leadership: merged main rejected
`IQ1_S` in CPU `QMatMul`, while the pinned llama.cpp ARM64 kernel consumes
Q8_K-quantized activations rather than GoAI's direct F32 input.

The unchanged IQ2_S M1/N4096 benchmark is the baseline/candidate negative
control. A separate baseline/candidate comparison proves that routing the
existing IQ1_S tensor decoder through caller-owned storage is
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
- Baseline source: merged main `46cf0883280379fa025e95b37f469870c9ca1784`
- Baseline binary SHA-256: `497de6c8514bafe7f71f84629ea76f6abe649dcdb8046b5441ed04dc0e204d03`
- Candidate source: this evidence directory's containing commit
- Candidate binary SHA-256: `ae64761e41facf6417aa7ede510224237fa59881fb60b092c8a2afa8aead0777`
- Spectackle v0.9.3; external perfscan module v1.71.0

## Method

- Short screens established the likely effect size but are not part of the
  retained comparison.
- Ten retained fresh-process samples use `-test.benchtime 500ms`; invocation
  order alternates scalar-first and NEON-first. Every final sample, including
  visible system-contention outliers, is committed and none was removed.
- IQ2_S and tensor-dequant controls use ten fresh-process baseline/candidate
  pairs at the same duration, alternating binary order.
- `benchstat` compares the committed normalized streams.
- A pre-adjusted signed-delta grid improved the leaf by 8.35% in a separate
  five-pair screen; a float odd-scale table then improved it by 1.58%.
  Moving coefficient preparation into the leaf improved N64 by 5.26%, and
  extending that leaf to one native call per complete row improved the leaf
  by another 6.82%. All retained forms and their measurements were reported
  upstream.

## Results

| Cell | Portable scalar | ARM64 fused | Speedup | Significance |
| --- | ---: | ---: | ---: | ---: |
| IQ1_S leaf, K=4096 | 5.1835 us | 0.6691 us | 7.75x | p=0.000, n=10 |
| IQ1_S QMatMul, M1/N64/K1024 | 87.94 us | 11.27 us | 7.80x | p=0.000, n=10 |
| IQ1_S QMatMul, M1/N4096/K1024 | 1,685.9 us | 300.2 us | 5.62x | p=0.000, n=10 |
| IQ2_S M1/N4096/K1024 negative control | 657.5 us | 649.6 us | flat | p=0.481, n=10 |
| Existing IQ1_S tensor dequant | 480.2 us | 473.7 us | flat | p=0.631, n=10 |

Every accelerated cell clears the proposal's hard 2x threshold. Allocation
counts are statistically flat: the leaf remains zero-allocation, N64 remains
at four allocations, and N4096 remains at 29 allocations. N4096 bytes/op are
also neutral (`p=0.404`); both controls are neutral in time and allocations.

## Correctness gates

- Exact all-zero-code block golden: `-224`.
- Maximum scalar-relative error over 100 arbitrary packed rows at K=256, 512,
  and 4096: `3.452027636539311e-14`.
- Cancellation-heavy paired-half input remains within the `1e-4` relative gate.
- Inputs are immutable and the ARM64 row wrapper allocates zero bytes.
- The portable fused row dot is bit-exact to materialized decode plus dot.
- Existing gguf-py IQ1_S golden remains bit-exact through caller-owned decode.
- Portable F32/F64 and M1/M3 `QMatMul` match fully dequantized references.
- Invalid activation widths and truncated weight matrices return clean errors.
- M>1 scratch allocations are invariant from N=1 to N=31.
- Selector tests restrict the ARM64 leaf to contiguous F32 M1; F32 M>1 and F64
  remain portable.

## Static analysis and specification

External perfscan v1.71.0 ran with `GOPROXY=direct` and isolated task caches.
The exact-main production scan reports 1,673 findings and the candidate 1,671;
a line-number-normalized set comparison reports zero new diagnostics and two
removed IQ1_S diagnostics. No finding remains in a new IQ1_S file. The three
diagnostics in touched shared `quant_matmul.go` are identical to baseline.

The generalized signed-delta lookup, odd-scale table, coefficient preparation,
and whole-row native-call opportunities are reported as
[perfscan issue #808](https://github.com/jxsl13/perfscan/issues/808),
[perfscan issue #809](https://github.com/jxsl13/perfscan/issues/809), and
[perfscan issue #810](https://github.com/jxsl13/perfscan/issues/810). The
remaining portable fixed-count endian-plane opportunity is tracked as
[perfscan issue #811](https://github.com/jxsl13/perfscan/issues/811).

Spectackle proposal `P-01M0KGTBCHF59` and task `T-01M0KGWNMSF72` govern the
change. The four rules are `IQ1S-PORTABLE-QMATMUL-001`,
`IQ1S-PORTABLE-SCRATCH-001`, `ARM64-IQ1S-FUSED-DOT-001`, and
`ARM64-IQ1S-FUSED-DOT-SCOPE-001`.

## Final validation

- Rebuilding the final annotated source reproduced the benchmarked candidate
  binary byte-for-byte at SHA-256
  `ae64761e41facf6417aa7ede510224237fa59881fb60b092c8a2afa8aead0777`.
- The complete GGUF test binary and a separately rebuilt race-enabled binary
  pass. CGO-disabled Linux ARM64 and AMD64 GGUF test binaries compile.
- `make preflight` and the native `make preflight-metal` lane pass.
- Disassembly confirms one row-level native call, native f16 and scale lookup,
  four FP64 FMA chains across every block, and no call inside the assembly leaf.
- The full external perfscan reports zero new production diagnostic as
  described above.
- Final Spectackle reindex and fully paged drift-check counts are captured in
  `manifest.json`.

## Reproduction

```sh
go test -c -o /tmp/goai-iq1s-gguf.test ./format/gguf
/tmp/goai-iq1s-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ1SPaths|BenchmarkQMatMulIQ1SPaths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
GOAI_GGUF_IQ1S_NEON_FIRST=1 /tmp/goai-iq1s-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ1SPaths|BenchmarkQMatMulIQ1SPaths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
```

Repeat those invocations five times each, alternating them. Normalize the
`/scalar` and `/neon` suffixes as in the committed streams, then run
`benchstat`. Direct test-binary filtering is intentional; repository policy
forbids `go test -run`.
