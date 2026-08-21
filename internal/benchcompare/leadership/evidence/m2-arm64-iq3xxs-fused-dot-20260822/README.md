# M2 ARM64 IQ3_XXS fused row dot

## Scope

This tranche adds portable F32/F64 `QMatMul` semantics for `IQ3_XXS` and
compares its exact scalar row dot with an Apple ARM64 fused grid/ksigns/scale
row dot in the same candidate binary. It does not claim llama.cpp or
cross-library leadership: merged main rejected `IQ3_XXS` in `QMatMul`, while
the pinned llama.cpp ARM64 kernel consumes Q8_K-quantized activations rather
than GoAI's direct F32 input.

The unchanged IQ4_XS M1 decode benchmark is the baseline/candidate negative
control. A separate baseline/candidate comparison proves that delegating the
existing IQ3_XXS tensor decoder to caller-owned storage is performance-neutral.

## Architecture inputs

- Historical `ARCHITECTURE-RESEARCH.md` from commit `eb8b5a7f`, blob
  `a4b5ce34ce8db73f4b4c1ae01e7fcb0c1067755e`; the file is absent from
  current main, so it is treated as a dated design input rather than an
  evergreen claim.
- Governing repository ADR: `docs/decisions/ADR-0016-quant-matmul-capability.md`.
- llama.cpp ARM quant audit pinned at
  `3af988fabcf79fd81f8720505e684d2aa5bfc786`; local
  `ggml/src/ggml-cpu/arch/arm/quants.c` blob
  `b988abf9963a192e16177661a7d99596effc0d36`, verified against upstream.
- Spectackle ADR `ADR-01M0K6C6PEF4J` preserves GoAI's direct-F32/F64 boundary.

## Environment

- Apple M2 Pro, macOS 26.5.1 (25F80), Darwin 25.5.0 arm64
- Go 1.26.6, `darwin/arm64`
- Baseline source: merged main `1daaf7163700f288337f36d17ed0fa4a2b71d910`
- Baseline binary SHA-256: `5c6a0546c293e0c01777ab3a4b04821635829a2c7c7f95b61701c1667491f567`
- Candidate source: this evidence directory's containing commit
- Candidate binary SHA-256: `b2bcdb74b0dd709839d9b363c053762731ec48ec0e775b1c089b10871f439ed5`
- Spectackle v0.9.3; external perfscan module v1.71.0

## Method

- A three-sample 500 ms pilot and one pre-final ten-sample run were excluded
  after perfscan motivated replacing a full uint32 decode with one scale-nibble
  byte load. The final binary was rebuilt and every retained sample rerun.
- Ten retained fresh-process samples use `-test.benchtime 500ms`; invocation
  order alternates scalar-first and NEON-first. Every final sample is committed
  and none was removed.
- IQ4_XS and tensor-dequant controls use ten fresh-process baseline/candidate
  pairs at the same duration, alternating binary order.
- `benchstat` compares the committed normalized streams.

## Results

| Cell | Portable scalar | ARM64 fused | Speedup | Significance |
| --- | ---: | ---: | ---: | ---: |
| IQ3_XXS leaf, K=4096 | 5.0805 us | 0.8041 us | 6.32x | p=0.000, n=10 |
| IQ3_XXS QMatMul, M1/N64/K1024 | 82.25 us | 13.42 us | 6.13x | p=0.000, n=10 |
| IQ3_XXS QMatMul, M1/N4096/K1024 | 810.2 us | 190.1 us | 4.26x | p=0.000, n=10 |
| IQ4_XS M1/N4096/K1024 negative control | 196.0 us | 196.1 us | flat | p=0.796, n=10 |
| Existing IQ3_XXS tensor dequant | 233.7 us | 232.2 us | flat | p=0.631, n=10 |

Every accelerated cell clears the proposal's hard 2x threshold. Bytes and
allocations are statistically flat: the leaf remains zero-allocation, N64
remains at four allocations, and N4096 remains at 29 allocations. Both
controls are neutral in time, bytes, and allocations.

## Correctness gates

- Exact all-positive block golden: `1024`.
- Maximum scalar-relative error over 100 arbitrary packed rows at K=256, 512,
  and 4096: `1.3257546909390963e-15`.
- Cancellation-heavy paired-half input remains within the `1e-4` relative gate.
- Inputs are immutable and the ARM64 row wrapper allocates zero bytes.
- The portable fused row dot is bit-exact to materialized decode plus dot.
- Existing gguf-py IQ3_XXS golden remains bit-exact through caller-owned decode.
- Portable F32/F64 and M1/M3 `QMatMul` match fully dequantized references.
- M>1 scratch allocations are invariant from N=1 to N=31.
- Selector tests restrict the ARM64 leaf to contiguous F32 M1; F32 M>1 and F64
  remain portable.
- Full GGUF tests pass; Linux ARM64 and AMD64 package binaries cross-compile.

## Static analysis and specification

External perfscan v1.71.0 ran with `GOPROXY=direct` and an isolated task cache.
The final focused scan reports no IQ3_XXS finding. Narrow PS4001 suppressions
cover eight unaligned sign/scale words in each heterogeneous 98-byte block;
a same-layout bulk copy would require unsafe, endian-specific aliasing.
The reusable full-width-decode-to-byte-subfield pattern is reported upstream as
[perfscan issue #803](https://github.com/jxsl13/perfscan/issues/803).

Spectackle reindexed 2,652 files, 18,024 nodes, 33,266 edges, including 6,750
typed calls. All four IQ3_XXS rules are bound to production and non-vacuous
test nodes. `spectackle check` reports no IQ3_XXS drift or vacuity; it retains
pre-existing warning debt and 16 old pending anchors whose nodes are absent.

The complete GGUF package passes normally and under the race detector. Linux
ARM64 and AMD64 GGUF test binaries cross-compile. `make preflight` passed build,
vet, tidy drift, meta-tests, and the full pure-Go short suite. On the M2 host,
`make preflight-metal` passed `backend/metal` in 43.822 s and `llamagpu` in
14.490 s. The full external perfscan remains advisory because it reports the
repository's pre-existing findings; none names a new IQ3_XXS file.

## Reproduction

```sh
go test -c -o /tmp/goai-iq3xxs-gguf.test ./format/gguf
/tmp/goai-iq3xxs-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ3XXSPaths|BenchmarkQMatMulIQ3XXSPaths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
GOAI_GGUF_IQ3XXS_NEON_FIRST=1 /tmp/goai-iq3xxs-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ3XXSPaths|BenchmarkQMatMulIQ3XXSPaths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
```

Repeat those invocations five times each, alternating them. Normalize the
`/scalar` and `/neon` suffixes as in the committed streams, then run
`benchstat`. Direct test-binary filtering is intentional; repository policy
forbids `go test -run`.
