# M2 ARM64 TQ1_0 fused row dot

## Scope

This tranche adds complete ggml type-34 `TQ1_0` support: 54-byte blocks for
256 weights, eager GGUF decode, `QuantTensor.Dequantize`, public encode and
decode, portable F32/F64 `QMatMul`, reused prefill scratch, and an ARM64 leaf
for contiguous F32 M1 decode. The leaf fuses f16 scale lookup, base-243 ternary
expansion, value lookup, and activation dot without materializing weights.

The primary result is a same-semantics GoAI scalar-versus-ARM64 comparison.
Pinned llama.cpp measurements are a separate boundary study, not a leadership
claim: llama.cpp's native TQ1_0 dot consumes Q8_K-quantized activations while
GoAI retains direct F32 activations and exact decoded-weight semantics.

## Reference pins

- Repository ADR: `docs/decisions/ADR-0016-quant-matmul-capability.md`.
- llama.cpp commit `3af988fabcf79fd81f8720505e684d2aa5bfc786`.
- `ggml-common.h` SHA-256
  `af255601767325f087313fa84b9435cb77aeec37df6b61b98d9ecc65f29fb4a0`.
- `ggml-quants.c` SHA-256
  `07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d`.
- ARM `quants.c` SHA-256
  `9fccd3897db24c9df89b8431b588175894e5f54697cf45768d0c6e6c5544093e`.

## Environment

- Apple M2 Pro (12 cores, 32 GiB), macOS 26.5.1 (25F80), Darwin 25.5.0 arm64
- Go 1.26.6, `darwin/arm64`; Apple clang 21.0.0
- Baseline source: merged main `449b67512af305c679a74bc0c014dac08e03ff42`
- Baseline binary SHA-256: `c9f9ac46727711e37ff3ef92f0af9ef6225debd03b620ba906bf8464bcb68e2f`
- Candidate source: this evidence directory's containing commit
- Candidate benchmark binary SHA-256: `1c0685d7077cbd0385502d3f910d284b3f398aa1da2502d2e6ad6ab11e4ed24f`
- Spectackle v0.9.3; external perfscan module v1.71.0

## Method

- Ten retained fresh-process samples use `-test.benchtime 500ms`; invocation
  order alternates scalar-first and NEON-first. Every final sample, including
  system-contention outliers, is committed and none was removed.
- Unchanged Q1_0 M1/N4096 and dequant benchmarks use ten fresh-process
  baseline/candidate pairs, alternating binary order.
- `benchstat` compares the committed normalized streams.
- The pinned llama.cpp harness uses a Release static CPU build with Metal,
  BLAS, and Accelerate disabled. Ten independent process samples are retained.
- An initial 1,280-byte package lookup table was rejected after ten-pair
  controls showed a 2.46% Q1_0 dequant regression (`p=0.001`). Direct
  mixed-radix arithmetic removes that global data, preserves portable scalar
  throughput, and restores both final controls to neutrality.

## Same-semantics GoAI results

| Cell | Portable scalar | ARM64 fused | Speedup | Significance |
| --- | ---: | ---: | ---: | ---: |
| TQ1_0 leaf, K=4096 | 4.8045 us | 0.8479 us | 5.67x | p=0.000, n=10 |
| TQ1_0 QMatMul, M1/N64/K1024 | 75.59 us | 13.90 us | 5.44x | p=0.000, n=10 |
| TQ1_0 QMatMul, M1/N4096/K1024 | 781.8 us | 199.5 us | 3.92x | p=0.000, n=10 |
| Existing Q1_0 dequant control | 255.5 us | 256.7 us | flat | p=0.315, n=10 |
| Existing Q1_0 M1/N4096 control | 149.2 us | 149.3 us | flat | p=0.971, n=10 |

Every accelerated cell clears the proposal's hard 2x threshold. Allocation
counts are flat: the leaf remains zero-allocation, N64 remains at four, and
N4096 remains at median 29. Both baseline/candidate controls are neutral in
time, bytes, and allocations.

## Pinned llama.cpp boundary study

At K=4096, the pinned native ARM path has a 6,657.201 ns median from an F32
caller boundary (Q8_K activation quantization plus TQ1_0/Q8_K dot), versus
GoAI's 847.9 ns direct-F32 median. GoAI is 7.85x faster in this boundary study
and avoids activation quantization, but the outputs do not have identical
semantics. At N64/K1024, llama.cpp's median is 3,752.603 ns versus GoAI's
13,900 ns; llama.cpp is 3.70x faster but uses the lower-precision Q8_K
activation boundary. The 110.995 ns llama.cpp dot-only median excludes the
required F32-to-Q8_K conversion and is not compared with the GoAI leaf.

These results expose a real remaining throughput gap for future work while
keeping the published leadership cell limited to identical GoAI semantics.
The raw samples and complete C harness are committed beside this document.

## Correctness gates

- Exact all-negative, zero, and all-positive block goldens pass.
- Maximum scalar-relative error over 100 arbitrary packed rows at K=256, 512,
  and 4096 is exactly zero.
- Cancellation-heavy paired-half input remains within the `1e-4` gate.
- Inputs are immutable and the ARM64 row wrapper allocates zero bytes.
- The portable fused row dot is bit-exact to materialized decode plus dot.
- Pinned TQ1_0 block layout and reference quantization bytes match exactly.
- Eager, raw-tensor, public decode, F32/F64, and M1/M3 QMatMul paths agree.
- Invalid widths, truncated bytes, and unsupported shapes return clean errors.
- M>1 scratch allocations are invariant from N=1 to N=31.
- Selector tests restrict the leaf to ARM64 contiguous F32 M1; F32 M>1 and
  F64 remain portable.

## Static analysis and specification

The generalized arithmetic-table SIMD diagnostic is reported as
[perfscan issue #816](https://github.com/jxsl13/perfscan/issues/816). The
unrelated-hot-data displacement caused by a small package lookup table is
reported as [perfscan issue #817](https://github.com/jxsl13/perfscan/issues/817).

External perfscan v1.71.0 ran with `GOPROXY=direct`. Exact merged main and the
final candidate each report 3,520 raw findings and 2,308 normalized findings
when tests are included: zero new and zero removed diagnostics.

Spectackle proposal `P-01M0KT0ECBEK5` and task `T-01M0KT35PPFRR` govern the
change. Their rules are `TQ1-FORMAT-001`, `TQ1-PORTABLE-QMATMUL-001`,
`TQ1-PORTABLE-SCRATCH-001`, `ARM64-TQ1-FUSED-DOT-001`, and
`ARM64-TQ1-FUSED-DOT-SCOPE-001`.

Final reindex records 2,692 files, 18,249 nodes, 33,654 edges, and 6,756
typed calls. The fully paged check has zero drift; lint retains 133 existing
warnings and zero errors. One additional VAC warning incorrectly treats
`range 100` as possibly empty; the isolated Go 1.22 integer-range defect is
reported as [Spectackle issue #277](https://github.com/jxsl13/spectackle/issues/277).

## Final validation

- Focused TQ1_0 and complete GGUF test-binary runs pass.
- A separately rebuilt race-enabled GGUF binary passes.
- CGO-disabled Linux ARM64 and AMD64 GGUF test binaries compile.
- `make preflight` and native `make preflight-metal` pass.
- Disassembly confirms byte-lane mixed-radix extraction, four VTBL value
  gathers, four persistent float64 FMA chains, and no call inside the leaf.
- External perfscan reports zero normalized candidate ratchet drift.
- Spectackle reindex, fully paged check, and lint finish without errors.

## Reproduction

```sh
go test -c -o /tmp/goai-tq1-gguf.test ./format/gguf
/tmp/goai-tq1-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotTQ1Paths|BenchmarkQMatMulTQ1Paths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
GOAI_GGUF_TQ1_NEON_FIRST=1 /tmp/goai-tq1-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotTQ1Paths|BenchmarkQMatMulTQ1Paths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
```

Repeat those invocations five times each, alternating them. Normalize the
`/scalar` and `/neon` suffixes as in the committed streams, then run
`benchstat`. Direct test-binary filtering is intentional; repository policy
forbids `go test -run`.
