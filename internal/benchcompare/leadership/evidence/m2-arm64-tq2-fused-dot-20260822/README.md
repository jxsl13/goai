# M2 ARM64 TQ2_0 fused row dot

## Scope

This tranche adds complete ggml type-35 `TQ2_0` support: exact 66-byte blocks
for 256 weights, eager GGUF decode, `QuantTensor.Dequantize`, public encode
and decode, portable F32/F64 `QMatMul`, reused prefill scratch, and an ARM64
leaf for contiguous F32 M1 decode. The leaf fuses f16 scale lookup, four-plane
two-bit unpack, code-minus-one mapping, and activation dot without
materializing weights. Raw code 3 correctly decodes to +2 even though the
reference encoder emits only codes 0 through 2.

The primary result is a same-semantics GoAI scalar-versus-ARM64 comparison.
Pinned llama.cpp measurements are a separate boundary study, not a leadership
claim: llama.cpp's native TQ2_0 dot consumes Q8_K-quantized activations while
GoAI retains direct F32 activations and exact decoded-weight semantics.

## Reference pins

- Repository ADR: `ADR-01M0KYBEDXFJW`.
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
- Baseline source: merged main `c3dad3dd7e8ab7646a73ffeb509e6f403616316d`
- Baseline binary SHA-256: `1c0685d7077cbd0385502d3f910d284b3f398aa1da2502d2e6ad6ab11e4ed24f`
- Candidate source: this evidence directory's containing commit
- Candidate benchmark binary SHA-256: `eeb5b4facf84d0356466fccc8d11340c493af0ec1387223fcab2b60e1000a408`
- Spectackle v0.9.3; external perfscan module v1.71.0

## Method

- Ten retained fresh-process samples use `-test.benchtime 500ms`; invocation
  order alternates scalar-first and NEON-first. No final sample was removed.
- Unchanged Q1_0 dequant, TQ1_0 dequant, and TQ1_0 M1/N4096 benchmarks use ten
  fresh-process baseline/candidate pairs with alternating binary order.
- `benchstat` compares the committed normalized streams.
- The pinned llama.cpp harness uses a Release static CPU build with Metal,
  BLAS, and Accelerate disabled. Ten independent process samples are retained.

## Same-semantics GoAI results

| Cell | Portable scalar | ARM64 fused | Speedup | Significance |
| --- | ---: | ---: | ---: | ---: |
| TQ2_0 leaf, K=4096 | 4.8055 us | 0.7345 us | 6.54x | p=0.000, n=10 |
| TQ2_0 QMatMul, M1/N64/K1024 | 76.28 us | 11.91 us | 6.41x | p=0.000, n=10 |
| TQ2_0 QMatMul, M1/N4096/K1024 | 716.4 us | 175.4 us | 4.09x | p=0.000, n=10 |

Every accelerated cell clears the proposal's hard 2x threshold. Allocation
counts are flat: the leaf remains zero-allocation, N64 remains at four, and
N4096 remains at median 29. All three unchanged controls are neutral in time
and allocations.

## Pinned llama.cpp boundary study

At K=4096, the pinned native ARM path has a 5,497.518 ns median from an F32
caller boundary (Q8_K activation quantization plus TQ2_0/Q8_K dot), versus
GoAI's 734.5 ns direct-F32 median. GoAI is approximately 7.48x faster in this
boundary study and avoids activation quantization, but the outputs do not have
identical semantics. At N64/K1024, llama.cpp's median is 2,311.625 ns versus
GoAI's 11,910 ns; llama.cpp is approximately 5.15x faster but uses one reused
lower-precision Q8_K activation. The 57.862 ns llama.cpp dot-only median
excludes the required F32-to-Q8_K conversion and is not compared with the
GoAI leaf.

These observations expose a remaining multi-row throughput gap without
overstating cross-library leadership. The raw samples and complete C harness
are committed beside this document.

## Correctness gates

- The exact pinned `quantize_row_tq2_0_ref` golden passes.
- Codes 0, 1, 2, and 3 map to -1, 0, +1, and +2 before scale.
- Maximum scalar-relative error over 100 arbitrary packed rows is
  `1.3461880134962722e-16`.
- Cancellation-heavy input remains within the `1e-4` gate.
- Inputs are immutable and the ARM64 row wrapper allocates zero bytes.
- Eager, raw-tensor, public decode, F32/F64, and M1/M3 QMatMul paths agree.
- Invalid widths, truncated bytes, and unsupported shapes return clean errors.
- M>1 scratch allocations are invariant from N=1 to N=31.
- Selector tests restrict the leaf to ARM64 contiguous F32 M1; F32 M>1 and
  F64 remain portable.

## Static analysis and specification

The generalized two-bit plane fusion is reported as
[perfscan issue #818](https://github.com/jxsl13/perfscan/issues/818).

External perfscan v1.71.0 ran with `GOPROXY=direct`. Exact merged main and
the final candidate each report 3,520 raw findings and 2,308 normalized
findings when tests are included. Their normalized record SHA-256 is exactly
`75dee2747115c751e8debf3f8cfb999f350649a7afb8dfcc41d2e26ac54bdefe`:
zero new and zero removed diagnostics.

Spectackle proposal `P-01M0KY9HK0FH8`, task `T-01M0KYDJXBEK0`, and ADR
`ADR-01M0KYBEDXFJW` govern the change. Their rules are `TQ2-FORMAT-001`,
`TQ2-CODES-001`, `TQ2-PORTABLE-QMATMUL-001`,
`TQ2-PORTABLE-SCRATCH-001`, `ARM64-TQ2-FUSED-DOT-001`, and
`ARM64-TQ2-FUSED-DOT-SCOPE-001`.

Final reindex records 2,699 files, 18,295 nodes, 33,739 edges, and 6,759 typed
calls. The fully paged check has zero drift; lint retains 133 existing
warnings and zero errors.

## Final validation

- Focused TQ2_0 and complete GGUF test-binary runs pass.
- A separately rebuilt race-enabled GGUF binary passes.
- CGO-disabled Linux ARM64 and AMD64 GGUF test binaries compile.
- `make preflight` and native `make preflight-metal` pass.
- Disassembly confirms four two-bit plane extracts, signed code mapping, four
  persistent float64 FMA chains, and no call inside the leaf.
- External perfscan reports zero normalized candidate ratchet drift.
- Spectackle reindex, fully paged check, and lint finish without errors.

## Reproduction

```sh
go test -c -o /tmp/goai-tq2-gguf.test ./format/gguf
/tmp/goai-tq2-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotTQ2Paths|BenchmarkQMatMulTQ2Paths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
GOAI_GGUF_TQ2_NEON_FIRST=1 /tmp/goai-tq2-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotTQ2Paths|BenchmarkQMatMulTQ2Paths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
```

Repeat those invocations five times each, alternating them. Normalize the
`/scalar` and `/neon` suffixes as in the committed streams, then run
`benchstat`. Direct test-binary filtering is intentional; repository policy
forbids `go test -run`.
