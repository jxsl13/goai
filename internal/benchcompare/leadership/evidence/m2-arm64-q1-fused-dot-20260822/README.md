# M2 ARM64 Q1_0 fused row dot

## Scope

This tranche adds complete ggml type-41 `Q1_0` support: 18-byte blocks for
128 weights, eager GGUF decode, `QuantTensor.Dequantize`, public encode and
decode, portable F32/F64 `QMatMul`, caller-owned prefill scratch, and an ARM64
leaf for contiguous F32 M1 decode. The leaf fuses f16 scale lookup, packed-sign
expansion, sign application, and activation dot without materializing weights.

This is a same-semantics GoAI scalar-versus-ARM64 comparison. It does not claim
llama.cpp or cross-library leadership because the pinned llama.cpp ARM kernel
uses Q8_0 activations while GoAI accepts direct F32 activations.

## Reference pins

- Historical `ARCHITECTURE-RESEARCH.md` from commit `eb8b5a7f`, blob
  `a4b5ce34ce8db73f4b4c1ae01e7fcb0c1067755e`; it is a dated design input.
- Repository ADR: `docs/decisions/ADR-0016-quant-matmul-capability.md`.
- llama.cpp commit `3af988fabcf79fd81f8720505e684d2aa5bfc786`.
- `ggml-common.h` SHA-256
  `af255601767325f087313fa84b9435cb77aeec37df6b61b98d9ecc65f29fb4a0`.
- `ggml-quants.c` SHA-256
  `07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d`.
- ARM `quants.c` SHA-256
  `9fccd3897db24c9df89b8431b588175894e5f54697cf45768d0c6e6c5544093e`.
- Spectackle ADR `ADR-01M0K6C6PEF4J` preserves GoAI's direct-F32/F64 boundary.

## Environment

- Apple M2 Pro, macOS 26.5.1 (25F80), Darwin 25.5.0 arm64
- Go 1.26.6, `darwin/arm64`
- Baseline source: merged main `c6ece6f1b4e686fd4e0f93278063e8623dc320e5`
- Baseline binary SHA-256: `11c67ef76e249d6e05bfc595216ffc61ed282e9358c264a32c14bb7ed075f803`
- Candidate source: this evidence directory's containing commit
- Candidate binary SHA-256: `c9f9ac46727711e37ff3ef92f0af9ef6225debd03b620ba906bf8464bcb68e2f`
- Spectackle v0.9.3; external perfscan module v1.71.0

## Method

- Ten retained fresh-process samples use `-test.benchtime 500ms`; invocation
  order alternates scalar-first and NEON-first. Every final sample, including
  system-contention outliers, is committed and none was removed.
- The unchanged IQ1_M M1/N4096 and dequant benchmarks use ten fresh-process
  baseline/candidate pairs, alternating binary order.
- `benchstat` compares the committed normalized streams.
- A separate twelve-pair sweep compares two exact sign tables. The initial
  8 KiB `[256][8]uint32` table measured 606.97 ns geomean; the retained 2 KiB
  `[256][8]byte` plus two `SHLL` stages measured 606.31 ns. It preserves
  throughput while reducing hot-table footprint by 75%.

## Results

| Cell | Portable scalar | ARM64 fused | Speedup | Significance |
| --- | ---: | ---: | ---: | ---: |
| Q1_0 leaf, K=4096 | 4.8895 us | 0.6153 us | 7.95x | p=0.000, n=10 |
| Q1_0 QMatMul, M1/N64/K1024 | 79.15 us | 10.16 us | 7.79x | p=0.000, n=10 |
| Q1_0 QMatMul, M1/N4096/K1024 | 859.1 us | 163.3 us | 5.26x | p=0.000, n=10 |
| Existing IQ1_M dequant control | 446.0 us | 330.3 us | flat | p=0.481, n=10 |
| Existing IQ1_M M1/N4096 control | 296.0 us | 226.2 us | flat | p=0.853, n=10 |

Every accelerated cell clears the proposal's hard 2x threshold. Allocation
counts are flat: the leaf remains zero-allocation, N64 remains at four, and
N4096 remains at 29. Both baseline/candidate controls are neutral in time,
bytes, and allocations.

## Correctness gates

- Exact all-positive and all-negative block goldens pass.
- Maximum scalar-relative error over 100 arbitrary packed rows at K=128, 256,
  and 4096 is `1.138970942829309e-16`.
- Cancellation-heavy paired-half input remains within the `1e-4` gate.
- Inputs are immutable and the ARM64 row wrapper allocates zero bytes.
- The portable fused row dot is bit-exact to materialized decode plus dot.
- Pinned Q1_0 block layout and reference quantization bytes match exactly.
- Eager, raw-tensor, public decode, F32/F64, and M1/M3 QMatMul paths agree.
- Invalid widths, truncated bytes, and unsupported shapes return clean errors.
- M>1 scratch allocations are invariant from N=1 to N=31.
- Selector tests restrict the leaf to ARM64 contiguous F32 M1; F32 M>1 and
  F64 remain portable.

## Static analysis and specification

External perfscan v1.71.0 ran with `GOPROXY=direct` and isolated caches. Exact
merged main and candidate each report 1,669 raw findings and 957 normalized
findings: zero new and zero removed diagnostics. The retained compact packed-bit
expansion technique is reported as
[perfscan issue #815](https://github.com/jxsl13/perfscan/issues/815).

Spectackle proposal `P-01M0KPSF31FYV` and task `T-01M0KPVJQVFCS` govern the
change. The five rules are `Q1-FORMAT-001`, `Q1-PORTABLE-QMATMUL-001`,
`Q1-PORTABLE-SCRATCH-001`, `ARM64-Q1-FUSED-DOT-001`, and
`ARM64-Q1-FUSED-DOT-SCOPE-001`. A fully paged check reports zero drift.
Reindexing records 2,685 files, 18,200 nodes, 33,566 edges, and 6,753 typed
calls. Spectackle lint reports 133 pre-existing warnings and zero errors.

## Final validation

- Focused Q1_0 and complete GGUF test-binary runs pass.
- A separately rebuilt race-enabled GGUF binary passes.
- CGO-disabled Linux ARM64 and AMD64 GGUF test binaries compile.
- `make preflight` and native `make preflight-metal` pass.
- Disassembly confirms one row-level native call, compact 2 KiB sign lookup,
  two widening stages, four FP64 FMA chains, and no call inside the leaf.
- External perfscan reports zero candidate ratchet drift.

## Reproduction

```sh
go test -c -o /tmp/goai-q1-gguf.test ./format/gguf
/tmp/goai-q1-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotQ1Paths|BenchmarkQMatMulQ1Paths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
GOAI_GGUF_Q1_NEON_FIRST=1 /tmp/goai-q1-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotQ1Paths|BenchmarkQMatMulQ1Paths)$' -test.benchmem -test.count 1 -test.benchtime 500ms
```

Repeat those invocations five times each, alternating them. Normalize the
`/scalar` and `/neon` suffixes as in the committed streams, then run
`benchstat`. Direct test-binary filtering is intentional; repository policy
forbids `go test -run`.
