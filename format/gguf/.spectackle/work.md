---
schema: v1
---

## P-01M0K1A86JFY3RHNM84CR28WKW Add MXFP4 QMatMul and fuse its ARM64 row dot
kind: proposal
state: done
created: 2026-08-21
refs: ADR-01M0K18S8HF46ARNK03RQMD0BE
grilled: 2026-08-21 open=1
targets: format/gguf/mxfp4.go, format/gguf/quant_matmul.go, format/gguf/mxfp4_test.go, format/gguf/dot_mxfp4_scalar.go, format/gguf/dot_mxfp4_asm_arm64.go, format/gguf/dot_mxfp4_asm_arm64.s, format/gguf/dot_mxfp4_asm_arm64_test.go, format/gguf/bench_test.go

Add portable MXFP4 QMatMul semantics for F32 and F64 inputs with caller-owned decode scratch, then add a contiguous-F32 M1 Apple ARM64 row-level fused E8M0-scale, nibble-codebook, and dot kernel. Preserve decoder float32 operation order before f64 accumulation, exact subnormal scale behavior, portable fallback, selector boundaries, and zero leaf allocations. Benchmark scalar versus NEON at K4096, M1/N64/K1024, and M1/N4096/K1024 with n=10 alternating fresh-process order and an unchanged IQ4_XS negative control. Retain only if all three accelerated cells exceed 2x with no allocation regression; otherwise redesign the kernel boundary or reject. Commit reproducible evidence, run race/cross-build/preflight/external perfscan/Spectackle gates, report the generalizable selector gap on jxsl13/perfscan, open a PR, and merge only after every CI lane passes.

## T-01M0K1CEY0FNEBG4A4TWD9VKMN Implement and benchmark MXFP4 QMatMul with ARM64 fused row dot
kind: task
state: done
created: 2026-08-21
parent: P-01M0K1A86JFY3RHNM84CR28WKW
refs: P-01M0K1A86JFY3RHNM84CR28WKW, ADR-01M0K18S8HF46ARNK03RQMD0BE
grilled: 2026-08-21 open=1
targets: format/gguf/mxfp4.go, format/gguf/quant_matmul.go, format/gguf/mxfp4_test.go, format/gguf/dot_mxfp4_scalar.go, format/gguf/dot_mxfp4_asm_arm64.go, format/gguf/dot_mxfp4_asm_arm64.s, format/gguf/dot_mxfp4_asm_arm64_test.go, format/gguf/bench_test.go

Implement caller-owned MXFP4 decode and portable QMatMul for F32/F64 and M1/M>1, preserving E8M0 subnormal scale conversion, low-half then high-half signed E2M1 codebook order, float32 scale multiplication, and f64 accumulation. Add a 256-entry exact E8M0-half table and an Apple ARM64 contiguous-F32 M1 row selector that performs one zero-allocation assembly call per output row, table-loads each 17-byte block scale, vector-looks up signed nibble weights, and accumulates f64 partials. Prove exact scale-table/decode identity, known packed golden, arbitrary and cancellation-heavy error at most 1e-4, input immutability, selector boundaries, zero row allocations, portable F32/F64 parity, and M>1 scratch invariance. Benchmark scalar/NEON at K4096, M1/N64/K1024, and M1/N4096/K1024 using n=10 alternating fresh-process order with no retained-sample removal and an untouched IQ4_XS baseline/candidate control. Retain only if every cell exceeds 2x; otherwise redesign or reject. Commit evidence, run GGUF/race/cross-build/preflight/external perfscan/Spectackle gates, report generalized findings on perfscan #799, open a PR, wait for every CI lane, and merge only when all pass.
