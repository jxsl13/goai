---
schema: v1
---

## P-01M0K1A86JFY3RHNM84CR28WKW Add MXFP4 QMatMul and fuse its ARM64 row dot
kind: proposal
state: active
created: 2026-08-21
refs: ADR-01M0K18S8HF46ARNK03RQMD0BE
grilled: 2026-08-21 open=1
targets: format/gguf/mxfp4.go, format/gguf/quant_matmul.go, format/gguf/mxfp4_test.go, format/gguf/dot_mxfp4_scalar.go, format/gguf/dot_mxfp4_asm_arm64.go, format/gguf/dot_mxfp4_asm_arm64.s, format/gguf/dot_mxfp4_asm_arm64_test.go, format/gguf/bench_test.go

Add portable MXFP4 QMatMul semantics for F32 and F64 inputs with caller-owned decode scratch, then add a contiguous-F32 M1 Apple ARM64 row-level fused E8M0-scale, nibble-codebook, and dot kernel. Preserve decoder float32 operation order before f64 accumulation, exact subnormal scale behavior, portable fallback, selector boundaries, and zero leaf allocations. Benchmark scalar versus NEON at K4096, M1/N64/K1024, and M1/N4096/K1024 with n=10 alternating fresh-process order and an unchanged IQ4_XS negative control. Retain only if all three accelerated cells exceed 2x with no allocation regression; otherwise redesign the kernel boundary or reject. Commit reproducible evidence, run race/cross-build/preflight/external perfscan/Spectackle gates, report the generalizable selector gap on jxsl13/perfscan, open a PR, and merge only after every CI lane passes.
