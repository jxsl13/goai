---
schema: v1
---

## P-01M0K6A4A6F0SAGEMT1937ZQQN M2-first exact IQ3_XXS fused row dot and portable QMatMul
kind: proposal
state: active
created: 2026-08-21
targets: go:gguf.dequantIQ3_XXS, go:gguf.QMatMul, format/gguf/iq3xxs.go, format/gguf/quant_matmul.go

Following ARCHITECTURE-RESEARCH.md CPU §§5.4-5.8 and benchmark §14, add caller-owned IQ3_XXS decoding and direct-F32/F64 QMatMul support, then specialize only contiguous F32 M=1 on Apple ARM64. Fuse 256x4 grid lookup, 7-bit ksigns expansion, packed sub-scale application, and activation dot without materializing weights or quantizing activations. Preserve exact materialized-reference semantics, input immutability, portable fallback, and output-row-independent scratch. Use llama.cpp commit 3af988fabcf79fd81f8720505e684d2aa5bfc786 as an executable layout/kernel reference; its Q8_K activation boundary is not a matched cross-library claim. Retain native code only if n=10 fresh-process, 500 ms, alternating-order samples show at least 2x improvement on K4096 leaf, M1/N64/K1024, and M1/N4096/K1024 with p<=0.01, flat allocation/byte profiles, and no statistically significant regression in an unrelated quantized control. Commit a reproducible evidence manifest and report any generalized finding to perfscan.
