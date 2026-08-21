---
schema: v1
---

## P-01M0K6A4A6F0SAGEMT1937ZQQN M2-first exact IQ3_XXS fused row dot and portable QMatMul
kind: proposal
state: active
created: 2026-08-21
grilled: 2026-08-21 open=0
needs: ADR-01M0K6C6PEF4JT8435XC34X2GQ
targets: go:gguf.dequantIQ3_XXS, go:gguf.QMatMul, format/gguf/iq3xxs.go, format/gguf/quant_matmul.go

Following ARCHITECTURE-RESEARCH.md CPU §§5.4-5.8 and benchmark §14, add caller-owned IQ3_XXS decoding and direct-F32/F64 QMatMul support, then specialize only contiguous F32 M=1 on Apple ARM64. Fuse 256x4 grid lookup, 7-bit ksigns expansion, packed sub-scale application, and activation dot without materializing weights or quantizing activations. Preserve exact materialized-reference semantics, input immutability, portable fallback, and output-row-independent scratch. Use llama.cpp commit 3af988fabcf79fd81f8720505e684d2aa5bfc786 as an executable layout/kernel reference; its Q8_K activation boundary is not a matched cross-library claim. Retain native code only if n=10 fresh-process, 500 ms, alternating-order samples show at least 2x improvement on K4096 leaf, M1/N64/K1024, and M1/N4096/K1024 with p<=0.01, flat allocation/byte profiles, and no statistically significant regression in an unrelated quantized control. Commit a reproducible evidence manifest and report any generalized finding to perfscan.

## ADR-01M0K6C6PEF4JT8435XC34X2GQ Which semantic boundary should the IQ3_XXS M2 tranche optimize?
kind: adr
state: submitted
created: 2026-08-21
context: ARCHITECTURE-RESEARCH.md requires one semantic definition, explicit numerical modes, portable fallback, shape-specific M=1 kernels, and matched benchmark cells. The GoAI public QMatMul accepts F32/F64 activations and accumulates portable results in F64. llama.cpp commit 3af988fabcf79fd81f8720505e684d2aa5bfc786 uses Q8_K activations for its ARM IQ3_XXS dot, so adopting that boundary would introduce activation quantization error and a different public semantic contract.
status: proposed

kind: radio
option: Preserve the GoAI direct-F32/F64 QMatMul semantics and add a portable decoder plus an Apple ARM64 exact row dot
option: Quantize activations to Q8_K and match the llama.cpp IQ3_XXS by Q8_K kernel boundary
option: Implement only a tensor dequantization optimization and defer QMatMul
blocks: P-01M0K6A4A6F0SAGEMT1937ZQQN
