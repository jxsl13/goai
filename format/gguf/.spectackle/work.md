---
schema: v1
---

## P-01M0K3NE11FSRAY6V751A2PC1K M2-first exact IQ3_S fused row dot and portable QMatMul
kind: proposal
state: active
created: 2026-08-21
targets: go:gguf.dequantIQ3_S, format/gguf/quant_matmul.go, format/gguf/iq3s.go

Add caller-owned IQ3_S decode and QMatMul support for F32/F64, then specialize only the F32 M=1 path on Apple ARM64 with a fused 9-bit grid/direct-sign row dot. Preserve exact materialized-reference semantics, input immutability, and constant scratch. Retain native code only when repeated fresh-process benchmarks show at least 2x speedup for K4096 leaf, M1/N64/K1024, and M1/N4096/K1024, with p<=0.01 and no statistically significant regression in an unrelated quantized negative control. IQ3_XXS remains a separate future tranche per ADR-01M0K3K391ERQ.
