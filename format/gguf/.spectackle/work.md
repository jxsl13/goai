---
schema: v1
---

## P-01M0K3NE11FSRAY6V751A2PC1K M2-first exact IQ3_S fused row dot and portable QMatMul
kind: proposal
state: active
created: 2026-08-21
targets: go:gguf.dequantIQ3_S, format/gguf/quant_matmul.go, format/gguf/iq3s.go

Add caller-owned IQ3_S decode and QMatMul support for F32/F64, then specialize only the F32 M=1 path on Apple ARM64 with a fused 9-bit grid/direct-sign row dot. Preserve exact materialized-reference semantics, input immutability, and constant scratch. Retain native code only when repeated fresh-process benchmarks show at least 2x speedup for K4096 leaf, M1/N64/K1024, and M1/N4096/K1024, with p<=0.01 and no statistically significant regression in an unrelated quantized negative control. IQ3_XXS remains a separate future tranche per ADR-01M0K3K391ERQ.

## T-01M0K3PD38FVWSMA8JTVVD2ERE Implement and statistically gate exact IQ3_S QMatMul and M2 ARM64 fused row dot
kind: task
state: active
created: 2026-08-21
parent: P-01M0K3NE11FSRAY6V751A2PC1K
targets: go:gguf.dequantIQ3_S, format/gguf/quant_matmul.go, format/gguf/iq3s.go

Freeze a merged-state baseline binary. Add caller-owned exact IQ3_S decoding, portable F32/F64 QMatMul, one-scratch-per-worker reuse, and an F32 M=1 selector. Implement a zero-allocation Apple ARM64 row-level fused 9-bit grid/direct-sign dot without changing inputs. Verify exact decoder and mapping parity, F64 reference correctness, M1/M3 behavior, selector scope, allocation invariants, cancellation and random packed blocks, race safety, cross-builds, and perfscan. Retain native code only if n=10 fresh-process 500ms alternating-order samples show at least 2x on K4096 leaf, M1/N64/K1024, and M1/N4096/K1024 with p<=0.01, while an unrelated quantized negative control does not regress significantly.
