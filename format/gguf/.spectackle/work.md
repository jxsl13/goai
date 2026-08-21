---
schema: v1
---

## P-01M0JZ33ZVEMKS1K6FYAEFJFK6 Add IQ4_XS QMatMul and fuse its ARM64 super-block dot
kind: proposal
state: done
created: 2026-08-21
refs: ADR-01M0JZ1Y2MF948ZEGM1K33ZWTM, P-01M0JWHMHZEDQVCFCT3GJVPME7
grilled: 2026-08-21 open=0
targets: go:gguf.QMatMul, format/gguf/iq4.go, format/gguf/quant_matmul.go, format/gguf/iq4_test.go, format/gguf/bench_test.go, format/gguf/dot_iq4xs_asm_arm64.go, format/gguf/dot_iq4xs_asm_arm64.s

Extend the importance-quant QMatMul family bottom-up from merged IQ4_NL to IQ4_XS. Add portable F32/F64 semantics for M greater than or equal to one, preserving f16 super-scale, signed six-bit sub-scale unpack, low-half then high-half nonlinear codebook order, and f64 accumulation. Add a zero-allocation scalar M1 row dot and an Apple ARM64 selector whose row wrapper computes exact float32 d*subscale coefficients and calls a fused 256-weight NEON nonlinear-lookup dot with f64 partials; if K/256 call overhead prevents at least a 2x retained gain at the K4096 leaf and both M1/N64/K1024 and M1/N4096/K1024 QMatMul cells, upgrade the same tranche to a row-level leaf or reject it. Measure scalar versus NEON in one candidate binary with n=10 alternating order, excluded warmups, no removed samples, unchanged allocations, and an untouched supported-quant negative control. Preserve IQ4_NL and every existing quant path, make no llama.cpp leadership claim without a matched harness, and require direct raw-byte, cancellation, selector-scope, full GGUF, race, cross-build, preflight, external perfscan, reproducible evidence, PR, all-green CI, and merge gates.
