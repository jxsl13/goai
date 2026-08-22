---
schema: v1
---

## ADR-01M0K6C6PEF4JT8435XC34X2GQ Which semantic boundary should the IQ3_XXS M2 tranche optimize?
kind: adr
state: done
created: 2026-08-21
context: ARCHITECTURE-RESEARCH.md requires one semantic definition, explicit numerical modes, portable fallback, shape-specific M=1 kernels, and matched benchmark cells. The GoAI public QMatMul accepts F32/F64 activations and accumulates portable results in F64. llama.cpp commit 3af988fabcf79fd81f8720505e684d2aa5bfc786 uses Q8_K activations for its ARM IQ3_XXS dot, so adopting that boundary would introduce activation quantization error and a different public semantic contract.
decision: Preserve the GoAI direct-F32/F64 QMatMul semantics and add a portable decoder plus an Apple ARM64 exact row dot
consequences: The portable path remains the semantic oracle and supports F32/F64 plus M greater than one. Only contiguous F32 M=1 dispatches the ARM64 leaf. The leaf expands grids, sign masks, and scale factors in registers and retains float64 partial accumulation. Cross-library leadership is not claimed against llama.cpp Q8_K activation kernels because the numerical boundary differs; the retained claim is an internal same-semantics M2 gain with a reproducible cell manifest.
status: accepted

kind: radio
option: Preserve the GoAI direct-F32/F64 QMatMul semantics and add a portable decoder plus an Apple ARM64 exact row dot
option: Quantize activations to Q8_K and match the llama.cpp IQ3_XXS by Q8_K kernel boundary
option: Implement only a tensor dequantization optimization and defer QMatMul
blocks: P-01M0K6A4A6F0SAGEMT1937ZQQN
choice: Preserve the GoAI direct-F32/F64 QMatMul semantics and add a portable decoder plus an Apple ARM64 exact row dot

## P-01M0KBF17FE9Z83X9MEJ23S1VM M2-first exact IQ2_XS fused row dot and portable QMatMul
kind: proposal
state: active
created: 2026-08-21
refs: ADR-01M0K6C6PEF4JT8435XC34X2GQ
grilled: 2026-08-21 open=0
targets: go:gguf.dequantIQ2_XS, go:gguf.QMatMul, format/gguf/iq2xs.go, format/gguf/quant_matmul.go

Close the bottom-up IQ2_XS CPU execution gap without weakening QMatMul semantics. Refactor the decoder through caller-owned storage, add an exact scalar row-dot oracle, support direct F32/F64 activations for all M with reusable worker scratch, and select an ARM64 fused M=1 F32 leaf only after numerical and allocation gates. Preserve the 512-entry eight-wide grid, per-eight 9-bit grid and 7-bit ksign indices, per-sixteen explicit 4-bit scales, d*(0.5+s)*0.25 float32 scaling, ascending element mapping, cancellation behavior, input immutability, and portable fallback. Benchmark matched M2 cells with neutral dequant and unrelated-quant controls, retain every final stream plus source and binary pins, inspect disassembly, cross-build Linux ARM64 and AMD64, run package, race, preflight, Metal, Spectackle, and external perfscan gates, report generalizable findings upstream, and ship only statistically validated leverage through a proper PR. Treat pinned llama.cpp ARM IQ2_XS at commit 3af988fabcf79fd81f8720505e684d2aa5bfc786 as a structural reference rather than a leadership baseline because it consumes Q8_K activations; preserve direct-activation semantics under ADR-01M0K6C6PEF4J and repository ADR-0016.

## T-01M0KBJMSPENETC1VE2QPSM91C Implement and statistically gate exact IQ2_XS QMatMul and M2 ARM64 fused row dot
kind: task
state: active
created: 2026-08-21
parent: P-01M0KBF17FE9Z83X9MEJ23S1VM
refs: ADR-01M0K6C6PEF4JT8435XC34X2GQ
targets: go:gguf.dequantIQ2_XS, go:gguf.QMatMul, format/gguf/iq2xs.go, format/gguf/quant_matmul.go, format/gguf/quant_matmul_test.go

Implement under ADR-01M0K6C6PEF4J and repository ADR-0016. Refactor dequantIQ2_XS through a caller-owned into decoder, add an exact portable row-dot oracle, admit IQ2_XS into QMatMul for direct F32/F64 activations and all M with one scratch set per worker, and select an ARM64 fused M=1 F32 leaf only. Preserve the 512-entry eight-wide grid, each uint16 low 9-bit grid index and high 7-bit ksign index, sixteen explicit four-bit scales shared per adjacent eight-weight groups, d*(0.5+s)*0.25 float32 scaling, ascending mapping, cancellation behavior, input immutability, and portable fallback. Add non-vacuous selector-scope, scratch-allocation, scalar/materialized parity, arbitrary raw-row error, known-block, F32/F64 M1/M3, invalid-shape, cancellation, immutability, and zero-leaf-allocation tests. Benchmark leaf K4096 and QMatMul M1/N64 and M1/N4096 with dequant and unrelated-quant controls in alternating fresh-process samples; preserve raw streams, benchstat, manifest, source pins, sample order, and binary hashes. Inspect disassembly, cross-build Linux ARM64 and AMD64, run package, race, preflight, Metal, Spectackle, and external perfscan gates with GOPROXY=direct, report generalizable findings upstream, and ship through a proper PR only after statistically validated leverage. Pin llama.cpp ARM quants commit 3af988fabcf79fd81f8720505e684d2aa5bfc786 as a structural reference only because its IQ2_XS kernel consumes Q8_K activations.
