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

## P-01M0K8Z3S1ER2BYJ5H815QYNM8 M2-first exact IQ2_XXS fused row dot and portable QMatMul
kind: proposal
state: active
created: 2026-08-21
targets: go:gguf.dequantIQ2_XXS, go:gguf.QMatMul, format/gguf/iq2xxs.go, format/gguf/quant_matmul.go

Close the bottom-up IQ2_XXS CPU execution gap without weakening QMatMul semantics. Add caller-owned portable dequantization and an exact scalar row-dot oracle, support F32 and F64 activations for all M through reusable scratch, and select an Apple ARM64 fused M=1 F32 leaf only after numerical and allocation gates. Benchmark fresh-process matched cells on M2, retain neutral dequant and unrelated-quant controls, inspect generated instructions, cross-build Linux ARM64 and AMD64, run the full package/race/Metal preflight matrix, and make no cross-library leadership claim against llama.cpp because its pinned ARM IQ2_XXS kernel consumes Q8_K activations rather than direct F32. Version evidence, source pins, sample order, and binary hashes. Report any generalizable perfscan improvement upstream.
