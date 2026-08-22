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

## T-01M0KGWNMSF72T3J11M86A89G3 Implement and statistically gate exact IQ1_S QMatMul and M2 ARM64 fused row dot
kind: task
state: done
created: 2026-08-22
parent: P-01M0KGTBCHF59VJBME52TDJ43A
targets: go:gguf.dequantIQ1_S, go:gguf.QMatMul, format/gguf/iq1.go, format/gguf/quant_matmul.go

Implement caller-owned IQ1_S decode, exact portable scalar row dot, portable F32/F64 QMatMul coverage, and an ARM64-only contiguous-F32-M1 fused leaf. Add exact decoder parity, materialized-reference, invalid-input, scratch-allocation, selector-scope, known-value, arbitrary-packed-row, cancellation, input-immutability, and zero-allocation gates. Retain ten fresh-process 500ms samples for leaf and M1 N64/N4096 cells; retain exact merged-main dequant and unchanged IQ2_S controls with alternating process order and no final sample removal. Inspect assembly, run full/race/Linux cross-build/preflight/Metal/external-perfscan/Spectackle/CI gates, commit evidence and pins, report generalizable optimizer findings to perfscan, ship through a proper PR, merge only with exact head and all CI green, then delete the remote branch.
