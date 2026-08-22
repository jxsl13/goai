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

## P-01M0KEEGADEWBRSZJ2KTWCT4FX M2-first exact IQ2_S fused row dot and portable QMatMul
kind: proposal
state: draft
created: 2026-08-22
refs: ADR-01M0K6C6PEF4JT8435XC34X2GQ
targets: go:gguf.dequantIQ2_S, go:gguf.QMatMul, format/gguf/iq2s.go, format/gguf/quant_matmul.go

Close the bottom-up IQ2_S CPU execution gap without weakening QMatMul semantics. Refactor the decoder through caller-owned storage, add an exact scalar row-dot oracle, support direct F32/F64 activations for all M with reusable worker scratch, and select an ARM64 fused M1 F32 leaf only after numerical and allocation gates. Preserve the 1024-entry eight-wide grid, per-eight low byte plus two packed qh index bits, one direct sign byte per eight weights, sixteen explicit four-bit scales shared by adjacent eight-weight groups, d*(0.5+s)*0.25 float32 scaling, ascending element mapping, cancellation behavior, input immutability, and portable fallback. Benchmark matched M2 cells with neutral decoder and unrelated-quant controls, retain every final stream plus source and binary pins, inspect disassembly, cross-build Linux ARM64 and AMD64, run package, race, preflight, Metal, Spectackle, and external perfscan gates, report generalizable findings upstream, and ship only statistically validated leverage through a proper PR. Require at least 2x acceleration in each retained M1 cell with p<0.05 and no material time, byte, or allocation regression in controls. Treat pinned llama.cpp ARM IQ2_S at commit 3af988fabcf79fd81f8720505e684d2aa5bfc786 as a structural reference rather than a leadership baseline because it consumes Q8_K activations; preserve direct-activation semantics under ADR-01M0K6C6PEF4J and repository ADR-0016.
