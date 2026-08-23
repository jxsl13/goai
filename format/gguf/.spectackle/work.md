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

## ADR-01M0M2ZMEDF86A4K2PP8HB176E Where should complete wire-format dispatch for already supported IQ and MXFP4 types live?
kind: adr
state: done
created: 2026-08-22
context: Read and QuantTensor.Dequantize already converge on decodeTensor, while public Dequantize separately proves each decoder and byteSize validates each layout.
decision: Extend the shared decodeTensor switch with the exact existing decoder functions and preserve unsupported-type errors
consequences: Read and QuantTensor.Dequantize gain identical coverage through one selector; public Dequantize remains an independent exact oracle; block layouts, decoder math, QMatMul routing, wire IDs, and unsupported-type behavior stay unchanged. Exhaustive synthetic-wire tests must cover every newly routed format.
status: accepted

kind: radio
option: Extend the shared decodeTensor switch with the exact existing decoder functions and preserve unsupported-type errors
option: Add format-specific entry-point wrappers or compatibility aliases outside decodeTensor
blocks: P-01M0M2XZRGFSWVT5G94ZXT61S8
choice: Extend the shared decodeTensor switch with the exact existing decoder functions and preserve unsupported-type errors

## T-01M0PKDX1FF0DARF9FTEKXN3EM Restore the measured block-aligned Q4_K row wrapper
kind: task
state: active
created: 2026-08-23
parent: P-01M0PGHM4TE7YAXSJT0Q56SSZ2
targets: go:gguf.dotQ4KPairRowASM

Remove the post-benchmark ceil-division guard from the internal paired ARM64 Q4_K wrapper. Q4_K callers already validate positive block-aligned K, so the additional add and rounding do not protect a valid input. Restore the exact k==0 and k/qkK wrapper used by the clean 7-of-7 leaf, FFN, and production campaigns; then rebuild and hash-compare the production binary to the recorded measured binary.
