# GGUF IQ and MXFP4 wire dispatch completion

## Scope

This tranche closes the remaining gap between GoAI's existing quantization
decoders and real GGUF file loading. The shared eager decoder now recognizes
all ten already-supported wire types: IQ2_XXS=16, IQ2_XS=17, IQ3_XXS=18,
IQ1_S=19, IQ4_NL=20, IQ3_S=21, IQ2_S=22, IQ4_XS=23, IQ1_M=29, and MXFP4=39.
The IQ1_S and IQ2_S IDs were corrected separately in PR #1151; this change
adds the eight missing eager dispatch cases and exhaustively gates all ten.

No quantization math, layout, block size, codebook, QMatMul kernel, or ARM64
selector changes. Unsupported type 24 remains rejected. This is a file-loading
correctness and interoperability result, not a performance leadership claim.

## Dispatch gates

- Each direct decoder produces the reference tensor for its exact synthetic
  single-tensor GGUF payload.
- Public `Dequantize`, internal `decodeTensor`, eager `Read`, and `ReadRaw`
  followed by `QuantTensor.Dequantize` match that reference bit-for-bit.
- `ReadRaw` preserves the exact wire type and raw payload bytes.
- The coverage table contains exactly ten entries, preventing an empty or
  accidentally shortened range from vacuously passing.
- Type 24 still returns `unsupported ggml type 24` through byte-size, public,
  eager, raw-tensor, raw-file, and QMatMul entry points.

## Performance ratchet

Ten retained fresh-process direct/wire pairs use 500 ms benchmarks,
`GOMAXPROCS=1`, alternating order, and no sample removal. All ten time
comparisons are neutral (`p>=0.315`). The time geomean is 310.9 ns for direct
decoder calls and 305.7 ns for wire dispatch (-1.67%). Every format remains at
1,256 bytes and four allocations per operation. `benchstat.txt` contains the
complete comparison and the raw streams are committed alongside it.

## Validation

- Focused exhaustive dispatch and unsupported-type tests pass.
- The complete native and separately built race-enabled GGUF short suites pass.
- CGO-disabled Linux ARM64 and AMD64 GGUF test binaries compile.
- External perfscan v1.71.0 ran through `GOPROXY=direct` against exact merged
  main and the candidate. Both contain 5,975 findings, the normalized streams
  are byte-identical, and the candidate adds zero findings.
- `make preflight` and native `make preflight-metal` pass.
- Spectackle proposal `P-01M0M2XZRGFSW`, task `T-01M0M30EFGEN4`, ADR
  `ADR-01M0M2ZMEDF86`, and rules `GGUF-IQ-MXFP4-WIRE-DISPATCH-001` and
  `GGUF-IQ-MXFP4-WIRE-SCOPE-001` govern the change.

No perfscan issue is filed for this tranche because it introduces no
generalizable performance improvement.
