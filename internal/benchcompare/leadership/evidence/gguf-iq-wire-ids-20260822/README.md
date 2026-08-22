# GGUF IQ1_S and IQ2_S wire ID correction

## Scope

This tranche repairs a wire-format ABI mismatch. Pinned ggml assigns IQ1_S to
type 19, IQ2_S to type 22, and I8 to type 24. GoAI previously assigned IQ2_S
to 19 and IQ1_S to 24. The corrected registry now uses the exact pinned IDs,
adds the missing eager and raw-tensor dispatch for both formats, and rejects
type 24 until I8 support exists.

The decoder layouts, codebooks, block sizes, QMatMul implementations, and
ARM64 kernels are unchanged. This is a correctness and interoperability
result, not a performance leadership claim.

## Reference pin

- llama.cpp commit: `3af988fabcf79fd81f8720505e684d2aa5bfc786`
- pinned `ggml.h` SHA-256:
  `6fe9b62d3ea48c2de82cce6e9e06d3ae4f0de34f4b5831399c49c099badefb09`
- pinned enum values: IQ1_S=19, IQ2_S=22, I8=24
- Spectackle ADR: `ADR-01M0M15D3HFHN`

## Dispatch gates

- Public `IQ1_S` and internal `tIQ1_S` equal 19.
- Public `IQ2_S` and internal `tIQ2_S` equal 22.
- `byteSize(19, 256)` is 50 bytes; `byteSize(22, 256)` is 82 bytes.
- Public `Dequantize`, eager `Read`, `ReadRaw`, and
  `QuantTensor.Dequantize` match the direct decoder bit-for-bit.
- Numeric `QMatMul` type 19 invokes only IQ1_S; type 22 invokes only IQ2_S.
- Type 24 returns `unsupported ggml type 24` from byte-size, public, eager,
  raw-tensor, raw-file, and QMatMul entry points.

## Performance ratchet

Ten retained fresh-process baseline/candidate pairs use 500 ms benchmarks and
alternate binary order. No sample was removed.

| Existing path | Exact merged main | Candidate | Result |
| --- | ---: | ---: | ---: |
| IQ2_S M1/N4096/K1024 | 198.2 us | 198.0 us | neutral, p=0.481 |
| IQ1_S M1/N4096/K1024 | 147.8 us | 148.0 us | neutral, p=0.218 |

Both paths remain at 29 allocations. The geomean time delta is +0.05%.

## Validation

- Focused wire-ID, dispatch, eager/raw, QMatMul, and type-24 rejection tests pass.
- Complete native and separately built race-enabled GGUF short suites pass.
- CGO-disabled Linux ARM64 and AMD64 GGUF test binaries compile.
- External perfscan v1.71.0 ran with `GOPROXY=direct`: exact merged main and
  candidate both have 3,520 raw and 2,308 normalized findings with identical
  normalized SHA-256
  `75dee2747115c751e8debf3f8cfb999f350649a7afb8dfcc41d2e26ac54bdefe`.
- `make preflight` and native `make preflight-metal` pass.
- Spectackle reindexed 2,700 files, 18,300 nodes, 33,764 edges, and 6,759
  typed calls. Its fully paged check has zero drift; lint retains 133 existing
  warnings and zero errors.
- Spectackle proposal `P-01M0M13SZGFBN` and task `T-01M0M1625WFQG`
  govern the change.

No perfscan issue is filed for this tranche because it introduces no
generalizable performance improvement; it is a wire correctness repair.
