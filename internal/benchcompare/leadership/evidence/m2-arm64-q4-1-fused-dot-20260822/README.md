# M2 ARM64 Q4_1 fused decode-dot evidence — 2026-08-22

GoAI now implements the exact 20-byte GGUF Q4_1 block and fuses its nibble
unpack, affine dequantization, f32 activation multiply, and row reduction in a
single Plan 9 ARM64 NEON call for M=1. The portable Go implementation remains
the reference and all non-ARM64 and M>1 paths remain portable.

## Internal result

Ten fresh benchmark processes alternated the scalar-first and NEON-first
sub-benchmark order. Values below are medians; no sample was removed.

| Cell | scalar/control | ARM64 NEON | speedup |
|---|---:|---:|---:|
| Q4_1 row dot, K=4096 | 4,990.5 ns | **826.35 ns** | **6.04x** |
| Materialize Q4_1 then dot, K=4096 | 7,611 ns | **826.35 ns** | **9.21x** |
| QMatMul M1/N64/K1024 | 78,553 ns | **13,811 ns** | **5.69x** |
| QMatMul M1/N4096/K1024 | 815,000 ns | **195,197.5 ns** | **4.18x** |

The fused leaf is allocation-free. The materialized reference uses 16,608
bytes and three allocations per operation. QMatMul allocation counts are
unchanged between its scalar and NEON arms: four at N=64 and 29 at N=4096.
The N=4096 scalar samples are visibly noisy, so only the large effect and
median are claimed; there is no p-value claim in this campaign.

## Pinned llama.cpp boundary

llama.cpp was pinned to commit
`3af988fabcf79fd81f8720505e684d2aa5bfc786`, built Release, CPU-only, with
native Apple ARM feature detection, and measured in ten fresh processes with
the two timed arms reversed on alternating invocations.

| K=4096 boundary | median |
|---|---:|
| llama.cpp Q4_1 x already-quantized Q8_1 dot | **120.595 ns** |
| llama.cpp F32 to Q8_1 activation quantization plus dot | 1,096.090 ns |
| GoAI direct Q4_1 x F32 NEON dot | **826.35 ns** |

The pre-quantized llama.cpp microkernel is 6.85x faster than GoAI's direct-F32
kernel and establishes the next CPU optimization target. For the boundary that
starts with F32 activations, GoAI is 1.33x faster than llama.cpp's activation
quantization plus dot. This is **not** promoted to a cross-library leadership
cell: Q8_1 changes activation precision, the generated data are not shared
byte-for-byte, and the accumulation contracts differ. The two boundaries are
reported together precisely to avoid hiding conversion work or overstating
semantic parity.

The benchmark source is retained as `llama_q41_bench.cpp`. llama.cpp's source
is the wire/algorithm oracle, not copied implementation code:

- <https://github.com/ggml-org/llama.cpp/tree/3af988fabcf79fd81f8720505e684d2aa5bfc786/ggml/src/ggml-cpu>
- Arm, *Coding for Neon*, Issue 04:
  <https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Neon%20Programmers%20Guide/102159_0104_01_CodingForNeon.pdf>

## Numerical and API gates

- The wire golden is `d=1`, `m=0`, followed by packed bytes `00 11 ... ff`;
  a second non-grid block is byte-identical to pinned llama.cpp output.
- Constant, arbitrary-raw, round-trip, writer, eager-reader, raw-reader, and
  `QuantTensor.Dequantize` paths are covered.
- 120 arbitrary raw rows span K=32, 64, 256, and 4096 and verify input
  immutability. Maximum mixed scalar-relative error was
  `6.294893722337496e-16`, below the `2e-5 absolute + 2e-5 relative` contract.
- A cancellation-heavy K=4096 row, a known-answer row, the full fused/general
  QMatMul matrix, zero leaf allocations, and scratch-allocation scaling pass.
- The race-selected Q4_1 suite passes.
- CGO-disabled test binaries compile for darwin/amd64, linux/arm64, and
  linux/amd64.
- Metal and Vulkan still return `backend.ErrQuantUnsupported` for wire type 3;
  their tagged test binaries compile. Runtime tests skipped because the
  devices were unavailable in the test process.
- External `github.com/jxsl13/perfscan/perfscan@v1.71.0` ran with
  `GOPROXY=direct`: both base and candidate have 79 findings, with zero added
  and zero removed. The generalizable opportunity is
  [perfscan issue 819](https://github.com/jxsl13/perfscan/issues/819).

## Reproduction

GoAI test and benchmark binaries are always compiled before selection; no
`go test -run` command is used:

```sh
GOCACHE=/private/tmp/goai-q41-gocache go test -c -o /private/tmp/gguf-q41.test ./format/gguf
/private/tmp/gguf-q41.test -test.run='Q41|TestQMatMulFusedDecodeMatchesGeneralPathExactly'
/private/tmp/gguf-q41.test -test.run='^$' -test.bench='^BenchmarkDotQ41Paths$' -test.benchtime=300ms -test.count=1
GOAI_GGUF_Q41_NEON_FIRST=1 /private/tmp/gguf-q41.test -test.run='^$' -test.bench='^BenchmarkDotQ41Paths$' -test.benchtime=300ms -test.count=1
/private/tmp/gguf-q41.test -test.run='^$' -test.bench='^BenchmarkQMatMulQ41Paths$' -test.benchtime=300ms -test.benchmem -test.count=1
GOAI_GGUF_Q41_NEON_FIRST=1 /private/tmp/gguf-q41.test -test.run='^$' -test.bench='^BenchmarkQMatMulQ41Paths$' -test.benchtime=300ms -test.benchmem -test.count=1
```

Pinned llama.cpp build and harness:

```sh
git -C /private/tmp/llama-q41.VxGYEf checkout 3af988fabcf79fd81f8720505e684d2aa5bfc786
cmake -S /private/tmp/llama-q41.VxGYEf -B /private/tmp/llama-q41.VxGYEf/build-goai-q41 -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=ON
cmake --build /private/tmp/llama-q41.VxGYEf/build-goai-q41 --target ggml-cpu -j8
c++ -O3 -DNDEBUG -std=c++17 -I/private/tmp/llama-q41.VxGYEf/ggml/include -I/private/tmp/llama-q41.VxGYEf/ggml/src -I/private/tmp/llama-q41.VxGYEf/ggml/src/ggml-cpu llama_q41_bench.cpp /private/tmp/llama-q41.VxGYEf/build-goai-q41/ggml/src/libggml-cpu.a /private/tmp/llama-q41.VxGYEf/build-goai-q41/ggml/src/libggml-base.a -framework Accelerate -framework Foundation -lpthread -ldl -o /private/tmp/q41-llama-bench
```

## Claim boundary

This evidence supports exact Q4_1 interoperability and an internal M2 ARM64
gain. It does not claim whole-model, GPU, or matched cross-library leadership.
Native Metal Q4_1 and a same-bytes, same-activation competitor cell are separate
follow-on work.
