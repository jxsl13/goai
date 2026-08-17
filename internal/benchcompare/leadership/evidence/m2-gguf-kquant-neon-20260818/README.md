# M2 GGUF Q4_K/Q6_K NEON evidence — 2026-08-18

## Verdict

GoAI now dispatches eager Q4_K and Q6_K dequantization to bit-exact ARM64 NEON
kernels while retaining the scalar Go implementation on other architectures.
On Apple M2 Pro, the zero-allocation 262,144-value kernels improve from 95.50 to
31.68 microseconds for Q4_K (**3.01x**) and from 139.42 to 24.84 microseconds
for Q6_K (**5.61x**), both `p=0.000`, n=10. The public allocation-inclusive
`Dequantize` API improves **2.01x** and **2.87x**, respectively, without changing
its three allocations.

Against pinned llama.cpp b10450 built CPU-only from commit
`ece963f41b0b02d7a0d61436ae365762c073a4c8`, the matched zero-allocation raw-byte
throughput is 4.654 versus 1.655 GB/s for Q4_K (**2.81x**) and 8.658 versus
6.800 GB/s for Q6_K (**1.27x**). This is a leaf-kernel leadership cell, not an
end-to-end inference claim.

## Claim cell and pins

- Hardware: Apple M2 Pro, arm64; macOS 26.5.1 (25F80).
- Go: 1.26.6; control `9c2477b5ad81f275e3c3a75ce83bfa0403008d76`;
  kernel commit `d9d75ae0175e93a1fdbe03224af96a01042ad069`.
- Shape: 262,144 output f32 values; Q4_K and Q6_K; one reusable output; warm
  process; 300 ms adaptive benchmark windows; ten samples per arm.
- llama.cpp: b10450 / commit `ece963f41b0b02d7a0d61436ae365762c073a4c8`,
  Release AppleClang 21 native-ARM build with Metal, Accelerate, and BLAS off;
  1,000 iterations per observation and ten observations.
- TinyLlama model: 668,788,096 bytes, SHA-256
  `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`;
  135 Q4_K tensors (514,031,616 encoded bytes), 21 Q6_K tensors (152,678,400
  bytes), and 45 F32 tensors (368,640 bytes).

The direct Go arms use deterministic raw blocks with finite scales. Q4_K and
Q6_K decoding is branch-free with respect to encoded values, so the synthetic
payload does not select a faster code path. `public-control.txt`,
`public-neon.txt`, `path-scalar.txt`, and `path-neon.txt` contain all Go samples;
`llamacpp.txt` contains every incumbent observation and the executable hash.

## Real-model boundary

`TestReadParsedKPathAB` parses the model once, then performs ten alternating
scalar/NEON rounds inside one process. Each arm uses fresh destination pages and
the production bounded work-stealing schedule; unchanged tensor formats still
use the production decoder. Its medians are 112.632 ms scalar and 104.846 ms
NEON (**1.074x**). A separate ten-process full-`ReadFile` campaign is directionally
similar at 113.115 versus 104.064 ms (**1.087x**), but `benchstat` reports
`p=0.393` because of 11–18% spread. These model-load numbers are supporting
attribution only, not a statistically significant standalone speedup claim.

## Correctness and portability

The assembly deliberately preserves scalar operation order: Q4_K computes
`step*q - offset` without fusion, while Q6_K computes `(d*scale)*q`. Randomized
tests compare every float32 bit for 19 blocks per type and cover positive and
negative zero, half subnormals, normals, and maximum finite half scales. NaN
payload behavior is outside the GGUF weight-scale contract. Non-ARM64 builds
dispatch directly to the unchanged scalar implementations.

Validated gates:

```sh
go test ./format/gguf -run '^TestDequantQ(4|6)KNeonBitExact$' -count=10
go test ./format/gguf -count=1
go test -race ./format/gguf -count=1
CGO_ENABLED=0 go test ./format/gguf -count=1
CGO_ENABLED=0 go vet ./format/gguf
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test -c ./format/gguf
make preflight
```

Benchmark commands:

```sh
go test ./format/gguf -run '^$' \
  -bench '^BenchmarkDequantQ(4|6)_K$' -count=10 -benchtime=300ms -benchmem
go test ./format/gguf -run '^$' \
  -bench '^BenchmarkDequantQ(4|6)KIntoPaths$' -count=10 -benchtime=300ms -benchmem
TINYLLAMA_GGUF=/absolute/path/to/tinyllama-1.1b-q4km.gguf \
  go test ./format/gguf -run '^TestReadParsedKPathAB$' -count=1 -v
```

## Rejected two-pass candidate

A faster warm-leaf prototype stored precomputed coefficients in the tail of the
new output before assembly overwrote the full tensor. It reached about 23.9
microseconds for Q6_K and 31.1 microseconds for Q4_K, but fresh model loads
regressed as far as 232.0 versus 109.5 ms because the prepass demand-zeroed and
wrote across roughly 4.4 GB of output pages before the real decode. The shipped
block-local design avoids that extra destination pass. The general benchmark
and static-analysis trap is tracked in
https://github.com/jxsl13/perfscan/issues/762.
