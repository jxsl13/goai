# M2 ARM64 IQ4_NL fused lookup dot

## Scope

This tranche adds portable `QMatMul` semantics for `IQ4_NL` and compares its
portable scalar row dot with the ARM64 row-level fused nonlinear-lookup dot in
the same candidate binary. It does not claim llama.cpp or cross-library
leadership; merged main rejected `IQ4_NL` in `QMatMul`, so there is no
semantically matched pre-change implementation to time.

The unchanged `Q5_K` M1 decode benchmark is the negative control. The existing
`Q4_K` M16 benchmark separately measures the fixed prefill scratch-allocation
coalescing needed by the new portable path.

## Environment

- Apple M2 Pro, macOS 26.5.1 (25F80), Darwin arm64
- Go 1.26.6, `darwin/arm64`
- Baseline source: `b2df5338db57b9bb30acc713546e868db32fc028`
- Candidate: this evidence directory's containing commit
- Benchmark binary was compiled once per source state before sampling

## Method

- A pilot/warmup run was excluded.
- IQ4_NL scalar/NEON: five scalar-first and five NEON-first samples from one
  candidate binary, `-test.benchtime 500ms`.
- Controls: ten baseline/candidate pairs, alternating which binary ran first,
  `-test.benchtime 500ms`.
- `benchstat` reports all ten retained samples per arm; no sample was removed.
- Raw outputs and normalized scalar/NEON inputs are committed beside this file.

## Results

| Cell | Control | Candidate | Speedup | Significance |
| --- | ---: | ---: | ---: | ---: |
| IQ4_NL leaf, K=4096 | 4.813 us | 0.7331 us | 6.57x | p=0.000, n=10 |
| IQ4_NL QMatMul, M1/N64/K1024 | 75.81 us | 12.04 us | 6.30x | p=0.000, n=10 |
| IQ4_NL QMatMul, M1/N4096/K1024 | 705.7 us | 175.5 us | 4.02x | p=0.000, n=10 |
| Q5_K M1/N4096/K1024 negative control | 163.1 us | 163.1 us | flat | p=0.436, n=10 |
| Q4_K M16 scratch control | 76.05 us | 74.31 us | 1.02x | p=0.000, n=10 |

The IQ4_NL leaf remains at 0 allocations. Both IQ4_NL QMatMul cells retain
their allocation counts. Coalescing the three fixed prefill scratch lanes into
one backing allocation reduces Q4_K M16 from 62 to 40 allocations per call
(-35.48%) while reducing time by 2.29%; bytes per operation stay flat.

## Correctness gates

- Exact packed-block golden: `-3696`.
- Maximum scalar-relative error over 100 arbitrary raw rows at K=32, 64, 256,
  and 4096: `4.548988124783841e-16`.
- Cancellation-heavy paired-half input: at most `1e-4` scalar-relative error.
- Inputs are immutable and the ARM64 leaf allocates zero bytes.
- Portable F32/F64 and M1/M3 `QMatMul` results match the fully dequantized
  reference within `1e-4`; caller-owned decode is bit-exact to tensor decode.
- M>1 allocation count is invariant between N=1 and N=31.

## Reproduction

```sh
go test -c -o /tmp/goai-iq4nl-gguf.test ./format/gguf
/tmp/goai-iq4nl-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ4NLPaths|BenchmarkQMatMulIQ4NLPaths)$' -test.benchmem -test.count 5 -test.benchtime 500ms
GOAI_GGUF_IQ4NL_NEON_FIRST=1 /tmp/goai-iq4nl-gguf.test -test.run '^$' -test.bench '^(BenchmarkDotIQ4NLPaths|BenchmarkQMatMulIQ4NLPaths)$' -test.benchmem -test.count 5 -test.benchtime 500ms
```
