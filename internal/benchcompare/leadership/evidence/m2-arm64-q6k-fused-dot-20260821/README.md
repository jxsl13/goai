# M2 ARM64 Q6_K fused decode-dot evidence — 2026-08-21

GoAI's ARM64 single-token Q6_K path now fuses six-bit unpack, signed
sub-block scaling, activation multiplication, and reduction in one NEON pass.
The portable scalar path, every non-ARM64 build, every M>1 path, and all other
quant formats are unchanged.

## Result

| Cell | scalar control | ARM64 NEON | change | speedup |
|---|---:|---:|---:|---:|
| Q6_K row dot, K=4096 | 4,815.5 ns | 452.2 ns | -90.61% | 10.65x |
| QMatMul M1, N=64, K=1024 | 76.437 us | 7.451 us | -90.25% | 10.26x |
| QMatMul M1, N=4096, K=1024 | 709.6 us | 117.2 us | -83.48% | 6.05x |
| Quantized Mamba2 recurrent decode, Q6_K | 364.3 us | 101.6 us | -72.12% | 3.59x |
| Quantized Mamba2 recurrent decode, Q4_K | 118.9 us | 118.9 us | statistically flat | 1.00x |

Every Q6_K comparison has p=0.000 and n=10. The Q4_K negative control has
p=0.853. Allocation counts are unchanged: 0 at the leaf, 4 and 29 in the
QMatMul cells, and 93 in recurrent decode.

The QMatMul benchmark's B/s counts logical f32 work (`M*N*K*4`), not encoded
Q6_K bytes. It is retained because it is the benchmark's established
throughput convention; it is not a physical memory-bandwidth claim.

## Protocol

- Hardware: Apple M2 Pro, darwin/arm64.
- OS: macOS 26.5.1 (25F80).
- Toolchain: Go 1.26.6; benchstat from golang.org/x/perf.
- Base: `89795e4a40d053148f5ce347aa157bd67b0188e2` (merged main after PR #1132).
- Branch: `codex/m2-arm64-q6k-fused-dot`.
- The leaf and QMatMul controls and candidates run in one binary. The committed
  campaign sets `GOAI_GGUF_Q6K_NEON_FIRST=1`, reversing the scalar-first pilot.
- Mamba2 uses separately precompiled base and candidate binaries, candidate first.
- Every binary ran eleven 250 ms samples. The first sample of every benchmark
  was discarded before benchstat, leaving n=10. No other sample was removed.
- Q4_K ran beside Q6_K as an untouched end-to-end negative control.
- Tests were selected by invoking already-compiled test binaries with
  `-test.run`; `go test -run` was not used.

## Numerical gate

The assembly preserves scalar weight construction order `(d*scale)*(q-32)`,
but uses four f32 vector accumulators and widens each 256-weight super-block
subtotal to f64. It therefore does not claim the scalar path's per-element
f64 summation order.

Three gates cover the contract:

1. A known-answer block whose 256 values all decode to one returns exactly 256.
2. Direct comparison at K=256, 512, and 4096 has maximum observed relative
   error 4.41e-7.
3. One hundred arbitrary raw-block trials, including negative int8 scales,
   have maximum scalar-relative error 1.20e-5 and verify input immutability.

Both error results are below the 1e-4 contract. The full fused-vs-general
QMatMul shape matrix also passes its architecture tolerance.

## Prior rejection and scope

The rejected Q6_K Metal packed-load experiment is not retried here. It changed
GPU load width and measured only 0.891x to 1.053x. This CPU change instead
removes the scalar unpack and per-element f64 accumulation that bypassed the
already-fast eager NEON dequantizer. The eager dequantizer itself measured
about 5.9x over scalar in the baseline context.

This is an internal M2/ARM64 leadership gain, not yet a cross-library
leadership claim. A matched llama.cpp CPU comparison requires equal activation
semantics, threads, quant bytes, shapes, and measurement boundaries. The
historical Q8_0 whole-model gap is not relabeled as closed by this Q6_K result.
