# M2 ARM64 Q4_K fused decode-dot evidence — 2026-08-21

GoAI's ARM64 single-token Q4_K path now fuses nibble unpack, affine dequantization,
activation multiplication, and reduction in one NEON pass. The portable scalar path,
all non-ARM64 builds, and every M>1 path are unchanged.

## Result

| Cell | scalar control | ARM64 NEON | change | speedup |
|---|---:|---:|---:|---:|
| Q4_K row dot, K=4096 | 4,792.0 ns | 599.2 ns | -87.50% | 8.00x |
| QMatMul M1, N=64, K=1024 | 75.524 us | 9.726 us | -87.12% | 7.77x |
| QMatMul M1, N=4096, K=1024 | 689.5 us | 145.9 us | -78.84% | 4.73x |
| Quantized Mamba2 recurrent decode, Q4_K | 349.2 us | 118.8 us | -65.96% | 2.94x |
| Quantized Mamba2 recurrent decode, Q6_K | 363.9 us | 364.1 us | statistically flat | 1.00x |

Every comparison has p=0.000 and n=10 except the Q6_K negative control
(p=0.853). Allocation counts are unchanged: 0 at the leaf, 4 and 29 in the
QMatMul cells, and 93 in recurrent decode.

The QMatMul benchmark's reported B/s counts logical f32 work
(`M*N*K*4`), not encoded Q4_K bytes. It is retained only because it is the
benchmark's existing throughput convention; it is not a physical memory-bandwidth
claim.

## Protocol

- Hardware: Apple M2 Pro, darwin/arm64.
- OS: macOS 26.5.1 (25F80).
- Toolchain: Go 1.26.6; benchstat from golang.org/x/perf.
- Base: `9a21d73b98be9b4cd21a75357a435a2e24cdbeb8` (merged main after PR #1131).
- Control and candidate were compiled into separate test binaries.
- The committed campaign ran the control first and candidate second, reversing
  the earlier pilot's candidate-first order.
- Each binary ran eleven 250 ms samples in a fresh process. The first sample of
  every benchmark was discarded before benchstat, leaving n=10.
- Q6_K ran beside Q4_K as an untouched negative control.
- Tests were selected by invoking the already-compiled test binaries with
  `-test.run`; `go test -run` was not used.

## Numerical gate

The assembly preserves the scalar weight construction order
`step*q-offset`, but uses four f32 vector accumulators and widens each
256-weight super-block subtotal to f64. It therefore does not claim the scalar
path's per-element f64 summation order.

Three gates cover the intended contract:

1. A known-answer block with 128 unit low nibbles returns exactly 128.
2. Direct comparison at K=256, 512, and 4096 has maximum observed relative
   error 7.10e-7, below the existing 1e-4 Q4_K SIMD gate.
3. `TestQMatMulFusedDecodeMatchesGeneralPathExactly` passes its architecture
   tolerance across the full shape matrix.

## Rejected control

Reusing the existing eager NEON unpacker and then running the scalar f64 dot
was measured before writing assembly. At N=4096/K=1024 it regressed Q4_K from
about 701 us to 781 us and increased memory from about 17 KiB/29 allocations
to 66 KiB/39 allocations. Materialization is therefore not part of this
implementation.

## Scope and claim

This is an internal M2/ARM64 leadership gain, not yet a cross-library
leadership claim. A matched llama.cpp CPU comparison requires equal activation
semantics, thread count, quant layout, shapes, and measurement boundaries.
The historical 8.8x Q8_0 whole-model gap is also not relabeled as closed by
this Q4_K-only result.
