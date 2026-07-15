//go:build goexperiment.simd && (amd64 || arm64)

package cpu

// f32NativeKernels routes the f32 MHA and Conv2D kernels through the gemmF32
// entry point on the perf builds, where gemmF32 is the f32-NATIVE SIMD kernel
// (AVX on amd64 per ADR-0021, NEON on arm64 per ADR-0026). The routed results
// carry the same K-scaled f32 accumulation error as the matmul op itself and
// are gated by the same tolerance parity tests (gemmF32Tolerant builds). The
// default build keeps the scalar f64-accumulating kernels bit-for-bit.
const f32NativeKernels = true
