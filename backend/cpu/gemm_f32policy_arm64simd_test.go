//go:build arm64 && goexperiment.simd

package cpu_test

// Experiment build (arm64): cpu F32 matmul uses the f32-NATIVE NEON tile
// kernel (ADR-0026, the arm64 twin of ADR-0021), which accumulates in f32 →
// within the K-scaled tolerance of the f64 reference, not bit-exact. F64
// matmul stays bit-exact on arm64 (the FMA f64 kernel is amd64-only).
const gemmF32Tolerant = true
const gemmF64Tolerant = false
