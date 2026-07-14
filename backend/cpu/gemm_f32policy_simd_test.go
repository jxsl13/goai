//go:build amd64 && goexperiment.simd

package cpu_test

// Experiment build: cpu F32 matmul uses the f32-NATIVE 8-wide SIMD kernel
// (ADR-0021), which accumulates in f32 → within a K-scaled tolerance of the
// f64 reference, not bit-exact. F64 matmul stays bit-exact in both builds.
const gemmF32Tolerant = true
