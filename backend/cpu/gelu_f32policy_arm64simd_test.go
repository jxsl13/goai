//go:build arm64 && goexperiment.simd

package cpu_test

// Experiment build (arm64): the cpu OpGELU F32 kernel runs the f32-native
// NEON pipeline (AS-7.1.26 erf on the vexp exp primitive, vexp.go) → within
// |err| ≤ 1e-6 + 2e-4·|ref| of the exact math.Erf reference
// (TestGeluF32Accuracy), not bit-exact. F64 GELU and every other build stay
// bit-exact.
const geluF32Tolerant = true
