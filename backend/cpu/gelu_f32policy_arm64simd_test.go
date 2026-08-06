//go:build arm64 && goexperiment.simd

package cpu_test

// Experiment build (arm64): the cpu OpGELU / OpSigmoid / OpSiLU F32 kernels
// run the f32-native NEON pipelines (AS-7.1.26 erf / stable-split sigmoid on
// the vexp exp primitive, vexp.go) → within |err| ≤ 1e-6 + 2e-4·|ref| of the
// exact f64 reference (TestGeluF32Accuracy / TestSigmoidF32Accuracy /
// TestSiluF32Accuracy), not bit-exact. F64 and every other build stay
// bit-exact.
const geluF32Tolerant = true

// arm64: F64 GELU stays on scalar math.Erf (vexpF64Fast=false; vgeluF64 SIMD is amd64-only).
const geluF64Tolerant = false
