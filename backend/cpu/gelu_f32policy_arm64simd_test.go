//go:build arm64 && goexperiment.simd

package cpu_test

// Experiment build (arm64): the cpu OpGELU / OpSigmoid / OpSiLU F32 kernels
// run the f32-native NEON pipelines (AS-7.1.26 erf / stable-split sigmoid on
// the vexp exp primitive, vexp.go) → within |err| ≤ 1e-6 + 2e-4·|ref| of the
// exact f64 reference (TestGeluF32Accuracy / TestSigmoidF32Accuracy /
// TestSiluF32Accuracy), not bit-exact. The separately gated F64 GELU intrinsic
// has its own tighter policy below; every default build remains bit-exact.
const geluF32Tolerant = true

// arm64 SIMD: the separately gated Float64x2 GELU intrinsic is held to
// 1e-12*max(1,abs(reference)); the default build remains exact.
const geluF64Tolerant = true
