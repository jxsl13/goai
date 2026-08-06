//go:build amd64 && goexperiment.simd

package cpu_test

// amd64 SIMD perf build: the cpu OpGELU F32 forward and OpGELUBackward run the
// f32-native 8-wide AVX2 pipeline (AS-7.1.26 erf on the vexp exp primitive,
// vexp_amd64.go) → within |err| ≤ 1e-6 + 2e-4·|ref| of the exact f64 reference
// (TestGeluF32Accuracy / TestGeluGradF32Accuracy), not bit-exact. Sigmoid/SiLU/
// exp/tanh/log stay on the scalar f64 paths here (bit-exact; the amd64 AVX
// campaign for those is separate), and a bit-exact result trivially passes the
// tolerant check, so this const is safe to share.
const geluF32Tolerant = true

// amd64 SIMD build: cpu OpGELU F64 forward runs the vectorized Cephes erf (vgeluF64,
// erfF64x4 on expF64x4) — ~1 ulp, not bit-exact vs ref's scalar math.Erf. Matches
// vexpF64Fast (true only here); every other build keeps scalar math.Erf (bit-exact).
const geluF64Tolerant = true
