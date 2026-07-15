//go:build !(arm64 && goexperiment.simd)

package cpu_test

// Default build (and the amd64 perf build): cpu F32 GELU keeps the scalar
// f64 math.Erf path → bit-exact vs ref.
const geluF32Tolerant = false
