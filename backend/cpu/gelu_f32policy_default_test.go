//go:build !(arm64 && goexperiment.simd)

package cpu_test

// Default build (and the amd64 perf build): cpu F32 GELU/sigmoid/SiLU keep
// the scalar f64 paths → bit-exact vs ref (and silu/gelu backward fall back
// to ref itself).
const geluF32Tolerant = false
