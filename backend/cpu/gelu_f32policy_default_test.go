//go:build !((arm64 || amd64) && goexperiment.simd)

package cpu_test

// Default (no-simd) build: cpu F32 GELU/sigmoid/SiLU keep the scalar f64 paths
// → bit-exact vs ref (and silu/gelu backward fall back to ref itself). The
// arm64 and amd64 SIMD perf builds have their own policy files (tolerant).
const geluF32Tolerant = false
