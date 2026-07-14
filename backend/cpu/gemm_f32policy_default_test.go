//go:build !(amd64 && goexperiment.simd)

package cpu_test

// Default build: cpu F32 matmul accumulates in f64 (§V10) → bit-exact vs ref.
const gemmF32Tolerant = false
