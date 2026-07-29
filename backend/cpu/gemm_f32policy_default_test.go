//go:build !(amd64 && goexperiment.simd) && !(arm64 && goexperiment.simd)

package cpu_test

// Default build: cpu F32 matmul accumulates in f64 (§V10) → bit-exact vs ref.
// F64 matmul uses the scalar mul+add kernel → bit-exact vs ref.
const gemmF32Tolerant = false
const gemmF64Tolerant = false
