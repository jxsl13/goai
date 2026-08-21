//go:build !arm64 || !goexperiment.simd

package cpu

// Keep the established scheduler grain on every non-ARM64-SIMD build. In
// particular, 30 is exactly divisible by the amd64 6-row AVX2 GEMM tile.
const mhaFwdBandRows = 30
