//go:build !(goexperiment.simd && (amd64 || arm64))

package cpu

// This file supplies the gemmF32Rows / gemmF32RowsCols worker-band entry points
// on every build that is NOT a SIMD amd64/arm64 perf build — i.e. the default
// (non-goexperiment.simd) build on any arch, and goexperiment.simd on arches
// without a NEON/AVX band. The gemm-routed MHA/Conv kernels gate their actual
// calls behind the f32NativeKernels const (false here), so these definitions are
// dead code at run time; they exist only so the references type-check (Go
// requires an identifier to be defined even inside a compile-time-false `if`).
// They forward to the portable f32-native scalar bodies in gemm_rows.go.

func gemmF32Rows(A, B, C []float32, loRow, hiRow, k, n int) {
	gemmF32RowsPortable(A, B, C, loRow, hiRow, k, n)
}

func gemmF32RowsCols(A, B, C []float32, loRow, hiRow, k, n, jLo, jHi int) {
	gemmF32RowsColsPortable(A, B, C, loRow, hiRow, k, n, jLo, jHi)
}
