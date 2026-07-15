//go:build !(goexperiment.simd && (amd64 || arm64))

package cpu

// Default build: the f32 MHA and Conv2D kernels keep their scalar
// f64-accumulating paths (bit-exact vs ref where promised — §V3/§V10/§V11).
// See f32native_simd.go for the perf-build override.
const f32NativeKernels = false
