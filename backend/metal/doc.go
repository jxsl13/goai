// Package metal is the Metal/MPS GPU backend (layer L1b, §T20).
//
// # For the AI professional
//
// It needs NO build tag (§T47): the file is guarded only by `//go:build darwin &&
// cgo`, and Metal/MetalPerformanceShaders/Foundation are macOS system frameworks
// always present, so a plain native `go build` on macOS (cgo is the default)
// compiles and links it. A `CGO_ENABLED=0` build contains no cgo (§V7, §I5) —
// then this package is just this doc file. The backend registers itself only when
// an MPS-capable GPU is present, so backend.Default() auto-selects it (§V18).
// Kernels: MatMul f32 via MPSMatrixMultiplication; every other op falls back to
// the Pure-Go backends through the dispatch layer (§I4). Merged through the cgo
// gate (§C2/§C3): benchmarked against the ceiling-optimized Pure-Go GEMM.
//
// # For the newcomer
//
// This makes your math run on the Mac's GPU. You do not have to do anything
// special: on a Mac, just importing the library (github.com/jxsl13/goai) is
// enough — GoAI finds the GPU and uses it. If there is no usable GPU, or you
// build with cgo turned off, it quietly falls back to the CPU.
package metal
