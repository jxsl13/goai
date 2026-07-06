// Package metal is the optional Metal/MPS GPU backend (layer L1b, §T20).
//
// It is strictly opt-in: build with `-tags metal` on darwin with cgo. The
// default CGO_ENABLED=0 build contains no cgo (§V7, §I5) — without the tag this
// package is just this doc file. Kernels: MatMul f32 via
// MPSMatrixMultiplication; every other op falls back to the Pure-Go backends
// through the dispatch layer (§I4). Merged through the cgo gate (§C2/§C3):
// benchmarked against the ceiling-optimized Pure-Go GEMM, numbers in
// docs/benchmarking.md.
package metal
