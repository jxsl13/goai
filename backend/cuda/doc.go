// Package cuda is the optional NVIDIA CUDA/cuBLAS GPU backend (layer L1b, §T42).
//
// It is strictly opt-in: build with `-tags cuda` on linux or windows with cgo
// and a CUDA toolkit (nvcc/cublas) installed. The default CGO_ENABLED=0 build
// contains no cgo (§V7, §I5) — without the tag this package is just this doc
// file, so `go build ./...` and every pure-Go platform stay green.
//
// Kernels: MatMul f32 via cublasSgemm; every other op falls back to the Pure-Go
// backends through the dispatch layer (§I4). Training GEMMs reach the GPU the
// same way the Metal backend does — autograd.NewTapeOn(cudaBackend) dispatches
// the forward/backward matmuls here, so the backend serves both inference and
// training (user priority: accel for training AND inference).
//
// Not host-verifiable on this arm64/macOS host (no CUDA) — the backend and its
// §V3 cross-reference + §C3 gate benchmarks are CI-gated on Linux/Windows CUDA
// runners (§B35, mirrors the archsimd CI gate §B13). Merged through the cgo gate
// (§C2/§C3): benchmarked against the ceiling-optimized Pure-Go GEMM in CI.
package cuda
