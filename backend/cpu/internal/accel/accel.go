// Package accel holds the cgo binding to Apple Accelerate's BLAS (ADR-0027
// Path A). It is a separate package because the Go toolchain forbids cgo and
// Go assembly files in the same package, and backend/cpu carries the Plan9
// NEON/AMX kernels. SGEMM is non-nil only on `darwin && cgo && arm64 &&
// goexperiment.simd` builds (accel_darwin.go); everywhere else — notably
// CGO_ENABLED=0 — it stays nil and backend/cpu keeps its pure-Go paths.
package accel

// SGEMM, when non-nil, computes c[m,n] = a[m,k]·b[k,n] in f32 (row-major,
// alpha=1, beta=0 — store semantics) via Accelerate cblas_sgemm, which runs on
// the Apple AMX coprocessor and self-threads. Call it once per matmul, never
// from inside a worker pool.
var SGEMM func(a, b, c []float32, m, k, n int)

// SGEMV, when non-nil, computes the decode GEMV y[n] = x[k]·w[k,n] in f32
// (w row-major [k,n], alpha=1, beta=0 — store semantics) via Accelerate
// cblas_sgemv with CblasTrans, i.e. y = wᵀ·x. GEMV is bandwidth-bound (w is
// read exactly once); the measured M2 Pro win over cblas_sgemm is at m=2
// (per-row calls, 1.40×) — at m=1 the two tie because sgemm's m=1 path is
// the same GEMV. Same calling discipline as SGEMM: once per op, never from
// inside a worker pool.
var SGEMV func(w, x, y []float32, k, n int)
