// Package simd is the internal SIMD wrapper (§B3).
//
// It isolates the experimental Go 1.26 simd/archsimd package (AMD64, via
// GOEXPERIMENT=simd) and Plan9-NEON kernels (ARM64) behind a thin API with a
// pure-scalar fallback, so the rest of the tree never depends on experimental
// APIs directly. Selection is by build tag + runtime feature detection (§I4).
// Populated by §T11; empty scaffold for now.
package simd
