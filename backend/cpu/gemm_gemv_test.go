//go:build darwin && cgo && arm64 && goexperiment.simd

package cpu

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cpu/internal/accel"
)

// TestGemvAccelMatchesReference pins the cblas_sgemv ORIENTATION (RowMajor +
// CblasTrans over w[k,n], lda=n) against the f64-accumulating scalar
// reference: a transposed call would compute w·x instead of wᵀ·x and produce
// silently wrong decode logits, so this must be green before any benchmark of
// the path is trusted. Tolerance is the ADR-0021 f32-native contract (rtol
// 2e-3), same as every other f32 fast path. Non-square k≠n shapes (FFN
// up/down, vocab head) are the real orientation witnesses — a k↔n swap cannot
// even produce the right output length there.
func TestGemvAccelMatchesReference(t *testing.T) {
	if accel.SGEMV == nil {
		t.Skip("accel.SGEMV unavailable")
	}
	rng := rand.New(rand.NewSource(43))
	for _, s := range []struct{ k, n int }{
		{2048, 2048},  // decode hidden×hidden
		{2048, 8192},  // FFN up
		{8192, 2048},  // FFN down
		{2048, 32000}, // vocab head
		{129, 67},     // odd, non-aligned
		{1, 5}, {7, 1},
	} {
		w := make([]float32, s.k*s.n)
		x := make([]float32, s.k)
		y := make([]float32, s.n)
		for i := range w {
			w[i] = rng.Float32() - 0.5
		}
		for i := range x {
			x[i] = rng.Float32() - 0.5
		}
		accel.SGEMV(w, x, y, s.k, s.n)
		checkTolerance(t, y, gemmF32Ref(x, w, 1, s.k, s.n), s.k)
	}
}

// TestGemmF32SmallMDispatchGemv checks the full gemmF32 entry point around
// the m<3 sgemv cutoff: m=1..2 route through the per-row sgemv branch,
// m=3..4 must stay on the GEMM paths (sgemm / small-m fallback).
func TestGemmF32SmallMDispatchGemv(t *testing.T) {
	rng := rand.New(rand.NewSource(44))
	for _, s := range []struct{ m, k, n int }{
		{1, 2048, 2048}, {2, 512, 1024}, {3, 300, 700}, {4, 128, 256},
	} {
		A := make([]float32, s.m*s.k)
		B := make([]float32, s.k*s.n)
		C := make([]float32, s.m*s.n)
		for i := range A {
			A[i] = rng.Float32() - 0.5
		}
		for i := range B {
			B[i] = rng.Float32() - 0.5
		}
		gemmF32(A, B, C, s.m, s.k, s.n)
		checkTolerance(t, C, gemmF32Ref(A, B, s.m, s.k, s.n), s.k)
	}
}
