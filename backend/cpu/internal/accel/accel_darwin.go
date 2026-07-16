//go:build darwin && cgo && arm64 && goexperiment.simd

package accel

// The actual Accelerate binding. ACCELERATE_NEW_LAPACK opts into the modern
// (ILP64-clean) headers per Apple's guidance; row-major + NoTrans means no
// transpose copies. cgo is opt-in per §C2, precedented by backend/metal's
// `darwin && cgo` gating.

/*
#cgo CFLAGS: -DACCELERATE_NEW_LAPACK=1
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
*/
import "C"

func init() { SGEMM, SGEMV = sgemm, sgemv }

func sgemm(a, b, c []float32, m, k, n int) {
	C.cblas_sgemm(C.CblasRowMajor, C.CblasNoTrans, C.CblasNoTrans,
		C.int(m), C.int(n), C.int(k), 1.0,
		(*C.float)(&a[0]), C.int(k),
		(*C.float)(&b[0]), C.int(n), 0.0,
		(*C.float)(&c[0]), C.int(n))
}

// sgemv: y[n] = x[k]·w[k,n]. w is row-major [k,n] (lda=n), so the op is
// y = wᵀ·x → CblasTrans with M=k rows, N=n cols; BLAS then takes x of length
// M=k and writes y of length N=n. Orientation is pinned by the parity test in
// backend/cpu (gemm_gemv_test.go) against a scalar dot-product reference.
func sgemv(w, x, y []float32, k, n int) {
	C.cblas_sgemv(C.CblasRowMajor, C.CblasTrans,
		C.int(k), C.int(n), 1.0,
		(*C.float)(&w[0]), C.int(n),
		(*C.float)(&x[0]), 1, 0.0,
		(*C.float)(&y[0]), 1)
}
