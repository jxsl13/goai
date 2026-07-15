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

func init() { SGEMM = sgemm }

func sgemm(a, b, c []float32, m, k, n int) {
	C.cblas_sgemm(C.CblasRowMajor, C.CblasNoTrans, C.CblasNoTrans,
		C.int(m), C.int(n), C.int(k), 1.0,
		(*C.float)(&a[0]), C.int(k),
		(*C.float)(&b[0]), C.int(n), 0.0,
		(*C.float)(&c[0]), C.int(n))
}
