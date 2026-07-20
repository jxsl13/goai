//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"
import "unsafe"

// GemmW8A16 computes C[m,n] (f16) = A[m,k] (f16) · dequant(W[k,n] int8, per-column f32 scale) via a
// dequant-in-tile tensor-core mma GEMM. The lever for low-batch serving: decode GEMMs are
// weight-bandwidth-bound and int8 weights are half f16's bytes. Raw pointers: a16/c16 are u16 (f16),
// w8 is int8 [k,n] row-major, scale is f32 [n]. Requires m%16==0, n%8==0, k%16==0.
func GemmW8A16(a16, w8, scale, c16 unsafe.Pointer, m, k, n int) int {
	return int(C.cu_gemm_w8a16(a16, w8, scale, c16, C.int(m), C.int(k), C.int(n)))
}

// UploadI8 copies an int8 slice to a device buffer (caller frees via FreeDev).
func UploadI8(vals []int8) unsafe.Pointer {
	if len(vals) == 0 {
		return nil
	}
	return unsafe.Pointer(C.cu_upload_i8((*C.schar)(unsafe.Pointer(&vals[0])), C.int(len(vals))))
}
