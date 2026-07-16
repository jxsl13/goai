//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import "unsafe"

// Internal wrappers for the Tw61 f16-accumulate prefill-GEMM probe (used by the internal benchmark).

func i8Upload(b []int8) unsafe.Pointer {
	if len(b) == 0 {
		return nil
	}
	return C.cu_upload_i8((*C.schar)(unsafe.Pointer(&b[0])), C.int(len(b)))
}

func f32Upload(f []float32) unsafe.Pointer {
	if len(f) == 0 {
		return nil
	}
	return C.cu_upload_f32((*C.float)(&f[0]), C.int(len(f)))
}

func f16Upload(f []float32) unsafe.Pointer {
	if len(f) == 0 {
		return nil
	}
	return C.cu_upload_f16((*C.float)(&f[0]), C.long(len(f)))
}

func devAllocBytes(n int) unsafe.Pointer { return C.cu_alloc_i8(C.int(n)) }
func devFree(p unsafe.Pointer)           { C.cu_free_f32(p) }
func devSync()                           { C.cu_graph_sync() }

func matmulF16w(a, w, c unsafe.Pointer, m, k, n int) int {
	return int(C.cu_matmul_f16w(a, w, c, C.int(m), C.int(k), C.int(n), C.float(0)))
}

func matmulF16acc16(a, w, c unsafe.Pointer, m, k, n int) int {
	return int(C.cu_matmul_f16acc16(a, w, c, C.int(m), C.int(k), C.int(n)))
}

func downloadF32(src unsafe.Pointer, dst []float32) {
	if len(dst) > 0 {
		C.cu_download_f32(src, (*C.float)(&dst[0]), C.int(len(dst)))
	}
}

func downloadU16(src unsafe.Pointer, dst []uint16) {
	if len(dst) > 0 {
		C.cu_download_u16(src, (*C.ushort)(&dst[0]), C.int(len(dst)))
	}
}
