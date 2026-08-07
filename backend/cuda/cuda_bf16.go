//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// MatMulBf16 computes out[M,N]f32 = a[M,K]·b[K,N] using bf16 tensor cores with f32 accumulation: it
// rounds a and b to bf16 into scratch, then runs the bf16 GEMM. On Ampere the bf16 tensor cores run at
// ~2× the TF32 rate; bf16 keeps f32's exponent range so no loss scaling is needed. This is the
// mixed-precision training GEMM (bf16 compute, f32 accumulate, f32 output/master weights).
func MatMulBf16(a, b, out *DeviceF32) error {
	m, k := a.rows, a.cols
	if b.rows != k {
		return fmt.Errorf("cuda: MatMulBf16 inner mismatch %d != %d", b.rows, k)
	}
	n := b.cols
	if out.rows != m || out.cols != n {
		return fmt.Errorf("cuda: MatMulBf16 out[%d,%d], want [%d,%d]", out.rows, out.cols, m, n)
	}
	abf := unsafe.Pointer(C.cu_alloc_u16(C.int(m * k)))
	bbf := unsafe.Pointer(C.cu_alloc_u16(C.int(k * n)))
	if abf == nil || bbf == nil {
		if abf != nil {
			C.cu_free_f32(abf)
		}
		if bbf != nil {
			C.cu_free_f32(bbf)
		}
		return fmt.Errorf("cuda: MatMulBf16 scratch alloc failed")
	}
	defer C.cu_free_f32(abf)
	defer C.cu_free_f32(bbf)
	C.cu_cvt_f32_to_bf16(abf, a.ptr, C.long(m*k))
	C.cu_cvt_f32_to_bf16(bbf, b.ptr, C.long(k*n))
	if rc := C.cu_matmul_bf16_ddd(abf, bbf, out.ptr, C.int(m), C.int(k), C.int(n)); rc != 0 {
		return fmt.Errorf("cuda: MatMulBf16 rc=%d", int(rc))
	}
	return nil
}

// MatMulGradWBf16 computes the linear-layer weight gradient dW[K,N]f32 = xᵀ·dY on bf16 tensor cores
// (rounds x and dY to bf16, f32 accumulation) — the mixed-precision training weight-gradient GEMM.
func MatMulGradWBf16(x, dY, dW *DeviceF32) error {
	m, k := x.rows, x.cols
	if dY.rows != m {
		return fmt.Errorf("cuda: MatMulGradWBf16 x[%d,%d]·dY[%d,%d] row mismatch", x.rows, x.cols, dY.rows, dY.cols)
	}
	n := dY.cols
	if dW.rows != k || dW.cols != n {
		return fmt.Errorf("cuda: MatMulGradWBf16 dW[%d,%d], want [%d,%d]", dW.rows, dW.cols, k, n)
	}
	xbf := unsafe.Pointer(C.cu_alloc_u16(C.int(m * k)))
	ybf := unsafe.Pointer(C.cu_alloc_u16(C.int(m * n)))
	if xbf == nil || ybf == nil {
		if xbf != nil {
			C.cu_free_f32(xbf)
		}
		if ybf != nil {
			C.cu_free_f32(ybf)
		}
		return fmt.Errorf("cuda: MatMulGradWBf16 scratch alloc failed")
	}
	defer C.cu_free_f32(xbf)
	defer C.cu_free_f32(ybf)
	C.cu_cvt_f32_to_bf16(xbf, x.ptr, C.long(m*k))
	C.cu_cvt_f32_to_bf16(ybf, dY.ptr, C.long(m*n))
	if rc := C.cu_matmul_bf16_ddd_at(xbf, ybf, dW.ptr, C.int(m), C.int(k), C.int(n)); rc != 0 {
		return fmt.Errorf("cuda: MatMulGradWBf16 rc=%d", int(rc))
	}
	return nil
}
