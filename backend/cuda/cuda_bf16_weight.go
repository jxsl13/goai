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

// DeviceBf16 is a device-resident bf16 (u16) buffer. Its purpose in training is to CACHE a weight in
// bf16 across a step: the f32 master weight is converted once (FromF32) after the optimizer update, and
// the training GEMMs reuse the cached bf16 copy instead of re-converting the (large) weight on every
// call — the redundancy the end-to-end bf16 block bench (#1038) exposed.
type DeviceBf16 struct {
	ptr        unsafe.Pointer
	rows, cols int
}

// NewDeviceBf16 allocates an uninitialized [rows,cols] bf16 buffer.
func NewDeviceBf16(rows, cols int) (*DeviceBf16, error) {
	p := unsafe.Pointer(C.cu_alloc_u16(C.int(rows * cols)))
	if p == nil {
		return nil, fmt.Errorf("cuda: NewDeviceBf16 alloc failed")
	}
	return &DeviceBf16{ptr: p, rows: rows, cols: cols}, nil
}

// FromF32 (re)fills the cache by rounding src to bf16. Call once per training step after the optimizer.
func (d *DeviceBf16) FromF32(src *DeviceF32) error {
	if src.rows*src.cols != d.rows*d.cols {
		return fmt.Errorf("cuda: DeviceBf16.FromF32 size mismatch")
	}
	if rc := C.cu_cvt_f32_to_bf16(d.ptr, src.ptr, C.long(d.rows*d.cols)); rc != 0 {
		return fmt.Errorf("cuda: DeviceBf16.FromF32 rc=%d", int(rc))
	}
	return nil
}

// Free releases the buffer.
func (d *DeviceBf16) Free() {
	if d.ptr != nil {
		C.cu_free_f32(d.ptr)
		d.ptr = nil
	}
}

// MatMulWBf16 computes out[M,N]f32 = a[M,K]·w[K,N] on bf16 tensor cores, where w is a PRE-CACHED bf16
// weight and only the activation a is converted per call — the forward projection GEMM for mixed-
// precision training with a weight cache.
func MatMulWBf16(a *DeviceF32, w *DeviceBf16, out *DeviceF32) error {
	m, k := a.rows, a.cols
	if w.rows != k {
		return fmt.Errorf("cuda: MatMulWBf16 a[%d,%d]·w[%d,%d] inner mismatch", a.rows, a.cols, w.rows, w.cols)
	}
	n := w.cols
	if out.rows != m || out.cols != n {
		return fmt.Errorf("cuda: MatMulWBf16 out[%d,%d], want [%d,%d]", out.rows, out.cols, m, n)
	}
	abf := unsafe.Pointer(C.cu_alloc_u16(C.int(m * k)))
	if abf == nil {
		return fmt.Errorf("cuda: MatMulWBf16 scratch alloc failed")
	}
	defer C.cu_free_f32(abf)
	C.cu_cvt_f32_to_bf16(abf, a.ptr, C.long(m*k))
	if rc := C.cu_matmul_bf16_ddd(abf, w.ptr, out.ptr, C.int(m), C.int(k), C.int(n)); rc != 0 {
		return fmt.Errorf("cuda: MatMulWBf16 rc=%d", int(rc))
	}
	return nil
}

// MatMulGradXWBf16 computes the input gradient dX[M,K]f32 = dY·wᵀ on bf16 tensor cores with a PRE-CACHED
// bf16 weight w[K,N]; only dY is converted per call. The backward companion to MatMulWBf16.
func MatMulGradXWBf16(dY *DeviceF32, w *DeviceBf16, dX *DeviceF32) error {
	m, n := dY.rows, dY.cols
	if w.cols != n {
		return fmt.Errorf("cuda: MatMulGradXWBf16 dY[%d,%d]·w[%d,%d] inner mismatch", dY.rows, dY.cols, w.rows, w.cols)
	}
	k := w.rows
	if dX.rows != m || dX.cols != k {
		return fmt.Errorf("cuda: MatMulGradXWBf16 dX[%d,%d], want [%d,%d]", dX.rows, dX.cols, m, k)
	}
	ybf := unsafe.Pointer(C.cu_alloc_u16(C.int(m * n)))
	if ybf == nil {
		return fmt.Errorf("cuda: MatMulGradXWBf16 scratch alloc failed")
	}
	defer C.cu_free_f32(ybf)
	C.cu_cvt_f32_to_bf16(ybf, dY.ptr, C.long(m*n))
	if rc := C.cu_matmul_bf16_ddd_bt(ybf, w.ptr, dX.ptr, C.int(m), C.int(n), C.int(k)); rc != 0 {
		return fmt.Errorf("cuda: MatMulGradXWBf16 rc=%d", int(rc))
	}
	return nil
}
