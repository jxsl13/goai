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

// MatMulMMQDevice runs the int8 tensor-core MMQ prefill GEMM (a·dequant(B)) reusing THIS resident
// Q8 weight's own int8 [N][K] blocks + per-block scales — the exact buffers cu_matmul_i8_mmq_r wants.
// So a single ResidentBQ8 serves BOTH the decode GEMV (QMatMulDevice) and the batched prefill GEMM
// with NO second weight copy: the prefill's int8 tensor cores read the same bytes the decode GEMV
// does. Activations are dynamically per-row int8-quantized on device. f32 accumulate; [M,N] f32 out.
//
// Requires the MMQ tiling constraints K%32==0 and N%64==0 (Q8 GEMV has none); returns an error
// otherwise so callers fall back to the f16/GEMV path for those projections. M is padded to 64.
func (r *ResidentBQ8) MatMulMMQDevice(a *DeviceF32) (*DeviceF32, error) {
	if r.q == nil || a.ptr == nil {
		return nil, fmt.Errorf("cuda: Q8 MatMulMMQDevice on a freed handle")
	}
	if a.cols != r.k {
		return nil, fmt.Errorf("cuda: Q8 MatMulMMQDevice inner dim mismatch a[%d,%d]·B[%d,%d]", a.rows, a.cols, r.k, r.n)
	}
	if r.k%32 != 0 || r.n%64 != 0 {
		return nil, fmt.Errorf("cuda: Q8 MMQ prefill needs K%%32==0 && N%%64==0, got K=%d N=%d", r.k, r.n)
	}
	m := a.rows
	mPad := ((m + 63) / 64) * 64
	dA8 := C.cu_alloc_i8(C.int(mPad * r.k))
	dAs := C.cu_alloc_f32(C.int(mPad))
	out := C.cu_alloc_f32(C.int(mPad * r.n))
	if dA8 == nil || dAs == nil || out == nil {
		C.cu_free_f32(unsafe.Pointer(dA8))
		C.cu_free_f32(dAs)
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: Q8 MMQ scratch alloc failed")
	}
	defer C.cu_free_f32(unsafe.Pointer(dA8))
	defer C.cu_free_f32(dAs)
	if rc := C.cu_quant_rows_i8(a.ptr, unsafe.Pointer(dA8), unsafe.Pointer(dAs), C.int(m), C.int(r.k)); rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: Q8 MMQ activation quant failed (code %d)", int(rc))
	}
	if rc := C.cu_matmul_i8_mmq_r(unsafe.Pointer(dA8), r.q, unsafe.Pointer(dAs), r.scales, out, C.int(mPad), C.int(r.k), C.int(r.n)); rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: Q8 MMQ matmul failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: m, cols: r.n}, nil
}

// MatMulMMQAccInto computes c += a·dequant(B) in place via the MMQ prefill GEMM (residual-fused) —
// the int8-tensor-core twin of QMatMulAccInto, reusing this weight's resident Q8 blocks.
func (r *ResidentBQ8) MatMulMMQAccInto(a, c *DeviceF32) error {
	if c.ptr == nil {
		return fmt.Errorf("cuda: Q8 MatMulMMQAccInto on a freed handle")
	}
	if c.rows != a.rows || c.cols != r.n {
		return fmt.Errorf("cuda: Q8 MatMulMMQAccInto shape a[%d,%d]·B[%d,%d]→c[%d,%d]", a.rows, a.cols, r.k, r.n, c.rows, c.cols)
	}
	tmp, err := r.MatMulMMQDevice(a)
	if err != nil {
		return err
	}
	defer tmp.Free()
	if rc := C.cu_add_f32(c.ptr, tmp.ptr, C.int(a.rows*r.n)); rc != 0 {
		return fmt.Errorf("cuda: Q8 MMQ residual add failed (code %d)", int(rc))
	}
	return nil
}
