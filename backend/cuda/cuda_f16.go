//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/jxsl13/goai/tensor"
)

// ResidentBF16 is a weight matrix resident on the GPU in f16 — the tensor-core prefill
// path (lever b1). The prefill FFN/projection GEMMs run at the cuBLAS f32 ceiling
// (PERF-PREFILL-PROFILE: ffn-gemm 53.8% of prefill), and Ampere executes f16-input /
// f32-accumulate GEMMs (cublasGemmEx) at ~2× the f32 FMA rate. Activations stay f32
// end-to-end (converted to f16 in a stream-ordered scratch per call — a bandwidth-cheap
// elementwise pass next to a compute-bound GEMM). Accuracy: weights round to f16
// (~1e-3 rel), accumulation stays f32. PREFILL-ONLY by design — decode stays on the
// Q4_K/Q8 GEMV (memory-bound, where f16 weights would be 2.8× the Q4_K bytes).
type ResidentBF16 struct {
	ptr  unsafe.Pointer // device f16 weight [K,N]
	k, n int
}

// NewResidentBF16 uploads B[K,N] (any float dtype) as a resident f16 weight.
func NewResidentBF16(b *tensor.Tensor) (*ResidentBF16, error) {
	if b.Ndim() != 2 {
		return nil, fmt.Errorf("cuda: NewResidentBF16 needs rank-2, got %dD", b.Ndim())
	}
	bc := b.Cast(tensor.F32).Contiguous()
	k, n := bc.Shape()[0], bc.Shape()[1]
	buf := bc.Storage().F32()
	p := C.cu_upload_f16((*C.float)(&buf[0]), C.long(len(buf)))
	if p == nil {
		return nil, fmt.Errorf("cuda: f16 weight upload failed [%d,%d]", k, n)
	}
	return &ResidentBF16{ptr: p, k: k, n: n}, nil
}

// MatMulDevice computes a·B on tensor cores (f16 inputs, f32 accumulate), returning a
// new resident f32 output — the drop-in prefill twin of ResidentB.MatMulDevice.
func (r *ResidentBF16) MatMulDevice(a *DeviceF32) (*DeviceF32, error) {
	if r.ptr == nil || a.ptr == nil {
		return nil, fmt.Errorf("cuda: f16 MatMulDevice on a freed handle")
	}
	if a.cols != r.k {
		return nil, fmt.Errorf("cuda: f16 MatMulDevice inner dim mismatch a[%d,%d]·B[%d,%d]", a.rows, a.cols, r.k, r.n)
	}
	out := C.cu_alloc_f32(C.int(a.rows * r.n))
	if out == nil {
		return nil, fmt.Errorf("cuda: f16 MatMulDevice output alloc failed")
	}
	var rc C.int
	if f16AccEnabled() {
		rc = C.cu_matmul_f16w_acc16(a.ptr, r.ptr, out, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(0))
	} else {
		rc = C.cu_matmul_f16w(a.ptr, r.ptr, out, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(0))
	}
	if rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: f16 matmul failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: a.rows, cols: r.n}, nil
}

// MatMulAccInto computes c += a·B in place (beta=1) on tensor cores — the residual-fused
// twin of ResidentB.MatMulAccInto.
func (r *ResidentBF16) MatMulAccInto(a, c *DeviceF32) error {
	if r.ptr == nil || a.ptr == nil || c.ptr == nil {
		return fmt.Errorf("cuda: f16 MatMulAccInto on a freed handle")
	}
	if a.cols != r.k || c.rows != a.rows || c.cols != r.n {
		return fmt.Errorf("cuda: f16 MatMulAccInto shape a[%d,%d]·B[%d,%d]→c[%d,%d]", a.rows, a.cols, r.k, r.n, c.rows, c.cols)
	}
	var rc C.int
	if f16AccEnabled() {
		rc = C.cu_matmul_f16w_acc16(a.ptr, r.ptr, c.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(1))
	} else {
		rc = C.cu_matmul_f16w(a.ptr, r.ptr, c.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(1))
	}
	if rc != 0 {
		return fmt.Errorf("cuda: f16 matmul-acc failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident f16 weight.
func (r *ResidentBF16) Free() {
	if r.ptr != nil {
		C.cu_free_f32(r.ptr)
		r.ptr = nil
	}
}
