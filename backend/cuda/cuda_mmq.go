//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/jxsl13/goai/tensor"
)

// ResidentMMQ is a weight matrix resident on the GPU as per-32-block int8 (Q8_0-style) — the
// int8 tensor-core MMQ prefill path. It mirrors ResidentBF16 but stores the weight as int8 [N][K]
// (native GGUF out×in layout) + per-block f32 scales instead of f16, ~half the weight VRAM. Per
// MatMul it dynamically per-row int8-quantizes the f32 activations on-device (cu_quant_rows_i8) and
// runs the MMQ GEMM (cu_matmul_i8_mmq_r), accumulating in f32. Accuracy ~0.9% norm-rel-RMS vs the
// f32 GEMM (Q8_0-class). PREFILL-ONLY by design (decode stays on the Q4_K/Q8 GEMV).
//
// Constraints from the tiled kernel: K%32==0 and N%64==0. M (sequence length) is arbitrary — it is
// padded up to a multiple of 64 internally (pad rows compute ignored garbage).
type ResidentMMQ struct {
	w8   unsafe.Pointer // device int8 [N][K]
	wSc  unsafe.Pointer // device f32  [N][K/32]
	k, n int
}

// NewResidentMMQ uploads B[K,N] (any float dtype) as a resident per-block-int8 [N][K] weight.
func NewResidentMMQ(b *tensor.Tensor) (*ResidentMMQ, error) {
	if b.Ndim() != 2 {
		return nil, fmt.Errorf("cuda: NewResidentMMQ needs rank-2, got %dD", b.Ndim())
	}
	bc := b.Cast(tensor.F32).Contiguous()
	k, n := bc.Shape()[0], bc.Shape()[1]
	if k%32 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentMMQ needs K%%32==0, got K=%d", k)
	}
	if n%64 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentMMQ needs N%%64==0, got N=%d", n)
	}
	src := bc.Storage().F32() // [K][N] row-major
	nb := k / 32
	w8 := make([]int8, n*k)      // [N][K]
	wSc := make([]float32, n*nb) // [N][K/32]
	// Transpose [K][N] -> [N][K] and quantize each output feature's K weights in 32-blocks.
	for row := 0; row < n; row++ { // output feature
		for blk := 0; blk < nb; blk++ {
			var amax float32
			for j := 0; j < 32; j++ {
				v := float32(math.Abs(float64(src[(blk*32+j)*n+row]))) // B[blk*32+j][row]
				if v > amax {
					amax = v
				}
			}
			s := amax / 127.0
			wSc[row*nb+blk] = s
			inv := float32(0)
			if s > 0 {
				inv = 1.0 / s
			}
			for j := 0; j < 32; j++ {
				q := int32(math.Round(float64(src[(blk*32+j)*n+row] * inv)))
				if q > 127 {
					q = 127
				} else if q < -128 {
					q = -128
				}
				w8[row*k+blk*32+j] = int8(q)
			}
		}
	}
	dW := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&w8[0])), C.int(len(w8)))
	if dW == nil {
		return nil, fmt.Errorf("cuda: MMQ weight upload failed [%d,%d]", k, n)
	}
	dWs := C.cu_upload_f32((*C.float)(unsafe.Pointer(&wSc[0])), C.int(len(wSc)))
	if dWs == nil {
		C.cu_free_f32(unsafe.Pointer(dW))
		return nil, fmt.Errorf("cuda: MMQ scale upload failed [%d,%d]", k, n)
	}
	return &ResidentMMQ{w8: unsafe.Pointer(dW), wSc: unsafe.Pointer(dWs), k: k, n: n}, nil
}

// MatMulDevice computes a·B via int8 MMQ tensor cores, returning a new resident f32 output — the
// int8 twin of ResidentBF16.MatMulDevice. a is [M,K] f32; output is [M,N] f32.
func (r *ResidentMMQ) MatMulDevice(a *DeviceF32) (*DeviceF32, error) {
	if r.w8 == nil || a.ptr == nil {
		return nil, fmt.Errorf("cuda: MMQ MatMulDevice on a freed handle")
	}
	if a.cols != r.k {
		return nil, fmt.Errorf("cuda: MMQ MatMulDevice inner dim mismatch a[%d,%d]·B[%d,%d]", a.rows, a.cols, r.k, r.n)
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
		return nil, fmt.Errorf("cuda: MMQ scratch alloc failed")
	}
	defer C.cu_free_f32(unsafe.Pointer(dA8))
	defer C.cu_free_f32(dAs)
	// Quantize only the m real activation rows (pad rows produce ignored output rows).
	if rc := C.cu_quant_rows_i8(a.ptr, unsafe.Pointer(dA8), unsafe.Pointer(dAs), C.int(m), C.int(r.k)); rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: MMQ activation quant failed (code %d)", int(rc))
	}
	if rc := C.cu_matmul_i8_mmq_r(unsafe.Pointer(dA8), r.w8, unsafe.Pointer(dAs), r.wSc, out, C.int(mPad), C.int(r.k), C.int(r.n)); rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: MMQ matmul failed (code %d)", int(rc))
	}
	// out buffer is [mPad,N]; the first m rows are the valid contiguous result.
	return &DeviceF32{ptr: out, rows: m, cols: r.n}, nil
}

// MatMulAccInto computes c += a·B in place (residual-fused, beta=1) — the int8 twin of
// ResidentBF16.MatMulAccInto used by the prefill o/down projections. Runs the MMQ into a scratch
// then a stream-ordered elementwise add (the MMQ output is f32, so no extra precision loss).
func (r *ResidentMMQ) MatMulAccInto(a, c *DeviceF32) error {
	if r.w8 == nil || a.ptr == nil || c.ptr == nil {
		return fmt.Errorf("cuda: MMQ MatMulAccInto on a freed handle")
	}
	if a.cols != r.k || c.rows != a.rows || c.cols != r.n {
		return fmt.Errorf("cuda: MMQ MatMulAccInto shape a[%d,%d]·B[%d,%d]→c[%d,%d]", a.rows, a.cols, r.k, r.n, c.rows, c.cols)
	}
	tmp, err := r.MatMulDevice(a)
	if err != nil {
		return err
	}
	defer tmp.Free()
	if rc := C.cu_add_f32(c.ptr, tmp.ptr, C.int(a.rows*r.n)); rc != 0 {
		return fmt.Errorf("cuda: MMQ residual add failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident int8 weight + scales.
func (r *ResidentMMQ) Free() {
	if r.w8 != nil {
		C.cu_free_f32(r.w8)
		r.w8 = nil
	}
	if r.wSc != nil {
		C.cu_free_f32(r.wSc)
		r.wSc = nil
	}
}
