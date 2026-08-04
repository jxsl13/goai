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
				//perfscan:ignore PS5007,PS6011 load-time weight quantization (NewResidentMMQ)
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

// QuantActs is a per-row int8-quantized activation matrix (int8 [mPad,K] + f32 scale [mPad]),
// resident on the GPU. The point is to quantize ONCE and feed several ResidentMMQ projections that
// share the same input — e.g. the q/k/v attention projections all consume the post-attn-norm hidden
// state, and gate/up both consume the post-ffn-norm state. cu_quant_rows_i8 is a full pass over the
// activations per call, so quantizing once instead of per-projection removes 2/3 of the attn-proj and
// 1/2 of the ffn-proj activation-quant work in prefill.
type QuantActs struct {
	a8      unsafe.Pointer // device int8 [mPad,K]
	sc      unsafe.Pointer // device f32  [mPad]
	m, mPad int
	k       int
}

// QuantizeActs quantizes a [M,K] f32 activation to per-row int8 (the cu_matmul_i8_mmq_r contract),
// once, so multiple ResidentMMQ.MatMulPreQuant calls can reuse it. M is padded to 64 internally.
func QuantizeActs(a *DeviceF32) (*QuantActs, error) {
	if a.ptr == nil {
		return nil, fmt.Errorf("cuda: QuantizeActs on a freed handle")
	}
	m, k := a.rows, a.cols
	mPad := ((m + 63) / 64) * 64
	a8 := C.cu_alloc_i8(C.int(mPad * k))
	sc := C.cu_alloc_f32(C.int(mPad))
	if a8 == nil || sc == nil {
		C.cu_free_f32(unsafe.Pointer(a8))
		C.cu_free_f32(sc)
		return nil, fmt.Errorf("cuda: QuantizeActs alloc failed")
	}
	if rc := C.cu_quant_rows_i8(a.ptr, unsafe.Pointer(a8), unsafe.Pointer(sc), C.int(m), C.int(k)); rc != 0 {
		C.cu_free_f32(unsafe.Pointer(a8))
		C.cu_free_f32(sc)
		return nil, fmt.Errorf("cuda: QuantizeActs quant failed (code %d)", int(rc))
	}
	return &QuantActs{a8: unsafe.Pointer(a8), sc: unsafe.Pointer(sc), m: m, mPad: mPad, k: k}, nil
}

// Free releases the quantized activation buffers.
func (q *QuantActs) Free() {
	if q.a8 != nil {
		C.cu_free_f32(q.a8)
		q.a8 = nil
	}
	if q.sc != nil {
		C.cu_free_f32(q.sc)
		q.sc = nil
	}
}

// MatMulPreQuant computes qa·B via MMQ WITHOUT re-quantizing the activations — the shared-quant twin
// of MatMulDevice for projections whose input was already quantized once (e.g. q/k/v sharing the
// attn-norm hidden). Returns a new [M,N] f32 output.
func (r *ResidentMMQ) MatMulPreQuant(qa *QuantActs) (*DeviceF32, error) {
	if r.w8 == nil || qa.a8 == nil {
		return nil, fmt.Errorf("cuda: MMQ MatMulPreQuant on a freed handle")
	}
	if qa.k != r.k {
		return nil, fmt.Errorf("cuda: MMQ MatMulPreQuant inner dim mismatch a[.,%d]·B[%d,%d]", qa.k, r.k, r.n)
	}
	out := C.cu_alloc_f32(C.int(qa.mPad * r.n))
	if out == nil {
		return nil, fmt.Errorf("cuda: MMQ MatMulPreQuant output alloc failed")
	}
	if rc := C.cu_matmul_i8_mmq_r(qa.a8, r.w8, qa.sc, r.wSc, out, C.int(qa.mPad), C.int(r.k), C.int(r.n)); rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: MMQ MatMulPreQuant matmul failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: qa.m, cols: r.n}, nil
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
