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

// qBlock is the Q8_0 group size: 32 contiguous weights (along the contracted K
// dimension) share one f32 scale, matching llama.cpp's Q8_0 block layout.
const qBlock = 32

// ResidentBQ8 is a Q8_0-quantized resident weight for a·B matmuls. The f32 weight
// B[K,N] is stored TRANSPOSED and quantized: q[n,k] (int8, each output neuron's K
// weights contiguous) with a per-32-block scale, so B[k,n] ≈ scale[n,k/32]·q[n,k].
// Reading int8 weights is 4× less memory bandwidth than f32 — the decode
// (memory-bound GEMV) win that mirrors llama.cpp's quantized kernels.
//
// DECODE-ONLY: the warp-per-output kernel is a GEMV. For prefill's M=P GEMM it is
// 6–10× SLOWER than cuBLAS f32 (measured, §PERF) — prefill is compute-bound, so it
// stays on the f32 cuBLAS path. Q8 is the memory-bound single-token decode lever.
type ResidentBQ8 struct {
	q      unsafe.Pointer // device int8 [N*K], row n = quantized B[:,n]
	scales unsafe.Pointer // device f32 [N*nb]
	k, n   int
	nb     int // blocks per output row = ceil(K/32)
}

// NewResidentBQ8 quantizes a host f32 weight B[K,N] to Q8_0 and uploads it. The
// quantization is symmetric per 32-block along K: scale = max|w|/127, q = round(w/scale).
func NewResidentBQ8(b *tensor.Tensor) (*ResidentBQ8, error) {
	if b.Ndim() != 2 {
		return nil, fmt.Errorf("cuda: NewResidentBQ8 needs rank-2, got %dD", b.Ndim())
	}
	bc := b.Cast(tensor.F32).Contiguous()
	k, n := bc.Shape()[0], bc.Shape()[1]
	bf := bc.Storage().F32() // bf[row*n + col] = B[row,col]
	nb := (k + qBlock - 1) / qBlock
	q := make([]int8, n*k)
	scales := make([]float32, n*nb)
	for col := 0; col < n; col++ {
		for blk := 0; blk < nb; blk++ {
			k0 := blk * qBlock
			k1 := k0 + qBlock
			if k1 > k {
				k1 = k
			}
			var amax float32
			for kk := k0; kk < k1; kk++ {
				if a := float32(math.Abs(float64(bf[kk*n+col]))); a > amax {
					amax = a
				}
			}
			s := amax / 127
			scales[col*nb+blk] = s
			var inv float32
			if s > 0 {
				inv = 1 / s
			}
			for kk := k0; kk < k1; kk++ {
				qi := int32(math.Round(float64(bf[kk*n+col] * inv)))
				if qi > 127 {
					qi = 127
				} else if qi < -127 {
					qi = -127
				}
				q[col*k+kk] = int8(qi)
			}
		}
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&q[0])), C.int(len(q)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: Q8 weight upload failed")
	}
	ds := C.cu_upload_f32((*C.float)(&scales[0]), C.int(len(scales)))
	if ds == nil {
		C.cu_free_f32(dq)
		return nil, fmt.Errorf("cuda: Q8 scales upload failed")
	}
	return &ResidentBQ8{q: dq, scales: ds, k: k, n: n, nb: nb}, nil
}

// QMatMulDevice computes a·dequant(B) with the activation a resident and the
// weight Q8-quantized+resident, leaving the [a.rows, N] result on the GPU.
func (r *ResidentBQ8) QMatMulDevice(a *DeviceF32) (*DeviceF32, error) {
	if r.q == nil || a.ptr == nil {
		return nil, fmt.Errorf("cuda: QMatMulDevice on a freed handle")
	}
	if a.cols != r.k {
		return nil, fmt.Errorf("cuda: QMatMulDevice inner dim mismatch a[%d,%d]·B[%d,%d]", a.rows, a.cols, r.k, r.n)
	}
	out := C.cu_alloc_f32(C.int(a.rows * r.n))
	if out == nil {
		return nil, fmt.Errorf("cuda: QMatMulDevice output alloc failed")
	}
	if rc := C.cu_qmatmul_q8(a.ptr, r.q, r.scales, out, C.int(a.rows), C.int(r.k), C.int(r.n), C.int(r.nb), C.float(0)); rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: Q8 matmul failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: a.rows, cols: r.n}, nil
}

// QMatMulInto computes out = a·dequant(B) into the caller's fixed buffer (beta=0)
// — the persistent-buffer Q8 matmul for the fixed-buffer / graph decode path.
func (r *ResidentBQ8) QMatMulInto(a, out *DeviceF32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: QMatMulInto on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: QMatMulInto shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	if rc := C.cu_qmatmul_q8(a.ptr, r.q, r.scales, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.int(r.nb), C.float(0)); rc != 0 {
		return fmt.Errorf("cuda: Q8 matmul-into failed (code %d)", int(rc))
	}
	return nil
}

// QMatMulAccInto computes c += a·dequant(B) in place (Q8 weight, beta=1), fusing
// the transformer residual add into the quantized projection — the Q8 analogue of
// ResidentB.MatMulAccInto. c must hold the residual and be [a.rows, N].
func (r *ResidentBQ8) QMatMulAccInto(a, c *DeviceF32) error {
	if r.q == nil || a.ptr == nil || c.ptr == nil {
		return fmt.Errorf("cuda: QMatMulAccInto on a freed handle")
	}
	if a.cols != r.k {
		return fmt.Errorf("cuda: QMatMulAccInto inner dim mismatch a[%d,%d]·B[%d,%d]", a.rows, a.cols, r.k, r.n)
	}
	if c.rows != a.rows || c.cols != r.n {
		return fmt.Errorf("cuda: QMatMulAccInto accumulator must be [%d,%d], got [%d,%d]", a.rows, r.n, c.rows, c.cols)
	}
	if rc := C.cu_qmatmul_q8(a.ptr, r.q, r.scales, c.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.int(r.nb), C.float(1)); rc != 0 {
		return fmt.Errorf("cuda: Q8 matmul-acc failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the quantized weight and scale buffers.
func (r *ResidentBQ8) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
	if r.scales != nil {
		C.cu_free_f32(r.scales)
		r.scales = nil
	}
}
