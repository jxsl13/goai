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

// ResidentBQ4 is a weight matrix quantized to ASYMMETRIC 4-bit and made GPU-resident: per
// 32-element block along K it stores a scale + a min, and each weight as a 4-bit nibble,
// dequantized in-kernel as w = min + nibble·scale. Asymmetric (scale+min) is far more
// accurate than symmetric Q4_0 (which the earlier experiment rejected for token divergence),
// while reading ≈0.67× the bytes of the resident Q8 GEMV — the weight-bandwidth lever for
// decode. K must be a multiple of 256 (the kernel's 8-block vector step). DECODE-ONLY (GEMV).
type ResidentBQ4 struct {
	q      unsafe.Pointer // device uint8 [N*K/2], packed nibbles, row n = quantized B[:,n]
	scales unsafe.Pointer // device f32 [N*nb]
	mins   unsafe.Pointer // device f32 [N*nb]
	k, n   int
	nb     int // 32-blocks per output row = K/32
}

// NewResidentBQ4 quantizes a host f32 weight B[K,N] to asymmetric Q4 and uploads it.
func NewResidentBQ4(b *tensor.Tensor) (*ResidentBQ4, error) {
	if b.Ndim() != 2 {
		return nil, fmt.Errorf("cuda: NewResidentBQ4 needs rank-2, got %dD", b.Ndim())
	}
	bc := b.Cast(tensor.F32).Contiguous()
	k, n := bc.Shape()[0], bc.Shape()[1]
	if k%256 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBQ4 needs K%%256==0, got K=%d", k)
	}
	bf := bc.Storage().F32() // bf[row*n + col] = B[row,col]
	nb := k / qBlock         // qBlock == 32
	q := make([]byte, n*(k/2))
	scales := make([]float32, n*nb)
	mins := make([]float32, n*nb)
	for col := 0; col < n; col++ {
		for blk := 0; blk < nb; blk++ {
			k0 := blk * qBlock
			mn, mx := float32(math.MaxFloat32), float32(-math.MaxFloat32)
			for kk := k0; kk < k0+qBlock; kk++ {
				v := bf[kk*n+col]
				if v < mn {
					mn = v
				}
				if v > mx {
					mx = v
				}
			}
			s := (mx - mn) / 15
			scales[col*nb+blk] = s
			mins[col*nb+blk] = mn
			var inv float32
			if s > 0 {
				inv = 1 / s
			}
			for kk := k0; kk < k0+qBlock; kk++ {
				nib := int32(math.Round(float64((bf[kk*n+col] - mn) * inv)))
				if nib < 0 {
					nib = 0
				} else if nib > 15 {
					nib = 15
				}
				bi := col*(k/2) + kk/2
				if kk&1 == 0 {
					q[bi] = (q[bi] & 0xf0) | byte(nib)
				} else {
					q[bi] = (q[bi] & 0x0f) | byte(nib<<4)
				}
			}
		}
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&q[0])), C.int(len(q)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: Q4 weight upload failed")
	}
	ds := C.cu_upload_f32((*C.float)(&scales[0]), C.int(len(scales)))
	dm := C.cu_upload_f32((*C.float)(&mins[0]), C.int(len(mins)))
	if ds == nil || dm == nil {
		C.cu_free_f32(dq)
		C.cu_free_f32(ds)
		C.cu_free_f32(dm)
		return nil, fmt.Errorf("cuda: Q4 scale/min upload failed")
	}
	return &ResidentBQ4{q: dq, scales: ds, mins: dm, k: k, n: n, nb: nb}, nil
}

// QMatMulInto computes out = a·dequant(W4) into the caller's fixed buffer (beta=0).
func (r *ResidentBQ4) QMatMulInto(a, out *DeviceF32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: Q4 QMatMulInto on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: Q4 QMatMulInto shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	if rc := C.cu_qmatmul_q4(a.ptr, r.q, r.scales, r.mins, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.int(r.nb), C.float(0)); rc != 0 {
		return fmt.Errorf("cuda: Q4 matmul failed (code %d)", int(rc))
	}
	return nil
}

// QMatMulAccInto computes c += a·dequant(W4) in place (beta=1) — fuses a transformer
// residual add into the projection (the o and down projections).
func (r *ResidentBQ4) QMatMulAccInto(a, c *DeviceF32) error {
	if r.q == nil || a.ptr == nil || c.ptr == nil {
		return fmt.Errorf("cuda: Q4 QMatMulAccInto on a freed handle")
	}
	if a.cols != r.k || c.rows != a.rows || c.cols != r.n {
		return fmt.Errorf("cuda: Q4 QMatMulAccInto shape a[%d,%d]·B[%d,%d]→c[%d,%d]", a.rows, a.cols, r.k, r.n, c.rows, c.cols)
	}
	if rc := C.cu_qmatmul_q4(a.ptr, r.q, r.scales, r.mins, c.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.int(r.nb), C.float(1)); rc != 0 {
		return fmt.Errorf("cuda: Q4 matmul-acc failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident Q4 weight.
func (r *ResidentBQ4) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
	if r.scales != nil {
		C.cu_free_f32(r.scales)
		r.scales = nil
	}
	if r.mins != nil {
		C.cu_free_f32(r.mins)
		r.mins = nil
	}
}
