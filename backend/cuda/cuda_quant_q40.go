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

// ResidentBQ40 is a weight matrix in ggml's legacy Q4_0 format made GPU-resident:
// per output row, K/32 blocks of 18 bytes (f16 d, then 16 nibble bytes holding 32
// 4-bit quants). SYMMETRIC round quant, no min: dequant in-kernel as y = d·(nibble−8),
// where byte i carries element i (low nibble) and i+16 (high nibble). 0.5625 B/weight —
// the original 4-bit GGUF format; still shipped by some models. Distinct from the
// K-quant family and from cuda's asymmetric Q4 (which carries a per-block min).
// K must be a multiple of 32. DECODE-ONLY (GEMV).
type ResidentBQ40 struct {
	q    unsafe.Pointer // device bytes, row n = nblk consecutive 18-byte blocks
	k, n int
	nblk int // blocks per row = K/32
}

const q40BlockBytes = 18

// NewResidentBQ40FromBlocks uploads pre-encoded Q4_0 blocks for a weight with N
// output rows of K inputs each (raw layout: row-major [N][K/32] blocks — exactly the
// byte order of a gguf [out,in] Q4_0 tensor, and of gguf.Quantize on an [N,K] tensor).
func NewResidentBQ40FromBlocks(raw []byte, k, n int) (*ResidentBQ40, error) {
	if k%32 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBQ40FromBlocks needs K%%32==0, got K=%d", k)
	}
	nblk := k / 32
	if want := n * nblk * q40BlockBytes; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBQ40FromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&raw[0])), C.int(len(raw)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: Q4_0 weight upload failed")
	}
	return &ResidentBQ40{q: dq, k: k, n: n, nblk: nblk}, nil
}

// QMatMulInto computes out = a·dequant(W) into the caller's fixed buffer (beta=0).
func (r *ResidentBQ40) QMatMulInto(a, out *DeviceF32) error {
	return r.qmatmul(a, out, 0)
}

// QMatMulAccInto computes c += a·dequant(W) in place (beta=1) — fuses a transformer
// residual add into the projection (the o and down projections).
func (r *ResidentBQ40) QMatMulAccInto(a, c *DeviceF32) error {
	return r.qmatmul(a, c, 1)
}

func (r *ResidentBQ40) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: Q4_0 matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: Q4_0 matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	if rc := C.cu_qmatmul_q40(a.ptr, r.q, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: Q4_0 matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident Q4_0 weight.
func (r *ResidentBQ40) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
}
