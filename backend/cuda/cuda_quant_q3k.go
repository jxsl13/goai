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

// ResidentBQ3K is a weight matrix in ggml's Q3_K super-block format (§R103) made
// GPU-resident: per output row, K/256 blocks of 110 bytes (hmask[32] one high bit
// per quant, qs[64] two low bits per quant, scales[12] = 16 signed 6-bit sub-scales
// via the aux/kmask splice, f16 d last). SYMMETRIC (no min): dequant in-kernel as
// y = d·(sc6−32)·(q3−4) with q3 = (2 low bits) | (high bit)<<2. 0.4297 B/weight —
// the bulk tensors of a Q3_K_M / Q3_K_L / Q3_K_S mix (fit large models in limited
// VRAM). Extends the CUDA K-quant family beyond Q4_K/Q5_K/Q6_K. K%256==0. DECODE GEMV.
type ResidentBQ3K struct {
	q    unsafe.Pointer // device bytes, row n = sbs consecutive 110-byte super-blocks
	k, n int
	sbs  int // super-blocks per row = K/256
}

const q3kBlockBytes = 110

// NewResidentBQ3KFromBlocks uploads pre-encoded Q3_K blocks for a weight with N
// output rows of K inputs each (raw layout: row-major [N][K/256] super-blocks —
// exactly the byte order of a gguf [out,in] Q3_K tensor, and of gguf.Quantize on
// an [N,K] tensor).
func NewResidentBQ3KFromBlocks(raw []byte, k, n int) (*ResidentBQ3K, error) {
	if k%256 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBQ3KFromBlocks needs K%%256==0, got K=%d", k)
	}
	sbs := k / 256
	if want := n * sbs * q3kBlockBytes; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBQ3KFromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&raw[0])), C.int(len(raw)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: Q3_K weight upload failed")
	}
	return &ResidentBQ3K{q: dq, k: k, n: n, sbs: sbs}, nil
}

// QMatMulInto computes out = a·dequant(W) into the caller's fixed buffer (beta=0).
func (r *ResidentBQ3K) QMatMulInto(a, out *DeviceF32) error {
	return r.qmatmul(a, out, 0)
}

// QMatMulAccInto computes c += a·dequant(W) in place (beta=1) — fuses a transformer
// residual add into the projection (the o and down projections).
func (r *ResidentBQ3K) QMatMulAccInto(a, c *DeviceF32) error {
	return r.qmatmul(a, c, 1)
}

func (r *ResidentBQ3K) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: Q3_K matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: Q3_K matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	if rc := C.cu_qmatmul_q3k(a.ptr, r.q, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: Q3_K matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident Q3_K weight.
func (r *ResidentBQ3K) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
}
