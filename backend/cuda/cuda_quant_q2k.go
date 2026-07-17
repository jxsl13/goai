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

// ResidentBQ2K is a weight matrix in ggml's Q2_K super-block format (§R104) made
// GPU-resident: per output row, K/256 blocks of 84 bytes (scales[16] with a 4-bit
// scale in the low nibble and a 4-bit min in the high nibble, qs[64] two-bit quants,
// f16 d, f16 dmin). ASYMMETRIC AFFINE like Q4_K but coarser: dequant in-kernel as
// y = d·sc4·q2 − dmin·min4. 0.3281 B/weight — the smallest quant, used to fit very
// large models (70B+) in tight VRAM. Completes the CUDA K-quant family
// (Q2_K + Q3_K + Q4_K + Q5_K + Q6_K). K%256==0. DECODE-ONLY (GEMV).
type ResidentBQ2K struct {
	q    unsafe.Pointer // device bytes, row n = sbs consecutive 84-byte super-blocks
	k, n int
	sbs  int // super-blocks per row = K/256
}

const q2kBlockBytes = 84

// NewResidentBQ2KFromBlocks uploads pre-encoded Q2_K blocks for a weight with N
// output rows of K inputs each (raw layout: row-major [N][K/256] super-blocks —
// exactly the byte order of a gguf [out,in] Q2_K tensor, and of gguf.Quantize on
// an [N,K] tensor).
func NewResidentBQ2KFromBlocks(raw []byte, k, n int) (*ResidentBQ2K, error) {
	if k%256 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBQ2KFromBlocks needs K%%256==0, got K=%d", k)
	}
	sbs := k / 256
	if want := n * sbs * q2kBlockBytes; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBQ2KFromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&raw[0])), C.int(len(raw)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: Q2_K weight upload failed")
	}
	return &ResidentBQ2K{q: dq, k: k, n: n, sbs: sbs}, nil
}

// QMatMulInto computes out = a·dequant(W) into the caller's fixed buffer (beta=0).
func (r *ResidentBQ2K) QMatMulInto(a, out *DeviceF32) error {
	return r.qmatmul(a, out, 0)
}

// QMatMulAccInto computes c += a·dequant(W) in place (beta=1) — fuses a transformer
// residual add into the projection (the o and down projections).
func (r *ResidentBQ2K) QMatMulAccInto(a, c *DeviceF32) error {
	return r.qmatmul(a, c, 1)
}

func (r *ResidentBQ2K) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: Q2_K matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: Q2_K matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	// M>1 (prefill/batch): route to the weight-read-once M-tiled GEMM (bit-identical) so column
	// n's Q2_K block is decoded once, not re-read per row. M==1 decode stays on the GEMV.
	if a.rows >= 8 {
		if rc := C.cu_qmatmul_q2k_mt(a.ptr, r.q, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
			return fmt.Errorf("cuda: Q2_K m-tiled matmul failed (code %d)", int(rc))
		}
		return nil
	}
	if rc := C.cu_qmatmul_q2k(a.ptr, r.q, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: Q2_K matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident Q2_K weight.
func (r *ResidentBQ2K) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
}
