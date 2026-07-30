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

// ResidentBQ5K is a weight matrix in ggml's Q5_K super-block format (§R102) made
// GPU-resident: per output row, K/256 blocks of 176 bytes (f16 d, f16 dmin,
// scales[12] = the SAME get_scale_min_k4 6-bit scale+min packing as Q4_K, qh[32]
// one high bit per quant, qs[128] low nibbles), dequantized in-kernel as
// y = d·sc6·q5 − dmin·min6 with q5 = nibble | (highbit<<4). 0.6875 B/weight — the
// NATIVE path for the Q5_K bulk tensors of llama.cpp Q5_K_M / Q5_K_S mixes (the
// second-most-common download after Q4_K_M), which the direct loader previously
// re-encoded to Q8 (1.0625 B/w, requant loss). Completes the K-quant family
// (Q4_K + Q5_K + Q6_K). K must be a multiple of 256. DECODE-ONLY (GEMV).
type ResidentBQ5K struct {
	q    unsafe.Pointer // device bytes, row n = sbs consecutive 176-byte super-blocks
	k, n int
	sbs  int // super-blocks per row = K/256
}

const q5kBlockBytes = 176

// NewResidentBQ5KFromBlocks uploads pre-encoded Q5_K blocks for a weight with N
// output rows of K inputs each (raw layout: row-major [N][K/256] super-blocks —
// exactly the byte order of a gguf [out,in] Q5_K tensor, and of gguf.Quantize on
// an [N,K] tensor).
func NewResidentBQ5KFromBlocks(raw []byte, k, n int) (*ResidentBQ5K, error) {
	if k%256 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBQ5KFromBlocks needs K%%256==0, got K=%d", k)
	}
	sbs := k / 256
	if want := n * sbs * q5kBlockBytes; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBQ5KFromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&raw[0])), C.int(len(raw)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: Q5_K weight upload failed")
	}
	return &ResidentBQ5K{q: dq, k: k, n: n, sbs: sbs}, nil
}

// QMatMulInto computes out = a·dequant(W) into the caller's fixed buffer (beta=0).
func (r *ResidentBQ5K) QMatMulInto(a, out *DeviceF32) error {
	return r.qmatmul(a, out, 0)
}

// QMatMulAccInto computes c += a·dequant(W) in place (beta=1) — fuses a transformer
// residual add into the projection (the o and down projections).
func (r *ResidentBQ5K) QMatMulAccInto(a, c *DeviceF32) error {
	return r.qmatmul(a, c, 1)
}

func (r *ResidentBQ5K) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: Q5_K matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: Q5_K matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	// M>1 (prefill/batch): route to the weight-read-once M-tiled GEMM (bit-identical) so column
	// n's Q5_K block is decoded once, not re-read per row. M==1 decode stays on the GEMV. The
	// threshold is M>1 (matching Q4_K/Q6_K): the GEMV re-reads each Q5_K block M times, so the
	// MT kernel's read-once already wins at M=2 (benchmarked), not only at large batches.
	if a.rows >= 2 {
		if rc := C.cu_qmatmul_q5k_mt(a.ptr, r.q, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
			return fmt.Errorf("cuda: Q5_K m-tiled matmul failed (code %d)", int(rc))
		}
		return nil
	}
	if rc := C.cu_qmatmul_q5k(a.ptr, r.q, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: Q5_K matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident Q5_K weight.
func (r *ResidentBQ5K) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
}
