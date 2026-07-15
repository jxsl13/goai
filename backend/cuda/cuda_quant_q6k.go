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

// ResidentBQ6K is a weight matrix in ggml's Q6_K super-block format (§R99) made
// GPU-resident: per output row, K/256 blocks of 210 bytes (ql[128] low nibbles,
// qh[64] packed 2-bit highs, scales[16] int8 per 16-element sub-block, f16 d),
// dequantized in-kernel as y = d·sc·(q6−32). 0.8203 B/weight — the NATIVE path
// for the Q6_K minority tensors of llama.cpp Q4_K_M mixes (v/down on some
// layers + the output head), which the direct loader previously re-encoded to
// Q8 (1.0625 B/w, requant loss). K must be a multiple of 256. DECODE-ONLY (GEMV).
type ResidentBQ6K struct {
	q    unsafe.Pointer // device bytes, row n = sbs consecutive 210-byte super-blocks
	k, n int
	sbs  int // super-blocks per row = K/256
}

// NewResidentBQ6KFromBlocks uploads pre-encoded Q6_K blocks for a weight with N
// output rows of K inputs each (raw layout: row-major [N][K/256] super-blocks —
// exactly the byte order of a gguf [out,in] Q6_K tensor).
func NewResidentBQ6KFromBlocks(raw []byte, k, n int) (*ResidentBQ6K, error) {
	if k%256 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBQ6KFromBlocks needs K%%256==0, got K=%d", k)
	}
	sbs := k / 256
	if want := n * sbs * 210; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBQ6KFromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&raw[0])), C.int(len(raw)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: Q6_K weight upload failed")
	}
	return &ResidentBQ6K{q: dq, k: k, n: n, sbs: sbs}, nil
}

// QMatMulInto computes out = a·dequant(W) into the caller's fixed buffer (beta=0).
func (r *ResidentBQ6K) QMatMulInto(a, out *DeviceF32) error {
	return r.qmatmul(a, out, 0)
}

// QMatMulAccInto computes c += a·dequant(W) in place (beta=1) — fuses a transformer
// residual add into the projection (the o and down projections).
func (r *ResidentBQ6K) QMatMulAccInto(a, c *DeviceF32) error {
	return r.qmatmul(a, c, 1)
}

func (r *ResidentBQ6K) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: Q6_K matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: Q6_K matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	if rc := C.cu_qmatmul_q6k(a.ptr, r.q, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: Q6_K matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident Q6_K weight.
func (r *ResidentBQ6K) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
}
