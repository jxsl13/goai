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

// ResidentBQ4K is a weight matrix in ggml's Q4_K super-block format (§R100) made
// GPU-resident: per output row, K/256 blocks of 144 bytes (f16 d + f16 dmin + 12 B
// packed 6-bit sub-scales/mins + 128 B nibbles), dequantized in-kernel as
// y = d·sc6·nibble − dmin·min6 per 32-element sub-block. At 0.5625 B/weight it reads
// 25% fewer bytes than ResidentBQ4 (0.75) with far better accuracy (super-block-scaled
// 6-bit affine per sub-block vs one f32 scale+min per 32) — the format llama.cpp's
// Q4_K_M mixes use for their bulk tensors, so weights from a Q4_K_M GGUF upload AS-IS
// with zero requantization loss. K must be a multiple of 256. DECODE-ONLY (GEMV).
//
// Quantization stays out of this package (layering): produce blocks either by reading
// a Q4_K tensor from a GGUF (gguf.ReadRaw → QuantTensor.Data, already row-major per
// output row) or by encoding f32 weights with gguf.Quantize(w, gguf.Q4_K) on the
// [N,K] (output-major) orientation.
type ResidentBQ4K struct {
	q    unsafe.Pointer // device bytes, row n = sbs consecutive 144-byte super-blocks
	k, n int
	sbs  int // super-blocks per row = K/256
}

// NewResidentBQ4KFromBlocks uploads pre-encoded Q4_K blocks for a weight with N output
// rows of K inputs each (raw layout: row-major [N][K/256] super-blocks — exactly the
// byte order of a gguf [out,in] Q4_K tensor, and of gguf.Quantize on an [N,K] tensor).
func NewResidentBQ4KFromBlocks(raw []byte, k, n int) (*ResidentBQ4K, error) {
	if k%256 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBQ4KFromBlocks needs K%%256==0, got K=%d", k)
	}
	sbs := k / 256
	if want := n * sbs * 144; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBQ4KFromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&raw[0])), C.int(len(raw)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: Q4_K weight upload failed")
	}
	return &ResidentBQ4K{q: dq, k: k, n: n, sbs: sbs}, nil
}

// QMatMulInto computes out = a·dequant(W) into the caller's fixed buffer (beta=0).
func (r *ResidentBQ4K) QMatMulInto(a, out *DeviceF32) error {
	return r.qmatmul(a, out, 0)
}

// QMatMulSwiGLUInto computes out = silu(gate) ⊙ (a·dequant(W)) — see
// ResidentBQ8.QMatMulSwiGLUInto (Tw55 epilogue fusion).
func (r *ResidentBQ4K) QMatMulSwiGLUInto(a, gate, out *DeviceF32) error {
	if r.q == nil || a.ptr == nil || gate.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: Q4_K QMatMulSwiGLUInto on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n || gate.rows != out.rows || gate.cols != out.cols {
		return fmt.Errorf("cuda: Q4_K QMatMulSwiGLUInto shape a[%d,%d]·B[%d,%d] gate[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, gate.rows, gate.cols, out.rows, out.cols)
	}
	if rc := C.cu_qmatmul_q4k_swiglu(a.ptr, r.q, gate.ptr, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n)); rc != 0 {
		return fmt.Errorf("cuda: Q4_K swiglu matmul failed (code %d)", int(rc))
	}
	return nil
}

// QMatMulAccInto computes c += a·dequant(W) in place (beta=1) — fuses a transformer
// residual add into the projection (the o and down projections).
func (r *ResidentBQ4K) QMatMulAccInto(a, c *DeviceF32) error {
	return r.qmatmul(a, c, 1)
}

func (r *ResidentBQ4K) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: Q4_K matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: Q4_K matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	// M>=2 (prefill / small-batch / speculative decode): the per-(m,n) GEMV re-reads each
	// Route to the weight-read-once M-tiled GEMM (bit-identical arithmetic) — weight BW M/MT×
	// lower, the dominant cost in this weight-BW-bound path. M==1 decode stays on the GEMV.
	if a.rows >= 2 {
		if rc := C.cu_qmatmul_q4k_mt(a.ptr, r.q, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
			return fmt.Errorf("cuda: Q4_K m-tiled matmul failed (code %d)", int(rc))
		}
		return nil
	}
	if rc := C.cu_qmatmul_q4k(a.ptr, r.q, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: Q4_K matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident Q4_K weight.
func (r *ResidentBQ4K) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
}

// Close frees the resident weight, satisfying the llamagpu qweight interface (so a Q4_K resident can be
// carried by a quantLinear alongside Q8, dispatched by the recorder's QMatMulResidentQ4K).
func (r *ResidentBQ4K) Close() error {
	r.Free()
	return nil
}
