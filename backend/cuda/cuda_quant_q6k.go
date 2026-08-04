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

// QMatMulWMMAInto computes out[M,N] = a[M,K]·dequant(W) on TENSOR CORES: it dequantizes the
// Q6_K weight to a contiguous f16 [K,N] matrix once, converts the f32 activation to f16, and runs
// the f16 WMMA GEMM — replacing the scalar acc GEMV (cu_qmatmul_q6k_mt) for compute-bound PREFILL
// (M large; the O(K·N) dequant amortizes over M). Result rides the incumbent f16-accum tolerance
// vs the scalar path. Requires M,N,K %16==0 (K%256==0 already holds). beta=0 (overwrite).
func (r *ResidentBQ6K) QMatMulWMMAInto(a, out *DeviceF32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: Q6_K WMMA matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: Q6_K WMMA matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	m := a.rows
	if m%16 != 0 || r.n%16 != 0 || r.k%16 != 0 {
		return fmt.Errorf("cuda: Q6_K WMMA matmul needs M,N,K %%16==0 (got M=%d N=%d K=%d)", m, r.n, r.k)
	}
	bf16 := C.cu_alloc_u16(C.int(r.k * r.n)) // dequantized weight [K,N] f16
	if bf16 == nil {
		return fmt.Errorf("cuda: Q6_K WMMA weight scratch alloc failed")
	}
	defer C.cu_free_f32(bf16)
	if rc := C.cu_dequant_q6k_to_f16(r.q, bf16, C.int(r.k), C.int(r.n)); rc != 0 {
		return fmt.Errorf("cuda: Q6_K dequant-to-f16 failed (code %d)", int(rc))
	}
	// cuBLAS f16 tensor-core GEMM (cublasGemmEx, f16 in / f32 accum) — ~2.2x the hand
	// cu_wmma_gemm on prefill shapes; converts the activation internally (#906).
	if rc := C.cu_matmul_f16w(a.ptr, bf16, out.ptr, C.int(m), C.int(r.k), C.int(r.n), C.float(0)); rc != 0 {
		return fmt.Errorf("cuda: Q6_K cuBLAS f16 gemm failed (code %d)", int(rc))
	}
	return nil
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
	// M>1 (prefill/batch): route to the weight-read-once M-tiled GEMM (bit-identical) so column
	// n's Q6_K block is decoded once, not re-read per row. M==1 decode stays on the GEMV.
	if a.rows >= 2 {
		if rc := C.cu_qmatmul_q6k_mt(a.ptr, r.q, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
			return fmt.Errorf("cuda: Q6_K m-tiled matmul failed (code %d)", int(rc))
		}
		return nil
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

// Close frees the resident weight, satisfying the llamagpu qweight interface (so a Q6_K resident from
// a native Q6_K GGUF upload is dispatched by the recorder's QMatMulResidentQ6K alongside Q8/Q4_K).
func (r *ResidentBQ6K) Close() error {
	r.Free()
	return nil
}
