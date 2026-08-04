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

// QMatMulWMMAInto computes out[M,N] = a[M,K]·dequant(W) on TENSOR CORES: it dequantizes the
// Q5_K weight to a contiguous f16 [K,N] matrix once, converts the f32 activation to f16, and runs
// the f16 WMMA GEMM — replacing the scalar acc GEMV (cu_qmatmul_q5k) for compute-bound PREFILL
// (M large; the O(K·N) dequant amortizes over M). Result rides the incumbent f16-accum tolerance
// vs the scalar path. Requires M,N,K %16==0 (K%256==0 already holds). beta=0 (overwrite).
func (r *ResidentBQ5K) QMatMulWMMAInto(a, out *DeviceF32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: Q5_K WMMA matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: Q5_K WMMA matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	m := a.rows
	if m%16 != 0 || r.n%16 != 0 || r.k%16 != 0 {
		return fmt.Errorf("cuda: Q5_K WMMA matmul needs M,N,K %%16==0 (got M=%d N=%d K=%d)", m, r.n, r.k)
	}
	bf16 := C.cu_alloc_u16(C.int(r.k * r.n)) // dequantized weight [K,N] f16
	if bf16 == nil {
		return fmt.Errorf("cuda: Q5_K WMMA weight scratch alloc failed")
	}
	defer C.cu_free_f32(bf16)
	af16 := C.cu_alloc_u16(C.int(m * r.k)) // activation [M,K] f16
	if af16 == nil {
		return fmt.Errorf("cuda: Q5_K WMMA activation scratch alloc failed")
	}
	defer C.cu_free_f32(af16)
	if rc := C.cu_dequant_q5k_to_f16(r.q, bf16, C.int(r.k), C.int(r.n)); rc != 0 {
		return fmt.Errorf("cuda: Q5_K dequant-to-f16 failed (code %d)", int(rc))
	}
	if rc := C.cu_cvt_f32_to_f16(af16, a.ptr, C.long(m*r.k)); rc != 0 {
		return fmt.Errorf("cuda: Q5_K WMMA activation convert failed (code %d)", int(rc))
	}
	rc := C.cu_wmma_gemm(unsafe.Pointer(&wmmaGemmFatbin[0]), C.int(len(wmmaGemmFatbin)),
		af16, bf16, out.ptr, C.int(m), C.int(r.k), C.int(r.n))
	if rc != 0 {
		return fmt.Errorf("cuda: Q5_K WMMA gemm failed (code %d)", int(rc))
	}
	return nil
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

// Close frees the resident weight, satisfying the llamagpu qweight interface (so a Q5_K resident from
// a native Q5_K GGUF upload is dispatched by the recorder's QMatMulResidentQ5K alongside Q8/Q4_K/Q6_K).
func (r *ResidentBQ5K) Close() error {
	r.Free()
	return nil
}
