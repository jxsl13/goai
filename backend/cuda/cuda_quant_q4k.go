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

// QMatMulMoeInto is the Q4_K twin of ResidentBQ8.QMatMulMoeInto: a token (output row) whose expert
// routing weight moeGate[row*gateStride+expertIdx] is 0 skips the dequant+dot K-loop and writes 0 —
// bit-identical to the dense eval + RowAxpy(weight=0), ~E/K faster for Q4_K MoE decode. Explicit m for
// the flat llamagpu scratch buffers. beta=0.
func (r *ResidentBQ4K) QMatMulMoeInto(a, out, moeGate *DeviceF32, m, gateStride, expertIdx int) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil || moeGate.ptr == nil {
		return fmt.Errorf("cuda: Q4_K QMatMulMoeInto on a freed handle")
	}
	gp := unsafe.Pointer(uintptr(moeGate.ptr) + uintptr(expertIdx)*4)
	if rc := C.cu_qmatmul_q4k_moe(a.ptr, r.q, out.ptr, C.int(m), C.int(r.k), C.int(r.n), gp, C.int(gateStride)); rc != 0 {
		return fmt.Errorf("cuda: Q4_K moe-gated matmul failed (code %d)", int(rc))
	}
	return nil
}

// QMatMulWMMAInto computes out[M,N] = a[M,K]·dequant(W) on TENSOR CORES: it dequantizes the
// Q4_K weight to a contiguous f16 [K,N] matrix once, converts the f32 activation to f16, and
// runs the f16 WMMA GEMM — replacing the scalar acc[8] GEMV (cu_qmatmul_q4k_mt) for compute-bound
// PREFILL (M large; the O(K·N) dequant amortizes over M). Result rides the incumbent f16-accum
// tolerance vs the scalar path. Requires M,N %16==0 (K%256==0 already holds). beta=0 (overwrite).
func (r *ResidentBQ4K) QMatMulWMMAInto(a, out *DeviceF32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: Q4_K WMMA matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: Q4_K WMMA matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	m := a.rows
	if m%16 != 0 || r.n%16 != 0 || r.k%16 != 0 {
		return fmt.Errorf("cuda: Q4_K WMMA matmul needs M,N,K %%16==0 (got M=%d N=%d K=%d)", m, r.n, r.k)
	}
	bf16 := C.cu_alloc_u16(C.int(r.k * r.n)) // dequantized weight [K,N] f16
	if bf16 == nil {
		return fmt.Errorf("cuda: Q4_K WMMA weight scratch alloc failed")
	}
	defer C.cu_free_f32(bf16)
	if rc := C.cu_dequant_q4k_to_f16(r.q, bf16, C.int(r.k), C.int(r.n)); rc != 0 {
		return fmt.Errorf("cuda: Q4_K dequant-to-f16 failed (code %d)", int(rc))
	}
	if rc := matmulF16wPrefill(a.ptr, bf16, out.ptr, m, r.k, r.n, 0); rc != 0 {
		return fmt.Errorf("cuda: Q4_K cuBLAS f16 gemm failed (code %d)", int(rc))
	}
	return nil
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

// QMatMulWMMAAccInto computes c += a·dequant(W) on TENSOR CORES (dequant Q4_K→f16 once, then cuBLAS f16
// GEMM with beta=1) — the accumulate/residual-fuse twin of the recorder's q4kPrefillWMMA, for the o and
// down projections in PREFILL (m large). Several× the scalar-MT QMatMulAccInto that qmatmul routes m>=2
// to (MT has no WMMA path), matching what the eager Decoder already does for these projections. Explicit
// m for the flat llamagpu scratch buffers (no shape check). Rides the incumbent f16-accum tolerance.
func (r *ResidentBQ4K) QMatMulWMMAAccInto(a, c *DeviceF32, m int) error {
	if r.q == nil || a.ptr == nil || c.ptr == nil {
		return fmt.Errorf("cuda: Q4_K WMMA acc matmul on a freed handle")
	}
	bf16 := C.cu_alloc_u16(C.int(r.k * r.n))
	if bf16 == nil {
		return fmt.Errorf("cuda: Q4_K WMMA acc weight scratch alloc failed")
	}
	defer C.cu_free_f32(bf16)
	if rc := C.cu_dequant_q4k_to_f16(r.q, bf16, C.int(r.k), C.int(r.n)); rc != 0 {
		return fmt.Errorf("cuda: Q4_K WMMA acc dequant failed (code %d)", int(rc))
	}
	if rc := matmulF16wPrefill(a.ptr, bf16, c.ptr, m, r.k, r.n, 1); rc != 0 {
		return fmt.Errorf("cuda: Q4_K WMMA acc gemm failed (code %d)", int(rc))
	}
	return nil
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

// qmatmulGEMVForBench / qmatmulMTForBench force a specific kernel regardless of m — used by
// the in-package crossover benchmark to find the recorder's MT routing threshold (cgo is not
// permitted in _test.go).
func (r *ResidentBQ4K) qmatmulGEMVForBench(a, out *DeviceF32) int {
	return int(C.cu_qmatmul_q4k(a.ptr, r.q, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(0)))
}
func (r *ResidentBQ4K) qmatmulMTForBench(a, out *DeviceF32) int {
	return int(C.cu_qmatmul_q4k_mt(a.ptr, r.q, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(0)))
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
