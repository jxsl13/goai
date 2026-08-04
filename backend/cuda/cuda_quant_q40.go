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

// ResidentBQ40 is a weight matrix in ggml's legacy Q4_0 format made GPU-resident. The gguf
// on-disk block is 18 bytes (f16 d + 16 nibble bytes, 32 quants), but that granularity is not
// 4-aligned and forces ~8× the memory transactions of a Q4_K super-block at identical bytes
// (transactions-not-bytes). So at upload we REPACK each output row into two aligned regions:
// scale (nblk f16 block scales, contiguous) and nib (nblk×16 nibble bytes, block-major,
// 16-aligned), which lets the GEMV read nibbles as coalesced uints (8 blocks/iteration, 4
// lanes/block) — ~2× the naive per-block kernel. SYMMETRIC round quant: y = d·(nibble−8).
// K must be a multiple of 32. DECODE-ONLY (GEMV).
type ResidentBQ40 struct {
	scale unsafe.Pointer // N*nblk f16 (2 bytes each), block-major per row
	nib   unsafe.Pointer // N*nblk*16 nibble bytes, 16-aligned per block
	k, n  int
	nblk  int // blocks per row = K/32
}

const q40BlockBytes = 18

// NewResidentBQ40FromBlocks repacks + uploads pre-encoded Q4_0 blocks for a weight with N
// output rows of K inputs each (raw layout: row-major [N][K/32] 18-byte blocks — the byte
// order of a gguf [out,in] Q4_0 tensor, and of gguf.Quantize on an [N,K] tensor).
func NewResidentBQ40FromBlocks(raw []byte, k, n int) (*ResidentBQ40, error) {
	if k%32 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBQ40FromBlocks needs K%%32==0, got K=%d", k)
	}
	nblk := k / 32
	nb := n * nblk // total blocks
	if want := nb * q40BlockBytes; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBQ40FromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	// repack: [block][2 scale | 16 nibbles]  →  scales[nb*2] + nibbles[nb*16]
	scales := make([]byte, nb*2)
	nibs := make([]byte, nb*16)
	for i := 0; i < nb; i++ {
		blk := raw[i*18 : i*18+18]
		copy(scales[i*2:i*2+2], blk[0:2])
		copy(nibs[i*16:i*16+16], blk[2:18])
	}
	dScale := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&scales[0])), C.int(len(scales)))
	if dScale == nil {
		return nil, fmt.Errorf("cuda: Q4_0 scale upload failed")
	}
	dNib := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&nibs[0])), C.int(len(nibs)))
	if dNib == nil {
		C.cu_free_f32(dScale)
		return nil, fmt.Errorf("cuda: Q4_0 nibble upload failed")
	}
	return &ResidentBQ40{scale: dScale, nib: dNib, k: k, n: n, nblk: nblk}, nil
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

// QMatMulWMMAInto: tensor-core Q4_0 prefill (dequant Q4_0 -> f16 [K,N] + f16 WMMA GEMM),
// replacing the scalar acc GEMV for compute-bound M. Rides f16-accum tolerance. M,N %16==0.
func (r *ResidentBQ40) QMatMulWMMAInto(a, out *DeviceF32) error {
	if r.scale == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: Q4_0 WMMA matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: Q4_0 WMMA matmul shape a[%d,%d]-B[%d,%d]->out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	m := a.rows
	if m%16 != 0 || r.n%16 != 0 || r.k%16 != 0 {
		return fmt.Errorf("cuda: Q4_0 WMMA matmul needs M,N,K %%16==0 (got M=%d N=%d K=%d)", m, r.n, r.k)
	}
	bf16 := C.cu_alloc_u16(C.int(r.k * r.n))
	if bf16 == nil {
		return fmt.Errorf("cuda: Q4_0 WMMA weight scratch alloc failed")
	}
	defer C.cu_free_f32(bf16)
	af16 := C.cu_alloc_u16(C.int(m * r.k))
	if af16 == nil {
		return fmt.Errorf("cuda: Q4_0 WMMA activation scratch alloc failed")
	}
	defer C.cu_free_f32(af16)
	if rc := C.cu_dequant_q40_to_f16(r.scale, r.nib, bf16, C.int(r.k), C.int(r.n)); rc != 0 {
		return fmt.Errorf("cuda: Q4_0 dequant-to-f16 failed (code %d)", int(rc))
	}
	if rc := C.cu_cvt_f32_to_f16(af16, a.ptr, C.long(m*r.k)); rc != 0 {
		return fmt.Errorf("cuda: Q4_0 WMMA activation convert failed (code %d)", int(rc))
	}
	rc := C.cu_wmma_gemm(unsafe.Pointer(&wmmaGemmFatbin[0]), C.int(len(wmmaGemmFatbin)),
		af16, bf16, out.ptr, C.int(m), C.int(r.k), C.int(r.n))
	if rc != 0 {
		return fmt.Errorf("cuda: Q4_0 WMMA gemm failed (code %d)", int(rc))
	}
	return nil
}

func (r *ResidentBQ40) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.scale == nil || r.nib == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: Q4_0 matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: Q4_0 matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	if rc := C.cu_qmatmul_q40(a.ptr, r.scale, r.nib, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: Q4_0 matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident Q4_0 weight (both repacked buffers).
func (r *ResidentBQ40) Free() {
	if r.scale != nil {
		C.cu_free_f32(r.scale)
		r.scale = nil
	}
	if r.nib != nil {
		C.cu_free_f32(r.nib)
		r.nib = nil
	}
}
