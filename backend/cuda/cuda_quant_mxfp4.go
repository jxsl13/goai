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

// ResidentBMXFP4 is a weight matrix in ggml's MXFP4 format (OCP Microscaling FP4 — the
// format gpt-oss ships in) made GPU-resident: per output row, K/32 blocks of 17 bytes
// (1 E8M0 scale byte + 16 nibble bytes). Each 4-bit nibble indexes the FP4 E2M1 codebook
// (bit 3 = sign): y = e8m0(scale)·mxfp4kv[nibble]. Byte i carries element i (low nibble)
// and i+16 (high). Distinct from IQ4 (a different codebook + a byte-exponent block scale).
// K%32==0. DECODE-ONLY (GEMV).
type ResidentBMXFP4 struct {
	scale unsafe.Pointer // N*nblk E8M0 scale bytes, block-major per row
	nib   unsafe.Pointer // N*nblk*16 nibble bytes, 16-aligned per block
	k, n  int
	nblk  int // K/32
}

const mxfp4BlockBytes = 17

// NewResidentBMXFP4FromBlocks repacks + uploads pre-encoded MXFP4 blocks (row-major [N][K/32]
// 17-byte blocks) into two aligned regions — scale (nblk E8M0 bytes/row) + nib (nblk×16 nibble
// bytes/row, 16-aligned) — for the coalesced-read GEMV.
func NewResidentBMXFP4FromBlocks(raw []byte, k, n int) (*ResidentBMXFP4, error) {
	if k%32 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBMXFP4FromBlocks needs K%%32==0, got K=%d", k)
	}
	nblk := k / 32
	nb := n * nblk
	if want := nb * mxfp4BlockBytes; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBMXFP4FromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	scales := make([]byte, nb)  // 1 E8M0 byte per block
	nibs := make([]byte, nb*16) // 16 nibble bytes per block
	for i := 0; i < nb; i++ {
		blk := raw[i*17 : i*17+17]
		scales[i] = blk[0]
		copy(nibs[i*16:i*16+16], blk[1:17])
	}
	dScale := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&scales[0])), C.int(len(scales)))
	if dScale == nil {
		return nil, fmt.Errorf("cuda: MXFP4 scale upload failed")
	}
	dNib := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&nibs[0])), C.int(len(nibs)))
	if dNib == nil {
		C.cu_free_f32(dScale)
		return nil, fmt.Errorf("cuda: MXFP4 nibble upload failed")
	}
	return &ResidentBMXFP4{scale: dScale, nib: dNib, k: k, n: n, nblk: nblk}, nil
}

// QMatMulInto computes out = a·dequant(W) into the caller's fixed buffer (beta=0).
func (r *ResidentBMXFP4) QMatMulInto(a, out *DeviceF32) error { return r.qmatmul(a, out, 0) }

// QMatMulAccInto computes c += a·dequant(W) in place (beta=1).
func (r *ResidentBMXFP4) QMatMulAccInto(a, c *DeviceF32) error { return r.qmatmul(a, c, 1) }

func (r *ResidentBMXFP4) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.scale == nil || r.nib == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: MXFP4 matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: MXFP4 matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	// M>1 (prefill/batch): route to the weight-read-once M-tiled GEMM (bit-identical) so column
	// n's MXFP4 blocks are decoded once, not re-read per row. M==1 decode stays on the GEMV.
	if a.rows >= 8 {
		if rc := C.cu_qmatmul_mxfp4_mt(a.ptr, r.scale, r.nib, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
			return fmt.Errorf("cuda: MXFP4 m-tiled matmul failed (code %d)", int(rc))
		}
		return nil
	}
	if rc := C.cu_qmatmul_mxfp4(a.ptr, r.scale, r.nib, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: MXFP4 matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident MXFP4 weight (both repacked buffers).
func (r *ResidentBMXFP4) Free() {
	if r.scale != nil {
		C.cu_free_f32(r.scale)
		r.scale = nil
	}
	if r.nib != nil {
		C.cu_free_f32(r.nib)
		r.nib = nil
	}
}
