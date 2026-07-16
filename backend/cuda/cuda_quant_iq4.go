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

// ResidentBIQ4NL is a weight matrix in ggml's IQ4_NL format made GPU-resident: per
// output row, K/32 blocks of 18 bytes (f16 d + 16 nibble bytes). Each 4-bit nibble
// indexes a fixed NONLINEAR 16-value codebook (denser near zero) rather than the
// affine grid of Q4_0/Q4_K: y = d·kvals[nibble]. The first of the i-quant (codebook)
// family on the CUDA path. K%32==0. DECODE-ONLY (GEMV).
type ResidentBIQ4NL struct {
	q    unsafe.Pointer
	k, n int
	nblk int // K/32
}

// ResidentBIQ4XS is IQ4_XS: K/256 super-blocks of 136 bytes (f16 d, u16 scales_h,
// 4 scales_l bytes, 128 qs) — 8 sub-blocks of 32 with a signed 6-bit sub-scale each,
// nibbles indexing the same nonlinear codebook. The popular quality-4-bit i-quant.
// K%256==0. DECODE-ONLY (GEMV).
type ResidentBIQ4XS struct {
	q    unsafe.Pointer
	k, n int
	sbs  int // K/256
}

const (
	iq4nlBlockBytes = 18
	iq4xsBlockBytes = 136
)

// NewResidentBIQ4NLFromBlocks uploads pre-encoded IQ4_NL blocks (row-major [N][K/32]).
func NewResidentBIQ4NLFromBlocks(raw []byte, k, n int) (*ResidentBIQ4NL, error) {
	if k%32 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBIQ4NLFromBlocks needs K%%32==0, got K=%d", k)
	}
	nblk := k / 32
	if want := n * nblk * iq4nlBlockBytes; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBIQ4NLFromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&raw[0])), C.int(len(raw)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: IQ4_NL weight upload failed")
	}
	return &ResidentBIQ4NL{q: dq, k: k, n: n, nblk: nblk}, nil
}

// NewResidentBIQ4XSFromBlocks uploads pre-encoded IQ4_XS blocks (row-major [N][K/256]).
func NewResidentBIQ4XSFromBlocks(raw []byte, k, n int) (*ResidentBIQ4XS, error) {
	if k%256 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBIQ4XSFromBlocks needs K%%256==0, got K=%d", k)
	}
	sbs := k / 256
	if want := n * sbs * iq4xsBlockBytes; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBIQ4XSFromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&raw[0])), C.int(len(raw)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: IQ4_XS weight upload failed")
	}
	return &ResidentBIQ4XS{q: dq, k: k, n: n, sbs: sbs}, nil
}

func (r *ResidentBIQ4NL) QMatMulInto(a, out *DeviceF32) error  { return r.qmatmul(a, out, 0) }
func (r *ResidentBIQ4NL) QMatMulAccInto(a, c *DeviceF32) error { return r.qmatmul(a, c, 1) }
func (r *ResidentBIQ4XS) QMatMulInto(a, out *DeviceF32) error  { return r.qmatmul(a, out, 0) }
func (r *ResidentBIQ4XS) QMatMulAccInto(a, c *DeviceF32) error { return r.qmatmul(a, c, 1) }

func (r *ResidentBIQ4NL) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: IQ4_NL matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: IQ4_NL matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	if rc := C.cu_qmatmul_iq4nl(a.ptr, r.q, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: IQ4_NL matmul failed (code %d)", int(rc))
	}
	return nil
}

func (r *ResidentBIQ4XS) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: IQ4_XS matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: IQ4_XS matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	if rc := C.cu_qmatmul_iq4xs(a.ptr, r.q, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: IQ4_XS matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident IQ4_NL weight.
func (r *ResidentBIQ4NL) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
}

// Free releases the resident IQ4_XS weight.
func (r *ResidentBIQ4XS) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
}
