//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"unsafe"
)

// ResidentBQ3K is a weight matrix in ggml's Q3_K super-block format (§R103) made
// GPU-resident. The gguf on-disk super-block is 110 bytes (hmask[32], qs[64],
// scales[12] = 16 signed 6-bit sub-scales via the aux/kmask splice, f16 d), but that
// layout is not 4-aligned and makes the kernel byte-read qs/hmask and re-run the
// ~15-op scale splice per super-block per lane. So at upload we REPACK each super-block
// into three aligned regions: meta (16 PRE-DECODED 6-bit scales + f16 d = 18 B), qs
// (64 B), hmask (32 B) — the kernel then reads a scale byte directly and pulls qs/hmask
// as coalesced uints. SYMMETRIC (no min): y = d·(sc6−32)·(q3−4), q3 = 2 low bits | high
// bit<<2. K%256==0. DECODE-ONLY (GEMV).
type ResidentBQ3K struct {
	meta unsafe.Pointer // N*sbs*18 bytes: 16 scale bytes + f16 d, per super-block
	qs   unsafe.Pointer // N*sbs*64 bytes: qs plane, 64-aligned per super-block
	hm   unsafe.Pointer // N*sbs*32 bytes: hmask plane, 32-aligned per super-block
	k, n int
	sbs  int // super-blocks per row = K/256
}

const q3kBlockBytes = 110

// q3kUnpackScales replicates ggml's dequantize_row_q3_K aux/kmask splice: 12 packed
// scale bytes → 16 six-bit values (indexed sc[is] = byte is&3 of aux[is>>2]).
func q3kUnpackScales(sc12 []byte) [16]byte {
	const km1, km2 = uint32(0x03030303), uint32(0x0f0f0f0f)
	a0 := binary.LittleEndian.Uint32(sc12[0:])
	a1 := binary.LittleEndian.Uint32(sc12[4:])
	a2 := binary.LittleEndian.Uint32(sc12[8:])
	tmp := a2
	aux := [4]uint32{
		(a0 & km2) | (((tmp >> 0) & km1) << 4),
		(a1 & km2) | (((tmp >> 2) & km1) << 4),
		((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4),
		((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4),
	}
	var out [16]byte
	for k := 0; k < 4; k++ {
		binary.LittleEndian.PutUint32(out[k*4:], aux[k])
	}
	return out
}

// NewResidentBQ3KFromBlocks repacks + uploads pre-encoded Q3_K blocks for a weight with N
// output rows of K inputs each (raw layout: row-major [N][K/256] 110-byte super-blocks).
func NewResidentBQ3KFromBlocks(raw []byte, k, n int) (*ResidentBQ3K, error) {
	if k%256 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBQ3KFromBlocks needs K%%256==0, got K=%d", k)
	}
	sbs := k / 256
	ns := n * sbs
	if want := ns * q3kBlockBytes; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBQ3KFromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	meta := make([]byte, ns*18)
	qs := make([]byte, ns*64)
	hm := make([]byte, ns*32)
	for i := 0; i < ns; i++ {
		blk := raw[i*110 : i*110+110] // hmask[0:32], qs[32:96], scales[96:108], d[108:110]
		copy(hm[i*32:i*32+32], blk[0:32])
		copy(qs[i*64:i*64+64], blk[32:96])
		scv := q3kUnpackScales(blk[96:108])
		copy(meta[i*18:i*18+16], scv[:])
		meta[i*18+16] = blk[108]
		meta[i*18+17] = blk[109]
	}
	dMeta := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&meta[0])), C.int(len(meta)))
	if dMeta == nil {
		return nil, fmt.Errorf("cuda: Q3_K meta upload failed")
	}
	dQs := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&qs[0])), C.int(len(qs)))
	if dQs == nil {
		C.cu_free_f32(dMeta)
		return nil, fmt.Errorf("cuda: Q3_K qs upload failed")
	}
	dHm := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&hm[0])), C.int(len(hm)))
	if dHm == nil {
		C.cu_free_f32(dMeta)
		C.cu_free_f32(dQs)
		return nil, fmt.Errorf("cuda: Q3_K hmask upload failed")
	}
	return &ResidentBQ3K{meta: dMeta, qs: dQs, hm: dHm, k: k, n: n, sbs: sbs}, nil
}

// QMatMulInto computes out = a·dequant(W) into the caller's fixed buffer (beta=0).
func (r *ResidentBQ3K) QMatMulInto(a, out *DeviceF32) error {
	return r.qmatmul(a, out, 0)
}

// QMatMulAccInto computes c += a·dequant(W) in place (beta=1) — fuses a transformer
// residual add into the projection (the o and down projections).
func (r *ResidentBQ3K) QMatMulAccInto(a, c *DeviceF32) error {
	return r.qmatmul(a, c, 1)
}

func (r *ResidentBQ3K) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.meta == nil || r.qs == nil || r.hm == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: Q3_K matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: Q3_K matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	if rc := C.cu_qmatmul_q3k(a.ptr, r.meta, r.qs, r.hm, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: Q3_K matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident Q3_K weight (all three repacked buffers).
func (r *ResidentBQ3K) Free() {
	if r.meta != nil {
		C.cu_free_f32(r.meta)
		r.meta = nil
	}
	if r.qs != nil {
		C.cu_free_f32(r.qs)
		r.qs = nil
	}
	if r.hm != nil {
		C.cu_free_f32(r.hm)
		r.hm = nil
	}
}
