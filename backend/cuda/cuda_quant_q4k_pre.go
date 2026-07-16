//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"
)

// ResidentBQ4KPre is a Q4_K weight repacked with PRE-DECODED sub-block scales — the applied
// form of perf-notes-cuda.md rule R5 (the Q4_K decode-GEMV ceiling is the per-block scale-decode
// ALU: f16 d/dmin decode + 6-bit unpack + shfl, NOT the multiply — measured in Tw44/Tw56/Tw58).
// At upload we precompute c1[j] = d·sc6[j] and c2[j] = dmin·min6[j] (the exact f32 products the
// kernel forms in-loop) for the 8 sub-blocks, and store them as a 64-byte f32 scale plane +
// the 128-byte nibbles = 192 B/block (vs 144). The GEMV then reads c1/c2 directly and drops the
// entire get_sm unpack + f16 decode + the two shfls.
//
// This is BIT-EXACT vs the standard Q4_K kernel by construction (identical f32 d·sc / dmin·min
// products, identical accumulation order) — the only tradeoff is +33% weight bytes on a
// bandwidth-bound path. Whether the ALU relief beats the extra bandwidth is a measure-first A/B
// (roofline says it can go either way); this type exists to run that A/B. OPT-IN, K%256==0.
type ResidentBQ4KPre struct {
	q    unsafe.Pointer // device bytes, row n = sbs consecutive 192-byte blocks (64B scale plane + 128B nibbles)
	k, n int
	sbs  int // K/256
}

// f16bitsToF32 converts an IEEE half to f32 with the exact bit sequence the kernel's f16f device
// helper uses (2^112 exponent rebias — exact for normals and subnormals), so the pre-decoded c1/c2
// match the in-kernel d/dmin decode bit-for-bit.
func f16bitsToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	v := math.Float32frombits(uint32(h&0x7fff)<<13) * math.Float32frombits(0x77800000)
	return math.Float32frombits(math.Float32bits(v) | sign)
}

// q4kScaleMin mirrors the kernel's get_sm: the ggml get_scale_min_k4 6-bit unpack of the 12
// packed scale bytes into sub-block j's 6-bit scale and min.
func q4kScaleMin(j int, q []byte) (sc, mn float32) {
	if j < 4 {
		sc = float32(q[j] & 63)
		mn = float32(q[j+4] & 63)
	} else {
		sc = float32((q[j+4] & 0xF) | ((q[j-4] >> 6) << 4))
		mn = float32((q[j+4] >> 4) | ((q[j] >> 6) << 4))
	}
	return
}

// NewResidentBQ4KPreFromBlocks repacks pre-encoded Q4_K blocks (row-major [N][K/256], 144 B each)
// into the pre-decoded 192-byte layout and uploads them. Scale-plane layout per block: 8 f32 pairs
// (c1[j], c2[j]) so a lane with pair p reads {c1[2p], c2[2p], c1[2p+1], c2[2p+1]} as one float4.
func NewResidentBQ4KPreFromBlocks(raw []byte, k, n int) (*ResidentBQ4KPre, error) {
	if k%256 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBQ4KPreFromBlocks needs K%%256==0, got K=%d", k)
	}
	sbs := k / 256
	if want := n * sbs * 144; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBQ4KPreFromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	out := make([]byte, n*sbs*192)
	for b := 0; b < n*sbs; b++ {
		src := raw[b*144 : b*144+144]
		dst := out[b*192 : b*192+192]
		d := f16bitsToF32(binary.LittleEndian.Uint16(src[0:2]))
		dmin := f16bitsToF32(binary.LittleEndian.Uint16(src[2:4]))
		scales := src[4:16]
		for j := 0; j < 8; j++ {
			sc, mn := q4kScaleMin(j, scales)
			binary.LittleEndian.PutUint32(dst[j*8:], math.Float32bits(d*sc))      // c1[j]
			binary.LittleEndian.PutUint32(dst[j*8+4:], math.Float32bits(dmin*mn)) // c2[j]
		}
		copy(dst[64:192], src[16:144]) // 128 B nibbles, unchanged
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&out[0])), C.int(len(out)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: Q4_K-pre weight upload failed")
	}
	return &ResidentBQ4KPre{q: dq, k: k, n: n, sbs: sbs}, nil
}

// QMatMulInto computes out = a·dequant(W) into the caller's fixed buffer (beta=0).
func (r *ResidentBQ4KPre) QMatMulInto(a, out *DeviceF32) error { return r.qmatmul(a, out, 0) }

// QMatMulAccInto computes c += a·dequant(W) in place (beta=1).
func (r *ResidentBQ4KPre) QMatMulAccInto(a, c *DeviceF32) error { return r.qmatmul(a, c, 1) }

func (r *ResidentBQ4KPre) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: Q4_K-pre matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: Q4_K-pre matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	if rc := C.cu_qmatmul_q4k_pre(a.ptr, r.q, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: Q4_K-pre matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident weight.
func (r *ResidentBQ4KPre) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
}
