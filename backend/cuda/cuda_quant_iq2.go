//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"sync"
	"unsafe"

	"github.com/jxsl13/goai/format/gguf"
)

// ResidentBIQ2XXS is a weight matrix in ggml's IQ2_XXS format (§T554) made GPU-resident: the
// first i-quant with a GRID codebook (an E8-lattice-derived 256×8 table) rather than a scalar
// codebook. Per output row, K/256 super-blocks of 66 bytes (f16 d + 8 (qs0,qs1) u32 pairs);
// each pair decodes 32 values as 4 groups of 8 — qs0 bytes index the grid, qs1 bits index the
// ksigns sign table + a 4-bit scale. 2.06 bits/weight — the smallest i-quant, used to run very
// large models (70B+) on tight VRAM. The 256×8 grid is reconstructed once (host-side, via the
// public gguf.Dequantize — no gguf internals needed) and uploaded to a shared device buffer.
// K%256==0. DECODE-ONLY (GEMV).
type ResidentBIQ2XXS struct {
	q    unsafe.Pointer // device bytes, row n = sbs consecutive 66-byte super-blocks
	k, n int
	sbs  int // K/256
}

const iq2xxsBlockBytes = 66

var (
	iq2GridOnce sync.Once
	iq2GridDev  unsafe.Pointer // shared [256*8]f32 device buffer
	iq2GridErr  error
)

// iq2xxsGrid returns the shared device grid, reconstructing + uploading it on first use.
// Reconstruction: the grid is recovered through the public gguf.Dequantize by decoding crafted
// blocks whose ksigns index is 0 (all-positive) and scale is known, so grid[idx][k] =
// dequant / (d·(0.5+s)·0.25). Verified exact against the format elsewhere (round-trip maxAbs 0).
func iq2xxsGrid() (unsafe.Pointer, error) {
	iq2GridOnce.Do(func() {
		const s = 15         // scale nibble → db = d·(0.5+s)·0.25
		const dbits = 0x3C00 // d = 1.0 as f16
		db := float32(1.0) * (0.5 + float32(s)) * 0.25
		nBlk := 256 / 32 // 8 blocks × 32 grid indices = 256
		raw := make([]byte, nBlk*66)
		for b := 0; b < nBlk; b++ {
			blk := raw[b*66 : b*66+66]
			binary.LittleEndian.PutUint16(blk[0:], dbits)
			for pair := 0; pair < 8; pair++ {
				var qs0 uint32
				for g := 0; g < 4; g++ {
					idx := b*32 + pair*4 + g
					qs0 |= uint32(byte(idx)) << (8 * g)
				}
				binary.LittleEndian.PutUint32(blk[2+pair*8:], qs0)
				binary.LittleEndian.PutUint32(blk[2+pair*8+4:], uint32(s)<<28) // ksigns idx 0
			}
		}
		deq, err := gguf.Dequantize(raw, gguf.IQ2_XXS, nBlk*256)
		if err != nil {
			iq2GridErr = fmt.Errorf("cuda: IQ2_XXS grid reconstruct: %w", err)
			return
		}
		df := deq.Storage().F32()
		grid := make([]float32, 256*8)
		for b := 0; b < nBlk; b++ {
			for pair := 0; pair < 8; pair++ {
				for g := 0; g < 4; g++ {
					idx := b*32 + pair*4 + g
					base := b*256 + pair*32 + g*8
					//perfscan:ignore PS5001 one-time IQ2 grid reconstruct (sync.Once)
					for k := 0; k < 8; k++ {
						grid[idx*8+k] = df[base+k] / db
					}
				}
			}
		}
		iq2GridDev = C.cu_upload_f32((*C.float)(&grid[0]), C.int(len(grid)))
		if iq2GridDev == nil {
			iq2GridErr = fmt.Errorf("cuda: IQ2_XXS grid upload failed")
		}
	})
	return iq2GridDev, iq2GridErr
}

// NewResidentBIQ2XXSFromBlocks uploads pre-encoded IQ2_XXS blocks (row-major [N][K/256]).
func NewResidentBIQ2XXSFromBlocks(raw []byte, k, n int) (*ResidentBIQ2XXS, error) {
	if k%256 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBIQ2XXSFromBlocks needs K%%256==0, got K=%d", k)
	}
	if _, err := iq2xxsGrid(); err != nil {
		return nil, err
	}
	sbs := k / 256
	if want := n * sbs * iq2xxsBlockBytes; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBIQ2XXSFromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&raw[0])), C.int(len(raw)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: IQ2_XXS weight upload failed")
	}
	return &ResidentBIQ2XXS{q: dq, k: k, n: n, sbs: sbs}, nil
}

// QMatMulInto computes out = a·dequant(W) into the caller's fixed buffer (beta=0).
func (r *ResidentBIQ2XXS) QMatMulInto(a, out *DeviceF32) error { return r.qmatmul(a, out, 0) }

// QMatMulAccInto computes c += a·dequant(W) in place (beta=1).
func (r *ResidentBIQ2XXS) QMatMulAccInto(a, c *DeviceF32) error { return r.qmatmul(a, c, 1) }

func (r *ResidentBIQ2XXS) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: IQ2_XXS matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: IQ2_XXS matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	grid, err := iq2xxsGrid()
	if err != nil {
		return err
	}
	// M>1 (prefill/batch): route to the weight-read-once M-tiled GEMM (bit-identical) so column
	// n's IQ2_XXS block is grid-decoded once, not re-read per row. M==1 decode stays on the GEMV.
	if a.rows >= 2 { // M>1: weight-read-once MT wins from M=2 (matches Q4_K/Q5_K/Q6_K)
		if rc := C.cu_qmatmul_iq2xxs_mt(a.ptr, r.q, grid, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
			return fmt.Errorf("cuda: IQ2_XXS m-tiled matmul failed (code %d)", int(rc))
		}
		return nil
	}
	if rc := C.cu_qmatmul_iq2xxs(a.ptr, r.q, grid, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: IQ2_XXS matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident IQ2_XXS weight (the shared grid buffer persists).
func (r *ResidentBIQ2XXS) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
}

// ResidentBIQ2XS is ggml's IQ2_XS (§T554): the 2.31-bit sibling of IQ2_XXS with a 512-entry grid
// and EXPLICIT per-16-element 4-bit scales. Per 74-byte super-block: f16 d + 32 u16 qs (low 9 bits
// = grid index, high 7 = ksigns index; each word = 8 elements) + 8 bytes = 16 four-bit scales.
// K%256==0. DECODE-ONLY (GEMV).
type ResidentBIQ2XS struct {
	q    unsafe.Pointer
	k, n int
	sbs  int // K/256
}

const iq2xsBlockBytes = 74

var (
	iq2xsGridOnce sync.Once
	iq2xsGridDev  unsafe.Pointer // shared [512*8]f32 device buffer
	iq2xsGridErr  error
)

// iq2xsGrid reconstructs + uploads IQ2_XS's 512×8 grid via the public gguf.Dequantize (crafted
// blocks: qs word = grid index with ksigns index 0, all scales = 15 → grid[idx][k] = dequant/db).
func iq2xsGrid() (unsafe.Pointer, error) {
	iq2xsGridOnce.Do(func() {
		const sc = 15
		db := float32(1.0) * (0.5 + float32(sc)) * 0.25
		nBlk := 512 / 32 // 16 blocks × 32 words = 512 grid entries
		raw := make([]byte, nBlk*74)
		for b := 0; b < nBlk; b++ {
			blk := raw[b*74 : b*74+74]
			binary.LittleEndian.PutUint16(blk[0:], 0x3C00) // d = 1.0
			for l := 0; l < 32; l++ {
				binary.LittleEndian.PutUint16(blk[2+l*2:], uint16(b*32+l)) // grid idx, ksigns 0
			}
			for j := 0; j < 8; j++ {
				blk[66+j] = 0xFF // both 4-bit scales = 15
			}
		}
		deq, err := gguf.Dequantize(raw, gguf.IQ2_XS, nBlk*256)
		if err != nil {
			iq2xsGridErr = fmt.Errorf("cuda: IQ2_XS grid reconstruct: %w", err)
			return
		}
		df := deq.Storage().F32()
		grid := make([]float32, 512*8)
		for b := 0; b < nBlk; b++ {
			for l := 0; l < 32; l++ {
				idx := b*32 + l
				base := b*256 + l*8
				//perfscan:ignore PS5001 one-time IQ2 grid reconstruct (sync.Once)
				for k := 0; k < 8; k++ {
					grid[idx*8+k] = df[base+k] / db
				}
			}
		}
		iq2xsGridDev = C.cu_upload_f32((*C.float)(&grid[0]), C.int(len(grid)))
		if iq2xsGridDev == nil {
			iq2xsGridErr = fmt.Errorf("cuda: IQ2_XS grid upload failed")
		}
	})
	return iq2xsGridDev, iq2xsGridErr
}

// NewResidentBIQ2XSFromBlocks uploads pre-encoded IQ2_XS blocks (row-major [N][K/256]).
func NewResidentBIQ2XSFromBlocks(raw []byte, k, n int) (*ResidentBIQ2XS, error) {
	if k%256 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBIQ2XSFromBlocks needs K%%256==0, got K=%d", k)
	}
	if _, err := iq2xsGrid(); err != nil {
		return nil, err
	}
	sbs := k / 256
	if want := n * sbs * iq2xsBlockBytes; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBIQ2XSFromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&raw[0])), C.int(len(raw)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: IQ2_XS weight upload failed")
	}
	return &ResidentBIQ2XS{q: dq, k: k, n: n, sbs: sbs}, nil
}

// QMatMulInto computes out = a·dequant(W) into the caller's fixed buffer (beta=0).
func (r *ResidentBIQ2XS) QMatMulInto(a, out *DeviceF32) error { return r.qmatmul(a, out, 0) }

// QMatMulAccInto computes c += a·dequant(W) in place (beta=1).
func (r *ResidentBIQ2XS) QMatMulAccInto(a, c *DeviceF32) error { return r.qmatmul(a, c, 1) }

func (r *ResidentBIQ2XS) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: IQ2_XS matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: IQ2_XS matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	grid, err := iq2xsGrid()
	if err != nil {
		return err
	}
	if rc := C.cu_qmatmul_iq2xs(a.ptr, r.q, grid, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: IQ2_XS matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident IQ2_XS weight (the shared grid buffer persists).
func (r *ResidentBIQ2XS) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
}
