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

// ResidentBIQ3XXS is a weight matrix in ggml's IQ3_XXS format (§T554) made GPU-resident: the
// 3.06-bit i-quant, a 256×4 grid codebook (vs IQ2_XXS's 256×8). Per output row, K/256 super-blocks
// of 98 bytes: f16 d + 64 grid-index bytes + 8 uint32 sign/scale words. Each word decodes 32 values
// as 4 sub-groups of 8 — two grid bytes index the 256×4 grid, the word's low 28 bits hold 4 ksigns
// indices (7 bits each), the top 4 bits a scale s (db = d·(0.5+s)·0.5). The grid is reconstructed
// once host-side (via the public gguf.Dequantize — no gguf internals) and uploaded to a shared
// device buffer, exactly like the IQ2_XXS path. K%256==0. DECODE-ONLY (GEMV).
type ResidentBIQ3XXS struct {
	q    unsafe.Pointer // device bytes, row n = sbs consecutive 98-byte super-blocks
	k, n int
	sbs  int // K/256
}

const iq3xxsBlockBytes = 98

var (
	iq3GridOnce sync.Once
	iq3GridDev  unsafe.Pointer // shared [256*4]f32 device buffer
	iq3GridErr  error
)

// iq3xxsGrid reconstructs + uploads IQ3_XXS's 256×4 grid via the public gguf.Dequantize. Crafted
// blocks: d=1.0, every sign/scale word = (s<<28) (ksigns index 0 = all-positive, known scale s), so
// each grid byte at position pos decodes to db·grid[idx][0..3] and grid[idx][k] = dequant/db. 4
// blocks (64 grid positions each) cover all 256 indices. Mirrors iq2xxsGrid.
func iq3xxsGrid() (unsafe.Pointer, error) {
	iq3GridOnce.Do(func() {
		const s = 15 // scale nibble → db = 1.0·(0.5+s)·0.5
		db := float32(1.0) * (0.5 + float32(s)) * 0.5
		const nBlk = 4 // 4 blocks × 64 grid positions = 256
		raw := make([]byte, nBlk*iq3xxsBlockBytes)
		for b := 0; b < nBlk; b++ {
			blk := raw[b*iq3xxsBlockBytes : (b+1)*iq3xxsBlockBytes]
			binary.LittleEndian.PutUint16(blk[0:], 0x3C00) // d = 1.0 as f16
			for pos := 0; pos < 64; pos++ {
				blk[2+pos] = byte(b*64 + pos) // grid index at this qs position
			}
			for g := 0; g < 8; g++ {
				binary.LittleEndian.PutUint32(blk[66+g*4:], uint32(s)<<28) // ksigns idx 0, scale s
			}
		}
		deq, err := gguf.Dequantize(raw, gguf.IQ3_XXS, nBlk*256)
		if err != nil {
			iq3GridErr = fmt.Errorf("cuda: IQ3_XXS grid reconstruct: %w", err)
			return
		}
		df := deq.Storage().F32()
		grid := make([]float32, 256*4)
		for b := 0; b < nBlk; b++ {
			for pos := 0; pos < 64; pos++ {
				// element base for grid position pos in block b: pos = g*8 + j*2 + half.
				idx := b*64 + pos
				g := pos >> 3
				r := pos & 7
				j := r >> 1
				half := r & 1
				base := b*256 + g*32 + j*8 + half*4
				for k := 0; k < 4; k++ {
					grid[idx*4+k] = df[base+k] / db
				}
			}
		}
		iq3GridDev = C.cu_upload_f32((*C.float)(&grid[0]), C.int(len(grid)))
		if iq3GridDev == nil {
			iq3GridErr = fmt.Errorf("cuda: IQ3_XXS grid upload failed")
		}
	})
	return iq3GridDev, iq3GridErr
}

// NewResidentBIQ3XXSFromBlocks uploads pre-encoded IQ3_XXS blocks (row-major [N][K/256]).
func NewResidentBIQ3XXSFromBlocks(raw []byte, k, n int) (*ResidentBIQ3XXS, error) {
	if k%256 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBIQ3XXSFromBlocks needs K%%256==0, got K=%d", k)
	}
	if _, err := iq3xxsGrid(); err != nil {
		return nil, err
	}
	sbs := k / 256
	if want := n * sbs * iq3xxsBlockBytes; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBIQ3XXSFromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&raw[0])), C.int(len(raw)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: IQ3_XXS weight upload failed")
	}
	return &ResidentBIQ3XXS{q: dq, k: k, n: n, sbs: sbs}, nil
}

// QMatMulInto computes out = a·dequant(W) into the caller's fixed buffer (beta=0).
func (r *ResidentBIQ3XXS) QMatMulInto(a, out *DeviceF32) error { return r.qmatmul(a, out, 0) }

// QMatMulAccInto computes c += a·dequant(W) in place (beta=1).
func (r *ResidentBIQ3XXS) QMatMulAccInto(a, c *DeviceF32) error { return r.qmatmul(a, c, 1) }

func (r *ResidentBIQ3XXS) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: IQ3_XXS matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: IQ3_XXS matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	grid, err := iq3xxsGrid()
	if err != nil {
		return err
	}
	if rc := C.cu_qmatmul_iq3xxs(a.ptr, r.q, grid, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: IQ3_XXS matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident IQ3_XXS weight (the shared grid buffer persists).
func (r *ResidentBIQ3XXS) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
}

// ResidentBIQ3S is ggml's IQ3_S (§T554): the 3.44-bit i-quant — a 512×4 grid over the odd-value
// codebook 1..15, with 9-bit grid indices (qs byte + a qh high bit), DIRECT sign bytes (no ksigns
// table) and explicit per-32 4-bit sub-scales (db = d·(1+2s)). Per output row, K/256 super-blocks
// of 110 bytes: f16 d + 64 qs + 8 qh + 32 signs + 4 scale bytes. K%256==0. DECODE-ONLY (GEMV).
type ResidentBIQ3S struct {
	q    unsafe.Pointer // device bytes, row n = sbs consecutive 110-byte super-blocks
	k, n int
	sbs  int // K/256
}

const iq3sBlockBytes = 110

var (
	iq3sGridOnce sync.Once
	iq3sGridDev  unsafe.Pointer // shared [512*4]f32 device buffer
	iq3sGridErr  error
)

// iq3sGrid reconstructs + uploads IQ3_S's 512×4 grid via the public gguf.Dequantize. Crafted
// blocks: d=1.0, scales=0 (db=1.0), signs=0 (all positive) → each grid position decodes to
// grid[idx][0..3] directly. The 9-bit index idx is set by qs[pos]=idx&0xFF plus the qh bit
// (idx>>8)&1; 8 blocks (64 positions each) cover all 512 indices.
func iq3sGrid() (unsafe.Pointer, error) {
	iq3sGridOnce.Do(func() {
		const nBlk = 8 // 8 blocks × 64 grid positions = 512
		raw := make([]byte, nBlk*iq3sBlockBytes)
		for b := 0; b < nBlk; b++ {
			blk := raw[b*iq3sBlockBytes : (b+1)*iq3sBlockBytes]
			binary.LittleEndian.PutUint16(blk[0:], 0x3C00) // d = 1.0
			for pos := 0; pos < 64; pos++ {
				idx := b*64 + pos
				blk[2+pos] = byte(idx & 0xFF)      // qs low 8 bits
				g, p := pos>>3, pos&7              // qh byte g, bit p
				blk[66+g] |= byte((idx>>8)&1) << p // qh high bit
			}
			// signs (blk[74:106]) and scales (blk[106:110]) left 0 → all positive, db = 1.0
		}
		deq, err := gguf.Dequantize(raw, gguf.IQ3_S, nBlk*256)
		if err != nil {
			iq3sGridErr = fmt.Errorf("cuda: IQ3_S grid reconstruct: %w", err)
			return
		}
		df := deq.Storage().F32()
		grid := make([]float32, 512*4)
		for b := 0; b < nBlk; b++ {
			for pos := 0; pos < 64; pos++ {
				idx := b*64 + pos
				g := pos >> 3
				r := pos & 7
				j := r >> 1
				half := r & 1
				base := b*256 + g*32 + j*8 + half*4
				for k := 0; k < 4; k++ {
					grid[idx*4+k] = df[base+k] // db = 1.0
				}
			}
		}
		iq3sGridDev = C.cu_upload_f32((*C.float)(&grid[0]), C.int(len(grid)))
		if iq3sGridDev == nil {
			iq3sGridErr = fmt.Errorf("cuda: IQ3_S grid upload failed")
		}
	})
	return iq3sGridDev, iq3sGridErr
}

// NewResidentBIQ3SFromBlocks uploads pre-encoded IQ3_S blocks (row-major [N][K/256]).
func NewResidentBIQ3SFromBlocks(raw []byte, k, n int) (*ResidentBIQ3S, error) {
	if k%256 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBIQ3SFromBlocks needs K%%256==0, got K=%d", k)
	}
	if _, err := iq3sGrid(); err != nil {
		return nil, err
	}
	sbs := k / 256
	if want := n * sbs * iq3sBlockBytes; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBIQ3SFromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&raw[0])), C.int(len(raw)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: IQ3_S weight upload failed")
	}
	return &ResidentBIQ3S{q: dq, k: k, n: n, sbs: sbs}, nil
}

// QMatMulInto computes out = a·dequant(W) into the caller's fixed buffer (beta=0).
func (r *ResidentBIQ3S) QMatMulInto(a, out *DeviceF32) error { return r.qmatmul(a, out, 0) }

// QMatMulAccInto computes c += a·dequant(W) in place (beta=1).
func (r *ResidentBIQ3S) QMatMulAccInto(a, c *DeviceF32) error { return r.qmatmul(a, c, 1) }

func (r *ResidentBIQ3S) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: IQ3_S matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: IQ3_S matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	grid, err := iq3sGrid()
	if err != nil {
		return err
	}
	// M>1 (prefill/batch): route to the weight-read-once M-tiled GEMM (bit-identical) so column
	// n's IQ3_S block is grid-decoded once, not re-read per row. M==1 decode stays on the GEMV.
	if a.rows >= 8 {
		if rc := C.cu_qmatmul_iq3s_mt(a.ptr, r.q, grid, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
			return fmt.Errorf("cuda: IQ3_S m-tiled matmul failed (code %d)", int(rc))
		}
		return nil
	}
	if rc := C.cu_qmatmul_iq3s(a.ptr, r.q, grid, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: IQ3_S matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident IQ3_S weight (the shared grid buffer persists).
func (r *ResidentBIQ3S) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
}
