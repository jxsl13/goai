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

// ResidentBIQ1S is a weight matrix in ggml's IQ1_S format (§T554) made GPU-resident: the 1.56-bit
// TERNARY i-quant — a shared 2048×8 grid over {−1,0,+1} with a ±0.125 delta added to every element
// (so values never hit exact zero) and odd per-32 multipliers dl = d·(2s+1). Per output row, K/256
// super-blocks of 50 bytes: f16 d + 32 qs + 16 qh (8 u16). Each qh u16 covers 32 elements: bits
// 0..11 = four 3-bit grid-index-high triples, bits 12..14 = the 3-bit odd multiplier s, bit 15 =
// the delta sign. The 2048×8 grid is reconstructed once host-side (via the public gguf.Dequantize —
// no gguf internals) and uploaded to a shared device buffer. K%256==0. DECODE-ONLY (GEMV).
type ResidentBIQ1S struct {
	q    unsafe.Pointer // device bytes, row n = sbs consecutive 50-byte super-blocks
	k, n int
	sbs  int // K/256
}

const iq1sBlockBytes = 50

var (
	iq1GridOnce sync.Once
	iq1GridDev  unsafe.Pointer // shared [2048*8]f32 device buffer
	iq1GridErr  error
)

// iq1sGrid reconstructs + uploads IQ1_S's 2048×8 ternary grid via the public gguf.Dequantize.
// Crafted blocks: d=1.0, multiplier s=0 (dl=1), delta sign=+ (δ=+0.125) → each grid position
// decodes to (grid[idx][k] + 0.125), so grid[idx][k] = dequant − 0.125. 64 blocks (32 grid rows
// each) cover all 2048 indices. The 11-bit index is set by qs[pos]=idx&0xFF plus the qh triple
// (idx>>8)&7 for that j.
func iq1sGrid() (unsafe.Pointer, error) {
	iq1GridOnce.Do(func() {
		const nBlk = 64 // 64 blocks × 32 grid rows = 2048
		raw := make([]byte, nBlk*iq1sBlockBytes)
		for b := 0; b < nBlk; b++ {
			blk := raw[b*iq1sBlockBytes : (b+1)*iq1sBlockBytes]
			binary.LittleEndian.PutUint16(blk[0:], 0x3C00) // d = 1.0
			for g := 0; g < 8; g++ {
				var qh uint16 // s=0 (bits 12..14), delta sign=+ (bit 15 clear)
				for j := 0; j < 4; j++ {
					idx := b*32 + g*4 + j
					blk[2+g*4+j] = byte(idx & 0xFF)     // qs low 8 bits
					qh |= uint16((idx>>8)&7) << (3 * j) // grid-index high 3 bits
				}
				binary.LittleEndian.PutUint16(blk[34+g*2:], qh)
			}
		}
		deq, err := gguf.Dequantize(raw, gguf.IQ1_S, nBlk*256)
		if err != nil {
			iq1GridErr = fmt.Errorf("cuda: IQ1_S grid reconstruct: %w", err)
			return
		}
		df := deq.Storage().F32()
		grid := make([]float32, 2048*8)
		for b := 0; b < nBlk; b++ {
			for g := 0; g < 8; g++ {
				for j := 0; j < 4; j++ {
					idx := b*32 + g*4 + j
					base := b*256 + g*32 + j*8
					for k := 0; k < 8; k++ {
						grid[idx*8+k] = df[base+k] - 0.125 // undo the +δ (dl=1)
					}
				}
			}
		}
		// Tw80 escape route: the grid is TERNARY {−1,0,+1}, so pack it to 2 bits/entry
		// (2048×8 → 4 KB). The kernel then reads ONE uint16/lane from shared (not 8 f32 from
		// L2) and decodes code→{−1,0,+1} in ALU — the small per-lane read dodges the 8-way
		// bank-conflict floor that caps the f32-grid path (perf-notes-cuda.md R6).
		packed := make([]byte, 2048*2) // 2048 rows × one uint16 (8 × 2-bit codes)
		for u := 0; u < 2048; u++ {
			var p uint16
			for k := 0; k < 8; k++ {
				p |= uint16(grid[u*8+k]+1) << (2 * k) // −1→0, 0→1, +1→2
			}
			binary.LittleEndian.PutUint16(packed[u*2:], p)
		}
		iq1GridDev = C.cu_upload_i8((*C.schar)(unsafe.Pointer(&packed[0])), C.int(len(packed)))
		if iq1GridDev == nil {
			iq1GridErr = fmt.Errorf("cuda: IQ1_S packed grid upload failed")
		}
	})
	return iq1GridDev, iq1GridErr
}

// NewResidentBIQ1SFromBlocks uploads pre-encoded IQ1_S blocks (row-major [N][K/256]).
func NewResidentBIQ1SFromBlocks(raw []byte, k, n int) (*ResidentBIQ1S, error) {
	if k%256 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBIQ1SFromBlocks needs K%%256==0, got K=%d", k)
	}
	if _, err := iq1sGrid(); err != nil {
		return nil, err
	}
	sbs := k / 256
	if want := n * sbs * iq1sBlockBytes; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBIQ1SFromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&raw[0])), C.int(len(raw)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: IQ1_S weight upload failed")
	}
	return &ResidentBIQ1S{q: dq, k: k, n: n, sbs: sbs}, nil
}

// QMatMulInto computes out = a·dequant(W) into the caller's fixed buffer (beta=0).
func (r *ResidentBIQ1S) QMatMulInto(a, out *DeviceF32) error { return r.qmatmul(a, out, 0) }

// QMatMulAccInto computes c += a·dequant(W) in place (beta=1).
func (r *ResidentBIQ1S) QMatMulAccInto(a, c *DeviceF32) error { return r.qmatmul(a, c, 1) }

func (r *ResidentBIQ1S) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: IQ1_S matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: IQ1_S matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	grid, err := iq1sGrid()
	if err != nil {
		return err
	}
	if rc := C.cu_qmatmul_iq1s(a.ptr, r.q, grid, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: IQ1_S matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident IQ1_S weight (the shared grid buffer persists).
func (r *ResidentBIQ1S) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
}

// ResidentBIQ1M is ggml's IQ1_M (§T554): the 1.75-bit ternary sibling of IQ1_S — the SAME 2048×8
// {−1,0,+1} grid + ±0.125 delta, but with per-16 (per-2-grid-row) 3-bit sub-scales and IQ1_M's
// quirk: the f16 super-scale d is SPLIT across the top nibbles of the four scale words. Per output
// row, K/256 super-blocks of 56 bytes: 32 qs + 16 qh + 8 scale bytes (4 u16). Each qh nibble carries
// a 3-bit grid-index-high triple + a delta sign. K%256==0. DECODE-ONLY (GEMV).
type ResidentBIQ1M struct {
	q    unsafe.Pointer // device bytes, row n = sbs consecutive 56-byte super-blocks
	k, n int
	sbs  int // K/256
}

const iq1mBlockBytes = 56

// NewResidentBIQ1MFromBlocks uploads pre-encoded IQ1_M blocks (row-major [N][K/256]). Shares the
// IQ1_S 2048×8 grid.
func NewResidentBIQ1MFromBlocks(raw []byte, k, n int) (*ResidentBIQ1M, error) {
	if k%256 != 0 {
		return nil, fmt.Errorf("cuda: NewResidentBIQ1MFromBlocks needs K%%256==0, got K=%d", k)
	}
	if _, err := iq1sGrid(); err != nil {
		return nil, err
	}
	sbs := k / 256
	if want := n * sbs * iq1mBlockBytes; len(raw) != want {
		return nil, fmt.Errorf("cuda: NewResidentBIQ1MFromBlocks got %d bytes, want %d (N=%d K=%d)", len(raw), want, n, k)
	}
	dq := C.cu_upload_i8((*C.schar)(unsafe.Pointer(&raw[0])), C.int(len(raw)))
	if dq == nil {
		return nil, fmt.Errorf("cuda: IQ1_M weight upload failed")
	}
	return &ResidentBIQ1M{q: dq, k: k, n: n, sbs: sbs}, nil
}

// QMatMulInto computes out = a·dequant(W) into the caller's fixed buffer (beta=0).
func (r *ResidentBIQ1M) QMatMulInto(a, out *DeviceF32) error { return r.qmatmul(a, out, 0) }

// QMatMulAccInto computes c += a·dequant(W) in place (beta=1).
func (r *ResidentBIQ1M) QMatMulAccInto(a, c *DeviceF32) error { return r.qmatmul(a, c, 1) }

func (r *ResidentBIQ1M) qmatmul(a, out *DeviceF32, beta float32) error {
	if r.q == nil || a.ptr == nil || out.ptr == nil {
		return fmt.Errorf("cuda: IQ1_M matmul on a freed handle")
	}
	if a.cols != r.k || out.rows != a.rows || out.cols != r.n {
		return fmt.Errorf("cuda: IQ1_M matmul shape a[%d,%d]·B[%d,%d]→out[%d,%d]", a.rows, a.cols, r.k, r.n, out.rows, out.cols)
	}
	grid, err := iq1sGrid()
	if err != nil {
		return err
	}
	if rc := C.cu_qmatmul_iq1m(a.ptr, r.q, grid, out.ptr, C.int(a.rows), C.int(r.k), C.int(r.n), C.float(beta)); rc != 0 {
		return fmt.Errorf("cuda: IQ1_M matmul failed (code %d)", int(rc))
	}
	return nil
}

// Free releases the resident IQ1_M weight (the shared grid buffer persists).
func (r *ResidentBIQ1M) Free() {
	if r.q != nil {
		C.cu_free_f32(r.q)
		r.q = nil
	}
}
