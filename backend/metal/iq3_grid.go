//go:build darwin && cgo

package metal

/*
#include "metal_bridge.h"
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
)

var (
	iq3XXSGridOnce sync.Once
	iq3XXSGridErr  error
	iq3SGridOnce   sync.Once
	iq3SGridErr    error
)

// ensureIQ3Grid reconstructs an IQ3 codebook through the existing exact GGUF
// decoder and copies it into one process-lifetime Metal buffer. Keeping this
// oracle at the format boundary avoids a second hand-maintained copy of either
// grid while direct, resident, and recorder kernels share the same bytes.
func ensureIQ3Grid(qt uint32) error {
	switch qt {
	case qtIQ3_XXS:
		iq3XXSGridOnce.Do(func() { iq3XXSGridErr = uploadIQ3XXSGrid() })
		return iq3XXSGridErr
	case qtIQ3_S:
		iq3SGridOnce.Do(func() { iq3SGridErr = uploadIQ3SGrid() })
		return iq3SGridErr
	default:
		return backend.ErrQuantUnsupported
	}
}

func uploadIQ3XXSGrid() error {
	const (
		blocks    = 4
		blockSize = 98
		scale     = 15
	)
	raw := make([]byte, blocks*blockSize)
	for b := range blocks {
		blk := raw[b*blockSize : (b+1)*blockSize]
		//perfscan:ignore PS4001 one-time exact f16 fixture construction; avoid unsafe host-layout coupling
		binary.LittleEndian.PutUint16(blk, 0x3c00) // f16(1)
		for pos := range 64 {
			blk[2+pos] = byte(b*64 + pos)
		}
		for group := range 8 {
			//perfscan:ignore PS4001 one-time exact packed fixture construction; no token-path amortization
			binary.LittleEndian.PutUint32(blk[66+group*4:], uint32(scale)<<28)
		}
	}
	deq, err := gguf.Dequantize(raw, gguf.IQ3_XXS, blocks*256)
	if err != nil {
		return fmt.Errorf("metal: reconstruct IQ3_XXS grid: %w", err)
	}
	db := float32(0.5+scale) * 0.5
	decoded := deq.Storage().F32()
	grid := make([]float32, 256*4)
	for b := range blocks {
		for pos := range 64 {
			idx := b*64 + pos
			group, within := pos>>3, pos&7
			pair, half := within>>1, within&1
			base := b*256 + group*32 + pair*8 + half*4
			for lane := range 4 {
				//perfscan:ignore PS5001 exact one-time oracle normalization must retain division semantics
				grid[idx*4+lane] = decoded[base+lane] / db
			}
		}
	}
	if rc := C.mtl_iq3_grid_upload(C.int(qtIQ3_XXS), (*C.float)(&grid[0]), C.int(len(grid))); rc != 0 {
		return fmt.Errorf("metal: upload IQ3_XXS grid failed (code %d)", int(rc))
	}
	return nil
}

func uploadIQ3SGrid() error {
	const (
		blocks    = 8
		blockSize = 110
	)
	raw := make([]byte, blocks*blockSize)
	for b := range blocks {
		blk := raw[b*blockSize : (b+1)*blockSize]
		//perfscan:ignore PS4001 one-time exact f16 fixture construction; avoid unsafe host-layout coupling
		binary.LittleEndian.PutUint16(blk, 0x3c00) // f16(1)
		for pos := range 64 {
			idx := b*64 + pos
			blk[2+pos] = byte(idx)
			group, bit := pos>>3, pos&7
			blk[66+group] |= byte(idx>>8) << bit
		}
	}
	deq, err := gguf.Dequantize(raw, gguf.IQ3_S, blocks*256)
	if err != nil {
		return fmt.Errorf("metal: reconstruct IQ3_S grid: %w", err)
	}
	decoded := deq.Storage().F32()
	grid := make([]float32, 512*4)
	for b := range blocks {
		for pos := range 64 {
			idx := b*64 + pos
			group, within := pos>>3, pos&7
			pair, half := within>>1, within&1
			base := b*256 + group*32 + pair*8 + half*4
			copy(grid[idx*4:idx*4+4], decoded[base:base+4])
		}
	}
	if rc := C.mtl_iq3_grid_upload(C.int(qtIQ3_S), (*C.float)(&grid[0]), C.int(len(grid))); rc != 0 {
		return fmt.Errorf("metal: upload IQ3_S grid failed (code %d)", int(rc))
	}
	return nil
}
