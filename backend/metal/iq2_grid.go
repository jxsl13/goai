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
	iq2XXSGridOnce sync.Once
	iq2XXSGridErr  error
	iq2XSGridOnce  sync.Once
	iq2XSGridErr   error
	iq2SGridOnce   sync.Once
	iq2SGridErr    error
)

// ensureIQ2Grid reconstructs an IQ2 eight-value codebook through the existing
// exact GGUF decoder and copies it into one process-lifetime Metal buffer. The
// direct, resident, and recorder kernels therefore share the format package's
// numerical truth without duplicating either packed table.
func ensureIQ2Grid(qt uint32) error {
	switch qt {
	case qtIQ2_XXS:
		iq2XXSGridOnce.Do(func() { iq2XXSGridErr = uploadIQ2XXSGrid() })
		return iq2XXSGridErr
	case qtIQ2_XS:
		iq2XSGridOnce.Do(func() { iq2XSGridErr = uploadIQ2XSGrid() })
		return iq2XSGridErr
	case qtIQ2_S:
		iq2SGridOnce.Do(func() { iq2SGridErr = uploadIQ2SGrid() })
		return iq2SGridErr
	default:
		return backend.ErrQuantUnsupported
	}
}

func uploadIQ2XXSGrid() error {
	const (
		blocks    = 8
		blockSize = 66
		scale     = 15
	)
	raw := make([]byte, blocks*blockSize)
	for b := range blocks {
		blk := raw[b*blockSize : (b+1)*blockSize]
		//perfscan:ignore PS4001 one-time exact f16 fixture construction; avoid unsafe host-layout coupling
		binary.LittleEndian.PutUint16(blk, 0x3c00) // f16(1)
		for pair := range 8 {
			pairBits := blk[2+pair*8:]
			for group := range 4 {
				pairBits[group] = byte(b*32 + pair*4 + group)
			}
			//perfscan:ignore PS4001 one-time exact packed fixture construction; no token-path amortization
			binary.LittleEndian.PutUint32(pairBits[4:], uint32(scale)<<28)
		}
	}
	deq, err := gguf.Dequantize(raw, gguf.IQ2_XXS, blocks*256)
	if err != nil {
		return fmt.Errorf("metal: reconstruct IQ2_XXS grid: %w", err)
	}
	db := float32(0.5+scale) * 0.25
	decoded := deq.Storage().F32()
	grid := make([]float32, 256*8)
	for b := range blocks {
		for pair := range 8 {
			for group := range 4 {
				idx := b*32 + pair*4 + group
				base := b*256 + pair*32 + group*8
				for lane := range 8 {
					//perfscan:ignore PS5001 exact one-time oracle normalization must retain division semantics
					grid[idx*8+lane] = decoded[base+lane] / db
				}
			}
		}
	}
	if rc := C.mtl_iq2_grid_upload(C.int(qtIQ2_XXS), (*C.float)(&grid[0]), C.int(len(grid))); rc != 0 {
		return fmt.Errorf("metal: upload IQ2_XXS grid failed (code %d)", int(rc))
	}
	return nil
}

func uploadIQ2XSGrid() error {
	const (
		blocks    = 16
		blockSize = 74
	)
	raw := make([]byte, blocks*blockSize)
	for b := range blocks {
		blk := raw[b*blockSize : (b+1)*blockSize]
		//perfscan:ignore PS4001 one-time exact f16 fixture construction; avoid unsafe host-layout coupling
		binary.LittleEndian.PutUint16(blk, 0x3c00) // f16(1)
		for group := range 32 {
			idx := b*32 + group
			//perfscan:ignore PS4001 one-time exact packed fixture construction; no token-path amortization
			binary.LittleEndian.PutUint16(blk[2+group*2:], uint16(idx))
		}
	}
	deq, err := gguf.Dequantize(raw, gguf.IQ2_XS, blocks*256)
	if err != nil {
		return fmt.Errorf("metal: reconstruct IQ2_XS grid: %w", err)
	}
	const db = float32(0.5) * 0.25
	decoded := deq.Storage().F32()
	grid := make([]float32, 512*8)
	for b := range blocks {
		for group := range 32 {
			idx := b*32 + group
			base := b*256 + group*8
			for lane := range 8 {
				//perfscan:ignore PS5001 exact one-time oracle normalization must retain division semantics
				grid[idx*8+lane] = decoded[base+lane] / db
			}
		}
	}
	if rc := C.mtl_iq2_grid_upload(C.int(qtIQ2_XS), (*C.float)(&grid[0]), C.int(len(grid))); rc != 0 {
		return fmt.Errorf("metal: upload IQ2_XS grid failed (code %d)", int(rc))
	}
	return nil
}

func uploadIQ2SGrid() error {
	const (
		blocks    = 32
		blockSize = 82
		db        = float32(0.5) * 0.25
	)
	raw := make([]byte, blocks*blockSize)
	for block := range blocks {
		blk := raw[block*blockSize : (block+1)*blockSize]
		//perfscan:ignore PS4001 one-time exact f16 fixture construction; avoid unsafe host-layout coupling
		binary.LittleEndian.PutUint16(blk, 0x3c00) // f16(1)
		for group := range 32 {
			index := block*32 + group
			blk[2+group] = byte(index)
			blk[66+group/4] |= byte(index>>8) << (2 * (group % 4))
		}
	}
	dequantized, err := gguf.Dequantize(raw, gguf.IQ2_S, blocks*256)
	if err != nil {
		return fmt.Errorf("metal: reconstruct IQ2_S grid: %w", err)
	}
	decoded := dequantized.Storage().F32()
	packed := make([]uint16, 1024)
	for block := range blocks {
		for group := range 32 {
			index := block*32 + group
			base := block*256 + group*8
			var word uint16
			for lane := range 8 {
				//perfscan:ignore PS5001 exact one-time oracle normalization must retain division semantics
				gridValue := decoded[base+lane] / db
				var code uint16
				switch gridValue {
				case 8:
					code = 0
				case 25:
					code = 1
				case 43:
					code = 2
				default:
					return fmt.Errorf("metal: IQ2_S grid[%d][%d]=%g is outside {8,25,43}", index, lane, gridValue)
				}
				word |= code << (2 * lane)
			}
			packed[index] = word
		}
	}
	if rc := C.mtl_iq2_s_grid_upload((*C.ushort)(&packed[0]), C.int(len(packed))); rc != 0 {
		return fmt.Errorf("metal: upload IQ2_S grid failed (code %d)", int(rc))
	}
	return nil
}
