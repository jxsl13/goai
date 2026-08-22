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

	"github.com/jxsl13/goai/format/gguf"
)

var (
	iq1GridOnce sync.Once
	iq1GridErr  error
)

// ensureIQ1Grid reconstructs the shared IQ1 ternary codebook through the exact
// public GGUF decoder and uploads one process-lifetime packed Metal buffer. Each
// uint16 stores eight two-bit {-1,0,+1} codes, reducing the immutable lookup
// footprint from 64 KiB of f32 values to 4 KiB.
func ensureIQ1Grid() error {
	iq1GridOnce.Do(func() { iq1GridErr = uploadIQ1Grid() })
	return iq1GridErr
}

func uploadIQ1Grid() error {
	const (
		blocks    = 64
		blockSize = 50
	)
	raw := make([]byte, blocks*blockSize)
	for block := range blocks {
		blk := raw[block*blockSize : (block+1)*blockSize]
		//perfscan:ignore PS4001 one-time exact f16 fixture construction; avoid unsafe host-layout coupling
		binary.LittleEndian.PutUint16(blk, 0x3c00) // f16(1)
		for group := range 8 {
			var high uint16
			for lane := range 4 {
				index := block*32 + group*4 + lane
				blk[2+group*4+lane] = byte(index)
				high |= uint16(index>>8) << (3 * lane)
			}
			// multiplier zero and positive delta produce grid+0.125 exactly.
			//perfscan:ignore PS4001 one-time packed fixture construction; no token-path amortization
			binary.LittleEndian.PutUint16(blk[34+group*2:], high)
		}
	}
	dequantized, err := gguf.Dequantize(raw, gguf.IQ1_S, blocks*256)
	if err != nil {
		return fmt.Errorf("metal: reconstruct IQ1 grid: %w", err)
	}
	decoded := dequantized.Storage().F32()
	packed := make([]uint16, 2048)
	for block := range blocks {
		for group := range 8 {
			for lane := range 4 {
				index := block*32 + group*4 + lane
				base := block*256 + group*32 + lane*8
				var word uint16
				for value := range 8 {
					gridValue := decoded[base+value] - 0.125
					var code uint16
					switch gridValue {
					case -1:
						code = 0
					case 0:
						code = 1
					case 1:
						code = 2
					default:
						return fmt.Errorf("metal: IQ1 grid[%d][%d]=%g is not ternary", index, value, gridValue)
					}
					word |= code << (2 * value)
				}
				packed[index] = word
			}
		}
	}
	if rc := C.mtl_iq1_grid_upload((*C.ushort)(&packed[0]), C.int(len(packed))); rc != 0 {
		return fmt.Errorf("metal: upload IQ1 grid failed (code %d)", int(rc))
	}
	return nil
}
