package gguf

import "encoding/binary"

// dotIQ1MRow folds split-f16 super-scale reconstruction, 11-bit ternary-grid
// lookup, paired odd multipliers, signed delta, and activation dot into one
// ascending-weight pass. Every float32 weight operation matches
// dequantIQ1_MInto before widening into the float64 accumulator.
func dotIQ1MRow(row []float32, raw []byte, k int) float64 {
	var acc float64
	for b := 0; b*qkK < k; b++ {
		blk := raw[b*iq1mBlockSize : (b+1)*iq1mBlockSize]
		qs, qh, scales := blk[0:32], blk[32:48], blk[48:56]
		var dbits uint16
		for i := range 4 {
			//perfscan:ignore PS4001 four split f16 nibbles require per-word bit extraction, not a same-layout bulk copy
			w := binary.LittleEndian.Uint16(scales[i*2:])
			dbits |= (w & 0xf000) >> (12 - 4*i)
		}
		d := f16ToF32(dbits)
		for i := range 32 {
			//perfscan:ignore PS4001 paired 3-bit scales are interleaved with split f16 nibbles in four words
			sw := binary.LittleEndian.Uint16(scales[(i/8)*2:])
			dl := d * float32(2*(sw>>(3*(i/2%4))&7)+1)
			nib := qh[i/2] >> (4 * (i % 2))
			u := uint16(qs[i]) | uint16(nib&7)<<8
			delta := float32(iq1Delta)
			if nib&8 != 0 {
				delta = -iq1Delta
			}
			gridRow := &iq1Grid[u]
			base := b*qkK + i*8
			for lane := range 8 {
				weight := dl * (gridRow[lane] + delta)
				acc += float64(row[base+lane]) * float64(weight)
			}
		}
	}
	return acc
}
