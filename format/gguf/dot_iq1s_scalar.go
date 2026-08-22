package gguf

import "encoding/binary"

// dotIQ1SRow folds the 11-bit ternary-grid lookup, odd qh multiplier, signed
// delta, and activation dot into one ascending-weight pass. Every float32
// weight operation matches dequantIQ1_SInto before widening into the float64
// accumulator.
func dotIQ1SRow(row []float32, raw []byte, k int) float64 {
	var acc float64
	for b := 0; b*qkK < k; b++ {
		blk := raw[b*iq1sBlockSize : (b+1)*iq1sBlockSize]
		//perfscan:ignore PS4001 one strided f16 scale per 50-byte quant block cannot use a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		qs := blk[2:34]
		for g := range 8 {
			qh := binary.LittleEndian.Uint16(blk[34+g*2:]) //perfscan:ignore PS4001 alignment/endian-safe oracle; architecture bulk view tracked in perfscan#811
			dl := d * float32(2*(qh>>12&7)+1)
			delta := float32(iq1Delta)
			if qh&0x8000 != 0 {
				delta = -iq1Delta
			}
			for j := range 4 {
				u := uint16(qs[g*4+j]) | (qh>>(3*j)&7)<<8
				gridRow := &iq1Grid[u]
				base := b*qkK + g*32 + j*8
				for lane := range 8 {
					weight := dl * (gridRow[lane] + delta)
					acc += float64(row[base+lane]) * float64(weight)
				}
			}
		}
	}
	return acc
}
