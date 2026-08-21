package gguf

import "encoding/binary"

// dotIQ4XSRow folds IQ4_XS super-scale, signed sub-scale, and nonlinear
// lookup into a single ascending-weight dot. Each float32 operation matches
// the materialized decoder before widening into the f64 accumulator.
func dotIQ4XSRow(row []float32, raw []byte, k int) float64 {
	var acc float64
	for b := 0; b*qkK < k; b++ {
		blk := raw[b*iq4xsBlockSize : (b+1)*iq4xsBlockSize]
		//perfscan:ignore PS4001 one strided f16 scale per 136-byte quant block cannot use a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		scalesH := binary.LittleEndian.Uint16(blk[2:])
		scalesL, qs := blk[4:8], blk[8:136]
		base := b * qkK
		for sb := range 8 {
			coefficient := d * float32(iq4XSSubscale(scalesH, scalesL, sb))
			q := qs[sb*16 : (sb+1)*16]
			xbase := base + sb*blockElems
			for i, packed := range q {
				weight := coefficient * iq4KValues[packed&0x0f]
				acc += float64(row[xbase+i]) * float64(weight)
			}
			for i, packed := range q {
				weight := coefficient * iq4KValues[packed>>4]
				acc += float64(row[xbase+16+i]) * float64(weight)
			}
		}
	}
	return acc
}
