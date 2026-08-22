package gguf

import (
	"encoding/binary"
	"math"
)

// dotIQ2XSRow folds the eight-wide grid lookup, ksign expansion, and explicit
// sub-scale application into one ascending-weight dot. Every float32 weight
// operation matches dequantIQ2_XSInto before widening into the float64
// accumulator.
func dotIQ2XSRow(row []float32, raw []byte, k int) float64 {
	var acc float64
	for b := 0; b*qkK < k; b++ {
		blk := raw[b*iq2xsBlockSize : (b+1)*iq2xsBlockSize]
		//perfscan:ignore PS4001 one strided f16 scale per 74-byte quant block cannot use a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		scales := blk[66:74]
		var codeScratch [32]uint16
		codes := iq2xsCodeWords(blk, &codeScratch)
		for g, u := range codes {
			s := g / 2
			scale := scales[s/2] >> (4 * (s % 2)) & 0x0f
			db := d * (0.5 + float32(scale)) * 0.25
			gridRow := &iq2xsGrid[u&0x1ff]
			signs := iq2xxsKSigns[u>>9]
			base := b*qkK + g*8
			for i := range 8 {
				sign := (uint32(signs>>i) & 1) << 31
				weight := math.Float32frombits(math.Float32bits(db*gridRow[i]) ^ sign)
				acc += float64(row[base+i]) * float64(weight)
			}
		}
	}
	return acc
}
