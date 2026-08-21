package gguf

import (
	"encoding/binary"
	"math"
)

// dotIQ3XXSRow folds the grid lookup, ksigns expansion, and sub-scale
// application into one ascending-weight dot. Every float32 weight operation
// matches dequantIQ3_XXSInto before widening into the float64 accumulator.
func dotIQ3XXSRow(row []float32, raw []byte, k int) float64 {
	var acc float64
	for b := 0; b*qkK < k; b++ {
		blk := raw[b*iq3xxsBlockSize : (b+1)*iq3xxsBlockSize]
		//perfscan:ignore PS4001 one strided f16 scale per 98-byte quant block cannot use a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		qs := blk[2:66]
		for g := range 8 {
			//perfscan:ignore PS4001 eight unaligned sign/scale words per heterogeneous 98-byte block cannot use a same-layout bulk copy
			sg := binary.LittleEndian.Uint32(blk[66+g*4:])
			db := d * (0.5 + float32(sg>>28)) * 0.5
			for j := range 4 {
				r1 := &iq3xxsGrid[qs[g*8+j*2]]
				r2 := &iq3xxsGrid[qs[g*8+j*2+1]]
				signs := iq2xxsKSigns[(sg>>(7*j))&0x7f]
				base := b*qkK + g*32 + j*8
				for i := range 4 {
					sign := (uint32(signs>>i) & 1) << 31
					weight := math.Float32frombits(math.Float32bits(db*r1[i]) ^ sign)
					acc += float64(row[base+i]) * float64(weight)
				}
				for i := range 4 {
					sign := (uint32(signs>>(i+4)) & 1) << 31
					weight := math.Float32frombits(math.Float32bits(db*r2[i]) ^ sign)
					acc += float64(row[base+4+i]) * float64(weight)
				}
			}
		}
	}
	return acc
}
