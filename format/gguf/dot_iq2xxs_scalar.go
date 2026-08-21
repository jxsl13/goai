package gguf

import (
	"encoding/binary"
	"math"
)

// dotIQ2XXSRow folds the eight-wide grid lookup, ksigns expansion, and
// sub-scale application into one ascending-weight dot. Every float32 weight
// operation matches dequantIQ2_XXSInto before widening into the float64
// accumulator.
func dotIQ2XXSRow(row []float32, raw []byte, k int) float64 {
	var acc float64
	for b := 0; b*qkK < k; b++ {
		blk := raw[b*iq2xxsBlockSize : (b+1)*iq2xxsBlockSize]
		//perfscan:ignore PS4001 one strided f16 scale per 66-byte quant block cannot use a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		for pair := range 8 {
			pairBits := blk[2+pair*8:]
			//perfscan:ignore PS4001 all 28 sign bits and the high scale nibble are consumed from this heterogeneous pair
			signsAndScale := binary.LittleEndian.Uint32(pairBits[4:])
			db := d * (0.5 + float32(signsAndScale>>28)) * 0.25
			for g := range 4 {
				gridRow := &iq2xxsGrid[pairBits[g]]
				signs := iq2xxsKSigns[(signsAndScale>>(7*g))&0x7f]
				base := b*qkK + pair*32 + g*8
				for i := range 8 {
					sign := (uint32(signs>>i) & 1) << 31
					weight := math.Float32frombits(math.Float32bits(db*gridRow[i]) ^ sign)
					acc += float64(row[base+i]) * float64(weight)
				}
			}
		}
	}
	return acc
}
