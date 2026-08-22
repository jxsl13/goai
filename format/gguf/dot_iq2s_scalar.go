package gguf

import (
	"encoding/binary"
	"math"
)

// dotIQ2SRow folds the 10-bit grid lookup, direct sign mapping, and explicit
// sub-scale application into one ascending-weight dot. Every float32 weight
// operation matches dequantIQ2_SInto before widening into the float64
// accumulator.
func dotIQ2SRow(row []float32, raw []byte, k int) float64 {
	var acc float64
	for b := 0; b*qkK < k; b++ {
		blk := raw[b*iq2sBlockSize : (b+1)*iq2sBlockSize]
		//perfscan:ignore PS4001 one strided f16 scale per 82-byte quant block cannot use a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		qs, signs, qh, scales := blk[2:34], blk[34:66], blk[66:74], blk[74:82]
		for g, lo := range qs {
			s := g / 2
			scale := scales[s/2] >> (4 * (s % 2)) & 0x0f
			db := d * (0.5 + float32(scale)) * 0.25
			u := uint16(lo) | uint16(qh[g/4]>>(2*(g%4))&0x3)<<8
			gridRow := &iq2sGrid[u]
			sg := signs[g]
			base := b*qkK + g*8
			for i := range 8 {
				sign := (uint32(sg>>i) & 1) << 31
				weight := math.Float32frombits(math.Float32bits(db*gridRow[i]) ^ sign)
				acc += float64(row[base+i]) * float64(weight)
			}
		}
	}
	return acc
}
