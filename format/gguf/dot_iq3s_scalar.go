package gguf

import (
	"encoding/binary"
	"math"
)

// dotIQ3SRow folds the 9-bit grid lookup, direct sign mapping, and sub-scale
// application into one ascending-weight dot. Every float32 weight operation
// matches dequantIQ3_SInto before widening into the float64 accumulator.
func dotIQ3SRow(row []float32, raw []byte, k int) float64 {
	var acc float64
	for b := 0; b*qkK < k; b++ {
		blk := raw[b*iq3sBlockSize : (b+1)*iq3sBlockSize]
		//perfscan:ignore PS4001 one strided f16 scale per 110-byte quant block cannot use a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		qs, qh, signs, scales := blk[2:66], blk[66:74], blk[74:106], blk[106:110]
		for g := range 8 {
			sc := scales[g/2] >> (4 * (g % 2)) & 0x0f
			db := d * (1 + 2*float32(sc))
			for j := range 4 {
				i1, i2 := g*8+j*2, g*8+j*2+1
				u1 := uint16(qs[i1]) | uint16(qh[g]>>(j*2)&1)<<8
				u2 := uint16(qs[i2]) | uint16(qh[g]>>(j*2+1)&1)<<8
				r1, r2 := &iq3sGrid[u1], &iq3sGrid[u2]
				sg := signs[g*4+j]
				base := b*qkK + g*32 + j*8
				for i := range 4 {
					sign := (uint32(sg>>i) & 1) << 31
					weight := math.Float32frombits(math.Float32bits(db*r1[i]) ^ sign)
					acc += float64(row[base+i]) * float64(weight)
				}
				for i := range 4 {
					sign := (uint32(sg>>(i+4)) & 1) << 31
					weight := math.Float32frombits(math.Float32bits(db*r2[i]) ^ sign)
					acc += float64(row[base+4+i]) * float64(weight)
				}
			}
		}
	}
	return acc
}
