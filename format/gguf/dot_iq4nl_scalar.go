package gguf

import "encoding/binary"

// dotIQ4NLRow folds IQ4_NL's nonlinear lookup and f16 scaling into one dot.
// The low-nibble half precedes the high-nibble half exactly as it does in the
// materialized decoder, and each float32 weight is widened before the f64
// multiply-accumulate.
func dotIQ4NLRow(row []float32, raw []byte, k int) float64 {
	var acc float64
	for b := 0; b*blockElems < k; b++ {
		blk := raw[b*iq4nlBlockSize : (b+1)*iq4nlBlockSize]
		//perfscan:ignore PS4001 one strided f16 scale per 18-byte quant block cannot use a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		qs := blk[2:18]
		base := b * blockElems
		for i, q := range qs {
			weight := d * iq4KValues[q&0x0f]
			acc += float64(row[base+i]) * float64(weight)
		}
		for i, q := range qs {
			weight := d * iq4KValues[q>>4]
			acc += float64(row[base+16+i]) * float64(weight)
		}
	}
	return acc
}
