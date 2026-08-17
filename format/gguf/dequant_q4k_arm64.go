//go:build arm64

package gguf

import "encoding/binary"

// dequantQ4KBlockNeon decodes one 144-byte Q4_K super-block into 256 f32
// values. coeff holds {stepLow, offsetLow, stepHigh, offsetHigh} for each of
// the four packed 64-value pairs.
//
//go:noescape
func dequantQ4KBlockNeon(dst *float32, qs *byte, coeff *float32, indexes *byte)

func dequantQ4_KIntoArch(dst []float32, raw []byte) {
	var coeff [16]float32
	for sb := 0; sb*qkK < len(dst); sb++ {
		blk := raw[sb*q4kBlockSize : (sb+1)*q4kBlockSize]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:]))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[2:]))
		scales := blk[4:16]
		for pair := range 4 {
			is := pair * 2
			sc1, m1 := getScaleMinK4(is, scales)
			sc2, m2 := getScaleMinK4(is+1, scales)
			ci := pair * 4
			coeff[ci+0] = d * float32(sc1)
			coeff[ci+1] = dmin * float32(m1)
			coeff[ci+2] = d * float32(sc2)
			coeff[ci+3] = dmin * float32(m2)
		}
		dequantQ4KBlockNeon(&dst[sb*qkK], &blk[16], &coeff[0], &qKByteToF32Indexes[0])
	}
}
