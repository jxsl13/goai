//go:build arm64

package gguf

import "encoding/binary"

// dotQ4KBlockNeon fuses one 256-weight Q4_K super-block's nibble unpack,
// affine dequantization, and activation dot. The vector kernel accumulates in
// f32; dotQ4_KRowASM widens each super-block subtotal to f64 before combining
// it, so this path is tolerance-gated rather than bit-identical to the scalar
// f64-per-element reference.
//
//go:noescape
func dotQ4KBlockNeon(x *float32, qs *byte, coeff *float32, indexes *byte) float32

func dotQ4_KRowASM(row []float32, raw []byte, k int) float64 {
	var coeff [16]float32
	var acc float64
	for sb := 0; sb*qkK < k; sb++ {
		blk := raw[sb*q4kBlockSize : (sb+1)*q4kBlockSize]
		//perfscan:ignore PS4001 one f16 scale per 256-weight block uses a lookup conversion, not a same-layout bulk copy
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
		acc += float64(dotQ4KBlockNeon(
			&row[sb*qkK], &blk[16], &coeff[0], &qKByteToF32Indexes[0],
		))
	}
	return acc
}

func init() { dotQ4KRowFn = dotQ4_KRowASM }

const q4kDotIsAsm = true
