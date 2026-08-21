//go:build arm64

package gguf

import "encoding/binary"

// dotQ5KBlockNeon fuses one 256-weight Q5_K super-block's nibble and
// fifth-bit unpack, affine dequantization, and activation dot. The vector
// kernel accumulates in f32; dotQ5_KRowASM widens each super-block subtotal
// to f64 before combining it, so this path is tolerance-gated rather than
// bit-identical to the scalar f64-per-element reference.
//
// raw points at the block's 32-byte qh plane; its 128-byte qs plane follows.
//
//go:noescape
func dotQ5KBlockNeon(x *float32, raw *byte, coeff *float32, indexes *byte) float32

func dotQ5_KRowASM(row []float32, raw []byte, k int) float64 {
	var coeff [16]float32
	var acc float64
	for sb := 0; sb*qkK < k; sb++ {
		blk := raw[sb*q5kBlockSize : (sb+1)*q5kBlockSize]
		//perfscan:ignore PS4001 two f16 scales per 256-weight block use lookup conversions, not a same-layout bulk copy
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
		acc += float64(dotQ5KBlockNeon(
			&row[sb*qkK], &blk[16], &coeff[0], &qKByteToF32Indexes[0],
		))
	}
	return acc
}

func init() { dotQ5KRowFn = dotQ5_KRowASM }

const q5kDotIsAsm = true
