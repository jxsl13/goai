//go:build arm64

package gguf

import "encoding/binary"

// dotQ2KBlockNeon fuses one 256-weight Q2_K super-block's two-bit unpack,
// affine dequantization, and activation dot. The vector kernel accumulates
// in f32; dotQ2_KRowASM widens each super-block subtotal to f64 before
// combining it, so this path is tolerance-gated rather than bit-identical
// to the scalar f64-per-element reference.
//
// qs points at the block's 64-byte two-bit plane.
//
//go:noescape
func dotQ2KBlockNeon(x *float32, qs *byte, coeff *float32, indexes *byte) float64

func dotQ2_KRowASM(row []float32, raw []byte, k int) float64 {
	var coeff [32]float32
	var acc float64
	for sb := 0; sb*qkK < k; sb++ {
		blk := raw[sb*q2kBlockSize : (sb+1)*q2kBlockSize]
		//perfscan:ignore PS4001 two f16 scales per 256-weight block use lookup conversions, not a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk[80:]))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[82:]))
		for i, sc := range blk[:16] {
			coeff[2*i] = d * float32(sc&0x0f)
			coeff[2*i+1] = dmin * float32(sc>>4)
		}
		acc += dotQ2KBlockNeon(
			&row[sb*qkK], &blk[16], &coeff[0], &qKByteToF32Indexes[0],
		)
	}
	return acc
}

func init() { dotQ2KRowFn = dotQ2_KRowASM }

const q2kDotIsAsm = true
