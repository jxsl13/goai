//go:build arm64

package gguf

import "encoding/binary"

// dotQ3KBlockNeon fuses one 256-weight Q3_K super-block's two-bit and
// inverted high-mask unpack, signed-scale dequantization, and activation
// dot. The vector kernel accumulates in f32; dotQ3_KRowASM widens each
// super-block subtotal to f64 before combining it, so this path is
// tolerance-gated rather than bit-identical to the scalar f64-per-element
// reference.
//
// raw points at the block's 32-byte high mask; its 64-byte qs plane follows.
//
//go:noescape
func dotQ3KBlockNeon(x *float32, raw *byte, coeff *float32, indexes *byte) float32

func dotQ3_KRowASM(row []float32, raw []byte, k int) float64 {
	var coeff [16]float32
	var acc float64
	for sb := 0; sb*qkK < k; sb++ {
		blk := raw[sb*q3kBlockSize : (sb+1)*q3kBlockSize]
		sc := q3kUnpackScales(blk[96:108])
		//perfscan:ignore PS4001 one f16 scale per 256-weight block uses a lookup conversion, not a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk[108:]))
		for i := range coeff {
			coeff[i] = d * float32(int(sc[i])-32)
		}
		acc += float64(dotQ3KBlockNeon(
			&row[sb*qkK], &blk[0], &coeff[0], &qKByteToF32Indexes[0],
		))
	}
	return acc
}

func init() { dotQ3KRowFn = dotQ3_KRowASM }

const q3kDotIsAsm = true
