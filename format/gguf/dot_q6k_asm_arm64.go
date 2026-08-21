//go:build arm64

package gguf

import "encoding/binary"

// dotQ6KBlockNeon fuses one 256-weight Q6_K super-block's six-bit unpack,
// signed scale dequantization, and activation dot. The vector kernel
// accumulates in f32; dotQ6_KRowASM widens each super-block subtotal to f64
// before combining it, so this path is tolerance-gated rather than
// bit-identical to the scalar f64-per-element reference.
//
//go:noescape
func dotQ6KBlockNeon(x *float32, raw *byte, d float32, indexes *byte) float32

func dotQ6_KRowASM(row []float32, raw []byte, k int) float64 {
	var acc float64
	for sb := 0; sb*qkK < k; sb++ {
		blk := raw[sb*q6kBlockSize : (sb+1)*q6kBlockSize]
		//perfscan:ignore PS4001 one f16 scale per 256-weight block uses a lookup conversion, not a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk[208:]))
		acc += float64(dotQ6KBlockNeon(
			&row[sb*qkK], &blk[0], d, &qKByteToF32Indexes[0],
		))
	}
	return acc
}

func init() { dotQ6KRowFn = dotQ6_KRowASM }

const q6kDotIsAsm = true
