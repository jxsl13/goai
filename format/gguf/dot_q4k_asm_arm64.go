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

// dotQ4KPairBlockNeon computes two independent Q4_K block dots while loading
// each activation vector once. Each output retains dotQ4KBlockNeon's exact
// instruction and reduction order.
//
//go:noescape
func dotQ4KPairBlockNeon(x *float32, qs0, qs1 *byte, coeff0, coeff1 *float32, indexes *byte) (out0, out1 float32)

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

func dotQ4KPairRowASM(row []float32, raw0, raw1 []byte, k int) (float64, float64) {
	var coeff0, coeff1 [16]float32
	var acc0, acc1 float64
	for sb := 0; sb*qkK < k; sb++ {
		blk0 := raw0[sb*q4kBlockSize : (sb+1)*q4kBlockSize]
		blk1 := raw1[sb*q4kBlockSize : (sb+1)*q4kBlockSize]
		//perfscan:ignore PS4001 four f16 scalars per paired 256-weight block are coefficient metadata, not a bulk-copy loop
		d0 := f16ToF32(binary.LittleEndian.Uint16(blk0[0:]))
		dmin0 := f16ToF32(binary.LittleEndian.Uint16(blk0[2:]))
		d1 := f16ToF32(binary.LittleEndian.Uint16(blk1[0:]))
		dmin1 := f16ToF32(binary.LittleEndian.Uint16(blk1[2:]))
		scales0, scales1 := blk0[4:16], blk1[4:16]
		for pair := range 4 {
			is := pair * 2
			sc00, m00 := getScaleMinK4(is, scales0)
			sc01, m01 := getScaleMinK4(is+1, scales0)
			sc10, m10 := getScaleMinK4(is, scales1)
			sc11, m11 := getScaleMinK4(is+1, scales1)
			ci := pair * 4
			coeff0[ci+0] = d0 * float32(sc00)
			coeff0[ci+1] = dmin0 * float32(m00)
			coeff0[ci+2] = d0 * float32(sc01)
			coeff0[ci+3] = dmin0 * float32(m01)
			coeff1[ci+0] = d1 * float32(sc10)
			coeff1[ci+1] = dmin1 * float32(m10)
			coeff1[ci+2] = d1 * float32(sc11)
			coeff1[ci+3] = dmin1 * float32(m11)
		}
		dot0, dot1 := dotQ4KPairBlockNeon(
			&row[sb*qkK], &blk0[16], &blk1[16], &coeff0[0], &coeff1[0], &qKByteToF32Indexes[0],
		)
		acc0 += float64(dot0)
		acc1 += float64(dot1)
	}
	return acc0, acc1
}

func init() {
	dotQ4KRowFn = dotQ4_KRowASM
	dotQ4KPairRowFn = dotQ4KPairRowASM
}

const q4kDotIsAsm = true
