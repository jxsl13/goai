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

// dotQ4KPairRowNeon keeps paired Q4_K header decode, coefficient staging, and
// all super-block dots inside one assembly call. Each block still reduces to
// f32 before the row accumulators widen and add it in order as f64.
//
//go:noescape
func dotQ4KPairRowNeon(x *float32, raw0, raw1 *byte, f16 *float32, indexes *byte, blocks int) (out0, out1 float64)

func dotQ4_KRowASM(row []float32, raw []byte, k int) float64 {
	var coeff [16]float32
	var acc float64
	for sb := 0; sb*qkK < k; sb++ {
		blk := raw[sb*q4kBlockSize : (sb+1)*q4kBlockSize]
		//perfscan:ignore PS4001 one f16 scale per 256-weight block uses a lookup conversion, not a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:]))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[2:]))
		scales := blk[4:16]
		for j := range 4 {
			s, m, hiBits := scales[j], scales[j+4], scales[j+8]
			lo, hi := j*2, (j+4)*2
			coeff[lo+0] = d * float32(s&63)
			coeff[lo+1] = dmin * float32(m&63)
			coeff[hi+0] = d * float32((hiBits&0xF)|((s>>6)<<4))
			coeff[hi+1] = dmin * float32((hiBits>>4)|((m>>6)<<4))
		}
		acc += float64(dotQ4KBlockNeon(
			&row[sb*qkK], &blk[16], &coeff[0], &qKByteToF32Indexes[0],
		))
	}
	return acc
}

func dotQ4KPairRowASM(row []float32, raw0, raw1 []byte, k int) (float64, float64) {
	if k <= 0 {
		return 0, 0
	}
	return dotQ4KPairRowNeon(
		&row[0], &raw0[0], &raw1[0], &f16Table[0], &qKByteToF32Indexes[0], (k+qkK-1)/qkK,
	)
}

func init() {
	dotQ4KRowFn = dotQ4_KRowASM
	dotQ4KPairRowFn = dotQ4KPairRowASM
}

const q4kDotIsAsm = true
