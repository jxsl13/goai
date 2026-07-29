//go:build amd64 && goexperiment.simd

package gguf

import "encoding/binary"

// dotQ4KRowAsm: dot of activation x (k floats) with one Q4_K weight row (raw, nsb
// super-blocks). scales holds, per sub-block-pair, [d1,-off1,d2,-off2] (negated offsets
// so the affine is two FMAs). Returns the f32 dot. Tol-gated: f32 accumulation vs the
// scalar's f64 (rides the Q4_K matmul tolerance).
//
//go:noescape
func dotQ4KRowAsm(x *float32, raw *byte, scales *float32, nsb int) float32

func dotQ4_KRowASM(row []float32, raw []byte, k int) float64 {
	nsb := (k + qkK - 1) / qkK
	scales := make([]float32, nsb*4*4) // 4 pairs × 4 vals per super-block
	for sb := 0; sb < nsb; sb++ {
		blk := raw[sb*q4kBlockSize : sb*q4kBlockSize+q4kBlockSize]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:]))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[2:]))
		sbs := blk[4:16]
		for pair := 0; pair < 4; pair++ {
			is := pair * 2
			sc1, m1 := getScaleMinK4(is+0, sbs)
			sc2, m2 := getScaleMinK4(is+1, sbs)
			o := (sb*4 + pair) * 4
			scales[o+0] = d * float32(sc1)
			scales[o+1] = -dmin * float32(m1)
			scales[o+2] = d * float32(sc2)
			scales[o+3] = -dmin * float32(m2)
		}
	}
	return float64(dotQ4KRowAsm(&row[0], &raw[0], &scales[0], nsb))
}

func init() { dotQ4KRowFn = dotQ4_KRowASM }

// q4kDotIsAsm is true when the tolerance-gated asm Q4_K row dot is active (simd build).
var q4kDotIsAsm = true
