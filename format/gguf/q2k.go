package gguf

import (
	"encoding/binary"

	"github.com/jxsl13/goai/tensor"
)

// Q2_K is a ggml "k-quant" format (§R104): a 256-element SUPER-BLOCK of 16 sub-blocks of 16,
// at ~2.63 bits/weight — the smallest mainstream quant, used by the Q2_K / Q2_K_S mixes to run
// very large models (70B–405B) on tight hardware. Like Q4_K it is ASYMMETRIC AFFINE (a scale
// AND a subtracted min per sub-block), but coarser everywhere: the quant q2 ∈ [0,3] is 2-bit,
// and the per-sub-block scale and min are plain 4-bit NIBBLES (no 6-bit get_scale_min_k4
// packing). Reconstruction y = d·sc4·q2 − dmin·min4. Block layout (84 bytes): scales[16]
// (low nibble = 4-bit scale, high nibble = 4-bit min), qs[64] (256 2-bit quants), f16 d
// (super-block scale-of-scales), f16 dmin (super-block scale-of-mins).
const q2kBlockSize = 84 // scales 16 + qs 64 + f16 d + f16 dmin

// dequantQ2_K decodes a Q2_K tensor to F32, exactly matching ggml's dequantize_row_q2_K
// (§R104): per 128-element block, shift=2j (j=0..3), two 16-groups per j; each group's scale4
// and min4 are the low/high nibbles of scales[is]; y = d·sc4·((q>>shift)&3) − dmin·min4.
func dequantQ2_K(shape tensor.Shape, raw []byte) (*tensor.Tensor, error) {
	t := tensor.New(tensor.F32, shape)
	dequantQ2_KInto(t.Storage().F32(), raw)
	return t, nil
}

// dequantQ2_KInto is dequantQ2_K writing into a caller-provided dst — lets QMatMul
// reuse one row buffer across all weight rows. Byte-for-byte the tensor form.
func dequantQ2_KInto(dst []float32, raw []byte) {
	for sb := 0; sb*qkK < len(dst); sb++ {
		blk := raw[sb*q2kBlockSize:]
		scales := blk[0:16]
		qsAll := blk[16:80]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[80:82]))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[82:84]))
		yi := sb * qkK
		is := 0
		for nb := range 2 { // n = 0, 128
			q := qsAll[nb*32 : nb*32+32]
			shift := 0
			for range 4 { // j = 0..3
				for _, g := range [2]int{0, 16} { // two groups of 16 (q[l], q[l+16])
					sc := scales[is]
					is++
					dl := d * float32(sc&0xF)
					ml := dmin * float32(sc>>4)
					y := dst[yi : yi+16]
					for l := range y {
						y[l] = dl*float32((q[l+g]>>shift)&3) - ml
					}
					yi += 16
				}
				shift += 2
			}
		}
	}
}

// quantizeQ2_K encodes f32 values into the Q2_K super-block layout — the inverse of
// dequantQ2_K. Each 16-element sub-block gets an affine (scale, min): scale = (hi−lo)/3 over
// the 2-bit grid and offset min = −min(lo,0) ≥ 0 (floor clamped non-positive like ggml); the
// 16 scales and mins are 4-bit-quantized through the super-block scales d = max(scale)/15 and
// dmin = max(min)/15. (A VALID Q2_K encoder; ggml's is an iterative best-fit make_qkx2_quants —
// not bit-identical, but round-trips through dequantQ2_K within the k-quant error, §V15.)
func quantizeQ2_K(x []float32) []byte {
	nb := len(x) / qkK
	out := make([]byte, nb*q2kBlockSize)
	for b := range nb {
		blk := x[b*qkK : (b+1)*qkK]
		var scales, mins [16]float32
		var maxScale, maxMin float32
		for is := range 16 {
			nblock, r := is/8, is%8
			j, g := r/2, (r%2)*16
			lo, hi := blk[nblock*128+j*32+g], blk[nblock*128+j*32+g]
			for l := 1; l < 16; l++ {
				v := blk[nblock*128+j*32+g+l]
				if v < lo {
					lo = v
				}
				if v > hi {
					hi = v
				}
			}
			if lo > 0 { // ggml clamps the affine offset non-positive
				lo = 0
			}
			scales[is] = (hi - lo) / 3 // 2-bit grid: q2 ∈ [0,3]
			mins[is] = -lo             // ≥ 0
			if scales[is] > maxScale {
				maxScale = scales[is]
			}
			if mins[is] > maxMin {
				maxMin = mins[is]
			}
		}
		d := maxScale / 15
		dmin := maxMin / 15
		var ids, idm float32
		if d != 0 {
			ids = 1 / d
		}
		if dmin != 0 {
			idm = 1 / dmin
		}
		o := b * q2kBlockSize
		scOut := out[o : o+16]
		qs := out[o+16 : o+80]
		binary.LittleEndian.PutUint16(out[o+80:], f32ToF16(d))
		binary.LittleEndian.PutUint16(out[o+82:], f32ToF16(dmin))
		for is := range 16 {
			sc4 := min(int(roundHalfAway(scales[is]*ids)), 15)
			mn4 := min(int(roundHalfAway(mins[is]*idm)), 15)
			scOut[is] = byte(sc4) | byte(mn4)<<4
			nblock, r := is/8, is%8
			j, g := r/2, (r%2)*16
			dl := d * float32(sc4)
			ml := dmin * float32(mn4)
			for l := range 16 {
				q2 := 0
				if dl != 0 {
					q2 = int(roundHalfAway((blk[nblock*128+j*32+g+l] + ml) / dl))
					if q2 < 0 {
						q2 = 0
					}
					if q2 > 3 {
						q2 = 3
					}
				}
				qs[nblock*32+g+l] |= byte(q2) << (2 * j)
			}
		}
	}
	return out
}

// dotQ2_KRow folds the Q2_K super-block dequant into a dot against one activation row,
// mirroring [dequantQ2_KInto] statement for statement: each weight goes to a register
// instead of a store and a reload. Same per-element float32 values, same ascending-k
// order, so the output is the general path's — checked by the fused-vs-general gate.
func dotQ2_KRow(row []float32, raw []byte, k int) float64 {
	var acc float64
	for sb := 0; sb*qkK < k; sb++ {
		blk := raw[sb*q2kBlockSize:]
		scales := blk[0:16]
		qsAll := blk[16:80]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[80:82]))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[82:84]))
		yi := sb * qkK
		is := 0
		for nb := range 2 { // n = 0, 128
			q := qsAll[nb*32 : nb*32+32]
			shift := 0
			for range 4 { // j = 0..3
				for _, g := range [2]int{0, 16} { // two groups of 16 (q[l], q[l+16])
					sc := scales[is]
					is++
					dl := d * float32(sc&0xF)
					ml := dmin * float32(sc>>4)
					x := row[yi : yi+16]
					for l := range x {
						acc += float64(x[l]) * float64(dl*float32((q[l+g]>>shift)&3)-ml)
					}
					yi += 16
				}
				shift += 2
			}
		}
	}
	return acc
}
