package gguf

import (
	"encoding/binary"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// Q6_K is a ggml "k-quant" format (§R99): a 256-element SUPER-BLOCK of 16 sub-blocks of
// 16, at ~6.56 bits/weight, used by modern GGUF models (standalone and as the
// higher-precision tensors of the Q4_K_M / Q5_K_M mixes). Block layout (210 bytes):
// ql[128] (low 4 bits of each of 256 quants), qh[64] (high 2 bits, 4 per byte),
// scales[16] (int8 per-sub-block scale), then the f16 super-block scale d.
const (
	qkK          = 256 // QK_K super-block size
	q6kBlockSize = 210 // ql 128 + qh 64 + scales 16 + f16 d
)

// dequantQ6_K decodes a Q6_K tensor to F32, exactly matching ggml's dequantize_row_q6_K
// (§R99): y = d · scales[sub] · (q6 − 32), the 6-bit quant q6 assembled from ql (low
// nibble) and qh (2 high bits), the sub-block scale int8·d.
func dequantQ6_K(shape tensor.Shape, raw []byte) (*tensor.Tensor, error) {
	t := tensor.New(tensor.F32, shape)
	dequantQ6_KInto(t.Storage().F32(), raw)
	return t, nil
}

// dequantQ6_KInto is dequantQ6_K writing into a caller-provided dst — lets QMatMul
// reuse one row buffer across all weight rows. Byte-for-byte the tensor-returning form.
func dequantQ6_KInto(dst []float32, raw []byte) {
	for sb := 0; sb*qkK < len(dst); sb++ {
		blk := raw[sb*q6kBlockSize:]
		ql, qh, sc := blk[0:128], blk[128:192], blk[192:208]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[208:]))
		base := sb * qkK
		for grp := range 2 { // two 128-element groups per super-block
			qlo, qho, sco, yo := grp*64, grp*32, grp*8, base+grp*128
			// The 8 sub-block scale products d·sc[k] are piecewise-constant across the 32-iter
			// loop (is = l/16 flips once), so recompute them once here instead of 16x each —
			// the hoist Q4_K/Q5_K already ship. Bit-exact: Go evaluates d*float32(int8(sc))*
			// float32(q-32) left-to-right, so dsc[k] holds the identical first product.
			var dsc [8]float32
			for k := range dsc {
				dsc[k] = d * float32(int8(sc[sco+k]))
			}
			for l := range 32 {
				is := l / 16
				q1 := int(ql[qlo+l]&0xF) | int(qh[qho+l]&3)<<4
				q2 := int(ql[qlo+l+32]&0xF) | int((qh[qho+l]>>2)&3)<<4
				q3 := int(ql[qlo+l]>>4) | int((qh[qho+l]>>4)&3)<<4
				q4 := int(ql[qlo+l+32]>>4) | int((qh[qho+l]>>6)&3)<<4
				dst[yo+l+0] = dsc[is+0] * float32(q1-32)
				dst[yo+l+32] = dsc[is+2] * float32(q2-32)
				dst[yo+l+64] = dsc[is+4] * float32(q3-32)
				dst[yo+l+96] = dsc[is+6] * float32(q4-32)
			}
		}
	}
}

// quantizeQ6_K encodes f32 values into the Q6_K super-block layout — the inverse packing
// of dequantQ6_K. Each 16-element sub-block gets a symmetric scale amax/32; those 16
// scales are themselves quantized to int8 through the super-block scale d = max|scale|/127.
// (A valid Q6_K encoder; ggml's is an iterative best-fit — this is not bit-identical to
// it, but round-trips through dequantQ6_K within the quant error, §V15.)
func quantizeQ6_K(x []float32) []byte {
	nb := len(x) / qkK
	out := make([]byte, nb*q6kBlockSize)
	for b := range nb {
		blk := x[b*qkK : (b+1)*qkK]
		// per-sub-block desired scale, then the super-block scale d over those 16 scales
		var scales [16]float32
		var maxScale float32
		for j := range 16 {
			var amax float32
			for i := range 16 {
				if a := math.Float32frombits(math.Float32bits(blk[j*16+i]) &^ (1 << 31)); a > amax {
					amax = a
				}
			}
			scales[j] = amax / 32
			if scales[j] > maxScale {
				maxScale = scales[j]
			}
		}
		d := maxScale / 127
		var id float32
		if d != 0 {
			id = 1 / d
		}
		o := b * q6kBlockSize
		ql, qh, sc := out[o:o+128], out[o+128:o+192], out[o+192:o+208]
		binary.LittleEndian.PutUint16(out[o+208:], f32ToF16(d))
		var q [256]byte // 6-bit quants
		for j := range 16 {
			s8 := min(int(roundHalfAway(scales[j]*id)), 127) // int8 sub-block scale
			sc[j] = byte(int8(s8))
			sub := d * float32(s8) // effective sub-block scale
			for i := range 16 {
				qi := 32
				if sub != 0 {
					qi = int(roundHalfAway(blk[j*16+i]/sub)) + 32
				}
				if qi < 0 {
					qi = 0
				}
				if qi > 63 {
					qi = 63
				}
				q[j*16+i] = byte(qi)
			}
		}
		// pack q back into ql/qh, inverting the dequant assembly
		for grp := range 2 {
			qlo, qho, yo := grp*64, grp*32, grp*128
			for l := range 32 {
				q1, q2, q3, q4 := q[yo+l], q[yo+l+32], q[yo+l+64], q[yo+l+96]
				ql[qlo+l] = q1&0xF | (q3&0xF)<<4
				ql[qlo+l+32] = q2&0xF | (q4&0xF)<<4
				qh[qho+l] = (q1>>4)&3 | ((q2>>4)&3)<<2 | ((q3>>4)&3)<<4 | ((q4>>4)&3)<<6
			}
		}
	}
	return out
}

// dotQ6_KRow folds the Q6_K super-block dequant straight into a dot against one
// activation row, mirroring [dequantQ6_KInto] statement for statement including its
// hoisted d·sc[k] products.
//
// Unlike the Q4_0/Q4_K fused paths, this one does NOT accumulate in ascending k: the
// dequant writes four interleaved streams (l, l+32, l+64, l+96), and following that
// traversal is what keeps the per-element float32 weights identical. The float64
// accumulator sums the same set of exact products in a different order, and the result
// narrows to float32 — so the output is unchanged, which the fused-vs-general gate
// checks rather than assumes.
func dotQ6_KRow(row []float32, raw []byte, k int) float64 {
	var acc float64
	for sb := 0; sb*qkK < k; sb++ {
		blk := raw[sb*q6kBlockSize:]
		ql, qh, sc := blk[0:128], blk[128:192], blk[192:208]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[208:]))
		base := sb * qkK
		for grp := range 2 { // two 128-element groups per super-block
			qlo, qho, sco, yo := grp*64, grp*32, grp*8, base+grp*128
			var dsc [8]float32
			for k := range dsc {
				dsc[k] = d * float32(int8(sc[sco+k]))
			}
			y := row[yo : yo+128]
			for l := range 32 {
				is := l / 16
				q1 := int(ql[qlo+l]&0xF) | int(qh[qho+l]&3)<<4
				q2 := int(ql[qlo+l+32]&0xF) | int((qh[qho+l]>>2)&3)<<4
				q3 := int(ql[qlo+l]>>4) | int((qh[qho+l]>>4)&3)<<4
				q4 := int(ql[qlo+l+32]>>4) | int((qh[qho+l]>>6)&3)<<4
				acc += float64(y[l+0]) * float64(dsc[is+0]*float32(q1-32))
				acc += float64(y[l+32]) * float64(dsc[is+2]*float32(q2-32))
				acc += float64(y[l+64]) * float64(dsc[is+4]*float32(q3-32))
				acc += float64(y[l+96]) * float64(dsc[is+6]*float32(q4-32))
			}
		}
	}
	return acc
}

// dotQ6_K4Rows is [dotQ6_KRow] over FOUR weight rows at once, sharing each activation
// load across four accumulators. Bit-exact against the single-row form: same per-element
// float32 weights, same traversal, independent accumulators.
func dotQ6_K4Rows(row []float32, r0, r1, r2, r3 []byte, k int) (float64, float64, float64, float64) {
	var a0, a1, a2, a3 float64
	for sb := 0; sb*qkK < k; sb++ {
		o := sb * q6kBlockSize
		b0, b1, b2, b3 := r0[o:], r1[o:], r2[o:], r3[o:]
		d0 := f16ToF32(binary.LittleEndian.Uint16(b0[208:]))
		d1 := f16ToF32(binary.LittleEndian.Uint16(b1[208:]))
		d2 := f16ToF32(binary.LittleEndian.Uint16(b2[208:]))
		d3 := f16ToF32(binary.LittleEndian.Uint16(b3[208:]))
		base := sb * qkK
		for grp := range 2 {
			qlo, qho, sco, yo := grp*64, grp*32, grp*8, base+grp*128
			var s0, s1, s2, s3 [8]float32
			for j := range 8 {
				s0[j] = d0 * float32(int8(b0[192+sco+j]))
				s1[j] = d1 * float32(int8(b1[192+sco+j]))
				s2[j] = d2 * float32(int8(b2[192+sco+j]))
				s3[j] = d3 * float32(int8(b3[192+sco+j]))
			}
			y := row[yo : yo+128]
			for l := range 32 {
				is := l / 16
				x0, x32 := float64(y[l+0]), float64(y[l+32])
				x64, x96 := float64(y[l+64]), float64(y[l+96])

				h0, l0 := b0[128+qho+l], b0[qlo+l]
				l0b := b0[qlo+l+32]
				a0 += x0 * float64(s0[is+0]*float32((int(l0&0xF)|int(h0&3)<<4)-32))
				a0 += x32 * float64(s0[is+2]*float32((int(l0b&0xF)|int((h0>>2)&3)<<4)-32))
				a0 += x64 * float64(s0[is+4]*float32((int(l0>>4)|int((h0>>4)&3)<<4)-32))
				a0 += x96 * float64(s0[is+6]*float32((int(l0b>>4)|int((h0>>6)&3)<<4)-32))

				h1, l1 := b1[128+qho+l], b1[qlo+l]
				l1b := b1[qlo+l+32]
				a1 += x0 * float64(s1[is+0]*float32((int(l1&0xF)|int(h1&3)<<4)-32))
				a1 += x32 * float64(s1[is+2]*float32((int(l1b&0xF)|int((h1>>2)&3)<<4)-32))
				a1 += x64 * float64(s1[is+4]*float32((int(l1>>4)|int((h1>>4)&3)<<4)-32))
				a1 += x96 * float64(s1[is+6]*float32((int(l1b>>4)|int((h1>>6)&3)<<4)-32))

				h2, l2 := b2[128+qho+l], b2[qlo+l]
				l2b := b2[qlo+l+32]
				a2 += x0 * float64(s2[is+0]*float32((int(l2&0xF)|int(h2&3)<<4)-32))
				a2 += x32 * float64(s2[is+2]*float32((int(l2b&0xF)|int((h2>>2)&3)<<4)-32))
				a2 += x64 * float64(s2[is+4]*float32((int(l2>>4)|int((h2>>4)&3)<<4)-32))
				a2 += x96 * float64(s2[is+6]*float32((int(l2b>>4)|int((h2>>6)&3)<<4)-32))

				h3, l3 := b3[128+qho+l], b3[qlo+l]
				l3b := b3[qlo+l+32]
				a3 += x0 * float64(s3[is+0]*float32((int(l3&0xF)|int(h3&3)<<4)-32))
				a3 += x32 * float64(s3[is+2]*float32((int(l3b&0xF)|int((h3>>2)&3)<<4)-32))
				a3 += x64 * float64(s3[is+4]*float32((int(l3>>4)|int((h3>>4)&3)<<4)-32))
				a3 += x96 * float64(s3[is+6]*float32((int(l3b>>4)|int((h3>>6)&3)<<4)-32))
			}
		}
	}
	return a0, a1, a2, a3
}
