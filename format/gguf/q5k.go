package gguf

import (
	"encoding/binary"

	"github.com/jxsl13/goai/tensor"
)

// Q5_K is a ggml "k-quant" format (§R102): a 256-element SUPER-BLOCK of 8 sub-blocks of 32,
// at ~5.5 bits/weight — the bulk tensors of the Q5_K_M / Q5_K_S mixes (the second-most-common
// download after Q4_K_M; paired with Q6_K §R99 for the few high-precision tensors). It is
// Q4_K's ASYMMETRIC AFFINE quant (§R100) — same 6-bit scale AND 6-bit min per sub-block, same
// get_scale_min_k4 packing — but 5-bit: each quant q5 ∈ [0,31] gains a high bit from a qh
// plane, so y = (d·sc6)·q5 − (dmin·min6), q5 = nibble | (highbit<<4). Block layout (176 bytes):
// f16 d, f16 dmin, scales[12] (scales+mins), qh[32] (1 high bit/quant), qs[128] (low nibbles).
const q5kBlockSize = 176 // f16 d + f16 dmin + scales 12 + qh 32 + qs 128

// dequantQ5_K decodes a Q5_K tensor to F32, exactly matching ggml's dequantize_row_q5_K
// (§R102): like dequantQ4_K but the quant is (qs nibble) + 16·(qh bit). Per 64-element pair p
// (is=2p): the low nibbles use qh bit 2p, the high nibbles use qh bit 2p+1.
func dequantQ5_K(shape tensor.Shape, raw []byte) (*tensor.Tensor, error) {
	t := tensor.New(tensor.F32, shape)
	dequantQ5_KInto(t.Storage().F32(), raw)
	return t, nil
}

// dequantQ5_KInto is dequantQ5_K writing into a caller-provided dst — lets QMatMul
// reuse one row buffer across all weight rows. Byte-for-byte the tensor form.
func dequantQ5_KInto(dst []float32, raw []byte) {
	for sb := 0; sb*qkK < len(dst); sb++ {
		blk := raw[sb*q5kBlockSize:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:]))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[2:]))
		scales := blk[4:16]
		qh := blk[16:48]
		qs := blk[48:176]
		yo := sb * qkK
		for pair := range 4 { // 4 pairs of (32 low, 32 high) = 8 sub-blocks
			is, ql := pair*2, qs[pair*32:pair*32+32]
			sc1, m1 := getScaleMinK4(is+0, scales)
			sc2, m2 := getScaleMinK4(is+1, scales)
			d1, off1 := d*float32(sc1), dmin*float32(m1)
			d2, off2 := d*float32(sc2), dmin*float32(m2)
			base := yo + pair*64
			y := dst[base : base+64]
			sh := 2 * pair
			for l, q := range ql {
				// branchless high bit: the qh bits are data-dependent, a
				// branch mispredicts ~50% (docs/perf-notes-lowlevel.md)
				h := qh[l] >> sh
				lo := int(q&0xF) | int(h&1)<<4
				hi := int(q>>4) | int(h>>1&1)<<4
				y[l] = d1*float32(lo) - off1
				y[l+32] = d2*float32(hi) - off2
			}
		}
	}
}

// quantizeQ5_K encodes f32 values into the Q5_K super-block layout — the inverse of
// dequantQ5_K. Identical to quantizeQ4_K but the affine grid is 5-bit (q5 ∈ [0,31], step
// (hi−lo)/31), and the 5th bit of each quant is split off into the qh plane. (A VALID Q5_K
// encoder; ggml's is an iterative best-fit make_qkx2_quants — this is not bit-identical, but
// round-trips through dequantQ5_K within the quant error, §V15.)
func quantizeQ5_K(x []float32) []byte {
	nb := len(x) / qkK
	out := make([]byte, nb*q5kBlockSize)
	for b := range nb {
		blk := x[b*qkK : (b+1)*qkK]
		var scales, mins [q4kSubs]float32
		var maxScale, maxMin float32
		for j := range q4kSubs {
			lo, hi := blk[j*32], blk[j*32]
			for i := 1; i < 32; i++ {
				v := blk[j*32+i]
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
			scales[j] = (hi - lo) / 31 // 5-bit grid: q5 ∈ [0,31]
			mins[j] = -lo              // ≥ 0
			if scales[j] > maxScale {
				maxScale = scales[j]
			}
			if mins[j] > maxMin {
				maxMin = mins[j]
			}
		}
		d := maxScale / 63
		dmin := maxMin / 63
		var ids, idm float32
		if d != 0 {
			ids = 1 / d
		}
		if dmin != 0 {
			idm = 1 / dmin
		}
		var sc6, min6 [q4kSubs]byte
		for j := range q4kSubs {
			sc6[j] = byte(min(int(roundHalfAway(scales[j]*ids)), 63))
			min6[j] = byte(min(int(roundHalfAway(mins[j]*idm)), 63))
		}
		o := b * q5kBlockSize
		binary.LittleEndian.PutUint16(out[o+0:], f32ToF16(d))
		binary.LittleEndian.PutUint16(out[o+2:], f32ToF16(dmin))
		putScaleMinK4(&sc6, &min6, out[o+4:o+16])
		qh := out[o+16 : o+48]
		qs := out[o+48 : o+176]
		for pair := range 4 {
			is := pair * 2
			u1, u2 := byte(1)<<(2*pair), byte(2)<<(2*pair)
			d1, off1 := d*float32(sc6[is+0]), dmin*float32(min6[is+0])
			d2, off2 := d*float32(sc6[is+1]), dmin*float32(min6[is+1])
			base := pair * 64
			for l := range 32 {
				qlo := q5quant(blk[base+l], d1, off1)
				qhi := q5quant(blk[base+l+32], d2, off2)
				qs[pair*32+l] = byte(qlo&0xF | (qhi&0xF)<<4)
				if qlo >= 16 {
					qh[l] |= u1
				}
				if qhi >= 16 {
					qh[l] |= u2
				}
			}
		}
	}
	return out
}

// q5quant inverts y = step·q5 − off → q5 = round((y+off)/step), clamped [0,31].
func q5quant(y, step, off float32) int {
	if step == 0 {
		return 0
	}
	q := int(roundHalfAway((y + off) / step))
	if q < 0 {
		q = 0
	}
	if q > 31 {
		q = 31
	}
	return q
}

// dotQ5_KRow folds the Q5_K super-block dequant into a dot against one activation row,
// mirroring [dequantQ5_KInto] statement for statement, including its branchless high-bit
// extraction (the qh bits are data-dependent and a branch mispredicts about half the
// time). Low and high nibbles are two sequential passes, as in the Q4_K twin.
func dotQ5_KRow(row []float32, raw []byte, k int) float64 {
	var acc float64
	for sb := 0; sb*qkK < k; sb++ {
		blk := raw[sb*q5kBlockSize:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:]))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[2:]))
		scales := blk[4:16]
		qh := blk[16:48]
		qs := blk[48:176]
		yo := sb * qkK
		for pair := range 4 { // 4 pairs of (32 low, 32 high) = 8 sub-blocks
			is, ql := pair*2, qs[pair*32:pair*32+32]
			sc1, m1 := getScaleMinK4(is+0, scales)
			sc2, m2 := getScaleMinK4(is+1, scales)
			d1, off1 := d*float32(sc1), dmin*float32(m1)
			d2, off2 := d*float32(sc2), dmin*float32(m2)
			base := yo + pair*64
			xlo, xhi := row[base:base+32], row[base+32:base+64]
			sh := 2 * pair
			for l, q := range ql {
				h := qh[l] >> sh
				lo := int(q&0xF) | int(h&1)<<4
				acc += float64(xlo[l]) * float64(d1*float32(lo)-off1)
			}
			for l, q := range ql {
				h := qh[l] >> sh
				hi := int(q>>4) | int(h>>1&1)<<4
				acc += float64(xhi[l]) * float64(d2*float32(hi)-off2)
			}
		}
	}
	return acc
}

// dotQ5_K4Rows is [dotQ5_KRow] over FOUR weight rows at once, sharing each activation
// load across four accumulators. Keeps the branchless high-bit extraction of the
// single-row form. Bit-exact against it: same per-element float32 weights, same
// ascending traversal, independent accumulators.
func dotQ5_K4Rows(row []float32, r0, r1, r2, r3 []byte, k int) (float64, float64, float64, float64) {
	var a0, a1, a2, a3 float64
	for sb := 0; sb*qkK < k; sb++ {
		o := sb * q5kBlockSize
		b0, b1, b2, b3 := r0[o:], r1[o:], r2[o:], r3[o:]
		d0, n0 := f16ToF32(binary.LittleEndian.Uint16(b0[0:])), f16ToF32(binary.LittleEndian.Uint16(b0[2:]))
		d1, n1 := f16ToF32(binary.LittleEndian.Uint16(b1[0:])), f16ToF32(binary.LittleEndian.Uint16(b1[2:]))
		d2, n2 := f16ToF32(binary.LittleEndian.Uint16(b2[0:])), f16ToF32(binary.LittleEndian.Uint16(b2[2:]))
		d3, n3 := f16ToF32(binary.LittleEndian.Uint16(b3[0:])), f16ToF32(binary.LittleEndian.Uint16(b3[2:]))
		yo := sb * qkK
		for pair := range 4 {
			is, sh := pair*2, 2*pair
			qh0, qh1, qh2, qh3 := b0[16:48], b1[16:48], b2[16:48], b3[16:48]
			ql0, ql1 := b0[48+pair*32:48+pair*32+32], b1[48+pair*32:48+pair*32+32]
			ql2, ql3 := b2[48+pair*32:48+pair*32+32], b3[48+pair*32:48+pair*32+32]
			sc00, mn00 := getScaleMinK4(is+0, b0[4:16])
			sc01, mn01 := getScaleMinK4(is+1, b0[4:16])
			sc10, mn10 := getScaleMinK4(is+0, b1[4:16])
			sc11, mn11 := getScaleMinK4(is+1, b1[4:16])
			sc20, mn20 := getScaleMinK4(is+0, b2[4:16])
			sc21, mn21 := getScaleMinK4(is+1, b2[4:16])
			sc30, mn30 := getScaleMinK4(is+0, b3[4:16])
			sc31, mn31 := getScaleMinK4(is+1, b3[4:16])
			dl0, of0, dh0, og0 := d0*float32(sc00), n0*float32(mn00), d0*float32(sc01), n0*float32(mn01)
			dl1, of1, dh1, og1 := d1*float32(sc10), n1*float32(mn10), d1*float32(sc11), n1*float32(mn11)
			dl2, of2, dh2, og2 := d2*float32(sc20), n2*float32(mn20), d2*float32(sc21), n2*float32(mn21)
			dl3, of3, dh3, og3 := d3*float32(sc30), n3*float32(mn30), d3*float32(sc31), n3*float32(mn31)
			base := yo + pair*64
			xlo, xhi := row[base:base+32], row[base+32:base+64]
			for l := range 32 {
				xv := float64(xlo[l])
				a0 += xv * float64(dl0*float32(int(ql0[l]&0xF)|int((qh0[l]>>sh)&1)<<4)-of0)
				a1 += xv * float64(dl1*float32(int(ql1[l]&0xF)|int((qh1[l]>>sh)&1)<<4)-of1)
				a2 += xv * float64(dl2*float32(int(ql2[l]&0xF)|int((qh2[l]>>sh)&1)<<4)-of2)
				a3 += xv * float64(dl3*float32(int(ql3[l]&0xF)|int((qh3[l]>>sh)&1)<<4)-of3)
			}
			for l := range 32 {
				xv := float64(xhi[l])
				a0 += xv * float64(dh0*float32(int(ql0[l]>>4)|int((qh0[l]>>sh)>>1&1)<<4)-og0)
				a1 += xv * float64(dh1*float32(int(ql1[l]>>4)|int((qh1[l]>>sh)>>1&1)<<4)-og1)
				a2 += xv * float64(dh2*float32(int(ql2[l]>>4)|int((qh2[l]>>sh)>>1&1)<<4)-og2)
				a3 += xv * float64(dh3*float32(int(ql3[l]>>4)|int((qh3[l]>>sh)>>1&1)<<4)-og3)
			}
		}
	}
	return a0, a1, a2, a3
}
