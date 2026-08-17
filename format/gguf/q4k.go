package gguf

import (
	"encoding/binary"

	"github.com/jxsl13/goai/tensor"
)

// Q4_K is a ggml "k-quant" format (§R100): a 256-element SUPER-BLOCK of 8 sub-blocks of 32,
// at ~4.5 bits/weight — the DOMINANT modern GGUF weight format (the bulk tensors of the
// Q4_K_M / Q4_K_S mixes; Q6_K, §R99, carries their few high-precision tensors). Unlike the
// symmetric Q6_K (scale only), Q4_K is an ASYMMETRIC AFFINE quant: each sub-block has both a
// 6-bit scale AND a 6-bit min, so y = (d·sc6)·nibble − (dmin·min6), nibble ∈ [0,15]. Block
// layout (144 bytes): f16 d (super-block scale-of-scales), f16 dmin (super-block scale-of-
// mins), scales[12] (8 six-bit scales + 8 six-bit mins bit-packed), qs[128] (256 4-bit quants).
const (
	q4kBlockSize = 144 // f16 d + f16 dmin + scales 12 + qs 128
	q4kSubs      = 8   // sub-blocks of 32 per super-block
)

// getScaleMinK4 unpacks the 6-bit scale d and 6-bit min m of sub-block j (0..7) from the
// 12-byte packed scales array — verbatim ggml get_scale_min_k4 (§R100, research-lite
// CONFIRMED unanimous). j<4 reads a byte's low 6 bits; j≥4 splices a low nibble with the
// two high bits stashed in the j<4 bytes.
func getScaleMinK4(j int, q []byte) (d, m byte) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
	m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4)
	return d, m
}

// putScaleMinK4 is the inverse of getScaleMinK4: it packs the 8 six-bit scales and 8 six-bit
// mins into the 12-byte scales array so getScaleMinK4 recovers them exactly. Derived by
// inverting the bit splices: scales[0:4] hold sc[0:4] (low 6) with sc[4:8]'s high 2 bits in
// their top 2; scales[4:8] hold min[0:4] (low 6) with min[4:8]'s high 2 bits; scales[8:12]
// hold sc[4:8] and min[4:8] low nibbles.
func putScaleMinK4(sc, mn *[q4kSubs]byte, out []byte) {
	for j := range 4 {
		out[j] = (sc[j] & 63) | ((sc[j+4] >> 4) << 6)
		out[j+4] = (mn[j] & 63) | ((mn[j+4] >> 4) << 6)
		out[j+8] = (sc[j+4] & 0xF) | ((mn[j+4] & 0xF) << 4)
	}
}

// dequantQ4_K decodes a Q4_K tensor to F32, exactly matching ggml's dequantize_row_q4_K
// (§R100): 8 sub-blocks of 32, y = d·sc6·nibble − dmin·min6. Per 64-element pair the low
// nibbles of 32 qs bytes give sub-block is+0, the high nibbles give is+1.
func dequantQ4_K(shape tensor.Shape, raw []byte) (*tensor.Tensor, error) {
	t := tensor.New(tensor.F32, shape)
	dequantQ4_KInto(t.Storage().F32(), raw)
	return t, nil
}

// dequantQ4_KInto is dequantQ4_K writing into a caller-provided dst (len = element
// count) — lets QMatMul reuse one row buffer across all weight rows instead of
// allocating a tensor per row. The fill is byte-for-byte the tensor-returning form.
func dequantQ4_KInto(dst []float32, raw []byte) {
	dequantQ4_KIntoArch(dst, raw)
}

// dequantQ4_KIntoScalar is the architecture-independent reference used by the
// portable path and as the bit-exact oracle for architecture-specific kernels.
func dequantQ4_KIntoScalar(dst []float32, raw []byte) {
	for sb := 0; sb*qkK < len(dst); sb++ {
		// Fixed-length reslices, not open-ended ones: they let the compiler prove the
		// inner stores in range, so the per-element IsInBounds on dst[base+l] and
		// dst[base+l+32] disappears. They also concentrate the length check at one
		// named site — a short raw or dst now panics here with a slice-bounds message
		// instead of mid-loop. Callers already validate (byteSize enforces
		// len%qkK == 0, and decodeTensor range-checks the offset), so this is
		// defense in depth, not a new contract.
		o := sb * q4kBlockSize
		blk := raw[o : o+q4kBlockSize]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:]))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[2:]))
		scales := blk[4:16]
		qs := blk[16:144]
		yo := sb * qkK
		for pair := range 4 { // 4 pairs of (32 low, 32 high) = 8 sub-blocks
			is, q := pair*2, qs[pair*32:pair*32+32]
			sc1, m1 := getScaleMinK4(is+0, scales)
			sc2, m2 := getScaleMinK4(is+1, scales)
			d1, off1 := d*float32(sc1), dmin*float32(m1)
			d2, off2 := d*float32(sc2), dmin*float32(m2)
			base := yo + pair*64
			y := dst[base : base+64]
			for l := range 32 {
				y[l] = d1*float32(q[l]&0xF) - off1
				y[l+32] = d2*float32(q[l]>>4) - off2
			}
		}
	}
}

// quantizeQ4_K encodes f32 values into the Q4_K super-block layout — the inverse of
// dequantQ4_K. Each 32-element sub-block gets an affine (scale, min): scale = (hi−lo)/15 and
// the offset min = −min(lo,0) ≥ 0 (ggml clamps the offset non-positive, so nibble 0 maps to
// min(lo,0)). The 8 scales and 8 mins are themselves 6-bit-quantized via super-block scales
// d = max(scale)/63 and dmin = max(min)/63. (A VALID Q4_K encoder; ggml's is an iterative
// best-fit (make_qkx2_quants) — this is not bit-identical, but round-trips through
// dequantQ4_K within the quant error, §V15.)
func quantizeQ4_K(x []float32) []byte {
	nb := len(x) / qkK
	out := make([]byte, nb*q4kBlockSize)
	for b := range nb {
		blk := x[b*qkK : (b+1)*qkK]
		var scales, mins [q4kSubs]float32
		var maxScale, maxMin float32
		//perfscan:ignore PS3067 offline quantizeQ4_K min/max loop, cold encode path
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
			scales[j] = (hi - lo) / 15
			mins[j] = -lo // ≥ 0
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
		o := b * q4kBlockSize
		binary.LittleEndian.PutUint16(out[o+0:], f32ToF16(d))
		binary.LittleEndian.PutUint16(out[o+2:], f32ToF16(dmin))
		putScaleMinK4(&sc6, &min6, out[o+4:o+16])
		qs := out[o+16 : o+144]
		for pair := range 4 {
			is := pair * 2
			d1, off1 := d*float32(sc6[is+0]), dmin*float32(min6[is+0])
			d2, off2 := d*float32(sc6[is+1]), dmin*float32(min6[is+1])
			base := pair * 64
			for l := range 32 {
				qs[pair*32+l] = q4nibbleAffine(blk[base+l], d1, off1) |
					q4nibbleAffine(blk[base+l+32], d2, off2)<<4
			}
		}
	}
	return out
}

// q4nibbleAffine inverts y = step·nibble − off → nibble = round((y+off)/step), clamped [0,15].
func q4nibbleAffine(y, step, off float32) byte {
	if step == 0 {
		return 0
	}
	q := int(roundHalfAway((y + off) / step))
	if q < 0 {
		q = 0
	}
	if q > 15 {
		q = 15
	}
	return byte(q)
}

// dotQ4_KRow folds the Q4_K super-block dequant straight into a dot against one
// activation row — the fused single-token path Q8_0 and Q4_0 already had. It mirrors
// [dequantQ4_KInto] statement for statement, reading each weight into a register
// instead of storing it and loading it back, so the per-element float32 values and the
// ascending-k accumulation order are unchanged.
func dotQ4_KRow(row []float32, raw []byte, k int) float64 {
	var acc float64
	for sb := 0; sb*qkK < k; sb++ {
		o := sb * q4kBlockSize
		blk := raw[o : o+q4kBlockSize]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:]))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[2:]))
		scales, qs := blk[4:16], blk[16:144]
		yo := sb * qkK
		for pair := range 4 { // 4 pairs of (32 low, 32 high) = 8 sub-blocks
			is, q := pair*2, qs[pair*32:pair*32+32]
			sc1, m1 := getScaleMinK4(is+0, scales)
			sc2, m2 := getScaleMinK4(is+1, scales)
			d1, off1 := d*float32(sc1), dmin*float32(m1)
			d2, off2 := d*float32(sc2), dmin*float32(m2)
			base := yo + pair*64
			xlo, xhi := row[base:base+32], row[base+32:base+64]
			//perfscan:ignore PS3010 fused Q4_K dot hot path; unroll/fusion levers bench-rejected #904
			for l := range 32 {
				acc += float64(xlo[l]) * float64(d1*float32(q[l]&0xF)-off1)
			}
			//perfscan:ignore PS3010 same fused Q4_K dot; SIMD/unroll/dp4a bench-rejected #904
			for l := range 32 {
				acc += float64(xhi[l]) * float64(d2*float32(q[l]>>4)-off2)
			}
		}
	}
	return acc
}
