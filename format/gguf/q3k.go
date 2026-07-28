package gguf

import (
	"encoding/binary"

	"github.com/jxsl13/goai/tensor"
)

// Q3_K is a ggml "k-quant" format (§R103): a 256-element SUPER-BLOCK of 16 sub-blocks of 16,
// at ~3.44 bits/weight — the bulk tensors of the Q3_K_M / Q3_K_L / Q3_K_S mixes (run large
// models on limited hardware; paired with higher-precision tensors per the mix). Unlike the
// affine Q4_K/Q5_K, Q3_K is SYMMETRIC (a single super-block scale d, no min): each 3-bit quant
// q3 ∈ [0,7] reconstructs as y = d·(sc6−32)·(q3−4), q3 = (2 low bits from qs) | (1 high bit
// from hmask)<<2. The per-sub-block scale is a SIGNED 6-bit value sc6−32 ∈ [−32,31]. Block
// layout (110 bytes, d LAST): hmask[32] (high bit/quant), qs[64] (2 low bits/quant),
// scales[12] (16 six-bit scales, bit-packed via the ggml aux/kmask splice), then f16 d.
const q3kBlockSize = 110 // hmask 32 + qs 64 + scales 12 + f16 d

// q3kUnpackScales extracts the 16 six-bit scales from the 12 packed bytes, verbatim ggml
// dequantize_row_q3_K (§R103): the aux[4]/kmask1/kmask2 splice. Each result byte is a 6-bit
// value in [0,63], used as (v−32) for the signed sub-block scale.
func q3kUnpackScales(raw []byte) [16]byte {
	const kmask1, kmask2 = uint32(0x03030303), uint32(0x0f0f0f0f)
	var aux [4]uint32
	aux[0] = binary.LittleEndian.Uint32(raw[0:])
	aux[1] = binary.LittleEndian.Uint32(raw[4:])
	aux[2] = binary.LittleEndian.Uint32(raw[8:])
	tmp := aux[2]
	aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
	aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
	aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
	aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
	var sc [16]byte
	for k := range 4 {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], aux[k])
		copy(sc[k*4:], b[:])
	}
	return sc
}

// packScalesQ3_K is the inverse of q3kUnpackScales: it packs 16 six-bit scales (0..63) into the
// 12-byte scales array so q3kUnpackScales recovers them exactly. Derived by inverting the aux
// splice — the low nibble of each of the 16 values goes to bytes 0..7 (aux[0],aux[1]), and the
// two high bits go, 4 packed per byte, into bytes 8..11 (aux[2]) at the source-specific shift.
func packScalesQ3_K(sc *[16]byte, out []byte) {
	// low 4 bits of scales 0..7 → out[0..7]; low 4 bits of scales 8..15 → high nibble of out[0..7]
	for i := range 8 {
		out[i] = (sc[i] & 0xF) | ((sc[i+8] & 0xF) << 4)
	}
	// high 2 bits of each scale → out[8..11], 4 scales per byte, matching the aux[2] unpack
	for i := range 4 {
		out[8+i] = ((sc[i] >> 4) & 3) | (((sc[i+4] >> 4) & 3) << 2) |
			(((sc[i+8] >> 4) & 3) << 4) | (((sc[i+12] >> 4) & 3) << 6)
	}
}

// dequantQ3_K decodes a Q3_K tensor to F32, exactly matching ggml's dequantize_row_q3_K
// (§R103). Per 128-element block the quant's 2 low bits come from qs[l] at shift 2j and its
// high bit from hmask[l] at bit m (m doubles per j → bits 0..7 over the super-block); the
// INVERTED hmask arithmetic subtracts 4 when the bit is CLEAR.
func dequantQ3_K(shape tensor.Shape, raw []byte) (*tensor.Tensor, error) {
	t := tensor.New(tensor.F32, shape)
	dequantQ3_KInto(t.Storage().F32(), raw)
	return t, nil
}

// dequantQ3_KInto is dequantQ3_K writing into a caller-provided dst — lets QMatMul
// reuse one row buffer across all weight rows. Byte-for-byte the tensor form.
func dequantQ3_KInto(dst []float32, raw []byte) {
	for sb := 0; sb*qkK < len(dst); sb++ {
		blk := raw[sb*q3kBlockSize:]
		hm := blk[0:32]
		qsAll := blk[32:96]
		sc := q3kUnpackScales(blk[96:108])
		dAll := f16ToF32(binary.LittleEndian.Uint16(blk[108:110]))
		yi := sb * qkK
		m := byte(1)
		is := 0
		for nb := range 2 { // n = 0, 128
			q := qsAll[nb*32 : nb*32+32]
			shift := 0
			for range 4 { // j = 0..3
				for _, g := range [2]int{0, 16} { // two groups of 16 (q[l], q[l+16])
					dl := dAll * float32(int(sc[is])-32)
					is++
					// NOTE: measured — this loop resists both a dst subslice and a
					// branchless h rewrite (the h-select already compiles to a
					// CSEL on arm64; both variants were ≥4% slower,
					// docs/perf-notes-lowlevel.md).
					for l := range 16 {
						h := 4
						if hm[l+g]&m != 0 {
							h = 0
						}
						dst[yi] = dl * float32(int((q[l+g]>>shift)&3)-h)
						yi++
					}
				}
				shift += 2
				m <<= 1
			}
		}
	}
}

// quantizeQ3_K encodes f32 values into the Q3_K super-block layout — the inverse of
// dequantQ3_K. Each 16-element sub-block gets a symmetric signed scale: q3−4 ∈ [−4,3], so the
// effective scale is max(posmax/3, negmax/4) (neither the +3 nor the −4 extreme clips); the 16
// scales are 6-bit-quantized through the super-block scale d = maxScale/31 and stored as
// sc6 = round(scale/d)+32. (A VALID Q3_K encoder using non-negative sub-scales; ggml's
// make_q3_quants is an iterative signed best-fit — this is not bit-identical, but round-trips
// through dequantQ3_K within the k-quant error, §V15.)
func quantizeQ3_K(x []float32) []byte {
	nb := len(x) / qkK
	out := make([]byte, nb*q3kBlockSize)
	for b := range nb {
		blk := x[b*qkK : (b+1)*qkK]
		// per-sub-block effective scale and the super-block max
		var scale [16]float32
		var maxScale float32
		for is := range 16 {
			nblock, r := is/8, is%8
			j, g := r/2, (r%2)*16
			var posmax, negmax float32
			for l := range 16 {
				v := blk[nblock*128+j*32+g+l]
				if v > posmax {
					posmax = v
				}
				if -v > negmax {
					negmax = -v
				}
			}
			s := max(posmax/3, negmax/4) // neither the +3 nor the −4 quant extreme clips
			scale[is] = s
			if s > maxScale {
				maxScale = s
			}
		}
		d := maxScale / 31
		var id float32
		if d != 0 {
			id = 1 / d
		}
		var sc6 [16]byte
		for is := range 16 {
			sc6[is] = byte(min(int(roundHalfAway(scale[is]*id)), 31) + 32)
		}
		o := b * q3kBlockSize
		hm := out[o : o+32]
		qs := out[o+32 : o+96]
		packScalesQ3_K(&sc6, out[o+96:o+108])
		binary.LittleEndian.PutUint16(out[o+108:], f32ToF16(d))
		for is := range 16 {
			nblock, r := is/8, is%8
			j, g := r/2, (r%2)*16
			dl := d * float32(int(sc6[is])-32)
			mbit := byte(1) << (4*nblock + j)
			for l := range 16 {
				p := nblock*128 + j*32 + g + l
				q3 := 4
				if dl != 0 {
					q3 = int(roundHalfAway(blk[p]/dl)) + 4
					if q3 < 0 {
						q3 = 0
					}
					if q3 > 7 {
						q3 = 7
					}
				}
				qs[nblock*32+g+l] |= byte(q3&3) << (2 * j)
				if q3&4 != 0 {
					hm[g+l] |= mbit
				}
			}
		}
	}
	return out
}

// dotQ3_KRow folds the Q3_K super-block dequant into a dot against one activation row,
// mirroring [dequantQ3_KInto] statement for statement. The h-select is left exactly as
// that loop writes it: it already compiles to a CSEL on arm64, and both a branchless
// rewrite and a dst subslice measured slower there.
func dotQ3_KRow(row []float32, raw []byte, k int) float64 {
	var acc float64
	for sb := 0; sb*qkK < k; sb++ {
		blk := raw[sb*q3kBlockSize:]
		hm := blk[0:32]
		qsAll := blk[32:96]
		sc := q3kUnpackScales(blk[96:108])
		dAll := f16ToF32(binary.LittleEndian.Uint16(blk[108:110]))
		yi := sb * qkK
		m := byte(1)
		is := 0
		for nb := range 2 { // n = 0, 128
			q := qsAll[nb*32 : nb*32+32]
			shift := 0
			for range 4 { // j = 0..3
				for _, g := range [2]int{0, 16} { // two groups of 16 (q[l], q[l+16])
					dl := dAll * float32(int(sc[is])-32)
					is++
					for l := range 16 {
						h := 4
						if hm[l+g]&m != 0 {
							h = 0
						}
						acc += float64(row[yi]) * float64(dl*float32(int((q[l+g]>>shift)&3)-h))
						yi++
					}
				}
				shift += 2
				m <<= 1
			}
		}
	}
	return acc
}
