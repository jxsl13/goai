//go:build arm64

package gguf

import "encoding/binary"

// iq3xxsSignMasks expands one ksigns byte into eight float32 sign masks. The
// 4 KiB table avoids branchy per-lane sign extraction in the row-dot leaf.
var iq3xxsSignMasks = func() (table [128][8]uint32) {
	for index := range table {
		signs := iq2xxsKSigns[index]
		for lane := range table[index] {
			table[index][lane] = uint32(signs>>lane&1) << 31
		}
	}
	return table
}()

// dotIQ3XXSBlockNeon fuses one 256-weight IQ3_XXS super-block's grid gathers,
// ksigns expansion, sub-scale multiplication, and activation dot. Go supplies
// the exact float32 d*(0.5+s)*0.5 coefficients; the leaf widens every product
// into four independent float64 partials.
//
// qs points at the block's 64-byte grid-index plane. The 32-byte packed
// sign/scale plane immediately follows it.
//
//go:noescape
func dotIQ3XXSBlockNeon(x *float32, qs *byte, coeff *float32, grid *float32, signMasks *uint32) float64

func dotIQ3XXSRowASM(row []float32, raw []byte, k int) float64 {
	var coeff [8]float32
	var acc float64
	for b := 0; b*qkK < k; b++ {
		blk := raw[b*iq3xxsBlockSize : (b+1)*iq3xxsBlockSize]
		//perfscan:ignore PS4001 one strided f16 scale per 98-byte quant block cannot use a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		for g := range coeff {
			scale := blk[69+g*4] >> 4
			coeff[g] = d * (0.5 + float32(scale)) * 0.5
		}
		acc += dotIQ3XXSBlockNeon(
			&row[b*qkK], &blk[2], &coeff[0], &iq3xxsGrid[0][0], &iq3xxsSignMasks[0][0],
		)
	}
	return acc
}

func init() { dotIQ3XXSRowFn = dotIQ3XXSRowASM }
