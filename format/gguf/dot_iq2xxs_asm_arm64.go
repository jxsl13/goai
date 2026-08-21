//go:build arm64

package gguf

import "encoding/binary"

// dotIQ2XXSBlockNeon fuses one 256-weight IQ2_XXS super-block's eight-wide
// grid gathers, ksigns expansion, sub-scale multiplication, and activation
// dot. Go supplies the exact float32 d*(0.5+s)*0.25 coefficients; the leaf
// widens every product into four independent float64 partials.
//
// pairs points at the block's eight interleaved uint32 grid-index/sign-scale
// pairs.
//
//go:noescape
func dotIQ2XXSBlockNeon(x *float32, pairs *byte, coeff *float32, grid *float32, signMasks *uint32) float64

func dotIQ2XXSRowASM(row []float32, raw []byte, k int) float64 {
	var coeff [8]float32
	var acc float64
	for b := 0; b*qkK < k; b++ {
		blk := raw[b*iq2xxsBlockSize : (b+1)*iq2xxsBlockSize]
		//perfscan:ignore PS4001 one strided f16 scale per 66-byte quant block cannot use a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		for pair := range coeff {
			scale := blk[9+pair*8] >> 4
			coeff[pair] = d * (0.5 + float32(scale)) * 0.25
		}
		acc += dotIQ2XXSBlockNeon(
			&row[b*qkK], &blk[2], &coeff[0], &iq2xxsGrid[0][0], &iqKSignMasks[0][0],
		)
	}
	return acc
}

func init() { dotIQ2XXSRowFn = dotIQ2XXSRowASM }
