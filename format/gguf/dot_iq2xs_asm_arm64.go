//go:build arm64

package gguf

import "encoding/binary"

// dotIQ2XSBlockNeon fuses one 256-weight IQ2_XS super-block's eight-wide grid
// gathers, ksign expansion, explicit sub-scale multiplication, and activation
// dot. Go supplies the exact float32 d*(0.5+s)*0.25 coefficients; the leaf
// widens every product into four independent float64 partials.
//
// codes points at the block's 32 little-endian uint16 grid/sign indices.
//
//go:noescape
func dotIQ2XSBlockNeon(x *float32, codes *byte, coeff *float32, grid *float32, signMasks *uint32) float64

func dotIQ2XSRowASM(row []float32, raw []byte, k int) float64 {
	var coeff [16]float32
	var acc float64
	for b := 0; b*qkK < k; b++ {
		blk := raw[b*iq2xsBlockSize : (b+1)*iq2xsBlockSize]
		//perfscan:ignore PS4001 one strided f16 scale per 74-byte quant block cannot use a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		scales := blk[66:74]
		for s := range coeff {
			scale := scales[s/2] >> (4 * (s % 2)) & 0x0f
			coeff[s] = d * (0.5 + float32(scale)) * 0.25
		}
		acc += dotIQ2XSBlockNeon(
			&row[b*qkK], &blk[2], &coeff[0], &iq2xsGrid[0][0], &iqKSignMasks[0][0],
		)
	}
	return acc
}

func init() { dotIQ2XSRowFn = dotIQ2XSRowASM }
