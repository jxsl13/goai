//go:build arm64

package gguf

import "encoding/binary"

// dotIQ4XSBlockNeon fuses one 256-weight IQ4_XS super-block's nonlinear
// lookup, sub-scale application, and activation dot. Go supplies the exact
// float32 d*subscale coefficients; the leaf widens products into f64 partials.
//
//go:noescape
func dotIQ4XSBlockNeon(x *float32, qs *byte, coeff *float32, codebook *int8, indexes *byte) float64

func dotIQ4XSRowASM(row []float32, raw []byte, k int) float64 {
	var coeff [8]float32
	var acc float64
	for b := 0; b*qkK < k; b++ {
		blk := raw[b*iq4xsBlockSize : (b+1)*iq4xsBlockSize]
		//perfscan:ignore PS4001 one strided f16 scale per 136-byte quant block cannot use a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		scalesH := binary.LittleEndian.Uint16(blk[2:])
		scalesL := blk[4:8]
		for sb := range coeff {
			coeff[sb] = d * float32(iq4XSSubscale(scalesH, scalesL, sb))
		}
		acc += dotIQ4XSBlockNeon(
			&row[b*qkK], &blk[8], &coeff[0], &iq4KValuesI8[0], &qKByteToF32Indexes[0],
		)
	}
	return acc
}

func init() { dotIQ4XSRowFn = dotIQ4XSRowASM }
