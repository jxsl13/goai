//go:build arm64

package gguf

import "encoding/binary"

// iq3sSignNibbles expands four direct sign bits into float32 sign masks. The
// 256-byte table stays hot while avoiding a branch or scalar sign operation
// for every decoded weight.
var iq3sSignNibbles = func() (table [16][4]uint32) {
	for nibble := range table {
		for lane := range table[nibble] {
			table[nibble][lane] = uint32(nibble>>lane&1) << 31
		}
	}
	return table
}()

// dotIQ3SBlockNeon fuses one 256-weight IQ3_S super-block's 9-bit grid
// gathers, direct sign expansion, sub-scale multiplication, and activation
// dot. Go supplies the exact float32 d*(1+2*s) coefficients; the leaf widens
// every product into four independent float64 partials.
//
// qs points at the block's 64-byte low-index plane. The 8-byte high-index
// plane and 32 direct-sign bytes immediately follow it.
//
//go:noescape
func dotIQ3SBlockNeon(x *float32, qs *byte, coeff *float32, grid *float32, signNibbles *uint32) float64

func dotIQ3SRowASM(row []float32, raw []byte, k int) float64 {
	var coeff [8]float32
	var acc float64
	for b := 0; b*qkK < k; b++ {
		blk := raw[b*iq3sBlockSize : (b+1)*iq3sBlockSize]
		//perfscan:ignore PS4001 one strided f16 scale per 110-byte quant block cannot use a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		scales := blk[106:110]
		for g := range coeff {
			sc := scales[g/2] >> (4 * (g % 2)) & 0x0f
			coeff[g] = d * (1 + 2*float32(sc))
		}
		acc += dotIQ3SBlockNeon(
			&row[b*qkK], &blk[2], &coeff[0], &iq3sGrid[0][0], &iq3sSignNibbles[0][0],
		)
	}
	return acc
}

func init() { dotIQ3SRowFn = dotIQ3SRowASM }
