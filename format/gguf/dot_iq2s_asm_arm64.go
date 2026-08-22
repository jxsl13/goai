//go:build arm64

package gguf

import "encoding/binary"

// iq2sSignMasks expands one direct sign byte into eight float32 sign masks.
// The 8 KiB table replaces per-group scalar bit extraction in the leaf.
var iq2sSignMasks = func() (table [256][8]uint32) {
	for signs := range table {
		for lane := range table[signs] {
			table[signs][lane] = uint32(signs>>lane&1) << 31
		}
	}
	return table
}()

// dotIQ2SBlockNeon fuses one 256-weight IQ2_S super-block's 10-bit grid
// gathers, direct sign expansion, explicit sub-scale multiplication, and
// activation dot. Go supplies the exact float32 d*(0.5+s)*0.25 coefficients;
// the leaf widens every product into four independent float64 partials.
//
// qs points at the block's 32-byte low-index plane. The 32 direct-sign bytes
// and 8-byte high-index plane immediately follow it.
//
//go:noescape
func dotIQ2SBlockNeon(x *float32, qs *byte, coeff *float32, grid *float32, signMasks *uint32) float64

func dotIQ2SRowASM(row []float32, raw []byte, k int) float64 {
	var coeff [16]float32
	var acc float64
	for b := 0; b*qkK < k; b++ {
		blk := raw[b*iq2sBlockSize : (b+1)*iq2sBlockSize]
		//perfscan:ignore PS4001 one strided f16 scale per 82-byte quant block cannot use a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		scales := blk[74:82]
		for s := range coeff {
			scale := scales[s/2] >> (4 * (s % 2)) & 0x0f
			coeff[s] = d * (0.5 + float32(scale)) * 0.25
		}
		acc += dotIQ2SBlockNeon(
			&row[b*qkK], &blk[2], &coeff[0], &iq2sGrid[0][0], &iq2sSignMasks[0][0],
		)
	}
	return acc
}

func init() { dotIQ2SRowFn = dotIQ2SRowASM }
