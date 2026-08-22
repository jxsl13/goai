//go:build arm64

package gguf

// iq1mQHOffsets expands both packed qh nibbles into byte offsets spanning the
// pre-adjusted grid's 11-bit row index and signed-delta plane. The 2 KiB table
// removes repeated high-index and sign bitfield assembly from the native leaf.
var iq1mQHOffsets = func() (table [256][2]uint32) {
	for packed := range 256 {
		for half := range 2 {
			nib := packed >> (4 * half) & 0x0f
			table[packed][half] = uint32(nib&7)<<13 | uint32(nib>>3)<<16
		}
	}
	return table
}()

// dotIQ1MRowNeon fuses complete IQ1_M rows: split-f16 super-scale lookup,
// paired odd multipliers, 11-bit grid gathers, signed deltas, and activation
// dot. The pre-adjusted IQ1_S grid performs the decoder's float32 delta add
// once; the leaf forms exact float32 d*(2*s+1) coefficients before widening
// products into four float64 partials that remain live across all blocks.
//
// raw points at the first 56-byte block. Each block stores 32 low-index bytes,
// 16 packed high-index/sign bytes, then four little-endian scale words whose
// top nibbles jointly encode the f16 super-scale.
//
//go:noescape
func dotIQ1MRowNeon(x *float32, raw *byte, f16 *float32, grid *float32, oddScales *float32, qhOffsets *uint32, blocks int) float64

func dotIQ1MRowASM(row []float32, raw []byte, k int) float64 {
	return dotIQ1MRowNeon(
		&row[0], &raw[0], &f16Table[0], &iq1sDeltaGrid[0][0][0], &iq1sOddScales[0], &iq1mQHOffsets[0][0], k/qkK,
	)
}

func init() { dotIQ1MRowFn = dotIQ1MRowASM }
