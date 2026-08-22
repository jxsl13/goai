//go:build arm64

package gguf

// iq1sDeltaGrid folds the two possible signed deltas into the ternary grid.
// The leaf trades 128 KiB of extra read-only codebook data for removing two
// vector additions from every eight-weight gather.
var iq1sDeltaGrid = func() (table [2][2048][8]float32) {
	unhex := func(c byte) byte {
		if c > 0x40 {
			c += 9
		}
		return c & 0x0f
	}
	gridMap := [3]float32{-1, 0, 1}
	for entry := range 2048 {
		for pair := range 2 {
			packed := unhex(iq1GridHex[entry*4+pair*2])<<4 |
				unhex(iq1GridHex[entry*4+pair*2+1])
			for lane := range 4 {
				v := gridMap[(packed>>(2*lane))&0x3]
				table[0][entry][pair*4+lane] = v + iq1Delta
				table[1][entry][pair*4+lane] = v - iq1Delta
			}
		}
	}
	return table
}()

var iq1sOddScales = [...]float32{1, 3, 5, 7, 9, 11, 13, 15}

// dotIQ1SRowNeon fuses complete IQ1_S rows: f16 block-scale lookup, 11-bit grid
// gathers, shared odd multipliers, signed deltas, and activation dot. The
// pre-adjusted grid performs the decoder's float32 delta add once; the leaf
// forms exact float32 d*(2*s+1) coefficients before widening products into
// four float64 partials that remain live across all blocks.
//
// raw points at the first 50-byte block. Each block stores f16 d, 32 low-index
// bytes, then eight little-endian qh words.
//
//go:noescape
func dotIQ1SRowNeon(x *float32, raw *byte, f16 *float32, grid *float32, oddScales *float32, blocks int) float64

func dotIQ1SRowASM(row []float32, raw []byte, k int) float64 {
	return dotIQ1SRowNeon(
		&row[0], &raw[0], &f16Table[0], &iq1sDeltaGrid[0][0][0], &iq1sOddScales[0], k/qkK,
	)
}

func init() { dotIQ1SRowFn = dotIQ1SRowASM }
