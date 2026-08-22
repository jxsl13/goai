//go:build arm64

package gguf

// dotTQ2RowNeon fuses complete TQ2_0 rows: vector two-bit unpack, trailing
// f16 scale lookup, and direct-F32 activation dot. Four float64 partials remain
// live across the entire row to protect cancellation-heavy inputs. raw points
// at the first 66-byte block.
//
//go:noescape
func dotTQ2RowNeon(x *float32, raw *byte, f16 *float32, indexes *byte, blocks int) float64

func dotTQ2RowASM(row []float32, raw []byte, k int) float64 {
	blocks := k / tq2BlockElems
	if blocks == 0 {
		return 0
	}
	return dotTQ2RowNeon(&row[0], &raw[0], &f16Table[0], &qKByteToF32Indexes[0], blocks)
}

func init() { dotTQ2RowFn = dotTQ2RowASM }
