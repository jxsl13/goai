//go:build arm64

package gguf

// dotTQ1RowNeon fuses complete TQ1_0 rows: vector base-243 trit expansion,
// trailing f16 scale lookup, and direct-F32 activation dot. Four float64
// partials remain live across the entire row to protect cancellation-heavy
// inputs. raw points at the first 54-byte block.
//
//go:noescape
func dotTQ1RowNeon(x *float32, raw *byte, f16 *float32, indexes *byte, blocks int) float64

func dotTQ1RowASM(row []float32, raw []byte, k int) float64 {
	blocks := k / tq1BlockElems
	if blocks == 0 {
		return 0
	}
	return dotTQ1RowNeon(&row[0], &raw[0], &f16Table[0], &qKByteToF32Indexes[0], blocks)
}

func init() { dotTQ1RowFn = dotTQ1RowASM }
