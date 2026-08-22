//go:build arm64

package gguf

// dotQ41RowNeon fuses a complete Q4_1 row's nibble expansion, affine
// dequantization, and activation dot. Float64 vector partials keep cancellation
// error bounded while the row-level call amortizes Go/assembly dispatch.
//
//go:noescape
func dotQ41RowNeon(x *float32, raw *byte, f16 *float32, indexes *byte, blocks int) float64

func dotQ41RowASM(row []float32, raw []byte, k int) float64 {
	blocks := k / blockElems
	if blocks == 0 {
		return 0
	}
	return dotQ41RowNeon(&row[0], &raw[0], &f16Table[0], &qKByteToF32Indexes[0], blocks)
}

func init() { dotQ41RowFn = dotQ41RowASM }

const q41DotIsAsm = true
