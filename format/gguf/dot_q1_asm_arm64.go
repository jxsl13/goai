//go:build arm64

package gguf

// q1SignBytes expands one packed Q1_0 sign byte into eight high-bit bytes. A
// clear quant bit selects -d, so the native leaf widens its 0x80 twice into an
// IEEE-754 sign mask. The 2 KiB table eliminates 128 scalar bit tests per block.
var q1SignBytes = func() (table [256][8]byte) {
	for packed := range 256 {
		for bit := range 8 {
			if packed&(1<<bit) == 0 {
				table[packed][bit] = 0x80
			}
		}
	}
	return table
}()

// dotQ1RowNeon fuses complete Q1_0 rows: f16 scale lookup, packed-sign
// expansion, sign application, and activation dot. Activations and the scale
// are widened before multiplication, and four float64 partials remain live
// across the row to protect cancellation-heavy inputs.
//
// raw points at the first 18-byte block. Each block stores one little-endian
// f16 scale followed by 16 LSB-first sign bytes for 128 weights.
//
//go:noescape
func dotQ1RowNeon(x *float32, raw *byte, f16 *float32, signBytes *byte, blocks int) float64

func dotQ1RowASM(row []float32, raw []byte, k int) float64 {
	blocks := k / q1BlockElems
	if blocks == 0 {
		return 0
	}
	return dotQ1RowNeon(&row[0], &raw[0], &f16Table[0], &q1SignBytes[0][0], blocks)
}

func init() { dotQ1RowFn = dotQ1RowASM }
