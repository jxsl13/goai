//go:build arm64

package gguf

// iq4KValuesI8 is the byte-native form of the shared nonlinear codebook. The
// assembly leaf table-looks up all sixteen nibbles before exact signed fixed-
// point conversion to float32.
var iq4KValuesI8 = [16]int8{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}

// dotIQ4NLRowNeon fuses an entire IQ4_NL row's nibble unpack, nonlinear table
// lookup, f16 scale application, and activation dot. The row-level boundary
// amortizes Go/assembly dispatch, while f64 vector partials protect the public
// 1e-4 relative-error gate under cancellation.
//
//go:noescape
func dotIQ4NLRowNeon(x *float32, raw *byte, f16 *float32, codebook *int8, indexes *byte, blocks int) float64

func dotIQ4NLRowASM(row []float32, raw []byte, k int) float64 {
	blocks := k / blockElems
	if blocks == 0 {
		return 0
	}
	return dotIQ4NLRowNeon(
		&row[0], &raw[0], &f16Table[0], &iq4KValuesI8[0], &qKByteToF32Indexes[0], blocks,
	)
}

func init() { dotIQ4NLRowFn = dotIQ4NLRowASM }
