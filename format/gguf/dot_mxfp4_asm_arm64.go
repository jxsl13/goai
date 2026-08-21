//go:build arm64

package gguf

var mxfp4KValuesI8 = [16]int8{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}

// dotMXFP4RowNeon fuses an entire MXFP4 row's nibble unpack, signed E2M1
// lookup, E8M0 scale application, and activation dot. The row-level boundary
// amortizes Go/assembly dispatch and float64 vector partials protect the public
// relative-error gate under cancellation.
//
//go:noescape
func dotMXFP4RowNeon(x *float32, raw *byte, scales *float32, codebook *int8, indexes *byte, blocks int) float64

func dotMXFP4RowASM(row []float32, raw []byte, k int) float64 {
	blocks := k / blockElems
	if blocks == 0 {
		return 0
	}
	return dotMXFP4RowNeon(
		&row[0], &raw[0], &e8m0HalfTable[0], &mxfp4KValuesI8[0], &qKByteToF32Indexes[0], blocks,
	)
}

func init() { dotMXFP4RowFn = dotMXFP4RowASM }
