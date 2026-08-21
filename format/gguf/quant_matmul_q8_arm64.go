//go:build arm64

package gguf

// dotQ8RowNeon fuses an entire Q8_0 row's signed-int8 widening, per-block
// f16 scale application, and activation dot. Keeping the block loop in
// assembly amortizes the Go/assembly boundary over the full output row.
// Accumulation is f32-vectorized, so this path is tolerance-gated rather than
// bit-identical to the portable f64-per-element reference.
//
//go:noescape
func dotQ8RowNeon(x *float32, raw *byte, f16 *float32, indexes *byte, blocks int) float32

func q8FusedDecodeM1Neon(row []float32, weight []byte, n, k, rowBytes int, outf []float32) {
	blocks := k / blockElems
	if blocks == 0 {
		return
	}
	for ni := range n {
		outf[ni] = dotQ8RowNeon(
			&row[0], &weight[ni*rowBytes], &f16Table[0], &qKByteToF32Indexes[0], blocks,
		)
	}
}

func init() { q8FusedDecodeM1 = q8FusedDecodeM1Neon }
