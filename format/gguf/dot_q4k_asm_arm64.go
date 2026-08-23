//go:build arm64

package gguf

// dotQ4KBlockNeon fuses one 256-weight Q4_K super-block's nibble unpack,
// affine dequantization, and activation dot. The vector kernel accumulates in
// f32; dotQ4_KRowASM widens each super-block subtotal to f64 before combining
// it, so this path is tolerance-gated rather than bit-identical to the scalar
// f64-per-element reference.
//
//go:noescape
func dotQ4KBlockNeon(x *float32, qs *byte, coeff *float32, indexes *byte) float32

// dotQ4KPairBlockNeon computes two independent Q4_K block dots while loading
// each activation vector once. Each output retains dotQ4KBlockNeon's exact
// instruction and reduction order.
//
//go:noescape
func dotQ4KPairBlockNeon(x *float32, qs0, qs1 *byte, coeff0, coeff1 *float32, indexes *byte) (out0, out1 float32)

// dotQ4KRowNeon keeps independent Q4_K header decode and all super-block dots
// inside one assembly call. Each block reduces to f32 before ordered f64 row
// accumulation, matching the former Go-orchestrated path bit-for-bit.
//
//go:noescape
func dotQ4KRowNeon(x *float32, raw *byte, f16 *float32, indexes *byte, blocks int) (out float64)

// dotQ4KPairRowNeon keeps paired Q4_K header decode, coefficient staging, and
// all super-block dots inside one assembly call. Each block still reduces to
// f32 before the row accumulators widen and add it in order as f64.
//
//go:noescape
func dotQ4KPairRowNeon(x *float32, raw0, raw1 *byte, f16 *float32, indexes *byte, blocks int) (out0, out1 float64)

func dotQ4_KRowASM(row []float32, raw []byte, k int) float64 {
	if k == 0 {
		return 0
	}
	return dotQ4KRowNeon(&row[0], &raw[0], &f16Table[0], &qKByteToF32Indexes[0], k/qkK)
}

func dotQ4KPairRowASM(row []float32, raw0, raw1 []byte, k int) (float64, float64) {
	if k == 0 {
		return 0, 0
	}
	return dotQ4KPairRowNeon(
		&row[0], &raw0[0], &raw1[0], &f16Table[0], &qKByteToF32Indexes[0], k/qkK,
	)
}

func init() {
	dotQ4KRowFn = dotQ4_KRowASM
	dotQ4KPairRowFn = dotQ4KPairRowASM
}

const q4kDotIsAsm = true
