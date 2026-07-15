package safetensors

import "math"

// FP8 decoding (§T577) for the two OCP 8-bit float formats safetensors ships
// (DeepSeek-V3-class FP8 checkpoints use F8_E4M3). Both decode exactly into
// float32 — every FP8 value is representable — so loading widens losslessly.
//
// F8_E4M3 (1 sign, 4 exponent, 3 mantissa, bias 7) is the "fn" variant: it has
// NO infinities; the all-ones pattern S.1111.111 is NaN and S.1111.110 is a
// regular finite value (±448, the format's max). F8_E5M2 (1/5/2, bias 15) is a
// truncated IEEE half: exponent 31 encodes ±Inf (mantissa 0) and NaN otherwise.
// Verified byte-for-byte against torch.float8_e4m3fn / torch.float8_e5m2 over
// all 256 encodings.

// Decode tables: only 256 encodings exist per format, so Load widens through a
// [256]float32 lookup instead of re-deriving the value per element (~6.4× on
// a whole FP8 Load, M2 Pro — docs/perf-notes-lowlevel.md). Built at init FROM the reference
// decoders below, so table and function cannot drift apart.
var f8e4m3Table, f8e5m2Table [256]float32

func init() {
	for i := range 256 {
		f8e4m3Table[i] = f8e4m3ToF32(byte(i))
		f8e5m2Table[i] = f8e5m2ToF32(byte(i))
	}
}

// f8e4m3ToF32 decodes one F8_E4M3 byte.
func f8e4m3ToF32(b byte) float32 {
	e := int(b>>3) & 0xF
	m := int(b) & 7
	var v float64
	switch {
	case e == 0xF && m == 7:
		return float32(math.NaN())
	case e == 0:
		v = float64(m) * 0x1p-9 // subnormal: m/8 · 2^-6
	default:
		v = (1 + float64(m)/8) * math.Ldexp(1, e-7)
	}
	if b&0x80 != 0 {
		v = -v
	}
	return float32(v)
}

// f8e5m2ToF32 decodes one F8_E5M2 byte.
func f8e5m2ToF32(b byte) float32 {
	e := int(b>>2) & 0x1F
	m := int(b) & 3
	var v float64
	switch {
	case e == 0x1F:
		if m != 0 {
			return float32(math.NaN())
		}
		v = math.Inf(1)
	case e == 0:
		v = float64(m) * 0x1p-16 // subnormal: m/4 · 2^-14
	default:
		v = (1 + float64(m)/4) * math.Ldexp(1, e-15)
	}
	if b&0x80 != 0 {
		v = -v
	}
	return float32(v)
}
