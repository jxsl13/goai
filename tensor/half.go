package tensor

import "math"

// 16-bit float conversions (§T27). f16 = IEEE-754 binary16 (1/5/10). bf16 =
// 1/8/7, the top 16 bits of a float32. Both f32→16 use round-to-nearest-even to
// match numpy (f16) and torch (bf16). Confirmed vs those references (§R41).

// f16ToF32 converts IEEE binary16 bits to float32 (subnormals, inf, NaN).
func f16ToF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1F
	frac := uint32(h) & 0x3FF
	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		e := uint32(127 - 15 + 1)
		for frac&0x400 == 0 {
			frac <<= 1
			e--
		}
		frac &= 0x3FF
		return math.Float32frombits(sign | e<<23 | frac<<13)
	case 0x1F:
		return math.Float32frombits(sign | 0xFF<<23 | frac<<13)
	default:
		return math.Float32frombits(sign | (exp+127-15)<<23 | frac<<13)
	}
}

// f32ToF16 converts float32 to IEEE binary16 bits with round-to-nearest-even.
func f32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int32(b>>23) & 0xFF
	mant := b & 0x007FFFFF

	if exp == 0xFF { // Inf / NaN
		if mant != 0 {
			return sign | 0x7E00 // quiet NaN
		}
		return sign | 0x7C00 // Inf
	}
	e := exp - 127 + 15 // rebias to f16
	if e >= 0x1F {
		return sign | 0x7C00 // overflow → Inf
	}
	if e <= 0 { // subnormal or zero
		if e < -10 {
			return sign // underflow → ±0
		}
		mant |= 0x00800000 // implicit leading 1
		shift := uint32(14 - e)
		lower := mant & ((1 << shift) - 1)
		half := uint32(1) << (shift - 1)
		res := mant >> shift
		if lower > half || (lower == half && res&1 == 1) {
			res++
		}
		return sign | uint16(res)
	}
	res := uint16(e<<10) | uint16(mant>>13)
	lower := mant & 0x1FFF
	if lower > 0x1000 || (lower == 0x1000 && res&1 == 1) {
		res++ // carry into exponent is correct (0x7BFF→0x7C00 = Inf)
	}
	return sign | res
}

// bf16ToF32 converts bfloat16 bits to float32 (a pure left shift).
func bf16ToF32(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }

// f32ToBF16 converts float32 to bfloat16 bits with round-to-nearest-even.
func f32ToBF16(f float32) uint16 {
	b := math.Float32bits(f)
	if b&0x7FFFFFFF > 0x7F800000 { // NaN → keep NaN (set a mantissa bit)
		return uint16(b>>16) | 0x0040
	}
	rounding := ((b >> 16) & 1) + 0x7FFF
	return uint16((b + rounding) >> 16)
}
