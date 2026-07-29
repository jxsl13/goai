package gguf

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// Quantize encodes a tensor's values into the ggml block-quantized byte layout for qt
// (Q8_0 or Q4_0, §R94), the inverse of the dequantization done on read. The element
// count must be a multiple of the 32-element block size. The result is the raw weight
// bytes accepted by QMatMul and produced by ggml — enabling GoAI to WRITE quantized
// models, not just read them. Encoding follows ggml-quants.c exactly: per 32-block,
// Q8_0 uses d=amax/127 with roundf quants; Q4_0 uses the signed max, d=max/−8 and
// nibble=min(15,⌊x/d+8.5⌋). Values already on the quantization grid round-trip exactly.
func Quantize(t *tensor.Tensor, qt QuantType) ([]byte, error) {
	n := t.Numel()
	be := blockElems // 32 for Q8_0/Q4_0
	if qt == Q6_K || qt == Q4_K || qt == Q5_K || qt == Q3_K || qt == Q2_K {
		be = qkK // k-quant super-block is 256 elements
	}
	if n%be != 0 {
		return nil, fmt.Errorf("gguf: Quantize numel %d not a multiple of %d", n, be)
	}
	// Bulk-extract the values: read contiguous storage directly instead of the
	// per-element AtF64(Unravel(...)) dispatch (§base-perf: the Unravel/AtF64
	// anti-pattern; ~2.4× on a whole Quantize call, docs/perf-notes-lowlevel.md).
	x := make([]float32, n)
	switch c := t.Contiguous(); c.Dtype() {
	case tensor.F32:
		copy(x, c.Storage().F32())
	case tensor.F64:
		for i, v := range c.Storage().F64() {
			x[i] = float32(v)
		}
	default:
		for i := range n {
			x[i] = float32(t.AtF64(tensor.Unravel(i, t.Shape())...))
		}
	}
	switch qt {
	case Q8_0:
		return quantizeQ8_0(x), nil
	case Q4_0:
		return quantizeQ4_0(x), nil
	case Q2_K:
		return quantizeQ2_K(x), nil
	case Q3_K:
		return quantizeQ3_K(x), nil
	case Q4_K:
		return quantizeQ4_K(x), nil
	case Q5_K:
		return quantizeQ5_K(x), nil
	case Q6_K:
		return quantizeQ6_K(x), nil
	case MXFP4:
		return quantizeMXFP4(x), nil
	default:
		return nil, fmt.Errorf("gguf: Quantize unsupported quant type %d", qt)
	}
}

// Dequantize decodes n block-quantized values of type qt back to an F32 tensor of
// shape [n] — the public form of the on-read dequantization, the inverse of Quantize.
func Dequantize(data []byte, qt QuantType, n int) (*tensor.Tensor, error) {
	need, err := byteSize(uint32(qt), n)
	if err != nil {
		return nil, err
	}
	if len(data) != need {
		return nil, fmt.Errorf("gguf: Dequantize %d bytes != %d for %d values", len(data), need, n)
	}
	switch qt {
	case Q8_0:
		return dequantQ8_0(tensor.Shape{n}, data)
	case Q4_0:
		return dequantQ4_0(tensor.Shape{n}, data)
	case Q2_K:
		return dequantQ2_K(tensor.Shape{n}, data)
	case Q3_K:
		return dequantQ3_K(tensor.Shape{n}, data)
	case Q4_K:
		return dequantQ4_K(tensor.Shape{n}, data)
	case Q5_K:
		return dequantQ5_K(tensor.Shape{n}, data)
	case Q6_K:
		return dequantQ6_K(tensor.Shape{n}, data)
	case IQ2_XXS:
		return dequantIQ2_XXS(tensor.Shape{n}, data)
	case IQ2_XS:
		return dequantIQ2_XS(tensor.Shape{n}, data)
	case IQ3_XXS:
		return dequantIQ3_XXS(tensor.Shape{n}, data)
	case IQ4_NL:
		return dequantIQ4_NL(tensor.Shape{n}, data)
	case IQ4_XS:
		return dequantIQ4_XS(tensor.Shape{n}, data)
	case IQ3_S:
		return dequantIQ3_S(tensor.Shape{n}, data)
	case IQ2_S:
		return dequantIQ2_S(tensor.Shape{n}, data)
	case IQ1_S:
		return dequantIQ1_S(tensor.Shape{n}, data)
	case IQ1_M:
		return dequantIQ1_M(tensor.Shape{n}, data)
	case MXFP4:
		return dequantMXFP4(tensor.Shape{n}, data)
	default:
		return nil, fmt.Errorf("gguf: Dequantize unsupported quant type %d", qt)
	}
}

// quantizeQ8_0: per 32-block, d = amax/127 (f16), qs[j] = roundf(x[j]/d) int8.
func quantizeQ8_0(x []float32) []byte {
	nb := len(x) / blockElems
	out := make([]byte, nb*34)
	for b := range nb {
		blk := x[b*blockElems : (b+1)*blockElems]
		var amax float32
		for _, v := range blk {
			// bit-exact abs via sign-bit clear (no f32->f64->f32 round-trip); |v| identical to math.Abs
			if a := math.Float32frombits(math.Float32bits(v) &^ (1 << 31)); a > amax {
				amax = a
			}
		}
		d := amax / 127
		var id float32
		if d != 0 {
			id = 1 / d
		}
		o := b * 34
		binary.LittleEndian.PutUint16(out[o:], f32ToF16(d))
		for j, v := range blk {
			out[o+2+j] = byte(int8(roundHalfAway(v * id)))
		}
	}
	return out
}

// quantizeQ4_0: per 32-block, d = max/−8 with the SIGNED max, nibble =
// min(15,⌊x/d+8.5⌋); byte j packs element j (low) and j+16 (high).
func quantizeQ4_0(x []float32) []byte {
	nb := len(x) / blockElems
	out := make([]byte, nb*18)
	for b := range nb {
		blk := x[b*blockElems : (b+1)*blockElems]
		var max float32 // the element with the largest absolute value, sign kept
		var amax float32
		for _, v := range blk {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax, max = a, v
			}
		}
		d := max / -8
		var id float32
		if d != 0 {
			id = 1 / d
		}
		o := b * 18
		binary.LittleEndian.PutUint16(out[o:], f32ToF16(d))
		for j := range 16 {
			lo := q4nibble(blk[j] * id)
			hi := q4nibble(blk[j+16] * id)
			out[o+2+j] = lo | hi<<4
		}
	}
	return out
}

// q4nibble = MIN(15, (int8_t)(v + 8.5f)) (ggml truncation toward zero; v+8.5 ≥ 0).
func q4nibble(v float32) byte {
	q := int(v + 8.5) // Go int() truncates toward zero; v+8.5 ≥ 0 so this is a floor
	if q > 15 {
		q = 15
	}
	if q < 0 {
		q = 0
	}
	return byte(q)
}

// roundHalfAway rounds to the nearest integer, ties away from zero (C roundf).
func roundHalfAway(v float32) float32 { return float32(math.Round(float64(v))) }

// f32ToF16 encodes an IEEE-754 float32 as binary16 with round-to-nearest-even
// (inverse of f16ToF32), handling overflow→inf, subnormals and NaN.
func f32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b >> 16 & 0x8000)
	exp := int32(b>>23&0xFF) - 127
	mant := b & 0x7FFFFF
	switch {
	case b&0x7FFFFFFF == 0: // ±0
		return sign
	case b>>23&0xFF == 0xFF: // inf / NaN
		if mant != 0 {
			return sign | 0x7E00 // NaN (quiet)
		}
		return sign | 0x7C00 // inf
	case exp > 15: // overflow → inf
		return sign | 0x7C00
	case exp < -24: // underflow → ±0
		return sign
	case exp < -14: // subnormal: drop (13 + (-14-exp)) low bits, round-to-nearest-even
		mant |= 0x800000 // implicit leading 1
		drop := uint32(13) + uint32(-14-exp)
		roundHalf := uint32(1) << (drop - 1)
		lower := mant & ((1 << drop) - 1)
		m := mant >> drop
		if lower > roundHalf || (lower == roundHalf && m&1 == 1) {
			m++ // may carry into the smallest normal exponent, which is correct
		}
		return sign | uint16(m)
	default: // normal
		e := uint16(exp+15) << 10
		roundHalf := uint32(1 << 12)
		lower := mant & 0x1FFF // 13 bits dropped (23→10)
		m := mant >> 13
		if lower > roundHalf || (lower == roundHalf && m&1 == 1) {
			m++
			if m == 0x400 { // mantissa overflow → bump exponent
				m = 0
				e += 1 << 10
				if e>>10 >= 0x1F {
					return sign | 0x7C00 // → inf
				}
			}
		}
		return sign | e | uint16(m)
	}
}
