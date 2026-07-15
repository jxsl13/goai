package gguf

import (
	"encoding/binary"
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// IQ4_NL / IQ4_XS importance-quant READ paths (§T554 part 4): 4-bit quants over a
// fixed NONLINEAR 16-value codebook (denser near zero — matched to the weight
// distribution) instead of the affine grid of Q4_0/Q4_K. IQ4_NL is the plain
// 32-element block (f16 scale + 16 nibble bytes); IQ4_XS is the 256-element
// super-block with 6-bit sub-scales ((low nibble | high 2 bits << 4) − 32).
// Ported verbatim from gguf-py.

const (
	tIQ4_NL = 20 // ggml type id
	tIQ4_XS = 23

	iq4nlBlockSize = 18  // f16 d + 16 nibble bytes per 32 elements
	iq4xsBlockSize = 136 // f16 d + u16 scales_h + 4 scales_l + 128 qs per 256 elements
)

// iq4KValues is the shared nonlinear codebook (kvalues_iq4nl).
var iq4KValues = [16]float32{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}

// iq4Nibbles decodes one 16-byte group: elements 0..15 from the LOW nibbles,
// 16..31 from the HIGH nibbles (gguf-py's unpack order).
func iq4Nibbles(qs []byte, scale float32, dst []float32) {
	y, q := dst[:32], qs[:16]
	for e, b := range q {
		y[e] = scale * iq4KValues[b&0x0F]
		y[16+e] = scale * iq4KValues[b>>4]
	}
}

// dequantIQ4_NL decodes IQ4_NL blocks into an F32 tensor of the given shape.
func dequantIQ4_NL(shape tensor.Shape, data []byte) (*tensor.Tensor, error) {
	n := shape.Numel()
	if n%blockElems != 0 {
		return nil, fmt.Errorf("gguf: IQ4_NL numel %d not multiple of %d", n, blockElems)
	}
	nb := n / blockElems
	if len(data) != nb*iq4nlBlockSize {
		return nil, fmt.Errorf("gguf: IQ4_NL data %dB, want %d", len(data), nb*iq4nlBlockSize)
	}
	out := tensor.New(tensor.F32, shape)
	dst := out.Storage().F32()
	for b := range nb {
		blk := data[b*iq4nlBlockSize : (b+1)*iq4nlBlockSize]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:]))
		iq4Nibbles(blk[2:18], d, dst[b*blockElems:])
	}
	return out, nil
}

// dequantIQ4_XS decodes IQ4_XS blocks into an F32 tensor of the given shape.
func dequantIQ4_XS(shape tensor.Shape, data []byte) (*tensor.Tensor, error) {
	n := shape.Numel()
	if n%qkK != 0 {
		return nil, fmt.Errorf("gguf: IQ4_XS numel %d not multiple of %d", n, qkK)
	}
	nb := n / qkK
	if len(data) != nb*iq4xsBlockSize {
		return nil, fmt.Errorf("gguf: IQ4_XS data %dB, want %d", len(data), nb*iq4xsBlockSize)
	}
	out := tensor.New(tensor.F32, shape)
	dst := out.Storage().F32()
	for b := range nb {
		blk := data[b*iq4xsBlockSize : (b+1)*iq4xsBlockSize]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:]))
		scalesH := binary.LittleEndian.Uint16(blk[2:])
		scalesL := blk[4:8]
		qs := blk[8:136]
		for sb := range 8 { // 8 sub-blocks of 32
			sl := scalesL[sb/2] >> (4 * (sb % 2)) & 0x0F
			sh := byte(scalesH>>(2*sb)) & 0x03
			scale := float32(int8(sl|sh<<4) - 32)
			iq4Nibbles(qs[sb*16:(sb+1)*16], d*scale, dst[b*qkK+sb*32:])
		}
	}
	return out, nil
}
