package gguf

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// MXFP4 (§T555): the OCP Microscaling FP4 format gpt-oss ships in (ggml type
// 39) — 32 E2M1 elements sharing one E8M0 (power-of-two) scale byte. The
// element table kvalues holds the E2M1 values DOUBLED; the e8m0-half decode
// folds the ×½ back in (ggml_e8m0_to_fp32_half). Ported verbatim from gguf-py,
// both directions.
//
// Block layout (17 bytes for 32 elements): e (E8M0 scale byte), 16 nibble
// bytes — low nibbles = elements 0..15, high = 16..31.

const (
	tMXFP4         = 39
	mxfp4BlockSize = 17
)

// mxfp4KValues are the E2M1 magnitudes doubled; nibble bit 3 selects the
// negative half (index 8 = −0 = 0).
var mxfp4KValues = [16]float32{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}

// e8m0ToF32Half decodes an E8M0 exponent byte to 2^(e−127)/2, with the sub-2
// range mapping to subnormal bit patterns exactly like ggml_e8m0_to_fp32_half.
func e8m0ToF32Half(e byte) float32 {
	var bits uint32
	if e < 2 {
		bits = 0x00200000 << e
	} else {
		bits = uint32(e-1) << 23
	}
	return math.Float32frombits(bits)
}

// quantizeMXFP4 encodes f32 values (len % 32 == 0) into MXFP4 blocks: the block
// scale is 2^(⌊log2(absmax)⌋−2) (E8M0-floored), each element snaps to the
// nearest scaled kvalue (first-match argmin, mirroring gguf-py's argmin).
func quantizeMXFP4(x []float32) []byte {
	nb := len(x) / blockElems
	out := make([]byte, nb*mxfp4BlockSize)
	for b := range nb {
		blk := x[b*blockElems : (b+1)*blockElems]
		var amax float64
		for _, v := range blk {
			if a := math.Abs(float64(v)); a > amax {
				amax = a
			}
		}
		var e byte
		if amax > 0 {
			e = byte(math.Floor(math.Log2(amax)) - 2 + 127)
		}
		d := e8m0ToF32Half(e)
		o := out[b*mxfp4BlockSize:]
		o[0] = e
		for i, v := range blk {
			best, bestErr := 0, math.Inf(1)
			for k, kv := range mxfp4KValues {
				if err := math.Abs(float64(d*kv) - float64(v)); err < bestErr {
					best, bestErr = k, err
				}
			}
			if i < 16 {
				o[1+i] |= byte(best)
			} else {
				o[1+i-16] |= byte(best) << 4
			}
		}
	}
	return out
}

// dequantMXFP4 decodes MXFP4 blocks into an F32 tensor of the given shape.
func dequantMXFP4(shape tensor.Shape, data []byte) (*tensor.Tensor, error) {
	n := shape.Numel()
	if n%blockElems != 0 {
		return nil, fmt.Errorf("gguf: MXFP4 numel %d not multiple of %d", n, blockElems)
	}
	nb := n / blockElems
	if len(data) != nb*mxfp4BlockSize {
		return nil, fmt.Errorf("gguf: MXFP4 data %dB, want %d", len(data), nb*mxfp4BlockSize)
	}
	out := tensor.New(tensor.F32, shape)
	dst := out.Storage().F32()
	for b := range nb {
		blk := data[b*mxfp4BlockSize : (b+1)*mxfp4BlockSize]
		d := e8m0ToF32Half(blk[0])
		for i := range 16 {
			dst[b*blockElems+i] = d * mxfp4KValues[blk[1+i]&0x0F]
			dst[b*blockElems+16+i] = d * mxfp4KValues[blk[1+i]>>4]
		}
	}
	return out, nil
}
