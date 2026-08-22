package gguf

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// TQ2_0 is ggml's 2.0625-bit ternary quant. Each 256-weight block stores
// sixty-four bytes with four 2-bit codes in 32-lane groups and one trailing
// f16 scale. Raw code 3 decodes to +2 even though the reference encoder emits
// only the ternary codes 0, 1, and 2.
const (
	tTQ2_0         = 35
	tq2BlockElems  = 256
	tq2PackedBytes = tq2BlockElems / 4
	tq2BlockSize   = tq2PackedBytes + 2
)

// dequantTQ2_0Into preserves ggml's 32-lane plane order. dst and raw contain
// complete blocks.
func dequantTQ2_0Into(dst []float32, raw []byte) {
	for b := 0; b*tq2BlockSize < len(raw); b++ {
		blk := raw[b*tq2BlockSize : (b+1)*tq2BlockSize]
		//perfscan:ignore PS4001 TQ2_0 scales are strided 66-byte f16 fields, not a same-layout bulk plane.
		d := f16ToF32(binary.LittleEndian.Uint16(blk[tq2PackedBytes:]))
		out := dst[b*tq2BlockElems : (b+1)*tq2BlockElems]
		pos := 0
		for packed := 0; packed < tq2PackedBytes; packed += 32 {
			for plane := range 4 {
				shift := uint(2 * plane)
				for lane := range 32 {
					q := int(blk[packed+lane]>>shift) & 3
					out[pos] = float32(q-1) * d
					pos++
				}
			}
		}
	}
}

func dequantTQ2_0(shape tensor.Shape, raw []byte) (*tensor.Tensor, error) {
	n := shape.Numel()
	if n%tq2BlockElems != 0 {
		return nil, fmt.Errorf("gguf: TQ2_0 numel %d not multiple of %d", n, tq2BlockElems)
	}
	if want := n / tq2BlockElems * tq2BlockSize; len(raw) != want {
		return nil, fmt.Errorf("gguf: TQ2_0 data %dB, want %d", len(raw), want)
	}
	out := tensor.New(tensor.F32, shape)
	dequantTQ2_0Into(out.Storage().F32(), raw)
	return out, nil
}

// quantizeTQ2_0 follows quantize_row_tq2_0_ref from the pinned llama.cpp
// source. Quantization uses the unrounded f32 absolute maximum; only the stored
// trailing scale is rounded to f16.
func quantizeTQ2_0(x []float32) []byte {
	out := make([]byte, len(x)/tq2BlockElems*tq2BlockSize)
	for b := 0; b*tq2BlockElems < len(x); b++ {
		in := x[b*tq2BlockElems : (b+1)*tq2BlockElems]
		blk := out[b*tq2BlockSize : (b+1)*tq2BlockSize]
		var amax float32
		for _, v := range in {
			a := math.Float32frombits(math.Float32bits(v) &^ (1 << 31))
			if a > amax {
				amax = a
			}
		}
		var id float32
		if amax != 0 {
			id = 1 / amax
		}
		//perfscan:ignore PS4001 Each strided f16 scale is computed for its block rather than copied from a bulk source.
		binary.LittleEndian.PutUint16(blk[tq2PackedBytes:], f32ToF16(amax))
		for packed := 0; packed < tq2PackedBytes; packed += 32 {
			base := packed * 4
			for lane := range 32 {
				var q byte
				for plane := range 4 {
					xi := int(math.Round(float64(in[base+lane+plane*32]*id))) + 1
					q |= byte(xi&3) << uint(2*plane)
				}
				blk[packed+lane] = q
			}
		}
	}
	return out
}

// dotTQ2Row fuses TQ2_0 decode and activation dot while retaining ascending
// activation order and float64 accumulation.
func dotTQ2Row(row []float32, raw []byte, k int) float64 {
	var acc float64
	for b := 0; b*tq2BlockElems < k; b++ {
		blk := raw[b*tq2BlockSize : (b+1)*tq2BlockSize]
		//perfscan:ignore PS4001 TQ2_0 scales are strided 66-byte f16 fields, not a same-layout bulk plane.
		d := f16ToF32(binary.LittleEndian.Uint16(blk[tq2PackedBytes:]))
		base := b * tq2BlockElems
		pos := 0
		for packed := 0; packed < tq2PackedBytes; packed += 32 {
			for plane := range 4 {
				shift := uint(2 * plane)
				for lane := range 32 {
					q := int(blk[packed+lane]>>shift) & 3
					w := float32(q-1) * d
					acc += float64(row[base+pos]) * float64(w)
					pos++
				}
			}
		}
	}
	return acc
}
