package gguf

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// TQ1_0 is ggml's 1.6875-bit ternary quant. Each 256-weight block stores
// forty-eight five-trit base-243 bytes, four four-trit tail bytes, and one
// trailing f16 scale. The byte layout deliberately follows ggml's vector-friendly
// element permutation rather than consecutive groups of five weights.
const (
	tTQ1_0         = 34
	tq1BlockElems  = 256
	tq1PackedBytes = 48
	tq1TailBytes   = 4
	tq1BlockSize   = tq1PackedBytes + tq1TailBytes + 2
)

func tq1Trit(packed byte, trit int) int8 {
	mul := [...]byte{1, 3, 9, 27, 81}[trit]
	q := packed * mul
	return int8(uint16(q)*3>>8) - 1
}

// dequantTQ1_0Into preserves ggml's output order: 32 interleaved five-trit
// groups, 16 interleaved five-trit groups, then four interleaved four-trit
// groups. dst and raw contain complete blocks.
func dequantTQ1_0Into(dst []float32, raw []byte) {
	for b := 0; b*tq1BlockSize < len(raw); b++ {
		blk := raw[b*tq1BlockSize : (b+1)*tq1BlockSize]
		//perfscan:ignore PS4001 TQ1_0 scales are strided 54-byte f16 fields, not a same-layout bulk plane.
		d := f16ToF32(binary.LittleEndian.Uint16(blk[tq1PackedBytes+tq1TailBytes:]))
		out := dst[b*tq1BlockElems : (b+1)*tq1BlockElems]
		pos := 0
		for trit := range 5 {
			for lane := range 32 {
				out[pos] = float32(tq1Trit(blk[lane], trit)) * d
				pos++
			}
		}
		for packed := 32; packed < tq1PackedBytes; packed += 16 {
			for trit := range 5 {
				for lane := range 16 {
					out[pos] = float32(tq1Trit(blk[packed+lane], trit)) * d
					pos++
				}
			}
		}
		for trit := range 4 {
			for lane := range tq1TailBytes {
				out[pos] = float32(tq1Trit(blk[tq1PackedBytes+lane], trit)) * d
				pos++
			}
		}
	}
}

func dequantTQ1_0(shape tensor.Shape, raw []byte) (*tensor.Tensor, error) {
	n := shape.Numel()
	if n%tq1BlockElems != 0 {
		return nil, fmt.Errorf("gguf: TQ1_0 numel %d not multiple of %d", n, tq1BlockElems)
	}
	if want := n / tq1BlockElems * tq1BlockSize; len(raw) != want {
		return nil, fmt.Errorf("gguf: TQ1_0 data %dB, want %d", len(raw), want)
	}
	out := tensor.New(tensor.F32, shape)
	dequantTQ1_0Into(out.Storage().F32(), raw)
	return out, nil
}

func tq1Pack5(x []float32, base, stride int, id float32) byte {
	q := 0
	for trit := range 5 {
		xi := int(math.Round(float64(x[base+trit*stride]*id))) + 1
		q = q*3 + xi
	}
	return byte((q*256 + 242) / 243)
}

// quantizeTQ1_0 follows quantize_row_tq1_0_ref from the pinned llama.cpp
// source. Quantization uses the unrounded f32 absolute maximum; only the stored
// trailing scale is rounded to f16.
func quantizeTQ1_0(x []float32) []byte {
	out := make([]byte, len(x)/tq1BlockElems*tq1BlockSize)
	for b := 0; b*tq1BlockElems < len(x); b++ {
		in := x[b*tq1BlockElems : (b+1)*tq1BlockElems]
		blk := out[b*tq1BlockSize : (b+1)*tq1BlockSize]
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
		binary.LittleEndian.PutUint16(blk[tq1PackedBytes+tq1TailBytes:], f32ToF16(amax))
		for lane := range 32 {
			blk[lane] = tq1Pack5(in, lane, 32, id)
		}
		for lane := range 16 {
			blk[32+lane] = tq1Pack5(in, 160+lane, 16, id)
		}
		for lane := range tq1TailBytes {
			q := 0
			for trit := range 4 {
				xi := int(math.Round(float64(in[240+lane+trit*tq1TailBytes]*id))) + 1
				q = q*3 + xi
			}
			q *= 3
			blk[tq1PackedBytes+lane] = byte((q*256 + 242) / 243)
		}
	}
	return out
}

// dotTQ1Row fuses TQ1_0 decode and activation dot while retaining ascending
// activation order and float64 accumulation.
func dotTQ1Row(row []float32, raw []byte, k int) float64 {
	var acc float64
	for b := 0; b*tq1BlockElems < k; b++ {
		blk := raw[b*tq1BlockSize : (b+1)*tq1BlockSize]
		//perfscan:ignore PS4001 TQ1_0 scales are strided 54-byte f16 fields, not a same-layout bulk plane.
		d := f16ToF32(binary.LittleEndian.Uint16(blk[tq1PackedBytes+tq1TailBytes:]))
		base := b * tq1BlockElems
		pos := 0
		for trit := range 5 {
			for lane := range 32 {
				w := float32(tq1Trit(blk[lane], trit)) * d
				acc += float64(row[base+pos]) * float64(w)
				pos++
			}
		}
		for packed := 32; packed < tq1PackedBytes; packed += 16 {
			for trit := range 5 {
				for lane := range 16 {
					w := float32(tq1Trit(blk[packed+lane], trit)) * d
					acc += float64(row[base+pos]) * float64(w)
					pos++
				}
			}
		}
		for trit := range 4 {
			for lane := range tq1TailBytes {
				w := float32(tq1Trit(blk[tq1PackedBytes+lane], trit)) * d
				acc += float64(row[base+pos]) * float64(w)
				pos++
			}
		}
	}
	return acc
}
