package gguf

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// Q1_0 is ggml's 1.125-bit binary quant: one f16 scale and one LSB-first
// sign bit per weight. A set bit selects +d and a clear bit selects -d.
const (
	tQ1_0        = 41
	q1BlockElems = 128
	q1BlockSize  = 18
)

// dequantQ1_0Into decodes complete Q1_0 blocks into caller-owned storage.
func dequantQ1_0Into(dst []float32, raw []byte) {
	for b := 0; b*q1BlockSize < len(raw); b++ {
		blk := raw[b*q1BlockSize : (b+1)*q1BlockSize]
		//perfscan:ignore PS4001 Q1 scales are strided every 18 bytes; no same-layout bulk decode exists
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		signs := blk[2:]
		out := dst[b*q1BlockElems : (b+1)*q1BlockElems]
		for i := range out {
			out[i] = -d
			if signs[i/8]&(1<<uint(i%8)) != 0 {
				out[i] = d
			}
		}
	}
}

func dequantQ1_0(shape tensor.Shape, raw []byte) (*tensor.Tensor, error) {
	n := shape.Numel()
	if n%q1BlockElems != 0 {
		return nil, fmt.Errorf("gguf: Q1_0 numel %d not multiple of %d", n, q1BlockElems)
	}
	if want := n / q1BlockElems * q1BlockSize; len(raw) != want {
		return nil, fmt.Errorf("gguf: Q1_0 data %dB, want %d", len(raw), want)
	}
	out := tensor.New(tensor.F32, shape)
	dequantQ1_0Into(out.Storage().F32(), raw)
	return out, nil
}

// quantizeQ1_0 follows ggml's reference encoder: d is the float32 mean
// absolute value of each 128-weight block and each non-negative input sets its
// LSB-first sign bit. Only the f16-rounded d is stored.
func quantizeQ1_0(x []float32) []byte {
	out := make([]byte, len(x)/q1BlockElems*q1BlockSize)
	for b := 0; b*q1BlockElems < len(x); b++ {
		in := x[b*q1BlockElems : (b+1)*q1BlockElems]
		blk := out[b*q1BlockSize : (b+1)*q1BlockSize]
		var sumAbs float32
		for _, v := range in {
			sumAbs += math.Float32frombits(math.Float32bits(v) &^ (1 << 31))
		}
		//perfscan:ignore PS5001 one reference-format divide per block feeds f16 quantization; preserve pinned ggml arithmetic
		d := sumAbs / q1BlockElems
		//perfscan:ignore PS4001 Q1 scales are strided every 18 bytes; no same-layout bulk encode exists
		binary.LittleEndian.PutUint16(blk, f32ToF16(d))
		for i, v := range in {
			if v >= 0 {
				blk[2+i/8] |= 1 << uint(i%8)
			}
		}
	}
	return out
}

// dotQ1Row fuses Q1_0 decode and activation dot while preserving the
// materialized decoder's ascending element order and float64 accumulation.
func dotQ1Row(row []float32, raw []byte, k int) float64 {
	var acc float64
	for b := 0; b*q1BlockElems < k; b++ {
		blk := raw[b*q1BlockSize : (b+1)*q1BlockSize]
		//perfscan:ignore PS4001 Q1 scales are strided every 18 bytes; no same-layout bulk decode exists
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		signs := blk[2:]
		base := b * q1BlockElems
		for i := range q1BlockElems {
			weight := -d
			if signs[i/8]&(1<<uint(i%8)) != 0 {
				weight = d
			}
			acc += float64(row[base+i]) * float64(weight)
		}
	}
	return acc
}
