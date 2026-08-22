package gguf

import (
	"encoding/binary"
	"math"

	"github.com/jxsl13/goai/tensor"
)

const q41BlockSize = 20

// quantizeQ4_1 follows quantize_row_q4_1_ref from llama.cpp commit
// 3af988fabcf79fd81f8720505e684d2aa5bfc786. Each block stores an affine
// grid spanning its minimum and maximum with fifteen intervals.
func quantizeQ4_1(x []float32) []byte {
	nb := len(x) / blockElems
	out := make([]byte, nb*q41BlockSize)
	for b := range nb {
		blk := x[b*blockElems : (b+1)*blockElems]
		minv := float32(math.MaxFloat32)
		maxv := -float32(math.MaxFloat32)
		for _, v := range blk {
			if v < minv {
				minv = v
			}
			if v > maxv {
				maxv = v
			}
		}
		d := (maxv - minv) / 15
		var id float32
		if d != 0 {
			id = 1 / d
		}
		o := b * q41BlockSize
		//perfscan:ignore PS4001 two strided f16 headers per 20-byte Q4_1 block cannot use a same-layout bulk copy
		binary.LittleEndian.PutUint16(out[o:], f32ToF16(d))
		binary.LittleEndian.PutUint16(out[o+2:], f32ToF16(minv))
		for j := range blockElems / 2 {
			lo := q41Nybble((blk[j] - minv) * id)
			hi := q41Nybble((blk[j+blockElems/2] - minv) * id)
			out[o+4+j] = lo | hi<<4
		}
	}
	return out
}

// q41Nybble is MIN(15, (int8_t)(v + 0.5f)) from the ggml reference.
// A valid finite block maps v into [0,15]; the lower clamp also keeps hostile
// non-finite input from wrapping through a Go integer conversion.
func q41Nybble(v float32) byte {
	if !(v >= 0) { // includes NaN
		return 0
	}
	if v >= 15 { // includes +Inf
		return 15
	}
	q := int(v + 0.5)
	return byte(q)
}

func dequantQ4_1(shape tensor.Shape, raw []byte) (*tensor.Tensor, error) {
	t := tensor.New(tensor.F32, shape)
	dequantQ4_1Into(t.Storage().F32(), raw)
	return t, nil
}

func dequantQ4_1Into(dst []float32, raw []byte) {
	for b := 0; b*blockElems < len(dst); b++ {
		blk := raw[b*q41BlockSize : (b+1)*q41BlockSize]
		//perfscan:ignore PS4001 two strided f16 headers per 20-byte Q4_1 block cannot use a same-layout bulk copy
		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
		m := f16ToF32(binary.LittleEndian.Uint16(blk[2:4]))
		y, qs := dst[b*blockElems:b*blockElems+blockElems], blk[4:20]
		for i, q := range qs {
			y[i] = d*float32(q&0x0f) + m
			y[i+blockElems/2] = d*float32(q>>4) + m
		}
	}
}

// dotQ41Row fuses Q4_1 affine decode into a matrix-vector row dot. It retains
// the materialized decoder's low-half/high-half order and widens every decoded
// float32 weight before multiplication, making it the portable ground truth.
func dotQ41Row(row []float32, raw []byte, k int) float64 {
	var acc float64
	for b := 0; b*blockElems < k; b++ {
		blk := raw[b*q41BlockSize : (b+1)*q41BlockSize]
		//perfscan:ignore PS4001 fused row dot consumes one strided scale/minimum pair per quant block
		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
		m := f16ToF32(binary.LittleEndian.Uint16(blk[2:4]))
		qs := blk[4:20]
		base := b * blockElems
		for i, q := range qs {
			weight := d*float32(q&0x0f) + m
			acc += float64(row[base+i]) * float64(weight)
		}
		for i, q := range qs {
			weight := d*float32(q>>4) + m
			acc += float64(row[base+blockElems/2+i]) * float64(weight)
		}
	}
	return acc
}
