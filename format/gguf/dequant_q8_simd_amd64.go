//go:build amd64 && goexperiment.simd

package gguf

import (
	"encoding/binary"
	"simd/archsimd"
	"unsafe"
)

// SIMD Q8_0 dequant: y[i] = d·float32(int8(q[i])) 8-wide via the int8→int32→float32
// widen chain. BIT-EXACT to the scalar (identical per-element float32 multiply, no
// accumulation). Registered as the dequantQ8Into0 hook on amd64+simd.
func dequantQ8_0IntoSIMD(dst []float32, raw []byte) {
	for b := 0; b*blockElems < len(dst); b++ {
		blk := raw[b*34 : b*34+34]
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		y := dst[b*blockElems : b*blockElems+blockElems]
		q := blk[2:34]
		bd := archsimd.BroadcastFloat32x8(d)
		qv := archsimd.LoadInt8x32(unsafe.Slice((*int8)(unsafe.Pointer(&q[0])), 32))
		lo := qv.GetLo().ExtendToInt16()
		hi := qv.GetHi().ExtendToInt16()
		lo.GetLo().ExtendToInt32().ConvertToFloat32().Mul(bd).Store(y[0:])
		lo.GetHi().ExtendToInt32().ConvertToFloat32().Mul(bd).Store(y[8:])
		hi.GetLo().ExtendToInt32().ConvertToFloat32().Mul(bd).Store(y[16:])
		hi.GetHi().ExtendToInt32().ConvertToFloat32().Mul(bd).Store(y[24:])
	}
}

func init() { dequantQ8Into0 = dequantQ8_0IntoSIMD }
