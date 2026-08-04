//go:build amd64 && goexperiment.simd

package gguf

import (
	"encoding/binary"
	"simd/archsimd"
	"unsafe"
)

// SIMD Q8_0 single-token (m==1) decode matmul: for each output row, dot the f32
// activation with the dequantized int8 weight row. int8 quants are widened
// int8→int16→int32→float32 (VPMOVSXBD chain) 8-wide and FMA'd against the activation;
// the block scale d is applied per 32-block. TOLERANCE-GATED (not bit-exact vs the
// scalar general path): the within-block sum is f32 and d is factored per-block — the
// same class of departure as an FMA kernel, well within the quantization error.
func q8FusedDecodeM1SIMD(row []float32, weight []byte, n, k, rowBytes int, outf []float32) {
	kb := k &^ 31
	for ni := 0; ni < n; ni++ {
		rb := weight[ni*rowBytes:]
		acc := archsimd.BroadcastFloat32x8(0)
		bi := 0
		for off := 0; off < kb; off += 32 {
			o := bi * 34
			d := f16ToF32(binary.LittleEndian.Uint16(rb[o : o+2]))
			q := rb[o+2:]
			qv := archsimd.LoadInt8x32((*[32]int8)(unsafe.Pointer(&q[0])))
			lo := qv.GetLo().ExtendToInt16()
			hi := qv.GetHi().ExtendToInt16()
			g0 := lo.GetLo().ExtendToInt32().ConvertToFloat32()
			g1 := lo.GetHi().ExtendToInt32().ConvertToFloat32()
			g2 := hi.GetLo().ExtendToInt32().ConvertToFloat32()
			g3 := hi.GetHi().ExtendToInt32().ConvertToFloat32()
			x0 := archsimd.LoadFloat32x8((*[8]float32)(unsafe.Pointer(&row[off])))
			x1 := archsimd.LoadFloat32x8((*[8]float32)(unsafe.Pointer(&row[off+8])))
			x2 := archsimd.LoadFloat32x8((*[8]float32)(unsafe.Pointer(&row[off+16])))
			x3 := archsimd.LoadFloat32x8((*[8]float32)(unsafe.Pointer(&row[off+24])))
			bacc := x0.Mul(g0)
			bacc = x1.MulAdd(g1, bacc)
			bacc = x2.MulAdd(g2, bacc)
			bacc = x3.MulAdd(g3, bacc)
			acc = bacc.MulAdd(archsimd.BroadcastFloat32x8(d), acc) // acc += (row·q)*d, f32 across blocks
			bi++
		}
		var buf [8]float32
		acc.StoreSlice(buf[:])
		accF := float64(buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7])
		//perfscan:ignore PS3010 hand SIMD Q8 matmul at-ceiling; absent in this worktree
		for i := kb; i < k; i++ {
			o := (i / 32) * 34
			d := f16ToF32(binary.LittleEndian.Uint16(rb[o : o+2]))
			accF += float64(row[i]) * float64(d*float32(int8(rb[o+2+(i%32)])))
		}
		outf[ni] = float32(accF)
	}
}

func init() { q8FusedDecodeM1 = q8FusedDecodeM1SIMD }
