//go:build arm64

package gguf

import "encoding/binary"

// qKByteToF32Indexes expands one vector of sixteen byte-sized integers into
// four vectors whose signed integer lives in bits 31:24 of each lane. A fixed-
// point SCVTF #24 then produces the exact float32 integer without the eight
// widening instructions that a conventional int8→int16→int32 path needs.
var qKByteToF32Indexes = [64]byte{
	16, 16, 16, 0, 16, 16, 16, 1, 16, 16, 16, 2, 16, 16, 16, 3,
	16, 16, 16, 4, 16, 16, 16, 5, 16, 16, 16, 6, 16, 16, 16, 7,
	16, 16, 16, 8, 16, 16, 16, 9, 16, 16, 16, 10, 16, 16, 16, 11,
	16, 16, 16, 12, 16, 16, 16, 13, 16, 16, 16, 14, 16, 16, 16, 15,
}

// dequantQ6KBlockNeon decodes one 210-byte Q6_K super-block into 256 f32
// values. Go passes the exact table-decoded half scale; assembly vectorizes the
// sixteen int8 scale products and all 256 q6 values.
//
//go:noescape
func dequantQ6KBlockNeon(dst *float32, raw *byte, d float32, indexes *byte)

func dequantQ6_KIntoArch(dst []float32, raw []byte) {
	for sb := 0; sb*qkK < len(dst); sb++ {
		blk := raw[sb*q6kBlockSize : (sb+1)*q6kBlockSize]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[208:]))
		dequantQ6KBlockNeon(&dst[sb*qkK], &blk[0], d, &qKByteToF32Indexes[0])
	}
}
