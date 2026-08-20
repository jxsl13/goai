//go:build arm64

#include "textflag.h"

// func negF32BlocksNeon(dst, src *float32, blocks int)
//
// Sixteen F32 lanes per iteration. XOR operates in the integer domain and
// changes only bit 31, preserving finite values, both zero encodings,
// infinities, signaling/quiet NaNs, and every NaN payload bit exactly.
TEXT ·negF32BlocksNeon(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD blocks+16(FP), R2
	MOVD $0x80000000, R3
	VDUP R3, V31.S4

loop:
	VLD1.P 64(R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	VEOR V31.B16, V0.B16, V0.B16
	VEOR V31.B16, V1.B16, V1.B16
	VEOR V31.B16, V2.B16, V2.B16
	VEOR V31.B16, V3.B16, V3.B16
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R0)
	ADD $64, R0, R0
	SUBS $1, R2, R2
	BNE loop
	RET
