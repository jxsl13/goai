//go:build arm64

#include "textflag.h"

// func absF32BlocksNeon(dst, src *float32, blocks int)
//
// Sixteen F32 lanes per iteration. Integer masking clears only bit 31, which
// matches Go 1.27's float32(math.Abs(float64(x))) lowering and preserves every
// finite, infinity, and NaN payload bit.
TEXT ·absF32BlocksNeon(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD blocks+16(FP), R2
	MOVD $0x7fffffff, R3
	VDUP R3, V28.S4

loop:
	VLD1.P 64(R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	VAND V28.B16, V0.B16, V4.B16
	VAND V28.B16, V1.B16, V5.B16
	VAND V28.B16, V2.B16, V6.B16
	VAND V28.B16, V3.B16, V7.B16
	VST1 [V4.S4, V5.S4, V6.S4, V7.S4], (R0)
	ADD $64, R0, R0
	SUBS $1, R2, R2
	BNE loop
	RET
