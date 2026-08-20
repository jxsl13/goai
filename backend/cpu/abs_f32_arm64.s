//go:build arm64

#include "textflag.h"

// func absF32BlocksNeon(dst, src *float32, blocks int)
//
// Sixteen F32 lanes per iteration. Each magnitude is compared as an unsigned
// integer with +Inf. Lanes above +Inf are NaNs, so their all-ones masks are
// narrowed to the quiet bit and ORed into the magnitude. This reproduces the
// incumbent FCVT-FABS-FCVT result: both zero signs and every negative sign are
// cleared, signaling NaNs are quieted, and all remaining payload bits remain.
TEXT ·absF32BlocksNeon(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD blocks+16(FP), R2
	MOVD $0x7fffffff, R3
	MOVD $0x7f800000, R4
	MOVD $0x00400000, R5
	VDUP R3, V28.S4
	VDUP R4, V29.S4
	VDUP R5, V30.S4

loop:
	VLD1.P 64(R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	VAND V28.B16, V0.B16, V4.B16
	VAND V28.B16, V1.B16, V5.B16
	VAND V28.B16, V2.B16, V6.B16
	VAND V28.B16, V3.B16, V7.B16
	WORD $0x6EBD3480 // CMHI V0.4S, V4.4S, V29.4S
	WORD $0x6EBD34A1 // CMHI V1.4S, V5.4S, V29.4S
	WORD $0x6EBD34C2 // CMHI V2.4S, V6.4S, V29.4S
	WORD $0x6EBD34E3 // CMHI V3.4S, V7.4S, V29.4S
	VAND V30.B16, V0.B16, V0.B16
	VAND V30.B16, V1.B16, V1.B16
	VAND V30.B16, V2.B16, V2.B16
	VAND V30.B16, V3.B16, V3.B16
	VORR V0.B16, V4.B16, V4.B16
	VORR V1.B16, V5.B16, V5.B16
	VORR V2.B16, V6.B16, V6.B16
	VORR V3.B16, V7.B16, V7.B16
	VST1 [V4.S4, V5.S4, V6.S4, V7.S4], (R0)
	ADD $64, R0, R0
	SUBS $1, R2, R2
	BNE loop
	RET
