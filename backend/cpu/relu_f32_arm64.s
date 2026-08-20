//go:build arm64

#include "textflag.h"

// func reluF32BlocksNeon(dst, src *float32, blocks int)
//
// Sixteen f32 lanes per iteration. FCMGT is ordered, so NaN lanes compare
// false. BSL then chooses the original bits only for x>0 and an integer-zero
// vector otherwise, preserving positive finite/+Inf bits while mapping
// negatives, NaNs, +0, and -0 to +0. FMAX is deliberately not used because
// its NaN and signed-zero semantics do not implement GoAI's ReLU contract.
TEXT ·reluF32BlocksNeon(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD blocks+16(FP), R2
	VMOVI $0, V31.B16

loop:
	VLD1.P 64(R1), [V0.S4, V1.S4, V2.S4, V3.S4]
	WORD $0x6EBFE404 // FCMGT V4.4S, V0.4S, V31.4S
	WORD $0x6EBFE425 // FCMGT V5.4S, V1.4S, V31.4S
	WORD $0x6EBFE446 // FCMGT V6.4S, V2.4S, V31.4S
	WORD $0x6EBFE467 // FCMGT V7.4S, V3.4S, V31.4S
	WORD $0x6E7F1C04 // BSL V4.16B, V0.16B, V31.16B
	WORD $0x6E7F1C25 // BSL V5.16B, V1.16B, V31.16B
	WORD $0x6E7F1C46 // BSL V6.16B, V2.16B, V31.16B
	WORD $0x6E7F1C67 // BSL V7.16B, V3.16B, V31.16B
	VST1 [V4.S4, V5.S4, V6.S4, V7.S4], (R0)
	ADD $64, R0, R0
	SUBS $1, R2, R2
	BNE loop
	RET
