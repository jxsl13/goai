//go:build arm64

#include "textflag.h"

// Convert sixteen unsigned q4 nibbles in SRC to four f32 vectors, apply the
// broadcast affine transform step*q-offset, and store 64 bytes at R4.
#define CONVERT_Q4(SRC) \
	VTBL V24.B16, [SRC.B16], V8.B16; \
	VTBL V25.B16, [SRC.B16], V9.B16; \
	VTBL V26.B16, [SRC.B16], V10.B16; \
	VTBL V27.B16, [SRC.B16], V11.B16; \
	WORD $0x6F28E508; /* UCVTF V8.4S, V8.4S, #24 */ \
	WORD $0x6F28E529; /* UCVTF V9.4S, V9.4S, #24 */ \
	WORD $0x6F28E54A; /* UCVTF V10.4S, V10.4S, #24 */ \
	WORD $0x6F28E56B; /* UCVTF V11.4S, V11.4S, #24 */ \
	WORD $0x6E34DD08; /* FMUL V8.4S, V8.4S, V20.4S */ \
	WORD $0x6E34DD29; /* FMUL V9.4S, V9.4S, V20.4S */ \
	WORD $0x6E34DD4A; /* FMUL V10.4S, V10.4S, V20.4S */ \
	WORD $0x6E34DD6B; /* FMUL V11.4S, V11.4S, V20.4S */ \
	WORD $0x4EB5D508; /* FSUB V8.4S, V8.4S, V21.4S */ \
	WORD $0x4EB5D529; /* FSUB V9.4S, V9.4S, V21.4S */ \
	WORD $0x4EB5D54A; /* FSUB V10.4S, V10.4S, V21.4S */ \
	WORD $0x4EB5D56B; /* FSUB V11.4S, V11.4S, V21.4S */ \
	VST1 [V8.S4, V9.S4, V10.S4, V11.S4], (R4)

// func dequantQ4KBlockNeon(dst *float32, qs *byte, coeff *float32, indexes *byte)
TEXT ·dequantQ4KBlockNeon(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD qs+8(FP), R1
	MOVD coeff+16(FP), R2
	MOVD indexes+24(FP), R3

	VLD1 (R3), [V24.B16, V25.B16, V26.B16, V27.B16]
	VMOVI $0x0f, V23.B16
	MOVD $4, R7

pair:
	VLD1 (R1), [V0.B16, V1.B16]
	ADD $32, R1, R1
	VAND V23.B16, V0.B16, V2.B16
	VAND V23.B16, V1.B16, V3.B16
	VUSHR $4, V0.B16, V4.B16
	VUSHR $4, V1.B16, V5.B16

	VLD1R.P 4(R2), [V20.S4]
	VLD1R.P 4(R2), [V21.S4]
	MOVD R0, R4
	CONVERT_Q4(V2)
	ADD $64, R0, R4
	CONVERT_Q4(V3)

	VLD1R.P 4(R2), [V20.S4]
	VLD1R.P 4(R2), [V21.S4]
	ADD $128, R0, R4
	CONVERT_Q4(V4)
	ADD $192, R0, R4
	CONVERT_Q4(V5)

	ADD $256, R0, R0
	SUBS $1, R7, R7
	BNE pair
	RET
