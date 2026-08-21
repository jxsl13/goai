//go:build arm64

#include "textflag.h"

// Convert sixteen unsigned q4 nibbles in SRC to four f32 vectors, apply the
// exact dequant operation order step*q-offset, load sixteen activations, and
// accumulate four independent vectors. Keeping four accumulators breaks the
// single-FMLA dependency chain without changing element mapping.
#define DOT_Q4(SRC) \
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
	VLD1 (R4), [V12.S4, V13.S4, V14.S4, V15.S4]; \
	VFMLA V8.S4, V12.S4, V16.S4; \
	VFMLA V9.S4, V13.S4, V17.S4; \
	VFMLA V10.S4, V14.S4, V18.S4; \
	VFMLA V11.S4, V15.S4, V19.S4

// func dotQ4KBlockNeon(x *float32, qs *byte, coeff *float32, indexes *byte) float32
TEXT ·dotQ4KBlockNeon(SB), NOSPLIT, $0-36
	MOVD x+0(FP), R0
	MOVD qs+8(FP), R1
	MOVD coeff+16(FP), R2
	MOVD indexes+24(FP), R3

	VLD1 (R3), [V24.B16, V25.B16, V26.B16, V27.B16]
	VMOVI $0x0f, V23.B16
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
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
	DOT_Q4(V2)
	ADD $64, R0, R4
	DOT_Q4(V3)

	VLD1R.P 4(R2), [V20.S4]
	VLD1R.P 4(R2), [V21.S4]
	ADD $128, R0, R4
	DOT_Q4(V4)
	ADD $192, R0, R4
	DOT_Q4(V5)

	ADD $256, R0, R0
	SUBS $1, R7, R7
	BNE pair

	WORD $0x4E31D610 // FADD V16.4S, V16.4S, V17.4S
	WORD $0x4E33D652 // FADD V18.4S, V18.4S, V19.4S
	WORD $0x4E32D610 // FADD V16.4S, V16.4S, V18.4S
	WORD $0x6E30D610 // FADDP V16.4S, V16.4S, V16.4S
	WORD $0x6E30D610 // FADDP V16.4S, V16.4S, V16.4S
	FMOVS F16, ret+32(FP)
	RET
