//go:build arm64

#include "textflag.h"

// Convert sixteen signed q8 bytes in SRC to four f32 vectors, apply the
// current block scale in V28, and accumulate their dot against sixteen
// consecutive activations at R5. Four accumulators hide the FMLA latency.
#define DOT_Q8(SRC) \
	VTBL V24.B16, [SRC.B16], V8.B16; \
	VTBL V25.B16, [SRC.B16], V9.B16; \
	VTBL V26.B16, [SRC.B16], V10.B16; \
	VTBL V27.B16, [SRC.B16], V11.B16; \
	WORD $0x4F28E508; /* SCVTF V8.4S, V8.4S, #24 */ \
	WORD $0x4F28E529; /* SCVTF V9.4S, V9.4S, #24 */ \
	WORD $0x4F28E54A; /* SCVTF V10.4S, V10.4S, #24 */ \
	WORD $0x4F28E56B; /* SCVTF V11.4S, V11.4S, #24 */ \
	WORD $0x6E3CDD08; /* FMUL V8.4S, V8.4S, V28.4S */ \
	WORD $0x6E3CDD29; /* FMUL V9.4S, V9.4S, V28.4S */ \
	WORD $0x6E3CDD4A; /* FMUL V10.4S, V10.4S, V28.4S */ \
	WORD $0x6E3CDD6B; /* FMUL V11.4S, V11.4S, V28.4S */ \
	VLD1 (R5), [V12.S4, V13.S4, V14.S4, V15.S4]; \
	VFMLA V8.S4, V12.S4, V16.S4; \
	VFMLA V9.S4, V13.S4, V17.S4; \
	VFMLA V10.S4, V14.S4, V18.S4; \
	VFMLA V11.S4, V15.S4, V19.S4

// func dotQ8RowNeon(x *float32, raw *byte, f16 *float32, indexes *byte, blocks int) float32
TEXT ·dotQ8RowNeon(SB), NOSPLIT, $0-44
	MOVD x+0(FP), R0
	MOVD raw+8(FP), R1
	MOVD f16+16(FP), R2
	MOVD indexes+24(FP), R3
	MOVD blocks+32(FP), R7

	VLD1 (R3), [V24.B16, V25.B16, V26.B16, V27.B16]
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16

block:
	MOVHU (R1), R4
	LSL $2, R4, R4
	ADD R2, R4, R4
	VLD1R (R4), [V28.S4]
	ADD $2, R1, R6
	VLD1 (R6), [V0.B16, V1.B16]

	MOVD R0, R5
	DOT_Q8(V0)
	ADD $64, R0, R5
	DOT_Q8(V1)

	ADD $128, R0, R0
	ADD $34, R1, R1
	SUBS $1, R7, R7
	BNE block

	WORD $0x4E31D610 // FADD V16.4S, V16.4S, V17.4S
	WORD $0x4E33D652 // FADD V18.4S, V18.4S, V19.4S
	WORD $0x4E32D610 // FADD V16.4S, V16.4S, V18.4S
	WORD $0x6E30D610 // FADDP V16.4S, V16.4S, V16.4S
	WORD $0x6E30D610 // FADDP V16.4S, V16.4S, V16.4S
	FMOVS F16, ret+40(FP)
	RET
