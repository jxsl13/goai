//go:build arm64

#include "textflag.h"

// Expand one packed sign byte through the 2 KiB high-byte table, widen those
// bytes twice into eight float32 sign masks, apply them to eight activations,
// then widen and accumulate scale*activation in four float64 vectors.
#define DOT_Q1_SIGNS() \
	MOVBU (R5), R9; \
	ADD $1, R5, R5; \
	LSL $3, R9, R9; \
	ADD R3, R9, R10; \
	WORD $0x0C407140; /* LD1 {V0.8B}, [R10] */ \
	WORD $0x2E213808; /* SHLL  V8.8H, V0.8B, #8 */ \
	WORD $0x6E613909; /* SHLL2 V9.4S, V8.8H, #16 */ \
	WORD $0x2E613908; /* SHLL  V8.4S, V8.4H, #16 */ \
	VLD1.P 32(R0), [V12.S4, V13.S4]; \
	VEOR V8.B16, V12.B16, V12.B16; \
	VEOR V9.B16, V13.B16, V13.B16; \
	WORD $0x0E617984; /* FCVTL  V4.2D, V12.2S */ \
	WORD $0x4E617985; /* FCVTL2 V5.2D, V12.4S */ \
	WORD $0x0E6179A6; /* FCVTL  V6.2D, V13.2S */ \
	WORD $0x4E6179A7; /* FCVTL2 V7.2D, V13.4S */ \
	VFMLA V28.D2, V4.D2, V16.D2; \
	VFMLA V28.D2, V5.D2, V17.D2; \
	VFMLA V28.D2, V6.D2, V18.D2; \
	VFMLA V28.D2, V7.D2, V19.D2

// func dotQ1RowNeon(x *float32, raw *byte, f16 *float32, signBytes *byte, blocks int) float64
TEXT ·dotQ1RowNeon(SB), NOSPLIT, $0-48
	MOVD x+0(FP), R0
	MOVD raw+8(FP), R1
	MOVD f16+16(FP), R2
	MOVD signBytes+24(FP), R3
	MOVD blocks+32(FP), R7

	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16

block_loop:
	MOVHU (R1), R4
	LSL $2, R4, R4
	ADD R2, R4, R4
	VLD1R (R4), [V29.S4]
	WORD $0x0E617BBC // FCVTL V28.2D, V29.2S

	ADD $2, R1, R5
	MOVD $16, R6
sign_loop:
	DOT_Q1_SIGNS()
	SUBS $1, R6, R6
	BNE sign_loop

	ADD $18, R1, R1
	SUBS $1, R7, R7
	BNE block_loop

	WORD $0x4E72D610 // FADD V16.2D, V16.2D, V18.2D
	WORD $0x4E73D631 // FADD V17.2D, V17.2D, V19.2D
	WORD $0x4E71D610 // FADD V16.2D, V16.2D, V17.2D
	WORD $0x7E70DA10 // FADDP V16.2D, F16
	FMOVD F16, ret+40(FP)
	RET
