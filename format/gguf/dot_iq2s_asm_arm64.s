//go:build arm64

#include "textflag.h"

// Multiply one signed eight-wide grid row by the current coefficient, load the
// matching activations, widen both operands to float64, and feed four
// independent accumulators. Each output lane keeps a fixed element mapping.
#define DOT_IQ2S() \
	WORD $0x6E3DDD08; /* FMUL V8.4S, V8.4S, V29.4S */ \
	WORD $0x6E3DDD29; /* FMUL V9.4S, V9.4S, V29.4S */ \
	VLD1.P 32(R0), [V12.S4, V13.S4]; \
	WORD $0x0E617904; /* FCVTL  V4.2D, V8.2S */ \
	WORD $0x4E617905; /* FCVTL2 V5.2D, V8.4S */ \
	WORD $0x0E617986; /* FCVTL  V6.2D, V12.2S */ \
	WORD $0x4E617987; /* FCVTL2 V7.2D, V12.4S */ \
	VFMLA V4.D2, V6.D2, V16.D2; \
	VFMLA V5.D2, V7.D2, V17.D2; \
	WORD $0x0E617924; /* FCVTL  V4.2D, V9.2S */ \
	WORD $0x4E617925; /* FCVTL2 V5.2D, V9.4S */ \
	WORD $0x0E6179A6; /* FCVTL  V6.2D, V13.2S */ \
	WORD $0x4E6179A7; /* FCVTL2 V7.2D, V13.4S */ \
	VFMLA V4.D2, V6.D2, V18.D2; \
	VFMLA V5.D2, V7.D2, V19.D2

#define GATHER_DOT_IQ2S(LSB) \
	MOVBU (R1), R10; \
	ADD $1, R1, R1; \
	UBFX $LSB, R8, $2, R11; \
	LSL $8, R11, R11; \
	ORR R11, R10, R10; \
	LSL $5, R10, R10; \
	ADD R3, R10, R10; \
	VLD1 (R10), [V8.S4, V9.S4]; \
	MOVBU (R5), R12; \
	ADD $1, R5, R5; \
	LSL $5, R12, R12; \
	ADD R4, R12, R12; \
	VLD1 (R12), [V10.S4, V11.S4]; \
	VEOR V10.B16, V8.B16, V8.B16; \
	VEOR V11.B16, V9.B16, V9.B16; \
	DOT_IQ2S()

// func dotIQ2SBlockNeon(x *float32, qs *byte, coeff *float32, grid *float32, signMasks *uint32) float64
TEXT ·dotIQ2SBlockNeon(SB), NOSPLIT, $0-48
	MOVD x+0(FP), R0
	MOVD qs+8(FP), R1
	MOVD coeff+16(FP), R2
	MOVD grid+24(FP), R3
	MOVD signMasks+32(FP), R4

	ADD $32, R1, R5  // signs
	ADD $64, R1, R6  // qh
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	MOVD $8, R7
high:
	MOVBU (R6), R8
	ADD $1, R6, R6
	VLD1R.P 4(R2), [V29.S4]
	GATHER_DOT_IQ2S(0)
	GATHER_DOT_IQ2S(2)
	VLD1R.P 4(R2), [V29.S4]
	GATHER_DOT_IQ2S(4)
	GATHER_DOT_IQ2S(6)
	SUBS $1, R7, R7
	BNE high

	WORD $0x4E72D610 // FADD V16.2D, V16.2D, V18.2D
	WORD $0x4E73D631 // FADD V17.2D, V17.2D, V19.2D
	WORD $0x4E71D610 // FADD V16.2D, V16.2D, V17.2D
	WORD $0x7E70DA10 // FADDP V16.2D, F16
	FMOVD F16, ret+40(FP)
	RET
