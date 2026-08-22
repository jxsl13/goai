//go:build arm64

#include "textflag.h"

// Multiply one pre-adjusted eight-wide ternary-grid row by the current odd
// coefficient, load the matching activations, widen both operands to float64,
// and feed four independent accumulators.
#define DOT_IQ1S() \
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

#define GATHER_DOT_IQ1S(LSB) \
	MOVBU (R13), R10; \
	ADD $1, R13, R13; \
	UBFX $LSB, R8, $3, R11; \
	LSL $8, R11, R11; \
	ORR R11, R10, R10; \
	LSL $5, R10, R10; \
	ORR R9, R10, R10; \
	ADD R3, R10, R10; \
	VLD1 (R10), [V8.S4, V9.S4]; \
	DOT_IQ1S()

// func dotIQ1SRowNeon(x *float32, raw *byte, f16 *float32, grid *float32, oddScales *float32, blocks int) float64
TEXT ·dotIQ1SRowNeon(SB), NOSPLIT, $0-56
	MOVD x+0(FP), R0
	MOVD raw+8(FP), R1
	MOVD f16+16(FP), R2
	MOVD grid+24(FP), R3
	MOVD oddScales+32(FP), R4
	MOVD blocks+40(FP), R7

	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16

block_loop:
	MOVHU (R1), R12
	LSL $2, R12, R12
	ADD R2, R12, R12
	VLD1R (R12), [V28.S4]
	ADD $2, R1, R13  // qs
	ADD $34, R1, R5 // qh
	MOVD $8, R6
groups:
	MOVHU (R5), R8
	ADD $2, R5, R5
	UBFX $15, R8, $1, R9
	LSL $16, R9, R9
	UBFX $12, R8, $3, R12
	LSL $2, R12, R12
	ADD R4, R12, R12
	VLD1R (R12), [V29.S4]
	WORD $0x6E3CDFBD // FMUL V29.4S, V29.4S, V28.4S
	GATHER_DOT_IQ1S(0)
	GATHER_DOT_IQ1S(3)
	GATHER_DOT_IQ1S(6)
	GATHER_DOT_IQ1S(9)
	SUBS $1, R6, R6
	BNE groups
	ADD $50, R1, R1
	SUBS $1, R7, R7
	BNE block_loop

	WORD $0x4E72D610 // FADD V16.2D, V16.2D, V18.2D
	WORD $0x4E73D631 // FADD V17.2D, V17.2D, V19.2D
	WORD $0x4E71D610 // FADD V16.2D, V16.2D, V17.2D
	WORD $0x7E70DA10 // FADDP V16.2D, F16
	FMOVD F16, ret+48(FP)
	RET
