//go:build arm64

#include "textflag.h"

// Multiply one pre-adjusted eight-wide ternary-grid row by the current odd
// coefficient, load the matching activations, widen both operands to float64,
// and feed four independent accumulators.
#define DOT_IQ1M() \
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

#define GATHER_DOT_IQ1M(OFFSET) \
	MOVBU (R14), R10; \
	ADD $1, R14, R14; \
	LSL $5, R10, R10; \
	UBFX $OFFSET, R9, $32, R11; \
	ADD R11, R10, R10; \
	ADD R3, R10, R10; \
	VLD1 (R10), [V8.S4, V9.S4]; \
	DOT_IQ1M()

#define PAIR_IQ1M(SCALE) \
	UBFX $SCALE, R8, $3, R12; \
	LSL $2, R12, R12; \
	ADD R4, R12, R12; \
	VLD1R (R12), [V29.S4]; \
	WORD $0x6E3CDFBD; /* FMUL V29.4S, V29.4S, V28.4S */ \
	MOVBU (R13), R9; \
	ADD $1, R13, R13; \
	LSL $3, R9, R9; \
	ADD R16, R9, R9; \
	MOVD (R9), R9; \
	GATHER_DOT_IQ1M(0); \
	GATHER_DOT_IQ1M(32)

// func dotIQ1MRowNeon(x *float32, raw *byte, f16 *float32, grid *float32, oddScales *float32, qhOffsets *uint32, blocks int) float64
TEXT ·dotIQ1MRowNeon(SB), NOSPLIT, $0-64
	MOVD x+0(FP), R0
	MOVD raw+8(FP), R1
	MOVD f16+16(FP), R2
	MOVD grid+24(FP), R3
	MOVD oddScales+32(FP), R4
	MOVD qhOffsets+40(FP), R16
	MOVD blocks+48(FP), R7

	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16

block_loop:
	// Reassemble the f16 super-scale from the top nibble of each scale word.
	MOVHU 48(R1), R15
	LSR $12, R15, R15
	MOVHU 50(R1), R12
	LSR $12, R12, R12
	LSL $4, R12, R12
	ORR R12, R15, R15
	MOVHU 52(R1), R12
	LSR $12, R12, R12
	LSL $8, R12, R12
	ORR R12, R15, R15
	MOVHU 54(R1), R12
	LSR $12, R12, R12
	LSL $12, R12, R12
	ORR R12, R15, R15
	LSL $2, R15, R12
	ADD R2, R12, R12
	VLD1R (R12), [V28.S4]

	MOVD R1, R14      // qs
	ADD $32, R1, R13 // qh
	ADD $48, R1, R5  // scales
	MOVD $4, R6
scale_groups:
	MOVHU (R5), R8
	ADD $2, R5, R5
	PAIR_IQ1M(0)
	PAIR_IQ1M(3)
	PAIR_IQ1M(6)
	PAIR_IQ1M(9)
	SUBS $1, R6, R6
	BNE scale_groups
	ADD $56, R1, R1
	SUBS $1, R7, R7
	BNE block_loop

	WORD $0x4E72D610 // FADD V16.2D, V16.2D, V18.2D
	WORD $0x4E73D631 // FADD V17.2D, V17.2D, V19.2D
	WORD $0x4E71D610 // FADD V16.2D, V16.2D, V17.2D
	WORD $0x7E70DA10 // FADDP V16.2D, F16
	FMOVD F16, ret+56(FP)
	RET
