//go:build arm64

#include "textflag.h"

// Convert sixteen signed ternary bytes in SRC to four float32 vectors, apply
// the current f16-derived block scale in V29, then widen weights and direct-F32
// activations before accumulating in eight float64 lanes.
#define DOT_TQ1(SRC) \
	VTBL V24.B16, [SRC.B16], V8.B16; \
	VTBL V25.B16, [SRC.B16], V9.B16; \
	VTBL V26.B16, [SRC.B16], V10.B16; \
	VTBL V27.B16, [SRC.B16], V11.B16; \
	WORD $0x4F28E508; /* SCVTF V8.4S, V8.4S, #24 */ \
	WORD $0x4F28E529; /* SCVTF V9.4S, V9.4S, #24 */ \
	WORD $0x4F28E54A; /* SCVTF V10.4S, V10.4S, #24 */ \
	WORD $0x4F28E56B; /* SCVTF V11.4S, V11.4S, #24 */ \
	WORD $0x6E3DDD08; /* FMUL V8.4S, V8.4S, V29.4S */ \
	WORD $0x6E3DDD29; /* FMUL V9.4S, V9.4S, V29.4S */ \
	WORD $0x6E3DDD4A; /* FMUL V10.4S, V10.4S, V29.4S */ \
	WORD $0x6E3DDD6B; /* FMUL V11.4S, V11.4S, V29.4S */ \
	VLD1 (R6), [V12.S4, V13.S4, V14.S4, V15.S4]; \
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
	VFMLA V5.D2, V7.D2, V19.D2; \
	WORD $0x0E617944; /* FCVTL  V4.2D, V10.2S */ \
	WORD $0x4E617945; /* FCVTL2 V5.2D, V10.4S */ \
	WORD $0x0E6179C6; /* FCVTL  V6.2D, V14.2S */ \
	WORD $0x4E6179C7; /* FCVTL2 V7.2D, V14.4S */ \
	VFMLA V4.D2, V6.D2, V16.D2; \
	VFMLA V5.D2, V7.D2, V17.D2; \
	WORD $0x0E617964; /* FCVTL  V4.2D, V11.2S */ \
	WORD $0x4E617965; /* FCVTL2 V5.2D, V11.4S */ \
	WORD $0x0E6179E6; /* FCVTL  V6.2D, V15.2S */ \
	WORD $0x4E6179E7; /* FCVTL2 V7.2D, V15.4S */ \
	VFMLA V4.D2, V6.D2, V18.D2; \
	VFMLA V5.D2, V7.D2, V19.D2

// Decode the current base-243 byte vector in V0. Multiplication by 3 wraps in
// byte lanes, and floor(3*q/256)-1 is produced by UHADD(q,q>>1)>>6 minus one,
// matching ggml's scalar mixed-radix decoder exactly.
#define EXTRACT_TQ1() \
	VORR V0.B16, V0.B16, V2.B16; \
	VUSHR $1, V2.B16, V5.B16; \
	WORD $0x6E250442; /* UHADD V2.16B, V2.16B, V5.16B */ \
	VUSHR $6, V2.B16, V2.B16; \
	VSUB V31.B16, V2.B16, V2.B16

// func dotTQ1RowNeon(x *float32, raw *byte, f16 *float32, indexes *byte, blocks int) float64
TEXT ·dotTQ1RowNeon(SB), NOSPLIT, $16-48
	MOVD x+0(FP), R0
	MOVD raw+8(FP), R1
	MOVD f16+16(FP), R2
	MOVD indexes+24(FP), R3
	MOVD blocks+32(FP), R7

	VLD1 (R3), [V24.B16, V25.B16, V26.B16, V27.B16]
	VMOVI $0x01, V31.B16
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16

block_loop:
	ADD $52, R1, R5
	MOVHU (R5), R5
	LSL $2, R5, R5
	ADD R2, R5, R5
	VLD1R (R5), [V29.S4]

	// Three full sixteen-byte streams: qs[0:16], qs[16:32], and qs[32:48].
	MOVD $0, R11
stream_setup:
	CMP $0, R11
	BEQ stream_zero
	CMP $1, R11
	BEQ stream_one
	ADD $32, R1, R5
	ADD $640, R0, R6
	MOVD $64, R10
	B stream_ready
stream_zero:
	MOVD R1, R5
	MOVD R0, R6
	MOVD $128, R10
	B stream_ready
stream_one:
	ADD $16, R1, R5
	ADD $64, R0, R6
	MOVD $128, R10
stream_ready:
	VLD1 (R5), [V0.B16]
	MOVD $5, R8
trit_loop:
	EXTRACT_TQ1()
	DOT_TQ1(V2)
	VADD V0.B16, V0.B16, V5.B16
	VADD V0.B16, V5.B16, V0.B16
	ADD R10, R6, R6
	SUBS $1, R8, R8
	BNE trit_loop
	ADD $1, R11, R11
	CMP $3, R11
	BNE stream_setup

	// qh[0:4] encodes four four-trit streams. Pack their decoded low words
	// into one vector so the final sixteen consecutive activations use DOT_TQ1.
	ADD $48, R1, R5
	MOVWU (R5), R9
	VEOR V0.B16, V0.B16, V0.B16
	VMOV R9, V0.S[0]
	MOVD RSP, R12
	MOVD $4, R8
tail_loop:
	EXTRACT_TQ1()
	VMOV V2.S[0], R9
	MOVW R9, (R12)
	ADD $4, R12, R12
	VADD V0.B16, V0.B16, V5.B16
	VADD V0.B16, V5.B16, V0.B16
	SUBS $1, R8, R8
	BNE tail_loop
	VLD1 (RSP), [V2.B16]
	ADD $960, R0, R6
	DOT_TQ1(V2)

	ADD $1024, R0, R0
	ADD $54, R1, R1
	SUBS $1, R7, R7
	BNE block_loop

	WORD $0x4E72D610 // FADD V16.2D, V16.2D, V18.2D
	WORD $0x4E73D631 // FADD V17.2D, V17.2D, V19.2D
	WORD $0x4E71D610 // FADD V16.2D, V16.2D, V17.2D
	WORD $0x7E70DA10 // FADDP V16.2D, F16
	FMOVD F16, ret+40(FP)
	RET
