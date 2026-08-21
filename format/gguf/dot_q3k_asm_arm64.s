//go:build arm64

#include "textflag.h"

// Convert sixteen signed q3 values in SRC to four f32 vectors, multiply by
// the current sub-block coefficient, load sixteen activations, and accumulate
// four independent vectors. Keeping four accumulators breaks the single-FMLA
// dependency chain without changing element mapping.
#define DOT_Q3(SRC) \
	VTBL V24.B16, [SRC.B16], V8.B16; \
	VTBL V25.B16, [SRC.B16], V9.B16; \
	VTBL V26.B16, [SRC.B16], V10.B16; \
	VTBL V27.B16, [SRC.B16], V11.B16; \
	WORD $0x4F28E508; /* SCVTF V8.4S, V8.4S, #24 */ \
	WORD $0x4F28E529; /* SCVTF V9.4S, V9.4S, #24 */ \
	WORD $0x4F28E54A; /* SCVTF V10.4S, V10.4S, #24 */ \
	WORD $0x4F28E56B; /* SCVTF V11.4S, V11.4S, #24 */ \
	WORD $0x6E34DD08; /* FMUL V8.4S, V8.4S, V20.4S */ \
	WORD $0x6E34DD29; /* FMUL V9.4S, V9.4S, V20.4S */ \
	WORD $0x6E34DD4A; /* FMUL V10.4S, V10.4S, V20.4S */ \
	WORD $0x6E34DD6B; /* FMUL V11.4S, V11.4S, V20.4S */ \
	VLD1 (R4), [V12.S4, V13.S4, V14.S4, V15.S4]; \
	VFMLA V8.S4, V12.S4, V16.S4; \
	VFMLA V9.S4, V13.S4, V17.S4; \
	VFMLA V10.S4, V14.S4, V18.S4; \
	VFMLA V11.S4, V15.S4, V19.S4

// Build the next two signed q3 groups. V0/V1 supply two low bits; the low
// bit currently visible in V6/V7 is promoted to bit two. Adding 0xfc then
// performs q-4 in byte arithmetic, exactly matching the inverted scalar
// high-mask rule. Shifting the source planes advances to the next sub-block.
#define EXTRACT_Q3() \
	VAND V22.B16, V0.B16, V2.B16; \
	VAND V22.B16, V1.B16, V3.B16; \
	VAND V21.B16, V6.B16, V28.B16; \
	VSHL $2, V28.B16, V28.B16; \
	VADD V28.B16, V2.B16, V2.B16; \
	VADD V23.B16, V2.B16, V2.B16; \
	VAND V21.B16, V7.B16, V29.B16; \
	VSHL $2, V29.B16, V29.B16; \
	VADD V29.B16, V3.B16, V3.B16; \
	VADD V23.B16, V3.B16, V3.B16; \
	VUSHR $2, V0.B16, V0.B16; \
	VUSHR $2, V1.B16, V1.B16; \
	VUSHR $1, V6.B16, V6.B16; \
	VUSHR $1, V7.B16, V7.B16

// func dotQ3KBlockNeon(x *float32, raw *byte, coeff *float32, indexes *byte) float32
TEXT ·dotQ3KBlockNeon(SB), NOSPLIT, $0-36
	MOVD x+0(FP), R0
	MOVD raw+8(FP), R1
	MOVD coeff+16(FP), R2
	MOVD indexes+24(FP), R3

	VLD1 (R3), [V24.B16, V25.B16, V26.B16, V27.B16]
	VMOVI $0x01, V21.B16
	VMOVI $0x03, V22.B16
	VMOVI $0xfc, V23.B16
	VLD1 (R1), [V6.B16, V7.B16]
	ADD $32, R1, R1
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	MOVD $2, R7

quant_half:
	VLD1 (R1), [V0.B16, V1.B16]
	ADD $32, R1, R1
	MOVD $4, R8

subblocks:
	EXTRACT_Q3()
	VLD1R.P 4(R2), [V20.S4]
	MOVD R0, R4
	DOT_Q3(V2)
	VLD1R.P 4(R2), [V20.S4]
	ADD $64, R0, R4
	DOT_Q3(V3)
	ADD $128, R0, R0
	SUBS $1, R8, R8
	BNE subblocks

	SUBS $1, R7, R7
	BNE quant_half

	WORD $0x4E31D610 // FADD V16.4S, V16.4S, V17.4S
	WORD $0x4E33D652 // FADD V18.4S, V18.4S, V19.4S
	WORD $0x4E32D610 // FADD V16.4S, V16.4S, V18.4S
	WORD $0x6E30D610 // FADDP V16.4S, V16.4S, V16.4S
	WORD $0x6E30D610 // FADDP V16.4S, V16.4S, V16.4S
	FMOVS F16, ret+32(FP)
	RET
